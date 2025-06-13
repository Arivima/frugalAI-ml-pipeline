import torch
import logging
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from peft import PeftModel, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer

from mlpipeline.config import setup_logging, Config
from mlpipeline.gcp import GCP
from mlpipeline.model.prompt import PromptTemplate

logger = logging.getLogger(__name__)


class LLMWrapper:

    def __init__(
            self,
            local_directory : str,
            adapter_name : str,
            model_name : str,
            project_id : str,
            bucket_name : str,
        ):
        
        try:
            logger.info(f"Loading model and tokenizer")

            # loading the adapter from gcs if not already done
            self.local_directory = local_directory
            adapter_dir = local_directory + '/' + adapter_name
            if os.path.exists(adapter_dir):
                logger.info(f"Loading adapter from local cache")
            else:
                logger.info(f"Loading adapter from gcs")
                GCP.load_adapter_gcs(
                    project_id = project_id,
                    bucket_name = bucket_name,
                    adapter_name = adapter_name,
                    local_directory= local_directory ,
                )

            self.model_name = model_name
            logger.info(f"Base model: {self.model_name}")
            logger.info(f"Adapter directory: {adapter_dir}")

            # setting device, precision, device map
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            self.torch_dtype = torch.float16 if self.device=="mps" else torch.float32
            device_map="auto" if self.device!="cpu" else None
            logger.info(f"Using device={self.device}, dtype={self.torch_dtype}, device_map={device_map}")

            # loading base model from hugging face
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.base_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path = self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=device_map
            ).to(self.device)

            # loading the adapter from local directory
            self.model = PeftModel.from_pretrained(
                model=self.base_model,
                model_id=adapter_dir,
                torch_dtype=self.torch_dtype,
                device_map=device_map,
            ).to(self.device)

            logger.info(f"✅ model loaded on {next(self.model.parameters()).device}")

        except Exception as e:
            logger.exception(f"❌ Error loading model: {e}.")


    def _apply_chat_template_generation(
            self, 
            quotes
        ):

        formatted_texts = []

        # only system and user prompts
        sys_msg = PromptTemplate.SYSTEM_MSG
        prompt = PromptTemplate.USER_TEMPLATE + PromptTemplate.GENERATION_TEMPLATE

        for quote in quotes:
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt.format(quote=quote)},
            ]

            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True # for generation
            )

            formatted_texts.append(formatted_text)

        return formatted_texts


    def _apply_chat_template_training(
            self, 
            ds : Dataset
        ):
        """
            - Takes a dataset object containing 'quote', 'label', and 'response' fields
            - Formats each example into a conversation using prefedined prompt templates
            - Applies the tokenizer's chat template
            - Returns a dictionary with a 'text' key containing the formatted examples
        """
            
        formatted_texts = []

        # fully filled Q/A : system, user and assistant prompt
        for quote, label, response in zip(ds['text'], ds['label_pred'], ds['explanation']):
            messages = [
                {"role": "system", "content": PromptTemplate.SYSTEM_MSG},
                {"role": "user", "content": PromptTemplate.USER_TEMPLATE.format(quote=quote)},
                {"role": "assistant", "content": PromptTemplate.ASSISTANT_TEMPLATE.format(response=response, label=label)}
            ]

            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False # for fine tuning
            )

            formatted_texts.append(formatted_text)

        return {'text': formatted_texts}


    def generate(
        self,
        quote: str = "Climate change is not happening",
        max_new_tokens: int = 2048,
    ):
        """
        Generate classification (category and explanation) from quote
        Returns category and explanation
        """
        assert self.model is not None

        logger.info(f"LLMWrapper.generate : {quote}")

        formatted_prompt = self._apply_chat_template_generation(quotes=quote)[0]
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        self.model.eval()
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        answer = output_text.split('assistant')[1]
        m = re.search(r"\d", answer)
        if m:
            category = m.group(0)
            explanation = answer.split(category)[1].strip()
        else:
            category = ''
            explanation = answer
        logger.info(f"category: {category}")
        logger.info(f"explanation: {explanation}")

        return category, explanation
    




    def train(
            self,
            data_train : Dataset,
            #data_val : Dataset
            ):
        
        # format training set with chat template
        formatted_ds = self._apply_chat_template_training(ds=data_train)
        logger.info(f'formatted train_ds sample {formatted_ds["text"][0]}')

        # tokenize
        tokenized = self.tokenizer(
            formatted_ds["text"],
            padding="max_length",
            truncation=True,
            max_length=1024,
        )
        tokenized["labels"] = tokenized["input_ids"] 
        tokenized_ds = Dataset.from_dict(tokenized) 


        # resume training on the current adapter
        model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=True
        )
        logger.info('model prepared')


        # override the default DataCollatorForLanguageModeling because we do a Sequence-to-Sequence task
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            return_tensors="pt",
            padding=True
        )

        logger.info('data_collator')

        is_bf16 = hasattr(torch.backends.mps, "is_bf16_supported") and torch.backends.mps.is_bf16_supported()

        training_args = TrainingArguments(
            output_dir="outputs/continue_adapter",
            per_device_train_batch_size=2,  # Number of examples per GPU/CPU during training
            gradient_accumulation_steps=4,  # Number of updates steps to accumulate before performing a backward/update pass.
                                            # Increases batch size to 2*4=8 without increasing memory usage
            #num_train_epochs=1
            max_steps=5,
            warmup_steps=2,
            learning_rate=2e-4,
            fp16=False,                         # MPS/CPU don’t support fp16
            bf16=is_bf16,
            logging_steps=10,
            optim="adamw_torch",                # native optimizer for MPS/CPU    #used "adamw_8bit" for the first adapter
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            #report_to = "wandb",
        )
        logger.info(f'training_args {training_args is not None}')

        trainer = SFTTrainer(
            model=model,
            train_dataset=tokenized_ds,
            data_collator=data_collator,
            args=training_args,
        )
        logger.info('trainer')

        self.model.train()

        stats = trainer.train()
        logger.info('trained !')
        print(stats)

        # save to mlflow



    def evaluate(
            self,
            ):
        pass



    def clear(self):
        """Free model and tokenizer from RAM"""
        try:
            del self.model
            del self.tokenizer

            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
            logger.info(f"Cleared model from memory")

        except Exception:
            pass




if __name__ == "__main__":

    setup_logging()

    llm = LLMWrapper(
        local_directory = Config.LOCAL_DIRECTORY,
        adapter_name = Config.ADAPTER_NAME,
        model_name = Config.MODEL_NAME,
        project_id = Config.GCP_PROJECT_ID,
        bucket_name = Config.GCS_BUCKET_NAME,
    )
    # llm.generate()

    from mlpipeline.data.data_processor import DataProcessor

    data = DataProcessor(
        project_id = Config.GCP_PROJECT_ID,
        dataset_id = Config.BQ_DATASET_ID,
        table_id = Config.BQ_TABLE_ID,
        start_date = None,
    )
    data.create_splits()
    print(data.df.shape, data.ds.shape, data.train_ds.shape, data.val_ds.shape, data.test_ds.shape)
    print(data.df.columns)
    print(data.train_ds[0])

    llm.train(
        data_train=data.train_ds
    )
