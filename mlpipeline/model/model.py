import torch
import logging
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from mlpipeline.config import setup_logging, Config
from mlpipeline.gcp import GCP
from mlpipeline.model.prompt import (
    SYSTEM_MSG, 
    USER_TEMPLATE,
    ASSISTANT_TEMPLATE, 
    GENERATION_TEMPLATE, 
    )


logger = logging.getLogger(__name__)


class LLMWrapper:

    def __init__(self):
        try:
            logger.info(f"LLMWrapper.__init__ : Loading model and tokenizer")

            adapter_dir = Config.DESTINATION_DIRECTORY + '/' + Config.ADAPTER_NAME
            if not os.path.exists(adapter_dir):
                GCP.load_adapter_gcs()

            self.model_name = Config.MODEL_NAME
            logger.info(f"Base model: {self.model_name}")
            logger.info(f"Adapter directory: {adapter_dir}")

            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            torch_dtype = torch.float16 if self.device=="mps" else torch.float32
            device_map="auto" if self.device!="cpu" else None
            logger.info(f"Using device={self.device}, dtype={torch_dtype}, device_map={device_map}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=device_map
            ).to(self.device)

            self.model = PeftModel.from_pretrained(
                self.base_model,
                adapter_dir,
                torch_dtype=torch_dtype,
                device_map=device_map,
            ).to(self.device)

            logger.info(f"✅ model loaded on {next(self.model.parameters()).device}")

        except Exception as e:
            logger.exception(f"❌ Error loading model: {e}.")


    def apply_chat_template_generation(self, quotes):
        formatted_texts = []

        for quote in quotes:
            messages = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": (USER_TEMPLATE+GENERATION_TEMPLATE).format(quote=quote)},
            ]

            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True # generation
            )

            formatted_texts.append(formatted_text)

        return formatted_texts


    def apply_chat_template_training(self, df):
        formatted_texts = []

        for quote, label, response in zip(df['quote'], df['label'], df['response']):
            messages = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": USER_TEMPLATE.format(quote=quote)},
                {"role": "assistant", "content": ASSISTANT_TEMPLATE.format(response=response, label=label)}
            ]

            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False # fine tuning
            )

            formatted_texts.append(formatted_text)

        return formatted_texts


    def generate(
        self,
        quote: str = "Who are you?",
        max_new_tokens: int = 2048,
    ):
        
        assert self.model is not None

        logger.info(f"LLMWrapper.generate : {quote}")

        self.model.eval()

        formatted_prompt = self.apply_chat_template_generation(quotes=quote)[0]

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

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


    def clear(self):
        try:
            del self.model
            del self.tokenizer

            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()

        except Exception:
            pass


if __name__ == "__main__":
    setup_logging()
    llm = LLMWrapper()
