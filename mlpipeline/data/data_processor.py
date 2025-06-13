# - load from BQ
# - preprocess
# - train test split
# - tokenization

import logging
from sklearn.model_selection import train_test_split
from datasets import Dataset, ClassLabel
from mlpipeline.gcp import GCP
from mlpipeline.config import setup_logging, Config


logger = logging.getLogger(__name__)

def mock_df():
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import random

    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Define climate disinformation categories and example texts
    climate_texts = {
        0: [  # Category 0: Climate denial
            "Climate change is a hoax perpetrated by scientists",
            "Global warming is completely natural and not caused by humans",
            "There's no scientific consensus on climate change",
            "The climate has always changed naturally throughout history",
            "CO2 is plant food, not a pollutant",
            "Climate scientists are just in it for the grant money",
            "The greenhouse effect doesn't exist",
            "Temperature records are unreliable and manipulated",
            "Climate change is a conspiracy to control the economy",
            "Ice ages prove climate always changes without human influence",
            "Volcanoes produce more CO2 than humans ever could",
            "Climate models are completely unreliable and always wrong"
        ],
        1: [  # Category 1: Extreme weather denial
            "Hurricanes have always existed, nothing to do with climate",
            "Wildfires are caused by poor forest management, not climate",
            "Heat waves are just normal summer weather patterns",
            "Floods have nothing to do with climate change",
            "Droughts are cyclical and always happened historically",
            "Tornadoes are not becoming more intense due to warming",
            "Arctic ice melting is just a natural cycle",
            "Sea level rise is barely measurable and insignificant",
            "Extreme weather events are actually decreasing over time",
            "Storm intensity hasn't changed in the past century",
            "Glaciers have been melting since the last ice age",
            "Ocean acidification is a myth propagated by environmentalists"
        ],
        2: [  # Category 2: Climate solutions denial
            "Renewable energy is too expensive and unreliable",
            "Solar panels create more pollution than they prevent",
            "Wind turbines kill more birds than fossil fuels",
            "Electric cars are worse for the environment than gas cars",
            "Carbon taxes will destroy the economy without helping climate",
            "Green energy transitions will cause massive job losses",
            "Nuclear power is too dangerous to be a climate solution",
            "Carbon capture technology is completely ineffective",
            "Switching to renewables will make the power grid unstable",
            "Climate action will make energy unaffordable for the poor",
            "Biofuels cause more environmental damage than fossil fuels",
            "Energy efficiency measures don't actually save energy"
        ],
        3: [  # Category 3: Climate impacts minimization
            "Climate change will be beneficial for agriculture",
            "Warmer temperatures will reduce winter deaths",
            "CO2 fertilization will make plants grow better",
            "Climate change impacts are centuries away, not urgent",
            "Humans can easily adapt to any climate changes",
            "Economic growth will outpace climate damages",
            "Technology will solve climate problems automatically",
            "Climate change will open new shipping routes in Arctic",
            "Warmer climates historically supported larger populations",
            "Climate impacts are exaggerated by alarmist scientists",
            "Migration due to climate change is a minor issue",
            "Climate change will make northern regions more habitable"
        ],
        4: [  # Category 4: Anti-climate policy
            "Climate policies are government overreach and tyranny",
            "Carbon regulations will make businesses uncompetitive",
            "Climate action is a threat to personal freedom",
            "Environmental regulations kill jobs and hurt workers",
            "Climate policies favor elite interests over regular people",
            "Green New Deal is socialist propaganda disguised as climate policy",
            "Climate regulations will make energy poverty widespread",
            "International climate agreements threaten national sovereignty",
            "Climate policies are designed to redistribute wealth globally",
            "Environmental protection hurts economic development",
            "Climate action will disproportionately harm rural communities",
            "Emissions standards are impossible for small businesses to meet"
        ],
        5: [  # Category 5: Pro-fossil fuel messaging
            "Fossil fuels lifted billions out of poverty",
            "Natural gas is clean energy that reduces emissions",
            "Coal provides reliable baseload power that renewables can't",
            "Oil and gas industry supports millions of good-paying jobs",
            "Fracking has made America energy independent",
            "Fossil fuel companies are investing heavily in clean technology",
            "Natural gas burns cleaner than other fossil fuels",
            "Oil and gas development boosts local economies",
            "Fossil fuels are essential for modern civilization",
            "Energy security requires domestic fossil fuel production",
            "Petrochemicals are essential for medicines and modern life",
            "Fossil fuel infrastructure represents trillions in investments"
        ],
        6: [  # Category 6: Climate science attack
            "Climate models have consistently been wrong about predictions",
            "The hockey stick graph was debunked years ago",
            "Climate data has been manipulated to show warming",
            "Medieval Warm Period was warmer than today",
            "Climategate emails proved scientists were conspiring",
            "Peer review process in climate science is corrupted",
            "Climate sensitivity to CO2 is much lower than claimed",
            "Urban heat island effect explains most warming trends",
            "Satellite data shows no significant warming trend",
            "Climate scientists suppress dissenting research",
            "IPCC reports are politically motivated, not scientific",
            "Temperature proxies are unreliable for historical climate"
        ],
        7: [  # Category 7: Delay and inaction
            "We need more research before taking drastic action",
            "Climate action should wait until developing countries act first",
            "Technology will solve climate change without lifestyle changes",
            "Economic costs of climate action outweigh the benefits",
            "Climate change is a long-term problem, not urgent",
            "Market forces will naturally drive clean energy adoption",
            "Individual actions can't make a difference on global scale",
            "We should adapt to climate change rather than prevent it",
            "Climate policies need to be gradual to avoid economic shock",
            "Innovation will make current climate concerns obsolete",
            "Climate action will hurt the poor more than help them",
            "We have decades to figure out climate solutions"
        ]
    }

    # Generate mock data
    data = []
    base_time = datetime.now()

    for category in range(8):
        texts = climate_texts[category]
        for i, text in enumerate(texts):
            # Generate timestamps with some variation
            time_offset = timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59),
                microseconds=random.randint(0, 999999)
            )
            
            # Generate predictions (0-7) with some realistic distribution
            label_pred = random.choices(range(8), weights=[1, 1, 1, 2, 1, 1, 1, 1])[0]
            
            data.append({
                'text': text,
                'label_pred': label_pred,
                'label_true': category,
                'explanation': 'blank for test',
                'created_at': (base_time - time_offset).isoformat() + '+00:00'
            })

    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    df['created_at'] = pd.to_datetime(df['created_at'])

    return df


# USE SCHEMA VALIDATION
class DataProcessor:
    def __init__(
            self,
            project_id:str,
            dataset_id:str,
            table_id:str,
            start_date:str,
            ):
        logger.info('Loading training dataset from bq')
        self.df = GCP.load_data_bq(
            project_id = project_id,
            dataset_id = dataset_id,
            table_id = table_id,
            start_date = start_date,
        )
        self.df = mock_df()
        self.ds = Dataset.from_pandas(self.df)
        self.train_ds = None
        self.test_ds = None
        self.val_ds = None

    def create_splits(self, test_size=0.2):
        logger.info("create_splits")

        try:
            if not isinstance(self.ds.features["label_true"], ClassLabel):
                unique_labels = sorted(self.ds.unique("label_true"))
                class_label = ClassLabel(names=unique_labels)
                self.ds = self.ds.cast_column("label_true", class_label)

            split1 = self.ds.train_test_split(
                test_size=test_size,
                seed=0,
                stratify_by_column="label_true"
            )
            split2 = split1["train"].train_test_split(
                test_size=test_size,
                seed=0,
                stratify_by_column="label_true"
            )

            self.test_ds = split1["test"]
            self.train_ds = split2["train"]
            self.val_ds   = split2["test"]

            return self.train_ds, self.val_ds, self.test_ds

        except Exception as e:
            logger.error(f"Error in create_splits: {e}")
            raise




if __name__ == '__main__':

    setup_logging()

    data = DataProcessor(
        project_id = Config.GCP_PROJECT_ID,
        dataset_id = Config.BQ_DATASET_ID,
        table_id = Config.BQ_TABLE_ID,
        start_date = None,
    )
    data.create_splits()
    print(data.df.shape, data.ds.shape, data.train_df.shape, data.val_df.shape, data.test_df.shape)
