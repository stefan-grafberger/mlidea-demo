import re
import pandas as pd
from utils import get_project_root

# Regular expressions
BENGALI_PATTERN = re.compile(r'[\u0980-\u09FF]')
PHRASE_PATTERN = re.compile(r'(no longer enjoy anything|was diagnosed with anhedonia)', re.IGNORECASE)


user_path = f'{get_project_root()}/pipelines/datasets/anhedonia_ml/data/users.pqt'
test_path = f'{get_project_root()}/pipelines/datasets/anhedonia_ml/data/expert_labeled.pqt'
user_df = pd.read_parquet(user_path)
test_df = pd.read_parquet(test_path)

# Update test file (directly checking Bengali content)
user_df['country'] = user_df['country'].replace('NL', 'CAN')
user_df['country'] = user_df['country'].replace('IN', 'US')
user_df.to_parquet(user_path, index=False)

test_df['country'] = test_df['country'].replace('NL', 'CAN')
test_df['country'] = test_df['country'].replace('IN', 'US')
test_df.to_parquet(test_path, index=False)

print("Data update complete.")
