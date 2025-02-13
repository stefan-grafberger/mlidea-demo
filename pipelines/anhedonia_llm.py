import pandas as pd
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma  # pylint: disable=no-name-in-module
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize

from pipelines.datasets.anhedonia_llm.pipeline_utils import initialize_seeds_and_llm_config, \
    get_langchain_rag_binary_classification
from utils import get_project_root

initialize_seeds_and_llm_config()

boolean_dictionary = {True: 'anhedonia', False: 'regular'}

def load_train_data(user_location, tweet_locations, included_countries):
    users = pd.read_parquet(user_location)
    users = users[users['country'].isin(included_countries)]

    tweets = []
    for tweet_location in tweet_locations:
        tweets.append(pd.read_parquet(tweet_location))
    tweets = pd.concat(tweets, ignore_index=True)

    return users.merge(tweets, on='user_id')

def load_test_data(test_location,included_countries):
    test = pd.read_parquet(test_location)
    test = test[test['country'].isin(included_countries)]
    return test

def weak_labeling(data):
    data['anhedonia'] = (
            (data['tweet'].str.contains(r'(0|no|zero) (motivation|interest)', regex=True)
             | data['tweet'].str.contains(r'(lose|losing|lost).{0,15} (interest|pleasure|motivation)', regex=True)
             # | data['tweet'].str.contains(r'I (no longer enjoy anything|was diagnosed with anhedonia)', regex=True)
            ) & ~data['tweet'].str.contains(r'recover.{0,15} from (0|no|zero) (motivation|interest)', regex=True)
    )
    # Addition to help the llm
    data['label'] = data['anhedonia'].replace(boolean_dictionary)
    return data

user_location = f'{str(get_project_root())}/pipelines/datasets/anhedonia_ml/data/users.pqt'
tweet_locations = [
    f'{str(get_project_root())}/pipelines/datasets/anhedonia_ml/data/tweets_march.pqt',
    # f'{str(get_project_root())}/pipelines/datasets/anhedonia_ml/data/tweets_april.pqt'
]
test_location = f'{str(get_project_root())}/pipelines/datasets/anhedonia_ml/data/expert_labeled.pqt'

included_countries = ['CAN']
# included_countries = ['CAN', 'US']

train = load_train_data(user_location, tweet_locations, included_countries=included_countries)
train = weak_labeling(train)
test = load_test_data(test_location, included_countries=included_countries)

vectorstore = Chroma.from_texts(texts=train['tweet'].to_list(), metadatas=train[['label']].to_dict('records'),
                                embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))

rag_chain = get_langchain_rag_binary_classification(list(boolean_dictionary.values()), vectorstore.as_retriever())

y_predicted = rag_chain.batch(test['tweet'].to_list(), test)
y_test_binarized = label_binarize(test['anhedonia'], classes=[True, False])
accuracy = accuracy_score(y_test_binarized, y_predicted)
print(f'Test accuracy is: {accuracy}')
