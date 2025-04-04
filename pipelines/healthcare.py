"""Predicting which patients are at a higher risk of complications"""
import warnings
import os

import pandas as pd
from fairlearn.metrics import equalized_odds_difference, MetricFrame, false_negative_rate
from scikeras.wrappers import KerasClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from example_pipelines.healthcare.healthcare_utils import MyW2VTransformer, \
    create_model

from utils import get_project_root

warnings.filterwarnings('ignore')

COUNTIES_OF_INTEREST = ['county2', 'county3']

patients = pd.read_csv(os.path.join(str(get_project_root()), "pipelines", "datasets", "healthcare",
                                    "patients.csv"), na_values='?')
histories = pd.read_csv(os.path.join(str(get_project_root()), "pipelines", "datasets", "healthcare",
                                     "histories.csv"), na_values='?')

data = patients.merge(histories, on=['ssn'])
complications = data.groupby('age_group') \
    .agg(mean_complications=('complications', 'mean'))
data = data.merge(complications, on=['age_group'])
data['label'] = data['complications'] > 1.2 * data['mean_complications']
data = data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income', 'label']]
data = data[data['county'].isin(COUNTIES_OF_INTEREST)]

impute_and_one_hot_encode = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])
featurisation = ColumnTransformer(transformers=[
    ("impute_and_one_hot_encode", impute_and_one_hot_encode, ['smoker', 'county', 'race']),
    ('word2vec', MyW2VTransformer(min_count=2), ['last_name']),
    ('numeric', StandardScaler(), ['num_children', 'income']),
], remainder='drop')
neural_net = KerasClassifier(model=create_model, epochs=10, batch_size=1, verbose=0,
                             hidden_layer_sizes=(9, 9,), loss="binary_crossentropy")
pipeline = Pipeline([
    ('features', featurisation),
    ('learner', neural_net)])

train_data, test_data = train_test_split(data)
model = pipeline.fit(train_data, train_data['label'])
test_predictions = model.predict(test_data)
print(f"Mean accuracy: {accuracy_score(test_data['label'], test_predictions)}")

sensitive_features = test_data[['race']]
sensitive_features['race'] = sensitive_features['race'].fillna("unknown")
fnr_by_group = MetricFrame(metrics=false_negative_rate, y_pred=test_predictions, y_true=test_data['label'],
                           sensitive_features=sensitive_features)
print(f"False-negative by group: {fnr_by_group.by_group}")
equalized_odds_diff = equalized_odds_difference(y_pred=test_predictions, y_true=test_data['label'],
                                                sensitive_features=sensitive_features)
print(f"Equalized odds difference: {equalized_odds_diff}")
