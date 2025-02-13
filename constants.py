import os

from utils import get_project_root


PIPELINE_CONFIG = {
    "anhedonia_llm": {
        "filename": os.path.join(str(get_project_root()), "pipelines", "anhedonia_llm.py"),
        "columns": [],
    },
    "anhedonia_ml": {
        "filename": os.path.join(str(get_project_root()), "pipelines", "anhedonia_ml.py"),
        "columns": [],
    },
    "healthcare": {
        "filename": os.path.join(str(get_project_root()), "pipelines", "healthcare.py"),
        "columns": ['smoker', 'last_name', 'county', 'num_children', 'race', 'income'],
    },
    # "reviews": {
    #     "filename": os.path.join(str(get_project_root()), "pipelines", "reviews.py"),
    #     "columns": ["total_votes", "star_rating", "vine", "category"],
    # },
    # "census": {
    #     "filename": os.path.join(str(get_project_root()), "pipelines", "census.py"),
    #     "columns": ['AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP'],
    # },
    "compas": {
        "filename": os.path.join(str(get_project_root()), "pipelines", "compas.py"),
        "columns": ['is_recid', 'c_charge_degree', 'age', 'priors_count'],
    },
    "adult": {
        "filename": os.path.join(str(get_project_root()), "pipelines", "adult_complex.py"),
        "columns": ['age','workclass','fnlwgt','education','education-num','marital-status','occupation',
                    'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country',
                    'income-per-year'],
    },
}
