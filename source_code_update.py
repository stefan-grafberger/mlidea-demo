from langchain_openai import ChatOpenAI

from pipelines.datasets.anhedonia_llm.pipeline_utils import initialize_seeds_and_llm_config

def get_source_code_integration(source_code, suggestion):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    result = llm.predict(f"""
        Can you please integrate a Python code snippet into my existing pipeline? Please directly reply with Python code only with the updated pipeline. Please don't wrap your response with backticks. I want to do the update directly in the column transformer, if possible, by using a scikit-learn pipeline, if the pipeline uses traditional ML and not an LLM. If it is a LLM pipeline, please apply the FunctionTransformer via `.fit_transform` directly to the dataframe column that is used to construct the LLM prompt and make sure to also update the test data with the `.transform` function. Please make sure that all code from the original pipeline is still included and the result is still executable.
    
        Here is an example:
        new_transformer = ...
        old_transformer = ...
        new_pipeline_transformer = Pipeline([
            ('new', new_transformer),
            ('old', old_transformer)
        ])
    
        And then the new_pipeline_transformer can be used in the ColumnTransformer instead of the old_transformer.
    
        __
        My current code:
        {source_code}
    
        __
        The code snippet to integrate:
        {suggestion}
        """)
    return result