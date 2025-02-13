"""
Some useful utils for the project
"""
from __future__ import annotations

# import getpass
import os
import random
from inspect import cleandoc

import numpy
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.language_models import FakeListChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from utils import get_project_root, get_open_ai_key


def initialize_seeds_and_llm_config():
    seed = 42
    numpy.random.seed(seed)
    random.seed(seed)
    set_llm_cache(SQLiteCache(database_path=f"{str(get_project_root())}/pipelines/datasets/anhedonia_llm/"
                                            f"offline/.langchain.db"))
    # os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
    # os.environ["OPENAI_API_KEY"] = "not_required_because_of_sqlite_caching"
    os.environ["OPENAI_API_KEY"] = get_open_ai_key()
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"


def get_langchain_rag_binary_classification(classes, retriever):
    # Prompt template taken from skllm
    FEW_SHOT_CLF_PROMPT_TEMPLATE = cleandoc(f"""
        You will be provided with the following information:
        1. An arbitrary text sample. The sample is delimited with triple backticks.
        2. List of categories the text sample can be assigned to. The list is delimited with square brackets. The categories in the list are enclosed in the single quotes and comma separated.
        3. Examples of text samples and their assigned categories. The examples are delimited with triple backticks. The assigned categories are enclosed in a list-like structure. These examples are to be used as training data.

        Perform the following tasks:
        1. Identify to which category the provided text belongs to with the highest probability.
        2. Assign the provided text to that category.
        3. Provide your response in a JSON format containing a single key `label` and a value corresponding to the assigned category. Do not provide any additional information except the JSON.

        List of categories: {classes}

        Training data:
        {{context}}

        Text sample: ```{{question}}```

        Your JSON response:
        """)
    prompt = ChatPromptTemplate.from_template(FEW_SHOT_CLF_PROMPT_TEMPLATE)
    # Make sure this pipeline is executable in Github Actions, but also uses a real LLM locally if needed
    # if os.getenv("GITHUB_ACTIONS") != "true":
    # FIXME: Quick and ugly way to develop w/o unneeded LLM calls
    if True or (os.getenv("GITHUB_ACTIONS") != "not_required_because_of_sqlite_caching" and os.getenv("GITHUB_ACTIONS") != "true"):  # pylint: disable=condition-evals-to-constant
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    else:
        llm = FakeListChatModel(responses=[f"{{\"label\": \"{classes[0]}\"}}""", f"{{\"label\": \"{classes[1]}\"}}"""])

    def format_docs(docs):
        retrieved_formatted = "\n\n".join(
            f"```{doc.page_content}```\nassigned category: ['{doc.metadata['label']}']" for doc in
            docs)  # Here we can also use label!
        # TODO: Create a copy of these functions and then modify the metadata in the llm mislabel pipeline to
        #  also contain the document id in the metadata. We don't have access to the id here, but as metadata
        #  we can access a static variable with the index datastructure
        #  We can infer the query id based on the order that we call the langchain with, I guess these are not
        #  randomly shuffled or anything like that. But we should verify this.
        #  However, we might not always want to have this additional id assign overhead, but maybe its fine
        return retrieved_formatted

    def format_response(json_object):
        if not isinstance(json_object, dict) or not "label" in json_object:
            return random.choice([0, 1])
        assigned_label = json_object["label"]
        if assigned_label not in classes:
            return random.choice([0, 1])
        return classes.index(assigned_label)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | JsonOutputParser()
            | format_response
    )
    return rag_chain
