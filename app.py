import os

import streamlit as st
import streamlit_extras as ste
from mlidea.shadow_pipelines._data_errors import DataErrorRobustness
from mlidea.shadow_pipelines._label_errors import LabelErrors
from mlidea.shadow_pipelines._slices import FairnessSlices
from streamlit_ace import st_ace
from streamlit_extras import stylable_container
from streamlit_javascript import st_javascript

from callbacks import scan_pipeline, \
    remove_column_specific_state, render_full_size_dag, get_readable_shadow_header, render_dag_slot, \
    render_dag_node_details, sanitize_dataframe
from constants import PIPELINE_CONFIG
from dag_visualisation import get_original_simple_dag, get_legend
from source_code_update import get_source_code_integration
from utils import get_project_root, get_open_ai_key, metrics_to_str

DATABASE_PATH_FUNC_TRANSFORMER = f"{str(get_project_root())}/function_transformer_cache/.cache"

LABEL_ERRORS_CLEANING_BATCH_SIZE = 20

slices_nice_fix_strategy_names = {
    "Num: IQR + Mean Impute": "Outlier Imputation (IQR)",
    "Cat: Isolation Forest + Simple Impute": "Outlier Imputation (IF)",
    "Text: LLM Codegen": "LLM Codegen Text",
    "Text: Spellcheck": "Spellcheck Text",
}


if 'PIPELINE_SOURCE_CODE' not in st.session_state:
    st.session_state['PIPELINE_SOURCE_CODE'] = ''
    st.session_state['PIPELINE_SCENARIO'] = ''

if 'ANALYSIS_RESULT' not in st.session_state:
    st.session_state['ANALYSIS_RESULT'] = None

if 'DAG_EXTRACTION_RESULT' not in st.session_state:
    st.session_state['DAG_EXTRACTION_RESULT'] = None

if 'ESTIMATION_RESULT' not in st.session_state:
    st.session_state['ESTIMATION_RESULT'] = None

if 'analyses' not in st.session_state:
    st.session_state['analyses'] = {}

if "is_running" not in st.session_state:
    st.session_state.is_running = False

if 'shadow_pipelines' not in st.session_state:
    label_errors = LabelErrors(proxy_model=True, cleaning_batch_size=LABEL_ERRORS_CLEANING_BATCH_SIZE)
    data_errors = DataErrorRobustness(corruption_significant_relative_threshold=0.95,
                                      database_path=DATABASE_PATH_FUNC_TRANSFORMER)
    slices = FairnessSlices(database_path=DATABASE_PATH_FUNC_TRANSFORMER)
    st.session_state['shadow_pipelines'] = [label_errors, data_errors, slices] # [label_errors, data_errors, slices]

st.set_page_config(page_title="mlidea", page_icon="🧐", layout="wide")
st.title("`mlidea` demo")

# st.markdown(
#     """
#     <style>
#     .stColumn {
#         background-color: rgb(40, 44, 52) !important;
#         border-radius: 0.5rem;
#         padding: 1rem;
#         border: 2px solid rgba(255, 255, 255, 0.1);
#     }
#     .stCustomComponentV1 {
#         background-color: rgb(40, 44, 52) !important;
#         border-radius: 0.5rem;
#         padding: 1rem;
#         border: 2px solid rgba(255, 255, 255, 0.1);
#     }
#
#     .stCustomComponentV1 body {
#         background-color: rgb(40, 44, 52) !important;
#     }
#
#     .stCustomComponentV1 .__________cytoscape_container {
#         background-color: rgb(40, 44, 52) !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

st.markdown(
    """
    <style>
    .stApp {
        background-color: #445056;
    }
    .st-key-main-column-0 {
        background-color:  #2a363d;
        border-radius: 0.5rem;
        padding: calc(-1px + 1rem);
    }
    
    .st-key-pipeline-code-output-container *:not(.stCode *) {
        border-radius: 0.5rem;
        padding: 2px;
    }
    
    .st-key-pipeline-code-run-button-container button {
        background-color:  #445056 !important;
    }
    
    .st-key-pipeline-code-run-button-container button p {
        background-color:  #445056 !important;
        font-color: black;
    }
    
     .st-key-pipeline-apply-llm-integrate-button-container-1-0-0 button {
        background-color:  #445056 !important;
    }
    
    .st-key-pipeline-apply-llm-integrate-button-container-1-0-0 button p {
        background-color:  #445056 !important;
        font-color: black;
    }
    
    .st-key-pipeline-apply-llm-integrate-button-container-1-1-0 button {
        background-color:  #445056 !important;
    }
    
    .st-key-pipeline-apply-llm-integrate-button-container-1-1-0 button p {
        background-color:  #445056 !important;
        font-color: black;
    }
    
    .st-key-pipeline-apply-llm-integrate-button-container-1-2-0 button {
        background-color:  #445056 !important;
    }
    
    .st-key-pipeline-apply-llm-integrate-button-container-1-2-0 button p {
        background-color:  #445056 !important;
        font-color: black;
    }
    
    .st-key-pipeline-apply-llm-integrate-button-container-2-0-0 button {
        background-color:  #445056 !important;
    }
    
    .st-key-pipeline-apply-llm-integrate-button-container-2-0-0 button p {
        background-color:  #445056 !important;
        font-color: black;
    }
    
    .st-key-pipeline-apply-llm-integrate-button-container-2-0-1 button {
        background-color:  #445056 !important;
    }
    
    .st-key-pipeline-apply-llm-integrate-button-container-2-0-1 button p {
        background-color:  #445056 !important;
        font-color: black;
    }
    
    .st-key-pipeline-apply-llm-run-button-container-1-0 button {
        background-color:  #445056 !important;
    }
    
    .st-key-pipeline-apply-llm-run-button-container-1-0 button p {
        background-color:  #445056 !important;
        font-color: black;
    }
    
    .st-key-pipeline-apply-llm-run-button-container-2-0 button {
        background-color:  #445056 !important;
    }
    
    .st-key-pipeline-apply-llm-run-button-container-2-0 button p {
        background-color:  #445056 !important;
        font-color: black;
    }
    
    .st-key-pipeline-show-prompt-button-container-1-0 button {
        background-color:  #445056 !important;
    }
    
    .st-key-pipeline-show-prompt-button-container-1-0 button p {
        background-color:  #445056 !important;
        font-color: black;
    }
    
    .st-key-pipeline-show-prompt-button-container-2-0 button {
        background-color:  #445056 !important;
    }
    
    .st-key-pipeline-show-prompt-button-container-2-0 button p {
        background-color:  #445056 !important;
        font-color: black;
    }
    
    .st-key-main-column-1 {
        background-color:  #2a363d;
        border-radius: 0.5rem;
        padding: calc(-1px + 1rem);
    }
    
    .st-key-main-column-2 {
        background-color:  #2a363d;
        border-radius: 0.5rem;
        padding: calc(-1px + 1rem);
    }
    
    .stAppHeader {
        background-color:  #445056;
    }
    
    .st-key-inner-pipeline-code-container *:not(.stButton .stButton>button .stButton>button>div):not(.stCode *)  {
        background-color:  #1d2329;
        border-color: black;
    }
    
    .st-key-inner-dag-container *:not(.legend *):not(.stSpinner):not(.stSpinner *) {
        background-color:  #1d2329 !important;
        border-color: black;
    }
    
    .st-key-shadow-pipeline-container-0 *:not(.stDataFrameGlideDataEditor):not(.stDataFrameGlideDataEditor *):not(.stButton):not(.stButton>button):not(.stButton>button>div):not(.stButton>button>div>p):not(.stCode *) {
        background-color:  #1d2329 !important;
        border-color: black;
    }
    
    .st-key-shadow-pipeline-container-1 *:not(.stDataFrameGlideDataEditor):not(.stDataFrameGlideDataEditor *):not(.stButton):not(.stButton>button):not(.stButton>button>div):not(.stButton>button>div>p):not(.stCode *) {
        background-color:  #1d2329 !important;
        border-color: black;
    }
    
    .st-key-shadow-pipeline-container-2 *:not(.stDataFrameGlideDataEditor):not(.stDataFrameGlideDataEditor *):not(.stButton):not(.stButton>button):not(.stButton>button>div):not(.stButton>button>div>p):not(.stCode *) {
        background-color:  #1d2329 !important;
        border-color: black;
    }
    
    .st-key-shadow-issue-container-0-0 *:not(.stDataFrameGlideDataEditor):not(.stDataFrameGlideDataEditor *):not(.stButton):not(.stButton>button):not(.stButton>button>div):not(.stButton>button>div>p):not(.stCode *) {
        background-color: #171c21 !important;
        border-color: black;
        border-radius: 0.5rem;
    }
    
    .st-key-shadow-issue-container-0-0-0 *:not(.stDataFrameGlideDataEditor):not(.stDataFrameGlideDataEditor *):not(.stButton):not(.stButton>button):not(.stButton>button>div):not(.stButton>button>div>p):not(.stCode *) {
        background-color: #13171c !important;
        border-color: black;
        border-radius: 0.5rem;
    }
    
    .st-key-shadow-issue-container-1-0 *:not(.stDataFrameGlideDataEditor):not(.stDataFrameGlideDataEditor *):not(.stButton):not(.stButton>button):not(.stButton>button>div):not(.stButton>button>div>p):not(.stCode *) {
        background-color: #171c21 !important;
        border-color: black;
        border-radius: 0.5rem;
    }
    
    .st-key-shadow-issue-container-1-1 *:not(.stDataFrameGlideDataEditor):not(.stDataFrameGlideDataEditor *):not(.stButton):not(.stButton>button):not(.stButton>button>div):not(.stButton>button>div>p):not(.stCode *) {
        background-color: #171c21 !important;
        border-color: black;
        border-radius: 0.5rem;
    }
    
     .st-key-shadow-issue-container-1-2 *:not(.stDataFrameGlideDataEditor):not(.stDataFrameGlideDataEditor *):not(.stButton):not(.stButton>button):not(.stButton>button>div):not(.stButton>button>div>p):not(.stCode *) {
        background-color: #171c21 !important;
        border-color: black;
        border-radius: 0.5rem;
    }
    
    .st-key-shadow-issue-container-1-0-0 *:not(.stDataFrameGlideDataEditor):not(.stDataFrameGlideDataEditor *):not(.stButton):not(.stButton>button):not(.stButton>button>div):not(.stButton>button>div>p):not(.stCode *) {
        background-color: #13171c !important;
        border-color: black;
        border-radius: 0.5rem;
    }
    
    .st-key-shadow-issue-container-1-1-1 *:not(.stDataFrameGlideDataEditor):not(.stDataFrameGlideDataEditor *):not(.stButton):not(.stButton>button):not(.stButton>button>div):not(.stButton>button>div>p):not(.stCode *) {
        background-color: #13171c !important;
        border-color: black;
        border-radius: 0.5rem;
    }
    
    .st-key-shadow-issue-container-1-2-2 *:not(.stDataFrameGlideDataEditor):not(.stDataFrameGlideDataEditor *):not(.stButton):not(.stButton>button):not(.stButton>button>div):not(.stButton>button>div>p):not(.stCode *) {
        background-color: #13171c !important;
        border-color: black;
        border-radius: 0.5rem;
    }
    
     .st-key-shadow-issue-container-2-0 *:not(.stDataFrameGlideDataEditor):not(.stDataFrameGlideDataEditor *):not(.stButton):not(.stButton>button):not(.stButton>button>div):not(.stButton>button>div>p):not(.stCode *) {
        background-color: #171c21 !important;
        border-color: black;
        border-radius: 0.5rem;
    }
    
    .st-key-shadow-issue-container-2-0-0 *:not(.stDataFrameGlideDataEditor):not(.stDataFrameGlideDataEditor *):not(.stButton):not(.stButton>button):not(.stButton>button>div):not(.stButton>button>div>p):not(.stCode *) {
        background-color: #13171c !important;
        border-color: black;
        border-radius: 0.5rem;
    }
    
    .st-key-shadow-issue-container-2-0-1 *:not(.stDataFrameGlideDataEditor):not(.stDataFrameGlideDataEditor *):not(.stButton):not(.stButton>button):not(.stButton>button>div):not(.stButton>button>div>p):not(.stCode *) {
        background-color: #13171c !important;
        border-color: black;
        border-radius: 0.5rem;
    }
    
    .st-key-ace-code-editor-container {
        border: 2px solid #1d2329 !important;
        border-radius: 0.5rem !important;
        overflow: hidden; 
        background-color: #1d2329 !important
        
    }
    
    .st-key-ace-code-editor-container * {
        border-radius: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# setTimeout(function() {
#             console.log("Hello, world!");
#             const iframe = document.querySelector('iframe');
#             if (iframe) {
#                 const iframeDocument = iframe.contentWindow.document;
#
#                 // Wait until the iframe's content is accessible (i.e., ACE is rendered)
#                 const aceEditor = iframeDocument.querySelector('.ace_editor');
#                 if (aceEditor) {
#                     aceEditor.style.borderRadius = '1rem';
#                 }
#             }
#         }, 1000); // 1-second delay to give the iframe time to load

### === SIDEBAR / CONFIGURATION ===
st.sidebar.title("Configuration")

# Pipeline
if 'pipeline_file_name_index' not in st.session_state:
    st.session_state['pipeline_file_name_index'] = 0
pipeline = st.sidebar.selectbox("Choose a pipeline", list(PIPELINE_CONFIG.keys()), key="pipeline-selection",
                                index=st.session_state['pipeline_file_name_index'])
new_pipeline_index = list(PIPELINE_CONFIG.keys()).index(pipeline)
if new_pipeline_index != st.session_state['pipeline_file_name_index']:
    remove_column_specific_state()
st.session_state['pipeline_file_name_index'] = new_pipeline_index

pipeline_filename = PIPELINE_CONFIG[pipeline]["filename"]
pipeline_columns = PIPELINE_CONFIG[pipeline]["columns"]
if st.sidebar.button("Load source code", key="source-code-loading"):
    with open(pipeline_filename) as file:
        st.session_state['PIPELINE_SOURCE_CODE'] = file.read()
        st.session_state['PIPELINE_SCENARIO'] = new_pipeline_index
    st.session_state['ANALYSIS_RESULT'] = None
    st.session_state['DAG_EXTRACTION_RESULT'] = None
    st.session_state['ESTIMATION_RESULT'] = None
    # st.rerun()
# pipeline_num_lines = len(st.session_state['PIPELINE_SOURCE_CODE'].splitlines())

### === LAYOUT ===
left, middle, right = st.columns([4, 4, 3])

with left:
    with st.container(key="main-column-0"):
        pipeline_code_container = st.container()
        with pipeline_code_container:
            st.subheader("Code Editor")
with middle:
    with st.container(key="main-column-1"):
        st.subheader("DAG & Shadow Pipelines")
        with st.container(key="inner-dag-container"):
            dag_container = st.container(border=True)
with right:
    with st.container(key="main-column-2"):
        st.subheader("Warnings & Suggestions")
        results_container = st.container()

### === ACTIONS ===

with st.sidebar:
    st.markdown("")
    st.markdown("")


def show_original_dag():
    runtime_messages = ""
    original_selected = {'nodes': []}
    if st.session_state.ANALYSIS_RESULT:
        # left, right = st.columns([2, 1])
        # with left:
        # with st.expander(f"**Original Pipeline DAG**", expanded=True):
        runtime_orig = st.session_state.ANALYSIS_RESULT.runtime_info.original_pipeline_estimated
        instrumentation_orig = st.session_state.ANALYSIS_RESULT.runtime_info.original_pipeline_importing_and_monkeypatching
        st.markdown(f"**Original Pipeline (`{runtime_orig:.2f} ms`)**")
        with st.container(border=False):
            original_selected.update(render_full_size_dag("Original"))
            # TODO: Should we use humanize here?
            #  from humanize import naturalsize
            #  naturaldelta or something like that
            # runtime_messages += f"Instrumentation: `{instrumentation_orig:.2f} ms`  \n"
            # runtime_messages += f"Execution: `{runtime_orig:.2f} ms` "
            # st.markdown(runtime_messages)
    return original_selected

### === MAIN CONTENT ===
with pipeline_code_container:
    with st.container(key="inner-pipeline-code-container"):
        with st.container(border=True):
            # tab1, tab2 = st.tabs(["Source Code", "Extracted DAG"])

            # with tab1:
            # st.code(pipeline_code)
            # Check out more themes: https://github.com/okld/streamlit-ace/blob/main/streamlit_ace/__init__.py#L36-L43
            with st.container(key="ace-code-editor-container"):
                # st.markdown('<div style="height: 25px;"></div>', unsafe_allow_html=True)
                st.session_state['PIPELINE_SOURCE_CODE'] = st_ace(value=st.session_state['PIPELINE_SOURCE_CODE'],
                                  language="python",
                                  theme="katzenmilch",
                                  key=f"pipeline_ace_editor_{st.session_state['PIPELINE_SCENARIO']}",
                                  auto_update=True,
                                  font_size=12)
                # st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)

            with st.container(key="pipeline-code-run-button-container"):
                scan_button = st.button("Run Pipeline", key="scan-button", disabled=st.session_state['is_running'])
            if st.session_state.ANALYSIS_RESULT:
                with st.container(key="pipeline-code-output-container"):
                    st.code(st.session_state.ANALYSIS_RESULT.dag_extraction_info.captured_orig_pipeline_stdout)

            # with tab2:
            #     st.markdown("Or show the full DAG here?")
                # show_original_dag()
                # with right:
                    #     with st.expander(f"**Execution Stats**", expanded=True):


if scan_button and not st.session_state.is_running:
    st.session_state.is_running = True  # Mark as running
    st.rerun()  # Trigger a rerun to disable the button immediately

if st.session_state.is_running:
    with results_container:
        with st.container(key=f"shadow-pipeline-container-0"):
            with st.container(border=True):
                st.markdown("Waiting for pipeline execution...")
    with dag_container:
        with st.spinner("Running the pipeline..."):
            analysis_result = scan_pipeline(st.session_state.PIPELINE_SOURCE_CODE,
                                            st.session_state.ANALYSIS_RESULT,
                                            shadow_pipelines=st.session_state.shadow_pipelines,
                                            database_path=DATABASE_PATH_FUNC_TRANSFORMER)

        st.session_state.ANALYSIS_RESULT = analysis_result
        st.session_state['is_running'] = False
        st.balloons()
        st.rerun()

with dag_container:
    # st.markdown("**Shadow Pipelines**")
    if st.session_state.ANALYSIS_RESULT:
        # FIXME: Shadow Pipeline Exec
        dag_left, dag_right = st.columns([3, 3])
        selected = {'nodes': []}
        with dag_left:
            selected_update = show_original_dag()
            if len(selected['nodes']) == 0 and len(selected_update['nodes']) != 0:
                selected.update(selected_update)
        with dag_right:
            for shadow_pipeline, report in st.session_state.ANALYSIS_RESULT.shadow_pipelines_to_result_reports.items():
                with st.container(border=False):
                    # metrics_frame_columns = report.select_dtypes('object')
                    # for column in metrics_frame_columns:
                    #     if len(report) != 0 and isinstance(report[column].iloc[0], MetricFrame):
                    #         def format_metric_frame(row):
                    #             pandas_df = row[column].by_group.reset_index(drop=False)
                    #             pandas_groups = pandas_df.iloc[:, 0].tolist()
                    #             pandas_values = pandas_df.iloc[:, 1].tolist()
                    #             results = []
                    #             for group, value in zip(pandas_groups, pandas_values):
                    #                 results.append(f"'{group}': {value:.3f}")
                    #             return ", ".join(results)
                    #         report[column] = report.apply(format_metric_frame, axis=1)
                    # for column in list(report.columns):
                    #     if pandas.isna(report[column][0]):
                    #         report.loc[0, column] = "(original pipeline)"
                    #     if "percentage" in column or "lineno" in column:
                    #         def format_percentage_column(row):
                    #             number = row[column]
                    #             if type(number) == float and not pandas.isna(number):
                    #                 if "percentage" in column:
                    #                     number = number * 100
                    #                 result = str(int(number))
                    #                 if "percentage" in column:
                    #                     result += "%"
                    #             elif type(number) == str:
                    #                 result = number
                    #             else:
                    #                 result = "<NA>"
                    #             return result
                    #
                    #         report[column] = report.apply(format_percentage_column, axis=1)
                    # TODO: Map mislabel cleaning None back to LABELS
                        header = get_readable_shadow_header(shadow_pipeline)
                        runtime_generation = st.session_state.ANALYSIS_RESULT.runtime_info.shadow_pipeline_generation[
                            shadow_pipeline]
                        runtime_execution = st.session_state.ANALYSIS_RESULT.runtime_info.shadow_pipeline_execution[
                            shadow_pipeline]
                        runtime_training = \
                            st.session_state.ANALYSIS_RESULT.runtime_info.shadow_pipeline_execution_combined_model_training[
                                shadow_pipeline]

                        st.markdown(f"**{header} (`{runtime_execution:.2f} ms`)**")

                        # with st.expander(f"Shadow DAG", expanded=True):
                        internal_dag = st.session_state.ANALYSIS_RESULT.shadow_pipeline_to_dags[shadow_pipeline]
                        reuse_info = st.session_state.ANALYSIS_RESULT.dag_extraction_info.reuse_info
                        # update mlidea to track full reuse as well in the reuse info
                        # then use this to color the DAG nodes
                        visualisation_dag = get_original_simple_dag(internal_dag, reuse_info)
                        selected_update = render_dag_slot(header, visualisation_dag, f"full-size-{header}", height='200px',
                                                   description=False)
                        if len(selected['nodes']) == 0 and len(selected_update['nodes']) != 0:
                            selected.update(selected_update)
                            selected["dag"] = internal_dag
                        # else:
                        #     st.markdown("Select a DAG Node for details")
                        # runtime_messages = f"Generation: `{runtime_generation:.2f} ms`  \n"
                        # runtime_messages += f"Execution: `{runtime_execution:.2f} ms` "
                        # st.markdown(runtime_messages)
                        # f"`{runtime_training:.2f} ms` of this execution time were model retraining.  \n"
                        # f"  \n ")
        if len(selected['nodes']) != 0:
            render_dag_node_details(selected["dag"], selected, width=1000, table_instead_of_df=True)
        else:
            st.markdown(
                "<div style='text-align: center;'>Select a DAG Node for details</div>",
                unsafe_allow_html=True
            )
            st.write("")
            st.markdown(
                "<div style='text-align: center;'>DAG Node Color Legend</div>",
                unsafe_allow_html=True
            )
            st.markdown(get_legend(), unsafe_allow_html=True)
            st.write("")
    else:
        st.markdown("Run the code to see more")


with results_container:
    # st.markdown("**Shadow Pipelines**")
    if st.session_state.ANALYSIS_RESULT:
        # FIXME: Shadow Pipeline Exec
        for shadow_index, (shadow_pipeline, report) in enumerate(st.session_state.ANALYSIS_RESULT.shadow_pipelines_to_result_reports.items()):
            with st.container(key=f"shadow-pipeline-container-{shadow_index}"):
                with st.container(border=True):

                    custom_container_start = """
                                    <style>
                                    .my-container {
                                        min-height: 45px;
                                        vertical-align: top;
                                    }
                                    </style>
                                    <div class="my-container">
                                    </div>
                                    """
                    # custom_container_end = """
                    #                 </div>
                    #                 """
                    #
                    # st.markdown(custom_container_start, unsafe_allow_html=True)
                    # metrics_frame_columns = report.select_dtypes('object')
                    # for column in metrics_frame_columns:
                    #     if len(report) != 0 and isinstance(report[column].iloc[0], MetricFrame):
                    #         def format_metric_frame(row):
                    #             pandas_df = row[column].by_group.reset_index(drop=False)
                    #             pandas_groups = pandas_df.iloc[:, 0].tolist()
                    #             pandas_values = pandas_df.iloc[:, 1].tolist()
                    #             results = []
                    #             for group, value in zip(pandas_groups, pandas_values):
                    #                 results.append(f"'{group}': {value:.3f}")
                    #             return ", ".join(results)
                    #         report[column] = report.apply(format_metric_frame, axis=1)
                    # for column in list(report.columns):
                    #     if pandas.isna(report[column][0]):
                    #         report.loc[0, column] = "(original pipeline)"
                    #     if "percentage" in column or "lineno" in column:
                    #         def format_percentage_column(row):
                    #             number = row[column]
                    #             if type(number) == float and not pandas.isna(number):
                    #                 if "percentage" in column:
                    #                     number = number * 100
                    #                 result = str(int(number))
                    #                 if "percentage" in column:
                    #                     result += "%"
                    #             elif type(number) == str:
                    #                 result = number
                    #             else:
                    #                 result = "<NA>"
                    #             return result
                    #
                    #         report[column] = report.apply(format_percentage_column, axis=1)
                    # TODO: Map mislabel cleaning None back to LABELS

                    # st.markdown("")
                    # st.markdown("")

                    header = get_readable_shadow_header(shadow_pipeline)
                    st.markdown(f"**{header}**")

                    # st.dataframe(report)
                    if type(shadow_pipeline) == LabelErrors:
                        screened_count = len(report.screened_issues)
                        passed_count = len([issue for issue in report.screened_issues if not issue.issue_found])
                        passed_fraction = int(passed_count / screened_count * 100)
                        failed_fraction = int((screened_count - passed_count) / screened_count * 100)
                        # st.progress(detected_fraction)

                        circle_html = f"""
                        <style>
                        .container {{
                          display: flex;
                          align-items: center;
                          gap: 10px;  /* Spacing between text and circle */
                        }}
    
                        .circular-progress-label {{
                          width: 40px;
                          height: 40px;
                          border-radius: 50%;
                          background: conic-gradient(green 0% {passed_fraction}%, red {passed_fraction}% 100%);
                          display: flex;
                          align-items: center;
                          justify-content: center;
                          font-weight: bold;
                          font-size: 12px;
                          color: black;
                          margin: 10px;
                        }}
    
                        .container p {{
                          margin: 0;  /* Remove default margin */
                          line-height: 1.2;  /* Optional: adjust line-height if needed */
                        }}
                        </style>
    
                        <div class="container">
                            <p><b>Screening: {passed_count}</b> / <b>{screened_count}</b> passed!</p>
                            <div class="circular-progress-label">{passed_fraction}%</div>
                        </div>
                        """

                        st.markdown(f"{circle_html}", unsafe_allow_html=True)

                        for issue_index, screened_issue in enumerate(report.screened_issues):
                            if screened_issue.issue_found:
                                if screened_issue.suggestion_found:
                                    fix_found_text = " - suggestion"
                                else:
                                    fix_found_text = ""
                                with st.container(key=f"shadow-issue-container-{shadow_index}-{issue_index}"):
                                    with st.expander(f"{screened_issue.description} (failed){fix_found_text}",
                                                     expanded=False):
                                        st.markdown(f"Original pipeline score: `{report.orig_metric_results}`")
                                        if report.orig_proxy_results is not None and len(report.orig_proxy_results) > 0:
                                            st.markdown(f"Proxy pipeline score: `{report.orig_proxy_results}`")
                                        st.markdown(f"Top `{LABEL_ERRORS_CLEANING_BATCH_SIZE}` lowest Shapley Values:")
                                        st.dataframe(sanitize_dataframe(screened_issue.issue_df))

                                        if screened_issue.suggestion_found:
                                            st.markdown(f"Suggestions from `mlidea`:")

                                        all_tried_suggestions = len(screened_issue.issue_suggestions)
                                        working_suggestions = 0
                                        for suggestion_index, suggestion in enumerate(screened_issue.issue_suggestions):
                                            if suggestion.improves_score:
                                                with st.container(
                                                        key=f"shadow-issue-container-{shadow_index}-{issue_index}-{suggestion_index}"):
                                                    with st.container(border=True):
                                                        working_suggestions += 1
                                                        st.markdown(f"Suggestion: `{suggestion.suggestion}`")
                                                        st.markdown(
                                                            f"Improved pipeline score: `{metrics_to_str(suggestion.suggestion_metric_results)}`.")
                                                        st.markdown(f"Relative change original: "
                                                                    f"`{metrics_to_str(suggestion.suggestion_max_score_improvement)}`.")
                                                        if suggestion.suggestion_df is not None:
                                                            st.markdown(f"Flipped predictions due to change:")
                                                            st.dataframe(sanitize_dataframe(suggestion.suggestion_df))

                                        if all_tried_suggestions != working_suggestions:
                                            if screened_issue.suggestion_found:
                                                st.markdown(f"Other strategies 'mlidea' tried:")
                                            else:
                                                st.markdown(
                                                    f"Unfortunately, `mlidea` could not automatically find a potential "
                                                    f"solution. Here is what it tried:")

                                            for suggestion_index, suggestion in enumerate(
                                                    screened_issue.issue_suggestions):
                                                    if not suggestion.improves_score:
                                                        with st.container(
                                                                key=f"shadow-issue-container-{shadow_index}-{issue_index}-{suggestion_index}"):
                                                            with st.container(border=True):
                                                                st.markdown(f"Suggestion: `{suggestion.suggestion}`")
                            else:
                                with st.container(key=f"shadow-issue-container-{shadow_index}-{issue_index}"):
                                    with st.expander(f"{screened_issue.description} (passed)", expanded=False):
                                        st.markdown(f"Original pipeline score: `{report.orig_metric_results}`")
                                        if report.orig_proxy_results is not None and len(report.orig_proxy_results) > 0:
                                            st.markdown(f"Proxy pipeline score: `{report.orig_proxy_results}`")
                                        st.markdown(f"Top `{LABEL_ERRORS_CLEANING_BATCH_SIZE}` lowest Shapley Values:")
                                        st.dataframe(sanitize_dataframe(screened_issue.issue_df))

                                        if screened_issue.suggestion_found:
                                            st.markdown(f"Suggestions from `mlidea`:")

                                        all_tried_suggestions = len(screened_issue.issue_suggestions)
                                        working_suggestions = 0
                                        for suggestion_index, suggestion in enumerate(screened_issue.issue_suggestions):
                                            if suggestion.improves_score:
                                                with st.container(
                                                        key=f"shadow-issue-container-{shadow_index}-{issue_index}-{suggestion_index}"):
                                                    with st.container(border=True):
                                                        working_suggestions += 1
                                                        st.markdown(f"Suggestion: `{suggestion.suggestion}`")
                                                        st.markdown(
                                                            f"Improved pipeline score: `{metrics_to_str(suggestion.suggestion_metric_results)}`.")
                                                        st.markdown(f"Relative change original: "
                                                                    f"`{metrics_to_str(suggestion.suggestion_max_score_improvement)}`.")
                                                        if suggestion.suggestion_df is not None:
                                                            st.markdown(f"Flipped predictions due to change:")
                                                            st.dataframe(sanitize_dataframe(suggestion.suggestion_df))

                                        if all_tried_suggestions != working_suggestions:
                                            if screened_issue.suggestion_found:
                                                st.markdown(f"Other strategies 'mlidea' tried:")
                                            else:
                                                st.markdown(
                                                    f"Unfortunately, `mlidea` could not automatically find a potential "
                                                    f"solution. Here is what it tried:")

                                            for suggestion_index, suggestion in enumerate(
                                                    screened_issue.issue_suggestions):
                                                if not suggestion.improves_score:
                                                    with st.container(
                                                            key=f"shadow-issue-container-{shadow_index}-{issue_index}-{suggestion_index}"):
                                                        with st.container(border=True):
                                                            st.markdown(f"Suggestion: `{suggestion.suggestion}`")
                        for placeholder_index in range(3-len(report.screened_issues)):
                            st.markdown(custom_container_start, unsafe_allow_html=True)


                    elif type(shadow_pipeline) == DataErrorRobustness:
                        screened_count = len(report.screened_issues)
                        passed_count = len([issue for issue in report.screened_issues if not issue.issue_found])
                        passed_fraction = int(passed_count / screened_count * 100)
                        failed_fraction = int((screened_count - passed_count) / screened_count * 100)
                        # st.progress(detected_fraction)

                        circle_html = f"""
                                      <style>
                                      .container {{
                                        display: flex;
                                        align-items: center;
                                        gap: 10px;  /* Spacing between text and circle */
                                      }}
    
                                      .circular-progress-robustness {{
                                        width: 40px;
                                        height: 40px;
                                        border-radius: 50%;
                                        background: conic-gradient(green 0% {passed_fraction}%, red {passed_fraction}% 100%);
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                        font-weight: bold;
                                        font-size: 12px;
                                        color: black;
                                        margin: 10px;
                                      }}
    
                                      .container p {{
                                        margin: 0;  /* Remove default margin */
                                        line-height: 1.2;  /* Optional: adjust line-height if needed */
                                      }}
                                      </style>
    
                                      <div class="container">
                                          <p><b>Screening: {passed_count}</b> / <b>{screened_count}</b> passed!</p>
                                          <div class="circular-progress-robustness">{passed_fraction}%</div>
                                      </div>
                                      """

                        st.markdown(f"{circle_html}", unsafe_allow_html=True)

                        for issue_index, screened_issue in enumerate(report.screened_issues):
                            if screened_issue.issue_found:
                                if screened_issue.suggestion_found:
                                    fix_found_text = " - suggestion"
                                else:
                                    fix_found_text = ""
                                with st.container(key=f"shadow-issue-container-{shadow_index}-{issue_index}"):
                                    with st.expander(f"{screened_issue.description} (failed){fix_found_text}",
                                                     expanded=False):
                                        st.markdown(f"Original pipeline score: `{report.orig_metric_results}`")
                                        st.markdown(
                                            f"Corrupted pipeline score: `{screened_issue.issue_metrics_result}`")
                                        st.markdown(
                                            f"This is a decrease by up to: `{screened_issue.issue_max_score_decrease}`")
                                        st.markdown("Sample of corrupted rows:")
                                        st.dataframe(sanitize_dataframe(screened_issue.issue_df))

                                        if screened_issue.suggestion_found:
                                            st.markdown(f"Suggestions from `mlidea`:")

                                        all_tried_suggestions = len(screened_issue.issue_suggestions)
                                        working_suggestions = 0
                                        for suggestion_index, suggestion in enumerate(screened_issue.issue_suggestions):
                                            if suggestion.improves_score:
                                                with st.container(
                                                        key=f"shadow-issue-container-{shadow_index}-{issue_index}-{suggestion_index}"):
                                                    with st.container(border=True):
                                                        working_suggestions += 1
                                                        st.markdown(f"Suggestion: `{suggestion.suggestion}`")
                                                        st.markdown(
                                                            f"Improves score to `{suggestion.suggestion_metric_results}`.")
                                                        st.markdown(f"This is an improvement up to "
                                                                    f"`{suggestion.suggestion_max_score_improvement}`.")
                                                        if suggestion.suggestion_df is not None:
                                                            st.markdown(f"Sample of fixed rows:")
                                                            st.dataframe(sanitize_dataframe(suggestion.suggestion_df))
                                                        # if suggestion.source_code_to_integrate is not None:
                                                        #     st.markdown(f"Code suggestion:")
                                                        #     st.code(suggestion.source_code_to_integrate)

                                        if all_tried_suggestions != working_suggestions:
                                            if screened_issue.suggestion_found:
                                                st.markdown(f"Other strategies 'mlidea' tried:")
                                            else:
                                                st.markdown(
                                                    f"Unfortunately, `mlidea` could not automatically find a potential "
                                                    f"solution. Here is what it tried:")

                                            for suggestion_index, suggestion in enumerate(
                                                    screened_issue.issue_suggestions):
                                                if not suggestion.improves_score:
                                                    with st.container(
                                                            key=f"shadow-issue-container-{shadow_index}-{issue_index}-{suggestion_index}"):
                                                        with st.container(border=True):
                                                            st.markdown(f"Suggestion: `{suggestion.suggestion}`")
                            else:
                                with st.container(key=f"shadow-issue-container-{shadow_index}-{issue_index}"):
                                    with st.expander(f"{screened_issue.description} (passed)", expanded=False):
                                        st.markdown(f"Original pipeline score: `{report.orig_metric_results}`")
                                        st.markdown(
                                            f"Corrupted pipeline score: `{screened_issue.issue_metrics_result}`")
                                        # st.markdown(f"This is a decrease by up to: {screened_issue.issue_max_score_decrease}")
                                        st.markdown("Sample of corrupted rows:")
                                        st.dataframe(sanitize_dataframe(screened_issue.issue_df))

                                        st.markdown(f"No suggestions necessary")
                        for placeholder_index in range(3-len(report.screened_issues)):
                            st.markdown(custom_container_start, unsafe_allow_html=True)
                    elif type(shadow_pipeline) == FairnessSlices:
                        screened_count = len(report.screened_issues)
                        passed_count = len([issue for issue in report.screened_issues if not issue.issue_found])
                        passed_fraction = int(passed_count / screened_count * 100)
                        failed_fraction = int((screened_count - passed_count) / screened_count * 100)
                        # st.progress(detected_fraction)

                        circle_html = f"""
                                      <style>
                                      .container {{
                                        display: flex;
                                        align-items: center;
                                        gap: 10px;  /* Spacing between text and circle */
                                      }}
    
                                      .circular-progress-slices {{
                                        width: 40px;
                                        height: 40px;
                                        border-radius: 50%;
                                        background: conic-gradient(green 0% {passed_fraction}%, red {passed_fraction}% 100%);
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                        font-weight: bold;
                                        font-size: 12px;
                                        color: black;
                                        margin: 10px;
                                      }}
    
                                      .container p {{
                                        margin: 0;  /* Remove default margin */
                                        line-height: 1.2;  /* Optional: adjust line-height if needed */
                                      }}
                                      </style>
    
                                      <div class="container">
                                          <p><b>Screening: {passed_count}</b> / <b>{screened_count}</b> passed!</p>
                                          <div class="circular-progress-slices">{passed_fraction}%</div>
                                      </div>
                                      """

                        st.markdown(f"{circle_html}", unsafe_allow_html=True)

                        for issue_index, screened_issue in enumerate(report.screened_issues):
                            if screened_issue.issue_found:
                                if screened_issue.suggestion_found:
                                    fix_found_text = " - suggestion"
                                else:
                                    fix_found_text = ""
                                with st.container(key=f"shadow-issue-container-{shadow_index}-{issue_index}"):
                                    with st.expander(f"{screened_issue.description} (failed){fix_found_text}",
                                                     expanded=False):

                                        st.markdown(f"Problematic slice: `{metrics_to_str(screened_issue.problematic_slice)}`")
                                        st.markdown(f"Original pipeline score: `{metrics_to_str(report.orig_metric_results)}`  \n"
                                                    f"Problematic slice score: `{metrics_to_str(screened_issue.slice_metric_result)}`")

                                        if screened_issue.problematic_slice_sample is not None:
                                            st.markdown(f"Sample from the slice:")
                                            st.dataframe(
                                                sanitize_dataframe(screened_issue.problematic_slice_sample))

                                        if screened_issue.suggestion_found:
                                            st.markdown(f"Suggestions from `mlidea`:")

                                        all_tried_suggestions = len(screened_issue.issue_suggestions)
                                        working_suggestions = 0
                                        for suggestion_index, suggestion in enumerate(screened_issue.issue_suggestions):
                                            if suggestion.improves_score:
                                                with st.container(
                                                        key=f"shadow-issue-container-{shadow_index}-{issue_index}-{suggestion_index}"):
                                                    with st.container(border=True):
                                                        working_suggestions += 1
                                                        nice_suggestion = slices_nice_fix_strategy_names[suggestion.suggestion]
                                                        st.markdown(f"Suggestion: `{nice_suggestion}`")
                                                        st.markdown(f"Improved pipeline score: `{metrics_to_str(suggestion.suggestion_metric_results)}`  \n"
                                                                    f"Improved slice score: `{metrics_to_str(suggestion.suggestion_metric_results_slice_only)}`")
                                                        st.markdown(f"Relative change original: "
                                                                    f"`{metrics_to_str(suggestion.suggestion_max_score_improvement)}`  \n"
                                                                    f"Relative change slice: "
                                                                    f"`{metrics_to_str(suggestion.suggestion_max_score_improvement_slice_only)}`")
                                                        if suggestion.source_code_to_integrate is not None:
                                                            st.markdown(f"Code suggestion:")
                                                            st.code(suggestion.source_code_to_integrate)

                                                            with st.container(
                                                                    key=f"pipeline-apply-llm-integrate-button-container-{shadow_index}-{issue_index}-{suggestion_index}"):
                                                                if st.button("Apply suggestion (via LLM)", key=f"llm-integration-{issue_index}-{suggestion_index}"):
                                                                    if "OPENAI_API_KEY" in os.environ:
                                                                        previous_key = os.environ["OPENAI_API_KEY"]
                                                                    else:
                                                                        previous_key = ""
                                                                    os.environ["OPENAI_API_KEY"] = get_open_ai_key()
                                                                    result = get_source_code_integration(
                                                                      st.session_state['PIPELINE_SOURCE_CODE'], suggestion.source_code_to_integrate)
                                                                    os.environ["OPENAI_API_KEY"] = previous_key
                                                                    # result = "Test"
                                                                    st.code(result)
                                                                    st.session_state["llm-suggestion"] = result
                                                            if "llm-suggestion" in st.session_state:
                                                                with st.container(
                                                                        key=f"pipeline-apply-llm-run-button-container-{shadow_index}-{issue_index}"):
                                                                    if st.button("Apply LLM Change", key="apply-llm-integration"):
                                                                        st.session_state['PIPELINE_SOURCE_CODE'] = "\n" + st.session_state["llm-suggestion"] + "\n\n"
                                                                        st.session_state['PIPELINE_SCENARIO'] = str(st.session_state['PIPELINE_SCENARIO']) + "+"
                                                                        del st.session_state["llm-suggestion"]
                                                                        st.rerun()

                                                                # st.rerun()
                                                                # print(st.session_state['PIPELINE_SCENARIO'])
                                                        if suggestion.suggestion_explanation_df is not None:
                                                            st.markdown(f"Sample of transformed rows:")
                                                            df = suggestion.suggestion_explanation_df
                                                            # df = df.rename(columns={"intermediate_before": "tweet_before",
                                                            #                         "intermediate_after": "tweet_after"})
                                                            # df = df.drop(columns=["tweet"])
                                                            st.dataframe(sanitize_dataframe(df))

                                                        if suggestion.prompt_used_to_generate_suggestion is not None:
                                                            with st.container(
                                                                    key=f"pipeline-show-prompt-button-container-{shadow_index}-{issue_index}"):
                                                                if st.button("Show internal LLM prompt", key="llm-prompt"):
                                                                    st.code(suggestion.prompt_used_to_generate_suggestion)


                                        if all_tried_suggestions != working_suggestions:
                                            if screened_issue.suggestion_found:
                                                st.markdown(f"Other strategies `mlidea` tried:")
                                            else:
                                                st.markdown(
                                                    f"Unfortunately, `mlidea` could not automatically find a potential "
                                                    f"solution. Here is what it tried:")

                                            for suggestion_index, suggestion in enumerate(
                                                    screened_issue.issue_suggestions):
                                                if not suggestion.improves_score:
                                                    # print(f"{shadow_index}-{issue_index}-{suggestion_index}")
                                                    with st.container(
                                                            key=f"shadow-issue-container-{shadow_index}-{issue_index}-{suggestion_index}"):
                                                        with st.container(border=True):
                                                            st.markdown(f"Suggestion: `{suggestion.suggestion}`")
                            else:
                                with st.container(key=f"shadow-issue-container-{shadow_index}-{issue_index}"):
                                    with st.expander(f"{screened_issue.description} (passed)", expanded=False):
                                        st.markdown(f"Original pipeline score: {report.orig_metric_results}")
                                        # st.markdown(f"Corrupted pipeline score: {screened_issue.issue_metrics_result}")
                                        # st.markdown(f"This is a decrease by up to: {screened_issue.issue_max_score_decrease}")
                                        # st.markdown("Sample of corrupted rows:")
                                        # st.dataframe(sanitize_dataframe(screened_issue.issue_df))

                                        st.markdown(f"No suggestions necessary")
                        for placeholder_index in range(3 - len(report.screened_issues)):
                            st.markdown(custom_container_start, unsafe_allow_html=True)
                    else:
                        with st.expander("Details", expanded=False):
                            st.text(report)

                    # st.markdown(custom_container_end, unsafe_allow_html=True)
    else:
        with st.container(key=f"shadow-pipeline-container-0"):
            with st.container(border=True):
                st.markdown("Run the code to see more")
# with details_left:
#     pipeline_code_container = st.container()
#     with pipeline_code_container:
#         st.header("Pipeline Code")
# with details_right:
#     results_container = st.container()
#     with results_container:
#         st.header("Shadow Pipelines")
#
# with details_left:
#     render_full_size_dag("Original")
# with details_right:
#     render_shadow_dags()
