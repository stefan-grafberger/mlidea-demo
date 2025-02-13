import os

from example_pipelines.healthcare import custom_monkeypatching
from mlidea import PipelineAnalyzer
from mlidea.shadow_pipelines._data_errors import DataErrorRobustness
from mlidea.shadow_pipelines._label_errors import LabelErrors
from mlidea.shadow_pipelines._slices import FairnessSlices

from utils import get_project_root

DATABASE_PATH_FUNC_TRANSFORMER = f"{str(get_project_root())}/.function_transformer_cache.db"
label_errors = LabelErrors(proxy_model=False)
data_errors = DataErrorRobustness(corruption_significant_relative_threshold=0.95)
slices = FairnessSlices(database_path=DATABASE_PATH_FUNC_TRANSFORMER)
shadow_pipelines = [label_errors, data_errors, slices]

analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(os.path.join(str(get_project_root()), "pipelines", "healthcare.py")) \
        .add_custom_monkey_patching_modules([custom_monkeypatching]) \
        .add_shadow_pipelines(shadow_pipelines) \
        .execute()

report_label_errors = analysis_result.shadow_pipelines_to_result_reports[label_errors]
report_data_errors = analysis_result.shadow_pipelines_to_result_reports[data_errors]
report_fairness_slices = analysis_result.shadow_pipelines_to_result_reports[slices]
