import random
from collections import Counter
from typing import List

import networkx
from mlidea import DagNode


def get_legend():
    # Define colors
    white = '#FFFFFF'
    black = '#000000'
    reuse_color = '#7d354c'
    reexecuted_color = '#355C7D'
    transitive_color = '#f67d72'

    # Create legend with color boxes
    legend_html = f"""
        <div class="legend" style="display: flex; flex-direction: row; justify-content: center; gap: 20px; margin-top: 10px;">
            <div style="display: flex; align-items: center; gap: 5px;">
                <div style="width: 15px; height: 15px; background-color: {white}; border-radius: 3px;"></div>
                <span style="color: white;">Initial execution</span>
            </div>
            <div style="display: flex; align-items: center; gap: 5px;">
                <div style="width: 15px; height: 15px; background-color: {reuse_color}; border-radius: 3px;"></div>
                <span style="color: white;">Full reuse</span>
            </div>
            <div style="display: flex; align-items: center; gap: 5px;">
                <div style="width: 15px; height: 15px; background-color: {reexecuted_color}; border-radius: 3px;"></div>
                <span style="color: white;">New operator</span>
            </div>
            <div style="display: flex; align-items: center; gap: 5px;">
                <div style="width: 15px; height: 15px; background-color: {transitive_color}; border-radius: 3px;"></div>
                <span style="color: white;">Partial reuse</span>
            </div>
        </div>
    """

    # Display legend below the graph
    return legend_html


def get_original_simple_dag(dag: networkx.DiGraph, reuse_info) -> networkx.DiGraph:
    plan = networkx.DiGraph()
    white = '#FFFFFF'
    black = '#000000'
    reuse_color = '#7d354c'
    reexecuted_color = '#355C7D'
    transitive_color = '#f67d72'
    # mlwhatif colors
    # colors = ['#7d354c', '#C06C84', '#355C7D', '#F67280', '#F9A2AB', '#f67d72']

    for node in dag.nodes:
        node_id = node.node_id
        operator_name = get_pretty_operator_name(node)

        if node in reuse_info.operator_reexecuted:
            plan.add_node(node_id, operator_name=operator_name, fillcolor=reexecuted_color, fontcolor=white, border_color=reexecuted_color,
                          style='filled', text_outline_color=reexecuted_color)
        elif node in reuse_info.operator_fully_reused:
            plan.add_node(node_id, operator_name=operator_name, fillcolor=reuse_color, fontcolor=white, border_color=reuse_color,
                          style='filled', text_outline_color=reuse_color)
        elif node in reuse_info.operator_transitive:
            plan.add_node(node_id, operator_name=operator_name, fillcolor=transitive_color, fontcolor=white, border_color=transitive_color,
                          style='filled', text_outline_color=transitive_color)
        else:
            plan.add_node(node_id, operator_name=operator_name, fillcolor=white, fontcolor=white, border_color=black,
                          style='filled', text_outline_color=black)

    for edge in dag.edges:
        plan.add_edge(edge[0].node_id, edge[1].node_id)

    # TODO: This is probably not needed for mlwhatif
    # while True:
    #     nodes_to_remove = [node for node, data in plan.nodes(data=True)
    #                        if data['operator_name'] == 'π' and len(list(plan.successors(node))) == 0]
    #     if len(nodes_to_remove) == 0:
    #         break
    #     else:
    #         plan.remove_nodes_from(nodes_to_remove)

    return plan


def get_colored_simple_dags(dags: List[networkx.DiGraph], with_reuse_coloring=True) -> List[networkx.DiGraph]:
    result_dags = []

    all_nodes = []
    for dag in dags:
        all_nodes.extend(list(dag.nodes))

    all_nodes_with_counts = Counter(all_nodes).items()
    shared_nodes = {node for (node, count) in all_nodes_with_counts if count >= 2}

    random.seed(42)
    # TODO: Pick enough suitable colors
    colors = random.choices(['#7d354c', '#C06C84', '#355C7D', '#F67280', '#F9A2AB', '#f67d72'], k=len(dags))
    white = '#FFFFFF'
    black = '#000000'

    for dag, variant_color in zip(dags, colors):
        plan = networkx.DiGraph()

        for node in dag.nodes:
            node_id = node.node_id
            operator_name = get_pretty_operator_name(node)

            if node not in shared_nodes or with_reuse_coloring is False:
                plan.add_node(node_id, operator_name=operator_name, fillcolor=variant_color, fontcolor=white,
                              border_color=variant_color, style='filled', text_outline_color=variant_color)
            else:
                plan.add_node(node_id, operator_name=operator_name, fillcolor=black, fontcolor=white, style='filled',
                              border_color=black, text_outline_color=black)

        for edge in dag.edges:
            plan.add_edge(edge[0].node_id, edge[1].node_id)

        # TODO: This is probably not needed for mlwhatif
        while True:
            nodes_to_remove = [node for node, data in plan.nodes(data=True)
                               if data['operator_name'] == 'π' and len(list(plan.successors(node))) == 0]
            if len(nodes_to_remove) == 0:
                break
            else:
                plan.remove_nodes_from(nodes_to_remove)
        result_dags.append(plan)

    return result_dags


def get_final_optimized_combined_colored_simple_dag(final_stage_dags: List[networkx.DiGraph]) -> networkx.DiGraph:
    colored_dags = get_colored_simple_dags(final_stage_dags)
    if len(colored_dags) != 0:
        big_execution_dag = networkx.compose_all(colored_dags)
    else:
        big_execution_dag = networkx.DiGraph()
    return big_execution_dag


def get_pretty_operator_name(node: DagNode) -> str:
    op_type = node.operator_info.operator
    operator_type = str(op_type).split('.')[1]
    operator_description = node.details.description

    operator_name = operator_type
    if operator_type == 'JOIN':
        operator_name = '⋈'
    elif operator_type == 'PROJECTION' or operator_type == 'PROJECTION_MODIFY':
        operator_name = 'π'
    elif operator_type == 'TRANSFORMER':
        operator_name = 'π'
    elif operator_type == 'SELECTION':
        operator_name = 'σ'
    elif operator_type == 'CONCATENATION':
        operator_name = '+'
    elif operator_type == 'DATA_SOURCE':
        operator_name = f'({node.node_id}) {node.details.description}'
    elif operator_type == 'ESTIMATOR':
        operator_name = 'Model Training'
    elif operator_type == 'SCORE':
        operator_name = 'Model Evaluation'
    elif operator_type == 'PREDICT':
        operator_name = 'Model Predictions'
    elif operator_type == 'TRAIN_DATA':
        operator_name = 'X_train'
    elif operator_type == 'TRAIN_LABELS':
        operator_name = 'y_train'
    elif operator_type == 'TEST_DATA':
        operator_name = 'X_test'
    elif operator_type == 'TEST_LABELS':
        operator_name = 'y_test'
    elif operator_type == 'EXTRACT_RESULT':
        operator_name = 'Result Extraction'
    elif operator_type == 'RAG_JOIN':
        operator_name = 'RAG Join'
    elif operator_type == 'SUBSCRIPT':
        # TODO: Maybe use a different symbol?
        operator_name = 'π'
    elif operator_type == 'GROUP_BY_AGG':
        operator_name = 'Γ'
    elif operator_type == 'TRAIN_TEST_SPLIT':
        if operator_description is None:
            operator_name = "Train-Test Split"
        elif operator_description == "(Test Data)":
            operator_name = "Train Side (Split)"
        else:
            operator_name = "Test Side (Split)"

    return operator_name
