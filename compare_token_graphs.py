import os
from logging import lastResort
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from pygments.modeline import modeline_re

from attribution_graph_creation import nodes_not_in, create_subgraph_from_selected_features, nodes_in, \
    select_features_by_links, create_or_load_graph, get_linked_sources, get_subgraphs
from utils import user_select_prompt, user_select_models, get_node_ids_from_features, create_node_id, Feature, \
    get_feature_from_node_id

load_dotenv()

DOWNLOAD_DATA = False
PROMPT = "THE SOFTWARE AND ANY ACCOMPANYING MATERIALS ARE PROVIDED \"AS IS\", WITHOUT ANY PROMISE OR GUARANTEE, INCLUDING THE WARRANTIES OF"
MODEL = "gemma-2-2b"
SUBMODEL = "clt-hp"
TOKEN_OF_INTEREST = "MERCHANTABILITY"

SAVE_PATH = "~/data/spar-memory/neuronpedia/"
ACT_DENSITY_THRESHOLD = 0.4
TOP_K = 4

FILTER_BY_LINKS = False

if __name__ == "__main__":
    base_save_path = os.path.expanduser(SAVE_PATH)
    model, submodel = user_select_models(model=MODEL, submodel=SUBMODEL)
    graph_dir = os.path.join(base_save_path, "graphs")

    graph_metadata = create_or_load_graph(graph_dir=graph_dir, model=model, submodel=submodel, prompt=PROMPT)
    get_subgraphs(graph_metadata)
    print(graph_metadata["metadata"]["info"]["url"])

    output_nodes = [node for node in graph_metadata["nodes"] if node["feature_type"] == "logit"]
    top_nodes = output_nodes[:TOP_K]
    output_token_to_id = {node["clerp"].split("\"")[1].strip(): node["node_id"] for node in top_nodes}
    output_token_to_features = {token: {node_id} for token, node_id in output_token_to_id.items()}
    newly_added = True
    while newly_added:
        newly_added = get_linked_sources(graph_metadata, output_token_to_features, positive_only=True)
    other_features = set()
    for token, node_ids in output_token_to_features.items():
        if token != TOKEN_OF_INTEREST:
            other_features.update(node_ids)
    unique_node_ids = output_token_to_features[TOKEN_OF_INTEREST].difference(other_features)
    unique_features = [get_feature_from_node_id(node_id, deliminator="_") for node_id in unique_node_ids]
    node_dict = [{"layer": feature.layer, "feature": feature.feature} for feature in unique_features if
                 int(feature.layer) > 1]
    node_df = pd.DataFrame(node_dict)
    url = create_subgraph_from_selected_features(node_df, graph_metadata, f"unique features for {TOKEN_OF_INTEREST}",
                                                 include_output_node=False)
