import os
from logging import lastResort
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from pygments.modeline import modeline_re

from attribution_graph_creation import nodes_not_in, create_subgraph_from_selected_features, nodes_in, \
    select_features_by_links
from utils import user_select_prompt, user_select_models, get_node_ids_from_features, create_node_id, Feature, \
    get_feature_from_node_id

load_dotenv()

DOWNLOAD_DATA = False
PROMPT = "THE SOFTWARE IS PROVIDED" #("THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, "
          #"INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF")
CONTRAST_PROMPTS = [
    "THE SOFTWARE AND ANY ACCOMPANYING MATERIALS ARE PROVIDED \"AS IS\", "
    "WITHOUT ANY PROMISE OR GUARANTEE OF PERFORMANCE, RELIABILITY, OR SUITABILITY AND THE WARRANTIES OF",
    "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT"
]
COMPARE_PROMPTS = ["The software is provided \"as is\", without warranty of any kind, express or implied, "
                   "including but not limited to the warranties of"]
MODEL = "gemma-2-2b"
SUBMODEL = "clt-hp"

SAVE_PATH = "~/data/spar-memory/neuronpedia/"
ACT_DENSITY_THRESHOLD = 0.4
TOP_K = 5

FILTER_BY_LINKS = False

if __name__ == "__main__":
    base_save_path = os.path.expanduser(SAVE_PATH)
    model, submodel = user_select_models(model=MODEL, submodel=SUBMODEL)
    graph_dir = os.path.join(base_save_path, "graphs")

    prompt = user_select_prompt(prompt_default=PROMPT, graph_dir=graph_dir)
    contrast_prompts = CONTRAST_PROMPTS  # looks for nodes in main prompt but NOT in these
    compare_prompts = COMPARE_PROMPTS  # looks for nodes in main prompt AND in these

    print(f"\nStarting run with model {model} and submodel {submodel}"
          f"\nPrompt1: '{prompt}'\n"
          f"\nContrasting {len(contrast_prompts)} prompts.\n"
          f"\nComparing {len(compare_prompts)} prompts.\n")

    graph_metadata: Optional[dict] = None
    unique_features: Optional[pd.DataFrame] = None
    if contrast_prompts:
        graph_metadata, unique_features = nodes_not_in(main_prompt=prompt, prompts2compare=contrast_prompts,
                                                       model=model, submodel=submodel, graph_dir=graph_dir)

    overlapping_features: Optional[pd.DataFrame] = None
    if compare_prompts:
        graph_metadata, overlapping_features = nodes_in(main_prompt=prompt, prompts2compare=compare_prompts,
                                                        model=model, submodel=submodel, graph_dir=graph_dir)

    if unique_features is not None and overlapping_features is not None:
        features_of_interest = pd.merge(overlapping_features, unique_features, how='inner', on=['layer', 'feature'])
    else:
        features_of_interest = unique_features if unique_features is not None else overlapping_features

    assert features_of_interest is not None, "Must provided either prompts to compare with or to contrast with."
    print(f"Neuronpedia Graph for 1st Prompt: {graph_metadata['metadata']['info']['url']}")

    output_node = graph_metadata["nodes"][-1]
    features_of_interest = features_of_interest[~features_of_interest['layer'].isin(['0', 'E', output_node['layer']])]

    if FILTER_BY_LINKS:
        node_ids = set(get_node_ids_from_features(features_of_interest))

        selected_features_by_output = select_features_by_links(graph_metadata, target_ids=output_node["node_id"],
                                                               source_ids=node_ids)

        most_recent_embd = [node["node_id"] for node in graph_metadata["nodes"]
                            if node["feature_type"].startswith("emb")][-1]
        last_embd_index =  most_recent_embd.split("_")[-1]
        last_embd_error_nodes = {node["node_id"] for node in graph_metadata["nodes"]
                                 if node["feature_type"].startswith("mlp") and
                                 node["node_id"].endswith(last_embd_index)}
        selected_features_by_embd = select_features_by_links(
            graph_metadata,
            target_ids={create_node_id(Feature(*feature)) for feature in selected_features_by_output},
            source_ids=last_embd_error_nodes.union({most_recent_embd})
        )
        selected_features = selected_features_by_embd.difference({get_feature_from_node_id(feature, deliminator="_")
                                                                  for feature in last_embd_error_nodes})
        filtered_df = pd.DataFrame(list(selected_features), columns=["layer", "feature"])
        filtered_df = features_of_interest.merge(filtered_df, on=["layer", "feature"])
        features_of_interest = filtered_df

    create_subgraph_from_selected_features(features_of_interest, graph_metadata, list_name="Features of Interest")
