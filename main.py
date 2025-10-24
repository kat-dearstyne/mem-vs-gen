import os

import pandas as pd
from dotenv import load_dotenv
from pandas.core.computation.expr import intersection
from pygments.modeline import modeline_re

from act_density_data_download import download_neuronpedia_act_density_data
from attribution_graph_creation import get_frequencies, create_graph, create_node_df, get_all_features, \
    add_features_to_list, create_subgraph_from_selected_features, get_overlap_scores_for_features
from constants import AVAILABLE_MODELS
from utils import load_json, save_json, user_select_prompt, user_select_models, create_prompt_id, get_top_k_from_df
from visualizations import freq_vs_act_density

load_dotenv()

DOWNLOAD_DATA = False
PROMPT = None
MODEL = "gemma-2-2b"
SUBMODEL = "clt-hp" #None

SAVE_PATH = "~/data/spar-memory/neuronpedia/"
ACT_DENSITY_THRESHOLD = 0.4
TOP_K = 5

if __name__ == "__main__":
    base_save_path = os.path.expanduser(SAVE_PATH)
    model, submodel = user_select_models(model=MODEL, submodel=SUBMODEL)
    graph_dir = os.path.join(base_save_path, "graphs")
    prompt = user_select_prompt(prompt_default=PROMPT, graph_dir=graph_dir)
    print(f"\nStarting run with model {model} and submodel {submodel}"
          f"\nPrompt: '{prompt}'\n")

    prompt_id = create_prompt_id(prompt)
    graph_path = os.path.join(graph_dir, prompt_id, f"{model}-{submodel}.json")

    if os.path.exists(graph_path):
        print(f"Loading graph from {graph_path}")
        graph_metadata = load_json(graph_path)
    else:
        print(f"Saving graph to {graph_path}")
        graph_metadata = create_graph(prompt=prompt, model=model, submodel=submodel)
        save_json(graph_metadata, graph_path)
    print(f"Neuronpedia Graph: {graph_metadata['metadata']['info']['url']}")

    node_df = create_node_df(graph_metadata)
    ctx_freq_df = get_frequencies(node_df)
    high_freq = ctx_freq_df[ctx_freq_df["ctx_freq"] > 1]
    high_freq = high_freq[high_freq["layer"] != '0']
    if len(high_freq) < 1:
        print("No Features of Interest")

    if DOWNLOAD_DATA:
        download_save_path = download_neuronpedia_act_density_data(
            foldername=base_save_path,
            model=model,
            submodel=submodel,
        )
        act_density_df = pd.read_csv(
            download_save_path,
            dtype={"layer": str, "feature": str},  # for compatibility with ctx_freq_df
        )

        merged_df = high_freq.merge(
            act_density_df,
            on=["layer", "feature"],
            how="inner"
        )[["layer", "feature", "ctx_freq", "act_density"]]
    else:
        features = get_all_features(high_freq, model=model, submodel=submodel, foldername=base_save_path)
        act_density = [feature["frac_nonzero"] for feature in features]
        merged_df = high_freq.copy()
        merged_df["act_density"] = act_density
        merged_df["overlap_scores"] = get_overlap_scores_for_features(graph_metadata["metadata"]["prompt_tokens"], features)

    filtered_features =  merged_df[merged_df["act_density"] <= ACT_DENSITY_THRESHOLD]
    top_features_by_freq = get_top_k_from_df(filtered_features, TOP_K, sort_by=['ctx_freq', 'act_density'], ascending=[False, True])
    top_features_by_overlap = get_top_k_from_df(filtered_features, TOP_K, sort_by=['overlap_scores',
                                                                                   'ctx_freq', 'act_density'],
                                                ascending=[False, False, True])
    top_features = pd.concat([top_features_by_freq, top_features_by_overlap])
    found_by_both = [f"{feature.layer}-{feature.feature}" for feature in top_features[top_features.duplicated(keep='first')].itertuples()]
    top_features = top_features.drop_duplicates()
    if found_by_both:
        print(f"Features of interest: {found_by_both}")
    url = add_features_to_list(top_features, prompt_id=prompt_id, model=model, submodel=submodel)
    print(f"Frequent features list: {url}")

    subgraph_id = create_subgraph_from_selected_features(top_features, graph_metadata)
    print(f"Subgraph of top features id: {subgraph_id}")

    save_path = os.path.join(os.path.join(base_save_path, "freq_vs_act_density"),
                             prompt_id, f"{model}-{submodel}.html")
    freq_vs_act_density(merged_df, save_path=save_path)
