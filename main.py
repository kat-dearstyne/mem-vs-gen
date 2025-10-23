import os
import uuid

import pandas as pd
import requests
from dotenv import load_dotenv

from attribution_graph_creation import get_frequencies, create_graph, create_node_df, get_feature, get_all_features, \
    add_features_to_list
from constants import SAVE_ACT_DENSITIES_FILENAME
from act_density_data_download import download_neuronpedia_act_density_data
from utils import load_json, save_json, get_most_recent_file
from visualizations import freq_vs_act_density

load_dotenv()

PROMPT = None
DOWNLOAD_DATA = False
SAVE_PATH = "~/data/spar-memory/neuronpedia/"
MODEL = "gemma-2-2b"
SUBMODEL = "gemmascope-transcoder-16k"
ACT_DENSITY_THRESHOLD = 0.4
TOP_K = 5


if __name__ == "__main__":
    base_save_path = os.path.expanduser(SAVE_PATH)
    graph_dir = os.path.join(base_save_path, MODEL, SUBMODEL, "graphs")
    latest_graph = get_most_recent_file(graph_dir)
    last_prompt = None
    if latest_graph:
        graph_metadata = load_json(latest_graph)
        last_prompt = graph_metadata["metadata"]["prompt"].replace("<bos>", "")
        print(f"Most recent prompt: {last_prompt}")
    prompt = input("Enter prompt or skip to use last prompt:\n").strip() if not PROMPT else PROMPT
    prompt = prompt or last_prompt
    assert prompt, "No prompt given"

    print(f"Starting run with model {MODEL} and submodel {SUBMODEL}\nPrompt: '{prompt}'")
    prompt_start = "_".join(prompt.lower().split()[:5])
    prompt_id = f"{prompt_start}:{uuid.uuid5(uuid.NAMESPACE_DNS, prompt)}"
    graph_path = os.path.join(graph_dir, f"{prompt_id}.json")
    print(f"Saving to {os.path.dirname(graph_dir)}")
    if os.path.exists(graph_path):
        graph_metadata = load_json(graph_path)
    else:
        graph_metadata = create_graph(prompt=prompt)
        save_json(graph_metadata, graph_path)
    print(f"Neuronpedia Graph: {graph_metadata['metadata']['info']['url']}")
    node_df = create_node_df(graph_metadata)
    ctx_freq_df = get_frequencies(node_df)
    high_freq = ctx_freq_df[ctx_freq_df["ctx_freq"] > 1]
    if len(high_freq) < 1:
        print("No Features of Interest")

    if DOWNLOAD_DATA:
        download_save_path = download_neuronpedia_act_density_data(
            foldername=base_save_path,
            model=MODEL,
            submodel=SUBMODEL,
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
        features = get_all_features(high_freq, foldername=base_save_path)
        act_density = [feature["frac_nonzero"] for feature in features]
        merged_df = high_freq.copy()
        merged_df["act_density"] = act_density

    top_features = merged_df[merged_df["act_density"] <= ACT_DENSITY_THRESHOLD].head(min(TOP_K, len(merged_df)))
    url = add_features_to_list(top_features, prompt_id=prompt_id, model=MODEL, submodel=SUBMODEL)
    print(f"Frequent features list: {url}")
    freq_vs_act_density(merged_df)


