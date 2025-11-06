
import os

import torch
from dotenv import load_dotenv
from circuit_tracer.graph_visualization import Supernode, Feature, InterventionGraph, create_graph_visualization

from attribution_graph_creation import create_or_load_graph, get_subgraphs
from utils import user_select_models

load_dotenv()

DOWNLOAD_DATA = False
PROMPT = "THE SOFTWARE AND ANY ACCOMPANYING MATERIALS ARE PROVIDED \"AS IS\", WITHOUT ANY PROMISE OR GUARANTEE, INCLUDING THE WARRANTIES OF"
MODEL = "gemma-2-2b"
SUBMODEL = "clt-hp"
SUBGRAPH_NAME = 'fake license subgraph'
SAVE_PATH = "~/data/spar-memory/neuronpedia/"

def create_feature_from_node_id(node_id: str):
    split_id = node_id.split("_")
    layer, feature_idx, pos = split_id
    if layer != "E":
        return Feature(layer=int(layer), feature_idx=int(feature_idx), pos=int(pos))

def find_best_match(approx_node_id: str, subgraph: dict):
    layer, _, pos = approx_node_id.split("_")
    for pinned_id in subgraph["pinnedIds"]:
        p_layer, _, p_pos = pinned_id.split("_")
        if layer == p_layer and pos == p_pos:
            return pinned_id
    return approx_node_id

if __name__ == "__main__":
    base_save_path = os.path.expanduser(SAVE_PATH)
    model, submodel = user_select_models(model=MODEL, submodel=SUBMODEL)
    graph_dir = os.path.join(base_save_path, "graphs")

    graph_metadata = create_or_load_graph(graph_dir=graph_dir, model=model, submodel=submodel, prompt=PROMPT)
    tokens = graph_metadata["metadata"]['prompt_tokens']
    node_id_to_node = {node["node_id"]: node for node in graph_metadata["nodes"]}

    subgraphs = get_subgraphs(graph_metadata)['subgraphs']
    selected_subgraph = [subgraph for subgraph in subgraphs if subgraph['displayName'] == SUBGRAPH_NAME]
    assert len(selected_subgraph) > 0, "No subgraphs found"
    selected_subgraph = selected_subgraph[0]
    supernode_to_id = {supernode[0]: supernode[1:] for supernode in selected_subgraph["supernodes"]}
    node_name_to_ids = {clerp[1]: [find_best_match(clerp[0], selected_subgraph)] for clerp in selected_subgraph['clerps']}
    node_name_to_ids.update(supernode_to_id)
    for node_id in selected_subgraph["pinnedIds"]:
        is_supernode = False
        for name, ids in node_name_to_ids.items():
            if node_id in ids:
                is_supernode = True
                break
        if is_supernode:
            continue
        clerp = node_id_to_node[node_id].get("clerp")
        if not clerp.strip():
            clerp = node_id
        if clerp in node_name_to_ids:
            node_name_to_ids[clerp].append(node_id)
        else:
            node_name_to_ids[clerp] = [node_id]
    all_node_ids = {node_id: name for name, ids in node_name_to_ids.items() for node_id in ids}
    links = {}
    for link in graph_metadata["links"]:
        if link["source"] in all_node_ids and link["target"] in all_node_ids and link["weight"] > 0:
            name = all_node_ids[link["source"]]
            if name not in links:
                links[name] = []
            links[name].append(link["target"])


    graph_nodes = {name: Supernode(name=name, features=[create_feature_from_node_id(node_id) for node_id in ids]) for name, ids in node_name_to_ids.items()}
    for graph_node in graph_nodes.values():
        if graph_node.name in links:
            graph_node.children = [graph_nodes[all_node_ids[l]] for l in links[graph_node.name]]
        activations = []
        graph_node.features = [feature for feature in graph_node.features if feature]
        if graph_node.features:
            for feature in graph_node.features:
                node_id = f"{feature.layer}_{feature.feature_idx}_{feature.pos}"
                act = node_id_to_node.get(node_id, {}).get("activation")
                activations.append(act if act else 0)
            graph_node.default_activations = torch.tensor(activations)
        else:
            graph_node.features = None

    nodes_by_layer = {}
    embedding_nodes = [None for _ in tokens]
    output_nodes = []
    for graph_node in graph_nodes.values():
        if  graph_node.name.startswith("E_"):
            pos = int(graph_node.name.split("_")[-1])
            graph_node.name = tokens[pos]
            embedding_nodes[pos] = graph_node
        elif graph_node.name.startswith("Output"):
            graph_node.name = graph_node.name.split("\"")[1]
            output_nodes.append(graph_node)
        else:
            layer = graph_node.features[0].layer
            if layer not in nodes_by_layer:
                nodes_by_layer[layer] = []
            nodes_by_layer[layer].append(graph_node)
    embedding_nodes = [e_node for e_node in embedding_nodes if e_node]
    ordered_nodes = [item[1] for item in sorted(nodes_by_layer.items())]
    ordered_nodes = [embedding_nodes] + ordered_nodes + [output_nodes]
    new_graph = InterventionGraph(ordered_nodes=ordered_nodes, prompt=PROMPT)
    [new_graph.initialize_node(graph_node) for graph_node in graph_nodes.values() if graph_node.activation is not None]
    visualize = create_graph_visualization(new_graph, [])
    with open("output.svg", "w") as f:
        f.write(visualize.data)
