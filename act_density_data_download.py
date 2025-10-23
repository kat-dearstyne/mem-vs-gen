import os
from typing import Optional

import pandas as pd
from IPython import get_ipython
from tqdm import tqdm

from constants import DEFAULT_SAVE_DIR, SAVE_ACT_DENSITIES_FILENAME
from utils import save_df


def get_neuronpedia_act_density_data_batch(
    layer: int,
    batch: int,
    model: str = "gemma-2-2b",
    submodel: str = "gemmascope-transcoder-16k",
    foldername: str = DEFAULT_SAVE_DIR,
) -> pd.DataFrame:
    """
    Load a single batch of feature activation density data directly from Neuronpedia's S3 bucket.
    """
    filename = os.path.join(foldername, f"batch-{batch}.jsonl.gz")
    if not os.path.exists(filename):
      curl_cmd = f"curl -L -s -o  https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/v1/{model}/{layer}-{submodel}/features/batch-{batch}.jsonl.gz"
      _ = get_ipython().system(curl_cmd)

    batch_df = pd.read_json(filename, lines=True, compression="gzip")[["index", "frac_nonzero"]]

    return batch_df

def download_neuronpedia_act_density_data(
    model: str = "gemma-2-2b",
    submodel: str = "gemmascope-transcoder-16k",
    num_layers: int = 26,
    num_batches: int = 64,
    foldername: Optional[str] = DEFAULT_SAVE_DIR,
    filename: Optional[str] = None,
) -> str:
    """
    Download feature activation density data directly from Neuronpedia's S3 bucket and save as CSV.
    """
    foldername = os.path.join(foldername, submodel)
    os.makedirs(foldername, exist_ok=True)

    total_df = pd.DataFrame()
    pbar = tqdm(
        total=num_layers*num_batches,
        desc="Downloading Neuronpedia activation density data"
    )

    for layer in range(num_layers):
        layer_df = pd.DataFrame()
        for batch in range(num_batches):
            batch_df = get_neuronpedia_act_density_data_batch(
                layer, batch, model=model, submodel=submodel, foldername=foldername,
            )
            layer_df = pd.concat([layer_df, batch_df])
            pbar.update(1)
        total_df = pd.concat([total_df, layer_df.assign(layer=layer)])
    pbar.close()

    total_df = total_df.rename(columns={"frac_nonzero": "act_density", "index": "feature"})

    if filename is None:
        filename = SAVE_ACT_DENSITIES_FILENAME.format(model=model, submodel=submodel)
    save_path = save_df(total_df, foldername, filename)
    return save_path

