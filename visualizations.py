import pandas as pd
import plotly.express as px


def freq_vs_act_density(merged_df: pd.DataFrame) -> None:
    """
    Create an interactive scatter plot showing feature context frequency vs activation density.
    """
    fig = px.scatter(
        merged_df,
        x="act_density",
        y="ctx_freq",
        hover_data=["layer", "feature"],
        title="Feature Context Frequency vs. Activation Density",
        labels={
            "act_density": "Activation Density",
            "ctx_freq": "Context Frequency"
        },
        width=600,
        height=600,
    )
    fig.show()