import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb
import numpy as np
import os
import itertools

sns.set_style("darkgrid")
sns.despine()
sns.set_context("paper")

metrics = {
    "config.grl_config.guide_in_actor_loss": [True, False],
    "config.grl_config.delay_training": [True, False],
    "config.algo_config.replay_buffer_kwargs.perc_guide_sampled": [
        [0.5, "cs"],
        [0.5, 0.5],
        ["cs", "cs"],
    ],
}

lengths = {k: len(v) for k, v in metrics.items()}

combos = itertools.product(*metrics.values())

config = {
    "config.env_name": "CombinationLock-v1",
    "config.eval_freq": 1000,
    "config.buffer_size": 1000000,
}

try:
    runs_df = pd.read_csv("wandb_plots/data/project.csv")
except FileNotFoundError:
    api = wandb.Api(timeout=60)
    runs_df = pd.DataFrame()
    for combo in combos:
        filters = dict(zip(metrics.keys(), combo)) | config

        runs = api.runs(
            path="lauren-taylor-the-university-of-adelaide/sb3-TD3", filters=filters
        )

        metric_dict = {}
        for run in runs:
            df = run.history(
                keys=["eval/mean_reward"],
                x_axis="global_step",
                pandas=(True),
                stream="default",
            )
            for k, v in filters.items():
                df[k] = [v] * len(df["eval/mean_reward"])

            runs_df = pd.concat([runs_df, df], ignore_index=True)
    print(runs_df)
    try:
        runs_df.to_csv("wandb_plots/data/project.csv")
    except FileNotFoundError:
        os.makedirs("wandb_plots/data", exist_ok=True)
        runs_df.to_csv("wandb_plots/data/project.csv")

runs_df["GuideInLoss-Delayed"] = (
    runs_df["config.grl_config.guide_in_actor_loss"].astype(str)
    + "-"
    + runs_df["config.grl_config.delay_training"].astype(str)
)
runs_df.columns = runs_df.columns.str.replace(
    "config.algo_config.replay_buffer_kwargs.perc_guide_sampled", "Sample Method"
)
runs_df["Sample Method"] = runs_df["Sample Method"].astype(str)
import pdb

pdb.set_trace()
ax = sns.lineplot(
    runs_df,
    x="global_step",
    y="eval/mean_reward",
    hue="GuideInLoss-Delayed",
    style="Sample Method",
    errorbar="sd",
)
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
plt.tight_layout()
plt.savefig("wandb_plots/plots/CombinationLock-v1.png")
