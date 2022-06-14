"""
Extract information logged to wandb in order to plot/analyze.
WandB help: https://docs.wandb.ai/guides/track/public-api-guide
"""
import os
from functools import partial
from multiprocessing import Pool

import pandas as pd
import wandb

# Fix entity
ENTITY = "l0-coin"
PROJECT = "kodak"


def get_metrics(filters, metric_keys, config_keys=None, x_axis="_step"):
    """
    Extract metric_keys from wandb runs given filters. Keep config_keys for reference
    Args:
        filters: Example {"$and": [{"config.run_group": "control"},
                            {"config.dual_optim": "SGD"}]}
        metric_keys: Example ["val/top1", "val/macs", "val/params"]
        config_keys: config elements to return: ["seed", "model_type"]
        x_axis: one of "_step" or "epoch"
    Returns:
        DataFrame with metrics, config list
    """
    api = wandb.Api(overrides={"entity": ENTITY, "project": PROJECT}, timeout=30)
    runs = api.runs(path=ENTITY + "/" + PROJECT, filters=filters, order="-created_at")
    print("Number of runs:", len(runs))

    all_frames = []
    for run in runs:
        # samples param: without replacement, if too large returns all.
        metrics = run.history(samples=10_000, keys=metric_keys, x_axis=x_axis)

        # Do not keep the whole config, only config_keys if provided by user
        filtered_config = {
            key: run.config[key] for key in config_keys if key in run.config
        }

        for key, val in filtered_config.items():
            metrics.insert(0, key, str(val))

        all_frames.append(metrics)

    return pd.concat(all_frames)


def main(image_id, final_bpp, baseline_hidden_dims, foldername):
    """
    Get various metrics across baseline, mp and loonie runs for the provided
    image_id and final bpp.
    """
    filters = {
        "$and": [
            {"config.train.image_id": image_id},
            {"state": "finished"},
            {
                "$or": [
                    {
                        "$and": [
                            {"config.wandb.run_group": "all_gated"},
                            {"config.train.target_bpp": final_bpp},
                        ]
                    },
                    {
                        "$and": [
                            {"config.wandb.run_group": "baseline"},
                            {"config.model.hidden_dims": baseline_hidden_dims},
                        ]
                    },
                    {
                        "$and": [
                            {"config.wandb.run_group": "new_mp"},
                            {"config.train.target_bpp": final_bpp},
                        ]
                    },
                ]
            },
        ]
    }
    metric_keys = [
        "compression/best_psnr",
        "_runtime",
        "compression/bpp",
        # "train/lambda_01",
    ]
    config_keys = [
        "task_type",
        "model.hidden_dims",
        "train.image_id",
        "train.target_bpp",
    ]

    # Use wandb api
    metrics_without_gated = get_metrics(filters, metric_keys, config_keys)
    # Delete gated runs as they are gathered twice
    mask = metrics_without_gated["task_type"] != "gated"
    metrics_without_gated = metrics_without_gated.loc[mask]

    metric_keys += ["train/lambda_01"]
    gated_metrics = get_metrics(filters, metric_keys, config_keys)

    # Merge gated and non-gated metrics
    metrics = pd.concat([metrics_without_gated, gated_metrics])

    # Save as csv
    try:
        os.makedirs(foldername)
    except FileExistsError:
        pass

    filename = foldername + "image_" + str(image_id).zfill(2)
    metrics.to_csv(filename + ".csv")


if __name__ == "__main__":

    image_ids = range(1, 24 + 1)

    # ---------------------------------- TBPP and Baseline Arch. Uncomment one
    # # 10x40 -> baseline bpp = 0.6
    # hidden_dims = 10 * [40]
    # final_bpp = 0.6

    # # 10x28 -> baseline bpp = 0.3
    # hidden_dims = 10 * [28]
    # final_bpp = 0.3

    # # 5x30 -> baseline bpp = 0.15
    hidden_dims = 5 * [30]
    final_bpp = 0.15

    # # 5x20 -> baseline bpp = 0.07
    # hidden_dims = 5 * [20]
    # final_bpp = 0.07

    foldername = "get_results/workshop/dataframes/bpp_" + str(final_bpp) + "/"

    aux_main = partial(
        main,
        final_bpp=final_bpp,
        baseline_hidden_dims=hidden_dims,
        foldername=foldername,
    )

    with Pool(5) as p:
        print(p.map(aux_main, image_ids))

    # for image_id in image_ids:
    #     main(image_id, final_bpp, hidden_dims, foldername)
