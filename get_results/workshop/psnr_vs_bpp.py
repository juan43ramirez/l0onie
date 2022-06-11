import os

import matplotlib.pyplot as plt
import pandas as pd
from plot_style import *


def main(x_label, y_label, save_dir):

    # ------------------- Get metrics from out WandB runs -------------------
    summary_frames = []
    image_ids = range(1, 24 + 1)
    for target_bpp in [0.07, 0.15, 0.3, 0.6]:
        for image_id in image_ids:

            img_str, bpp_str = str(image_id).zfill(2), str(target_bpp)
            path = "bpp_" + bpp_str + "/image_" + img_str
            df = pd.read_csv("get_results/workshop/dataframes/" + path + ".csv")

            # Getting metrics at the end of training
            summary = df.groupby("task_type").agg("last")
            summary = summary.drop(columns=["_runtime"])

            # Only keep achieved bpp "bpp" and psrn "best_psnr"
            summary = summary[["compression/bpp", "compression/best_psnr"]]

            # Add identifiers to summary
            summary["image_id"] = image_id
            summary["target_bpp"] = target_bpp

            summary_frames.append(summary)

    metrics = pd.concat(summary_frames)

    # Mean psnr and achieved bpp across images for each task_type and target bpp
    metrics = metrics.groupby(["task_type", "target_bpp"]).agg("mean")

    # ------------------- Get metrics from Codec baselines -------------------
    codec = pd.read_csv("codec_baselines/results/all_results.csv")
    codec_summary = codec.groupby(["codec", "target_bpp"]).agg("mean")

    # ------------------------------- Plot ------------------------------------

    fig, ax = plt.subplots(figsize=(3, 2))

    inr_tasks = ["gated", "coin_baseline", "magnitude_pruning"]
    codec_tasks = ["jpeg"]

    for task_type in inr_tasks + codec_tasks:

        if task_type in inr_tasks:
            group = metrics.query(f"task_type == '{task_type}'")
            y_axis = "compression/best_psnr"
            x_axis = "compression/bpp"

        elif task_type in codec_tasks:
            group = codec_summary.query(f"codec == '{task_type}'")
            y_axis = "psnr"
            x_axis = "achieved_bpp"

        group.plot(
            ax=ax,
            y=y_axis,
            ylabel=y_label,
            x=x_axis,
            xlabel=x_label,
            label=LABELS[task_type],
            legend=False,
            alpha=0.8,
            linewidth=LINEWIDTH,
            color=COLORS[task_type],
            zorder=ORDERS[task_type],
        )
        ax.scatter(group[x_axis], group[y_axis], c=COLORS[task_type], s=10)

    ax.set_xticks([0.0, 0.2, 0.4, 0.6])

    plt.grid("on", alpha=0.4)

    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.1))

    file_path = save_dir + "aggregated"
    plt.savefig(file_path + ".png", bbox_inches="tight", dpi=1000, transparent=True)
    plt.savefig(file_path + ".pdf", bbox_inches="tight", dpi=1000)
    plt.close()


if __name__ == "__main__":

    y_label = "Mean PSNR [dB]"
    x_label = "BPP"

    save_dir = "figs/psnr_vs_bpp/"
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        pass

    main(x_label, y_label, save_dir)
