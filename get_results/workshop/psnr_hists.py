import os

import matplotlib.pyplot as plt
import pandas as pd
from plot_style import *


def main(target_bpp, x_label, y_label, save_dir):

    # ------------------- Get metrics from out WandB runs -------------------
    summary_frames = []
    image_ids = range(1, 24 + 1)
    for image_id in image_ids:

        img_str, bpp_str = str(image_id).zfill(2), str(target_bpp)
        path = "bpp_" + bpp_str + "/image_" + img_str
        df = pd.read_csv("get_results/workshop/dataframes/" + path + ".csv")

        # Getting metrics at the end of training
        summary = df.groupby("task_type").agg("last")

        # Only keep achieved bpp "bpp" and psrn "best_psnr"
        summary = summary[["compression/bpp", "compression/best_psnr"]]

        # Add identifiers to summary
        summary["image_id"] = image_id

        summary_frames.append(summary)

    metrics = pd.concat(summary_frames)

    fig, ax = plt.subplots(figsize=(10, 2))

    inr_tasks = ["gated", "coin_baseline", "magnitude_pruning"]

    width = 0.25

    # Order from gated
    gated = metrics.query(f"task_type == 'gated'")
    gated = gated.sort_values("compression/best_psnr", ascending=False)
    gated = gated.reset_index().drop(columns="task_type")
    mapping_df = pd.DataFrame({"current": range(1, 25), "target": gated["image_id"]})
    mapping_df = mapping_df.set_index("target")

    for i, task_type in enumerate(["gated", "coin_baseline", "magnitude_pruning"]):

        group = metrics.query(f"task_type == '{task_type}'")
        sort_idx = group["image_id"].map(mapping_df["current"])
        group["sort_idx"] = sort_idx.values

        group = group.sort_values("sort_idx")

        ax.bar(
            group["sort_idx"] + (i - len(inr_tasks) / 2) * width,
            group["compression/best_psnr"],
            label=LABELS[task_type],
            color=COLORS[task_type],
            width=width,
        )

    ax.set_xticks(range(1, 24 + 1), group["image_id"])
    ax.set_ylim(ymin=15)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.grid("on", alpha=0.4)

    fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.1))

    file_path = save_dir + "bpp_" + bpp_str
    plt.savefig(file_path + ".png", bbox_inches="tight", dpi=1000, transparent=True)
    plt.savefig(file_path + ".pdf", bbox_inches="tight", dpi=1000)
    plt.close()


if __name__ == "__main__":

    y_axis = "compression/best_psnr"
    x_axis = "image_id"

    save_dir = "figs/histograms/"
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        pass

    for target_bpp in [0.07, 0.15, 0.3, 0.6]:

        main(target_bpp, "Image", "Max PSNR [dB]", save_dir)
