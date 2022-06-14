import os
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import pandas as pd
from plot_style import *


def main(image_id, target_bpp, x_axis, y_metrics, y_labels, df_file, save_dir):

    metrics = pd.read_csv(df_file + ".csv")
    metrics["_runtime"] = metrics["_runtime"] / 60

    # Three plots: PSNR, BPP and Lambda
    fig, axs = plt.subplots(1, 3, figsize=(9, 2))

    for task_type in ["gated", "coin_baseline", "magnitude_pruning"]:
        group = metrics.query(f"task_type == '{task_type}'")

        # Keeping some kwargs used both for psnr and bpp plots
        shared_kwargs = {
            "label": LABELS[task_type],
            "x": x_axis,
            "legend": False,
            "alpha": 0.8,
            "linewidth": LINEWIDTH,
            "color": COLORS[task_type],
            "zorder": ORDERS[task_type],
        }

        for i, (metric, ylab) in enumerate(zip(y_metrics, y_labels)):

            if metric == "train/lambda_01" and task_type != "gated":
                pass

            group.plot(ax=axs[i], y=metric, ylabel=ylab, **shared_kwargs)
            axs[i].set_xlabel("Step")

    # Set ymin dynamically based on the highest psnr achieved
    very_best_psnr = metrics["compression/best_psnr"].max()
    axs[0].set_ylim(ymin=int(0.8 * very_best_psnr))

    # Same for bpp
    axs[1].set_ylim(ymin=0.8 * target_bpp)

    # Xticks which do not clutter too much
    x_ticks = [0, 10_000, 20_000, 30_000, 40_000, 50_000]
    x_tick_labs = ["0", "10k", "20k", "30k", "40k", "50k"]
    for i in range(3):
        axs[i].set_xticks(x_ticks, x_tick_labs)
        axs[i].grid("on", alpha=0.4)

    fig.tight_layout(pad=0.3)

    # Legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.01))

    try:
        os.mkdir("figs/dynamics/")
    except FileExistsError:
        pass

    file_path = save_dir + "image_" + str(image_id).zfill(2)
    plt.savefig(file_path + ".png", bbox_inches="tight", dpi=1000, transparent=True)
    plt.savefig(file_path + ".pdf", bbox_inches="tight", dpi=1000)
    plt.close()


def aux_main(image_id, target_bpp, x_axis, y_metrics, y_labels, save_dir):

    print("Image: ", str(image_id))
    img_str, bpp_str = str(image_id).zfill(2), str(target_bpp)
    df_file = "get_results/workshop/dataframes/bpp_" + bpp_str + "/image_" + img_str

    main(image_id, target_bpp, x_axis, y_metrics, y_labels, df_file, save_dir)


if __name__ == "__main__":

    # Filters
    target_bpp = 0.3
    baseline_hidden_dims = 10 * [28]

    # Metrics for the plot axis
    y_metrics = ["compression/best_psnr", "compression/bpp", "train/lambda_01"]
    y_labels = ["Max PSNR [dB]", "BPP", r"$\lambda_{co}$"]
    x_axis = "_step"

    save_dir = "figs/dynamics/"
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        pass

    image_ids = range(1, 24 + 1)

    aux_callable = partial(
        aux_main,
        target_bpp=target_bpp,
        x_axis=x_axis,
        y_metrics=y_metrics,
        y_labels=y_labels,
        save_dir=save_dir,
    )
    with Pool(5) as p:
        print(p.map(aux_callable, image_ids))
