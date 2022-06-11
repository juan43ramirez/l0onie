import os

import matplotlib.pyplot as plt
import pandas as pd
from plot_style import *


def main(x_axis, y_axis, y_label, df_file, save_dir):

    metrics = pd.read_csv(df_file + ".csv")
    metrics["_runtime"] = metrics["_runtime"] / 60

    fig, ax = plt.subplots(figsize=(3, 2))

    for task_type in ["gated", "coin_baseline", "magnitude_pruning"]:
        group = metrics.query(f"task_type == '{task_type}'")

        group.plot(
            ax=ax,
            y=y_axis,
            ylabel=y_label,
            label=LABELS[task_type],
            x=x_axis,
            legend=False,
            alpha=0.8,
            linewidth=LINEWIDTH,
            color=COLORS[task_type],
            zorder=ORDERS[task_type],
        )

        ax.set_xlabel("Runtime [min]")

    # Set ymin dynamically based on the highest psnr achieved
    very_best_psnr = metrics["compression/best_psnr"].max().max()
    ax.set_ylim(ymin=int(0.8 * very_best_psnr))

    # Set ticks dynamically based on run length
    base_ticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    # First max works per group, second max works across the maxes for groups
    last_time = metrics["_runtime"].max().max()
    actual_ticks = [x for x in base_ticks if x <= last_time]
    ax.set_xticks(actual_ticks)

    plt.grid("on", alpha=0.4)

    fig.tight_layout(pad=0.3)

    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.01))

    file_path = save_dir + "image_" + str(image_id).zfill(2)
    plt.savefig(file_path + ".png", bbox_inches="tight", dpi=1000, transparent=True)
    plt.savefig(file_path + ".pdf", bbox_inches="tight", dpi=1000)
    plt.close()


if __name__ == "__main__":

    # Filters
    target_bpp = 0.3
    baseline_hidden_dims = 10 * [28]

    # Metrics for the plot axis
    y_axis = "compression/best_psnr"
    x_axis = "_runtime"

    y_label = "Max PSNR [dB]"

    save_dir = "figs/psnr_vs_time/"
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        pass

    image_ids = range(1, 24 + 1)
    for image_id in image_ids:

        print("Image: ", str(image_id))

        img_str, bpp_str = str(image_id).zfill(2), str(target_bpp)
        df_file = "get_results/workshop/dataframes/bpp_" + bpp_str + "/image_" + img_str

        main(x_axis, y_axis, y_label, df_file, save_dir)
