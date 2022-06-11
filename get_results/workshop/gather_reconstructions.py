import os
from multiprocessing import Pool
from functools import partial
import pdb

import wandb

# Fix entity
ENTITY = "l0-coin"
PROJECT = "kodak"


def get_image(image_id, target_bpp, hidden_dims):
    """
    Get the best compressed models of runs according to filters.
    """

    filters = make_filters(image_id, target_bpp, hidden_dims)

    api = wandb.Api(overrides={"entity": ENTITY, "project": PROJECT}, timeout=20)
    runs = api.runs(path=ENTITY + "/" + PROJECT, filters=filters, order="-created_at")

    for one_run in runs:

        # Getting the image name is tricky as it has a "random" suffix.

        prefix = "media/images/img_" + str(image_id).zfill(2) + "_49999"
        root = "figs/reconstructions/"
        files = one_run.files()
        for file in files:
            if file.name.startswith(prefix):
                file.download(root=root, replace=True)

                # Rename in human-readable, hierarchical format
                img_str = "kodim" + str(image_id).zfill(2) + "_"
                task_type = str(one_run.config["task_type"]) + "_"
                tbpp = "bpp_" + str(target_bpp)

                new_name = "figs/reconstructions/" + img_str + task_type + tbpp + ".png"
                os.rename(root + file.name, new_name)


def make_filters(image_id, target_bpp, hidden_dims):
    filters = {
        "$and": [
            {"config.train.image_id": image_id},
            {"state": "finished"},
            {
                "$or": [
                    {
                        "$and": [
                            {"config.wandb.run_group": "all_gated"},
                            {"config.train.target_bpp": target_bpp},
                        ]
                    },
                    {
                        "$and": [
                            {"config.wandb.run_group": "baseline"},
                            {"config.model.hidden_dims": hidden_dims},
                        ]
                    },
                    {
                        "$and": [
                            {"config.wandb.run_group": "new_mp"},
                            {"config.train.target_bpp": target_bpp},
                        ]
                    },
                ]
            },
        ]
    }
    return filters


if __name__ == "__main__":

    # NOTE: Ensure that figs>reconstructions>media>images exists before running.
    image_ids = [2, 8, 15, 23]

    # ----------------- TBPP and Baseline Arch. Uncomment desired lines

    all_configs = [
        (0.07, 5 * [20]),  # 5x20 -> baseline bpp = 0.07
        (0.15, 5 * [30]),  # 5x30 -> baseline bpp = 0.15
        (0.3, 10 * [28]),  # 10x28 -> baseline bpp = 0.3
        (0.6, 10 * [40]),  # 10x40 -> baseline bpp = 0.6
    ]

    for target_bpp, hidden_dims in all_configs:
        print("hidden_dims:", hidden_dims, "target_bpp:", target_bpp)

        aux_main = partial(get_image, target_bpp=target_bpp, hidden_dims=hidden_dims)
        with Pool(5) as p:
            print(p.map(aux_main, image_ids))
