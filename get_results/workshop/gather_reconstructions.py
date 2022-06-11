import os

import pandas as pd

import wandb

# Fix entity
ENTITY = "l0-coin"
PROJECT = "kodak"


def get_image(filters, image_id, target_bpp):
    """
    Get the best compressed models of runs according to filters.
    """

    api = wandb.Api(overrides={"entity": ENTITY, "project": PROJECT})
    runs = api.runs(path=ENTITY + "/" + PROJECT, filters=filters, order="-created_at")

    for one_run in runs:

        # Getting the image name is tricky as it has a "random" suffix.

        prefix = "media/images/img_" + str(image_id).zfill(2) + "_49999"

        files = one_run.files()
        for file in files:
            if file.name.startswith(prefix):
                file.download(replace=True)

                # Rename in human-readable, hierarchical format
                img_str = "kodim" + str(image_id).zfill(2) + "_"
                task_type = str(one_run.config["task_type"]) + "_"
                tbpp = "bpp_" + str(target_bpp)

                new_name = "media/images/" + img_str + task_type + tbpp + ".png"
                os.rename(file.name, new_name)


if __name__ == "__main__":

    image_ids = [2, 8, 15, 23]

    # ---------------------------------- TBPP and Baseline Arch. Uncomment one

    # # 10x40 -> baseline bpp = 0.6
    # hidden_dims = 10 * [40]
    # final_bpp = 0.6

    # # 10x28 -> baseline bpp = 0.3
    # hidden_dims = 10 * [28]
    # final_bpp = 0.3

    # # 5x30 -> baseline bpp = 0.15
    # hidden_dims = 5 * [30]
    # final_bpp = 0.15

    # # 5x20 -> baseline bpp = 0.07s
    # hidden_dims = 5 * [20]
    # final_bpp = 0.07

    for image_id in image_ids:
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
                                {"config.model.hidden_dims": hidden_dims},
                            ]
                        },
                        {
                            "$and": [
                                {"config.wandb.run_group": "magnitude_pruning"},
                                {"config.train.target_bpp": final_bpp},
                            ]
                        },
                    ]
                },
            ]
        }
        get_image(filters, image_id, final_bpp)
