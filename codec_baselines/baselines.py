import pdb

import pandas as pd
from compressai.utils.bench.codecs import AV1, BPG, HM, JPEG, JPEG2000, VTM, Codec, WebP
from compressai.utils.find_close.__main__ import find_closest

codec_classes = [JPEG, WebP, JPEG2000, BPG, VTM, HM, AV1]


def main(
    codec_str, image_folder, image_id, target_metric, metric, do_save: bool = False
):
    # Instantiate codec from given string
    codec_cls = next(c for c in codec_classes if c.__name__.lower() == codec_str)
    codec = codec_cls(None)

    fill_image_id = str(image_id).zfill(2)
    image_path = f"{image_folder}/kodim{fill_image_id}.png"

    quality, metrics, rec = find_closest(codec, image_path, target_metric, metric)
    achieved_metric = metrics[metric]

    if do_save:
        file_name = f"kodim{fill_image_id}_{codec_cls.__name__.lower()}_{metric}_{target_metric}.png"
        rec.save(f"./codec_baselines/results/{file_name}")

    return image_id, target_metric, achieved_metric, metrics["psnr"]


if __name__ == "__main__":

    image_folder = "./kodak_dataset/"
    metric = "bpp"
    codec_names = ["jpeg"]

    all_results = []
    for image_id in range(1, 24 + 1):
        print(f"Image ID: {image_id}")
        for target_metric in [0.15, 0.20, 0.3, 0.4, 0.5, 0.6]:
            for codec_str in codec_names:
                do_save = codec_str == "jpeg"
                foo = main(
                    codec_str, image_folder, image_id, target_metric, metric, do_save
                )
                all_results.append((codec_str,) + foo)

    df = pd.DataFrame(
        all_results,
        columns=["codec", "image_id", f"target_{metric}", f"achieved_{metric}", "psnr"],
    )
    df.to_csv("./codec_baselines/results/all_results.csv")
