import argparse
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join("..", ".."))),
)
from exp.data.conf import GPT_MODEL_STORE_DIR
from exp.gpt.augmentation import dialog_augmentation_with_gpt

if __name__ == "__main__":

    p = argparse.ArgumentParser(description="tool for augmenting using GPT-2")
    p.add_argument("source_data_dir")
    p.add_argument("export_dir")

    args = p.parse_args()
    data_dir = args.source_data_dir
    export_dir = args.export_dir
    dialog_augmentation_with_gpt(data_dir, export_dir, GPT_MODEL_STORE_DIR)
