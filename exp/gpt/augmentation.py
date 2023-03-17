import argparse
import csv
import glob
import logging
import os
import shutil
import sys

from transformers import pipeline

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join("..", ".."))),
)
from exp.data.conf import GPT_MODEL_STORE_DIR
from exp.data.dataset import GPTDataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s- %(name)s - %(levelname)s - %(message)s"
)


def dialog_augmentation_with_gpt(data_dir: str, export_dir: str, model_dir: str):
    gpt_dataset = GPTDataset("", "", "rinna/japanese-gpt2-xsmall")
    chef = pipeline(
        "text-generation",
        model=model_dir,
        tokenizer=gpt_dataset.get_tokenizer(),
        max_length=400,
    )
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    for file in files:
        arg_file_path = os.path.join(export_dir, "aug_" + os.path.basename(file))
        shutil.copy2(file, arg_file_path)
        dialog = ""
        last_speaker = ""
        with open(file) as fh:
            rows = csv.reader(fh, delimiter=",")
            is_head = True
            for row in rows:
                if is_head:
                    is_head = False
                    continue
                assert (
                    row[0] == "S" or row[0] == "U"
                ), f"wants row[0] value is s or u, but actual value is {row[0]}"
                speaker = "<system>" if row[0] == "S" else "<user>"
                sentence = row[1]
                dialog += speaker + sentence + "\n"
                last_speaker = "U" if row[0] == "S" else "S"
        dialog += gpt_dataset.get_tokenizer().sep_token + "<user>"
        logger.debug("this is dialog")
        logger.debug(dialog)
        logger.debug("------")
        generated_text = chef(dialog)[0]["generated_text"]
        logger.debug(generated_text)
        aug_reply = generated_text[len(dialog) :]
        logger.debug("======")
        logger.debug(aug_reply)

        with open(arg_file_path, "a") as fh:
            row = ["A", aug_reply, 30, 0, 0]
            csv.writer(fh, delimiter=",").writerow(row)
