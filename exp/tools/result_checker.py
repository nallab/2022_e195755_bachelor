import argparse
import logging
import os
import sys
import warnings

import conf

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join("..", ".."))),
)
import pytorch_lightning as pl
from ie_utils_tool import Mattermost
from jinja2 import Environment, FileSystemLoader
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from exp.bert import bert_conf
from exp.bert.model import SimpleExperimentLightningModel
from exp.data.dataset import __dataset_encoding_distribute

TOOLS_DIR = os.path.abspath(os.path.dirname(__file__))

TEMPLATES_DIR = os.path.join(TOOLS_DIR, "templates")
CHECKER_TEMPLATE = "checker.html"
CHECKER_OUTPUT = os.path.join(TOOLS_DIR, "result_checker.html")
MODEL_NAME = bert_conf.MODEL_NAME


def __generate_html(table, templates_dir: str, template_name: str, html_path: str):
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template(template_name)
    result = template.render(table=table)
    with open(html_path, "w") as fh:
        fh.write(result)


def bert_result_check(
    model_name: str,
    data_dir: str,
    cptk_path: str,
    templates_dir="templates",
    template_name="checker.html",
    output_html_path="result_checker.html",
):
    """It compares bert expected value and the actual value.

    Args:
        model_name: Bert model name.
        data_dir: Directory containing check data.
        cptk_path: checkpoint path of bert model.
        templates_dir: Directory of HTML templates. Default is templates.
        template_name: HTML template name. Default is checker.html.
        output_html_path: HTML path that you want to output. Default is result_checker.html.
    """
    trainer = pl.Trainer(accelerator="gpu", devices=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = DataLoader(__dataset_encoding_distribute(tokenizer, data_dir))
    model = SimpleExperimentLightningModel(model_name=model_name, num_labels=3, lr=1e-5)
    test = trainer.test(model, data, cptk_path, verbose=False)
    __generate_html(
        table=model.raw_html,
        templates_dir=templates_dir,
        template_name=template_name,
        html_path=output_html_path,
    )
    mm = Mattermost(url=conf.URL, token=conf.TOKEN)
    mm.file_upload(filepath=output_html_path, channel_id=conf.CHANNEL_ID)


if __name__ == "__main__":
    logging.getLogger("pytorch_lightning").setLevel(logging.NOTSET)
    logging.getLogger("torch.utils.data").setLevel(logging.NOTSET)
    logging.getLogger("transformers").setLevel(logging.NOTSET)
    warnings.simplefilter("ignore")

    p = argparse.ArgumentParser(description="学習したモデルを使用して破綻検出を行うツール")
    p.add_argument("checkpoint")
    p.add_argument("data_dir")
    args = p.parse_args()
    bert_result_check(
        model_name=MODEL_NAME,
        data_dir=args.data_dir,
        cptk_path=args.checkpoint,
        templates_dir=TEMPLATES_DIR,
        template_name=CHECKER_TEMPLATE,
        output_html_path=CHECKER_OUTPUT,
    )
