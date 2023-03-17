import logging
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join("..", ".."))),
)
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer

from exp.bert.model import SimpleExperimentLightningModel
from exp.data.conf import BERT_AUG_DATA
from exp.data.dataset import SupervisedType, specify_dataset_dir_dataloader

from .bert_conf import (
    CPTK_FILENAME,
    LARNING_RATE,
    MAX_EPOCHS,
    MODEL_NAME,
    SAVE_MODEL_DIR,
)


def train(is_aug, aug_data_dir=BERT_AUG_DATA):
    model_name = MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # dataloader_train, dataloader_val, dataloader_test  = dataset.mixed_dataloader(tokenizer, "dev", True,
    # SupervisedType.DISTRIBUTE)
    dataloader_train, dataloader_val, dataloader_test = specify_dataset_dir_dataloader(
        tokenizer, True, SupervisedType.DISTRIBUTE, is_aug, aug_data_dir
    )

    logging.warning("dataloader_train")
    logging.warning(len(dataloader_train))
    logging.warning("dataloader_val")
    logging.warning(len(dataloader_val))
    logging.warning("dataloader_test")
    logging.warning(len(dataloader_test))

    logger_filename = Path(CPTK_FILENAME).stem
    logger = TensorBoardLogger("lightning_logs", name=logger_filename)
    logger_dir = os.path.join("lightning_logs", logger_filename)

    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        dirpath=SAVE_MODEL_DIR,
        filename=CPTK_FILENAME,
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=int(MAX_EPOCHS),
        callbacks=[
            checkpoint,
            pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", mode="min"),
        ],
        logger=logger,
    )

    model = SimpleExperimentLightningModel(
        model_name=model_name, num_labels=3, lr=LARNING_RATE
    )

    trainer.fit(model, dataloader_train, dataloader_val)

    best_model_path = checkpoint.best_model_path
    print("score", checkpoint.best_model_score)

    test = trainer.test(model, dataloader_test)
    print("check", test[0]["mse"])
    return best_model_path, logger_dir
