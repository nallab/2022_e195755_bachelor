import argparse
import logging
import os
import sys
import tempfile

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join("..", ".."))),
)
from ie_utils_tool import Mattermost

from exp.bert.train import train
from exp.data.conf import BERT_TRAIN_DATA, GPT_MODEL_STORE_DIR
from exp.gpt.augmentation import dialog_augmentation_with_gpt
from exp.tools.conf import CHANNEL_ID, TOKEN, URL

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="一番すごいやつ")
    p.add_argument("model_dir")
    args = p.parse_args()
    model_dir = args.model_dir
    abs_model_dir = os.path.join(GPT_MODEL_STORE_DIR, model_dir)

    logger = logging.getLogger(__name__)

    temp_dir = tempfile.mkdtemp()

    logger.info("start data augmentation")
    dialog_augmentation_with_gpt(BERT_TRAIN_DATA, temp_dir, model_dir=abs_model_dir)
    logger.info("finish data augmentation")

    cptk_path, log_path = train(is_aug=True, aug_data_dir=temp_dir)

    mattermost = Mattermost(url=URL, token=TOKEN)
    abs_log_path = os.path.abspath(log_path)
    abs_cptk_path = os.path.abspath(cptk_path)
    message = f"""Training is just finished.
    To download the training results, you can execute the following command.
    ```
    rsync -avhz "braun:{abs_log_path}" ./{model_dir}
    rsync -avhz "braun:{abs_cptk_path}" ./{model_dir}
    """
    mattermost.send_message(CHANNEL_ID, message)
