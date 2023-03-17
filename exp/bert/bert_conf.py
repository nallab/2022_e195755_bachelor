import datetime
import os.path

MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
CPTK_FILENAME = f"bert_{str(datetime.datetime.today())}"
SAVE_MODEL_DIR = "../../model"
LARNING_RATE = 1e-6
MAX_EPOCHS = 100000
