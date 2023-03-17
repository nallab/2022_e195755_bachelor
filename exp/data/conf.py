import os

abspath = os.path.abspath(os.path.join(os.path.dirname(__file__)))
BERT_TRAIN_DATA = os.path.join(abspath, "bert_data", "train")
BERT_VAL_DATA = os.path.join(abspath, "bert_data", "val")
BERT_CHECK_DATA = os.path.join(abspath, "bert_data", "check")
BERT_AUG_DATA = os.path.join(abspath, "bert_data", "aug")
BERT_TEST_DATA = os.path.join(abspath, "bert_data", "test")

GPT_TRAIN_DATA = os.path.join(abspath, "gpt_data", "train")
GPT_VAL_DATA = os.path.join(abspath, "gpt_data", "val")
GPT_CHECK_DATA = os.path.join(abspath, "gpt_data", "check")
GPT_MODEL_STORE_DIR = os.path.join(abspath, "gpt2-train")
