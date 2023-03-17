import argparse
import glob
import os.path
import random
import shutil
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join("..", ".."))),
)

from exp.data.conf import BERT_TRAIN_DATA, BERT_VAL_DATA, BERT_CHECK_DATA, BERT_TEST_DATA, GPT_TRAIN_DATA, GPT_VAL_DATA, GPT_CHECK_DATA  

def copy_files(files, target_dir):
    for file in files:
        filename = os.path.basename(file)
        shutil.copyfile(file, os.path.join(target_dir, filename))

bert_train_data_path = BERT_TRAIN_DATA
bert_val_data_path = BERT_VAL_DATA
bert_check_data_path = BERT_CHECK_DATA
bert_test_data_path = BERT_TEST_DATA
  
gpt_train_data_path = GPT_TRAIN_DATA
gpt_val_data_path = GPT_VAL_DATA
gpt_check_data_path = GPT_CHECK_DATA


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="split data to train and evaluation.")
    p.add_argument("source_dir")

    args = p.parse_args()

    source_dir = args.source_dir

    files = glob.glob(os.path.join(source_dir, "*.csv"))
    len(files)
    bert_data_len = int(len(files) / 2)
    gpt_data_len = len(files) - bert_data_len
    
    bert_check_data_len = 1
    bert_train_data_len = int(bert_data_len * 0.8) + bert_check_data_len
    bert_val_data_len = int(bert_data_len * 0.1) + bert_train_data_len
    bert_test_data_len = bert_data_len

    bert_check_data = files[:bert_check_data_len]
    bert_train_data = files[bert_check_data_len:bert_train_data_len]
    bert_val_data = files[bert_train_data_len:bert_val_data_len]
    bert_test_data = files[bert_val_data_len:bert_test_data_len]

    gpt_train_data_len = int(gpt_data_len * 0.8) + bert_data_len
    gpt_val_data_len = int(gpt_data_len * 0.1) + gpt_train_data_len
    gpt_check_data_len = len(files)

    gpt_train_data = files[bert_data_len:gpt_train_data_len]
    gpt_val_data = files[gpt_train_data_len:gpt_val_data_len]
    gpt_check_data = files[gpt_val_data_len:]

    # print(len(files))
    # print(bert_check_data_len)
    # print(bert_train_data_len)
    # print(bert_val_data_len)
    # print(bert_test_data_len)

    # print(gpt_train_data_len)
    # print(gpt_val_data_len)
    # print(gpt_check_data_len)

    copy_files(bert_check_data ,bert_check_data_path)
    copy_files(bert_train_data ,bert_train_data_path)
    copy_files(bert_val_data ,bert_val_data_path)
    copy_files(bert_test_data ,bert_test_data_path)
    copy_files(gpt_train_data ,gpt_train_data_path)
    copy_files(gpt_val_data ,gpt_val_data_path)
    copy_files(gpt_check_data ,gpt_check_data_path)
