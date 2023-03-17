import csv
import glob
import logging
import os.path
import random
from enum import Enum
from logging import getLogger
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.dataset import T_co
from transformers import PreTrainedTokenizerBase, T5Tokenizer

from .conf import BERT_AUG_DATA, BERT_CHECK_DATA, BERT_TRAIN_DATA, BERT_VAL_DATA

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s- %(name)s - %(levelname)s - %(message)s"
)
logger = getLogger(__name__)


class SupervisedType(Enum):
    ACCURACY = 1
    DISTRIBUTE = 2


def specify_dataset_dir_dataloader(
    tokenizer,
    is_train_data_shuffle: bool,
    supervised_type: SupervisedType,
    is_aug: bool,
    aug_data_dir: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """BERTで学習するためのデータセットを取得する関数。

    Args:
        tokenizer:
        is_train_data_shuffle:
        supervised_type:
        is_aug:
        aug_data_dir:

    Returns:

    """
    train_csv_dir_path = ""
    val_csv_dir_path = BERT_VAL_DATA
    test_csv_dir_path = BERT_CHECK_DATA
    if is_aug:
        # is_aug が true じゃないと aug_data_dir が使われないのが気になる
        # バグの原因になりそう
        train_csv_dir_path = aug_data_dir
    else:
        train_csv_dir_path = BERT_TRAIN_DATA

    train_encoded = ""
    val_encoded = ""
    test_encoded = ""
    logger.info(f"train_path = {train_csv_dir_path}")
    logger.info(f"val_path = {val_csv_dir_path}")
    logger.info(f"test_path = {test_csv_dir_path}")

    assert os.path.exists(train_csv_dir_path), f"{train_csv_dir_path} is not exists."
    assert os.path.exists(val_csv_dir_path), f"{val_csv_dir_path} is not exists."
    assert os.path.exists(test_csv_dir_path), f"{test_csv_dir_path} is not exists."

    logger.debug(f"supervised type is {supervised_type}")
    if supervised_type == SupervisedType.ACCURACY:
        train_encoded = __dataset_encoding_accuracy(tokenizer, train_csv_dir_path)
        val_encoded = __dataset_encoding_accuracy(tokenizer, val_csv_dir_path)
        test_encoded = __dataset_encoding_accuracy(tokenizer, test_csv_dir_path)

    elif supervised_type == SupervisedType.DISTRIBUTE:
        train_encoded = __dataset_encoding_distribute(tokenizer, train_csv_dir_path)
        val_encoded = __dataset_encoding_distribute(tokenizer, val_csv_dir_path)
        test_encoded = __dataset_encoding_distribute(tokenizer, test_csv_dir_path)

    dataloader_train = DataLoader(
        train_encoded, batch_size=40, shuffle=is_train_data_shuffle
    )
    dataloader_val = DataLoader(val_encoded, batch_size=10)
    dataloader_test = DataLoader(test_encoded, batch_size=10)
    return dataloader_train, dataloader_val, dataloader_test


def __dataset_encoding_distribute(tokenizer, csv_dir_path: str):
    def fuga(nb, pb, b):
        sum = nb + pb + b
        index = [float(nb / sum), float(pb / sum), float(b / sum)]
        return index

    return hoge(tokenizer, csv_dir_path, fuga)


def hoge(tokenizer, csv_dir_path: str, exec_labels):
    dataset_for_loader = []

    for file in glob.glob(os.path.join(csv_dir_path, "*.csv")):
        with open(file) as fh:
            rows = csv.reader(fh, delimiter=",")
            texts = ""
            next(rows)
            for row in rows:
                speaker = row[0]
                text = row[1].strip().replace("　", "") + "\n"
                old_texts = texts
                texts += text
                if speaker == "U":
                    continue
                nb = int(row[2])
                pb = int(row[3])
                b = int(row[4])
                index = exec_labels(nb, pb, b)
                n = len(old_texts)
                text_len = len(text)
                if n + text_len + 3 > 512:
                    new_text = texts
                    while True:
                        new_text = new_text.split("\n")[1:]
                        new_text = "\n".join(new_text)
                        n = len(new_text)
                        if n <= 512 - 3:
                            input_tokenizer = new_text
                            break
                else:
                    input_tokenizer = old_texts + text
                encoding = tokenizer(
                    input_tokenizer,
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                )
                encoding["labels"] = index
                encoding = {k: torch.tensor(v) for k, v in encoding.items()}
                encoding["text"] = input_tokenizer
                dataset_for_loader.append(encoding)
    return dataset_for_loader


def __dataset_encoding_accuracy(tokenizer, csv_dir_path: str):
    def fuga(nb, pb, b):
        tmp = [nb, pb, b]
        # labels = ["NB", "PB", "B"]
        index = tmp.index(max(tmp))
        return index

    return hoge(tokenizer, csv_dir_path, fuga)


class GPTDataset:
    def __init__(self, train_path: str, val_path: str, tokenizer_name: str):
        self.train_path = train_path
        self.val_path = val_path
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.do_lower_case = (
            True  # due to some bug of tokenizer config loading
        )

        special_token = {"additional_special_tokens": ["<user>", "<system>"]}
        self.tokenizer.add_special_tokens(special_token)
        self.user_token = self.tokenizer("<user>")["input_ids"][0]
        self.system_token = self.tokenizer("<system>")["input_ids"][0]

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer

    def load_dataset(self):
        """旧学習方法で学習する場合はこの関数を呼んでください

        Returns:

        """
        dataset_train = self.__load_dataset(self.train_path)
        dataloader_train = GPTDataSetWrapper(dataset_train)
        dataset_val = self.__load_dataset(self.val_path)
        dataloader_val = GPTDataSetWrapper(dataset_val)
        return dataloader_train, dataloader_val

    def load_multiple_supervised_dataset(self, supervised_num):
        """新学習方法で学習する場合はこの関数を呼んでください

        Args:
            supervised_num: 実際に学習する対話の数

        Returns:

        """
        dataset_train = self.__multiple_rows_dataset(self.train_path, supervised_num)
        dataloader_train = GPTDataSetWrapper(dataset_train)
        dataset_val = self.__multiple_rows_dataset(self.val_path, supervised_num)
        dataloader_val = GPTDataSetWrapper(dataset_val)
        return dataloader_train, dataloader_val

    def __encode(self, dialog: str, reply: str):
        """渡された会話とリプライをエンコーディングします

        dialog には reply を進める必要があるので、以下のようにしてください。
        (例)
        - dialog: <u>こんにちは、良い天気ですね。<s>そうですね、久しぶりに晴れて気持ちいいです。
        - reply: <s>そうですね、久しぶりに晴れて気持ちいいです。

        Args:
            dialog:
            reply:

        Returns:

        """
        encoded_reply = self.tokenizer(text=reply)
        reply_length = len(encoded_reply["input_ids"])
        encoding = self.tokenizer(text=dialog, truncation=True)
        dialog_length = len(encoding["input_ids"])

        # debug code
        # hoge = self.tokenizer.convert_tokens_to_string(encoding["input_ids"])

        encoding = self.__create_labels(dialog_length, reply_length, encoding)
        encoding = self.__create_token_type_ids(encoding)
        return encoding

    def __create_labels(self, dialog_length: int, reply_length: int, encoding: dict):
        """labelsを付与します。encodingにはinput_idsが辞書の要素として必要です。

        labelsの説明はこっち。
        https://songstudio.info/tech/tech-35/
        https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel.forward.labels

        Args:
            dialog_length:
            reply_length:
            encoding:

        Returns:

        """
        ignore_length = dialog_length - reply_length
        ignore_labels = [-100] * ignore_length
        encoding["labels"] = ignore_labels + encoding["input_ids"][ignore_length:]
        return encoding

    def __create_token_type_ids(self, encoding: dict):
        token_type_ids = []
        speaker = ""
        for input_id in encoding["input_ids"]:
            if input_id == self.user_token:
                speaker = self.user_token
            elif input_id == self.system_token:
                speaker = self.system_token
            token_type_ids = token_type_ids + [speaker]
        encoding["token_type_ids"] = token_type_ids
        return encoding

    def padding(self, encoding):
        """input_idsとattention_mask, labels, token_type_idsの長さを 1024にpaddingします。

        Args:
            encoding:

        Returns:

        """
        max_length = 1024
        input_length = len(encoding["input_ids"])
        pad_length = max_length - input_length

        input_ids_len = len(encoding["input_ids"])
        attention_mask_len = len(encoding["attention_mask"])
        labels_len = len(encoding["labels"])
        token_type_ids_len = len(encoding["token_type_ids"])
        assert (
            input_ids_len == attention_mask_len == labels_len == token_type_ids_len
        ), f"input_ids_len = {input_ids_len}, attention_mask_len = {attention_mask_len}, labels_len = {labels_len}, token_type_ids_len = {token_type_ids_len}"

        encoding["input_ids"] = (
            encoding["input_ids"] + [self.tokenizer.pad_token_id] * pad_length
        )
        encoding["labels"] = encoding["labels"] + [-100] * pad_length
        encoding["attention_mask"] = encoding["attention_mask"] + [1] * pad_length
        encoding["token_type_ids"] = (
            encoding["token_type_ids"] + [encoding["token_type_ids"][-1]] * pad_length
        )
        return encoding

    def __load_dataset(self, data_dir: str):
        """

        内部でencodeを読み込んでいるので、encodeされた結果がlistで返ってきます。
        Args:
            data_dir:

        Returns:

        """
        encoded_dataset = []
        files = glob.glob(os.path.join(data_dir, "*.csv"))
        for file in files:
            with open(file) as fh:
                rows = csv.reader(fh, delimiter=",")
                dialog = ""
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
                    reply = speaker + sentence + "\n"
                    if dialog == "":
                        dialog += reply
                        continue
                    encoded_dataset.append(
                        self.encode(
                            dialog + self.tokenizer.sep_token + reply,
                            reply,
                        )
                    )
                    dialog += reply
        return encoded_dataset

    def __multiple_rows_dataset(self, data_dir: str, supervised_num: int):
        """

        内部でencodeを読み込んでいるので、encodeされた結果がlistで返ってきます。
        Args:
            data_dir:

        Returns:

        """
        encoded_dataset = []
        files = glob.glob(os.path.join(data_dir, "*.csv"))
        for file in files:
            with open(file) as fh:
                rows = csv.reader(fh, delimiter=",")
                dialog = []
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
                    dialog.append(speaker + sentence)

                hoge = self.__create_multiple_dataset(dialog, supervised_num)
                for datum in hoge:
                    encoded_dataset.append(
                        self.encode(datum["dialog_with_sep_token"], datum["reply"])
                    )
        return encoded_dataset

    def __create_multiple_dataset(self, dialog: list, supervised_num: int):
        loop_num = len(dialog) - supervised_num
        raw_dataset = []
        for i in range(1, loop_num + 1, 1):
            before_sep_sentences = dialog[:i]
            after_sep_sentences = dialog[i : i + supervised_num]
            before_sep = "\n".join(before_sep_sentences)
            after_sep = "\n".join(after_sep_sentences)
            raw_dataset.append(
                {
                    "dialog_with_sep_token": before_sep
                    + self.tokenizer.sep_token
                    + after_sep,
                    "reply": after_sep,
                }
            )
        return raw_dataset

    def encode(self, dialog: str, reply):
        """渡された会話とリプライをエンコーディングします

        dialog には reply を進める必要があるので、以下のようにしてください。
        (例)
        - dialog: <u>こんにちは、良い天気ですね。<s>そうですね、久しぶりに晴れて気持ちいいです。
        - reply: <s>そうですね、久しぶりに晴れて気持ちいいです。

        Args:
            dialog:
            reply:

        Returns:

        """
        dialog_length = len(dialog)
        if dialog_length > 1024:
            while True:
                split_dialog = dialog.split("\n")[1:]
                truncate_dialog = "\n".join(split_dialog)
                dialog = truncate_dialog
                truncate_length = len(truncate_dialog)
                logging.debug(f"truncate_length={truncate_length}")
                if truncate_length < 1024:
                    break
        encoding = self.__encode(dialog=dialog, reply=reply)
        encoding = self.padding(encoding)
        encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        return encoding


class GPTDataSetWrapper(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
