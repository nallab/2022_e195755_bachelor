import os.path
import unittest

from .dataset import GPTDataset

model_name = "rinna/japanese-gpt2-xsmall"


def test_発話者トークンが変換できているか確認する():
    gpt_dataset = GPTDataset("", "", model_name)
    system_token = gpt_dataset.get_tokenizer()("<system>")["input_ids"]
    assert len(system_token) == 2  # system token + bos
    user_token = gpt_dataset.get_tokenizer()("<user>")["input_ids"]
    assert len(user_token) == 2  # user token + bos


def test_GPTDataset__encode():
    system_sentence = "<system>こんばんは。\n"
    user_sentence = "<user>こんばんは。\n"
    dialog = system_sentence + user_sentence

    gpt_dataset = GPTDataset("test", "./hoge", model_name)
    encode = gpt_dataset._GPTDataset__encode(dialog, user_sentence)
    print(encode["input_ids"])
    print(encode["labels"])

    # tokenizer = gpt_dataset.get_tokenizer()
    # system_sentence_encode = tokenizer(system_sentence)
    # user_sentence_encode = tokenizer(user_sentence)
    # remove eos(eos id is 2)
    # print(system_sentence_encode["input_ids"][:-1])
    # [32001, 9, 11055, 8936, 11, 8]
    # print(user_sentence_encode["input_ids"])
    # [32000, 9, 11055, 8936, 11, 8, 2]
    want_label_ids = [
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        32000,
        9,
        11055,
        8936,
        11,
        8,
        2,
    ]
    assert encode["labels"] == want_label_ids
    assert len(encode["labels"]) == len(encode["input_ids"])


def test_labelsが期待通り作られるか():
    gpt_dataset = GPTDataset("", "", model_name)
    dialog = [32001, 9, 11055, 8936, 11, 8, 32000, 9, 11055, 8936, 11, 8, 2]
    reply = [32000, 9, 11055, 8936, 11, 8, 2]
    sample_dict = {"input_ids": dialog}
    sample_dict = gpt_dataset._GPTDataset__create_labels(
        len(dialog), len(reply), sample_dict
    )

    want_label_ids = [
        -100,
        -100,
        -100,
        -100,
        -100,
        -100,
        32000,
        9,
        11055,
        8936,
        11,
        8,
        2,
    ]
    assert sample_dict["labels"] == want_label_ids


def test_学習用データを読み込みデータセット作成を行う():
    train = os.path.abspath(os.path.join(os.path.dirname(__file__), "train"))
    gpt_dataset = GPTDataset(train_path=train, val_path="", tokenizer_name=model_name)
    dataset = gpt_dataset._GPTDataset__load_dataset(train)
    print(dataset)


def test_検証用データを読み込みデータセット作成を行う():
    val = os.path.abspath(os.path.join(os.path.dirname(__file__), "val"))
    gpt_dataset = GPTDataset(train_path=val, val_path="", tokenizer_name=model_name)
    dataset = gpt_dataset.load_dataset()
    print(dataset)


def test_token_type_idsが期待通り作られるか():
    gpt_dataset = GPTDataset("", "", model_name)
    system_sentence = "<system>こんばんは。\n"
    user_sentence = "<user>こんばんは。\n"
    dialog = system_sentence + user_sentence

    encoding = gpt_dataset.get_tokenizer()(dialog)

    sample_dict = gpt_dataset._GPTDataset__create_token_type_ids(encoding)
    assert encoding["input_ids"] == [
        32001,
        9,
        11055,
        8936,
        11,
        8,
        32000,
        9,
        11055,
        8936,
        11,
        8,
        2,
    ]
    want_token_type_ids = [gpt_dataset.system_token] * 6 + [gpt_dataset.user_token] * 7
    assert sample_dict["token_type_ids"] == want_token_type_ids


def test_padding後のencodingの各要素の長さが等しいか():
    gpt_dataset = GPTDataset("", "", model_name)
    dialog = "<user>こんばんは。\n<system>こんばんは。\n"
    reply = "<system>こんばんは。\n"
    user_token = gpt_dataset.get_tokenizer()("<user>")["input_ids"][
        0
    ]  # 'input_ids': [32000, 2]
    encoding = gpt_dataset.encode(dialog, reply)
    print(encoding)
    input_ids_len = len(encoding["input_ids"])
    attention_mask_len = len(encoding["attention_mask"])
    labels_len = len(encoding["labels"])
    token_type_ids_len = len(encoding["token_type_ids"])
    assert (
        input_ids_len == attention_mask_len == labels_len == token_type_ids_len
    ), f"input_ids_len = {input_ids_len}, attention_mask_len = {attention_mask_len}, labels_len = {labels_len}, token_type_ids_len = {token_type_ids_len}"


def test_複数行の対話データを教師データとしたデータセットの作成():
    gpt_dataset = GPTDataset("", "", model_name)
    sep_token = gpt_dataset.get_tokenizer().sep_token
    dialog = ["sample", "hogehoge", "fugafuga", "piyopiyo", "funnfunn"]
    supervised_num = 2
    expected = gpt_dataset._GPTDataset__create_multiple_dataset(dialog, supervised_num)
    wants = [
        {
            "dialog_with_sep_token": "sample" + sep_token + "hogehoge\n" + "fugafuga",
            "reply": "hogehoge\nfugafuga",
        },
        {
            "dialog_with_sep_token": "sample\n"
            + "hogehoge"
            + sep_token
            + "fugafuga\n"
            + "piyopiyo",
            "reply": "fugafuga\npiyopiyo",
        },
        {
            "dialog_with_sep_token": "sample\n"
            + "hogehoge\n"
            + "fugafuga"
            + sep_token
            + "piyopiyo\n"
            + "funnfunn",
            "reply": "piyopiyo\nfunnfunn",
        },
    ]

    assert expected == wants


if __name__ == "__main__":
    unittest.main()
