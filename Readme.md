## 実行環境

- Manjaro Linux
- Python 3.9.7
- Go 1.18 以上


## 環境構築

### とりあえずやっておいてコマンド
```
cp exp/tools/sample_conf.py exp/tools/conf.py
```

### Pythonのセットアップとpackageのインストール
```
python -m venv venv
source venv/bin/activate
pip install -r requirements-linux-dev.txt
```

### データセット構築ツールのインストール
```
go install gitlab.ie.u-ryukyu.ac.jp/e195755/dataset_builder@latest
```

### データセット構築

[こちら](https://sites.google.com/site/dialoguebreakdowndetection/chat-dialogue-corpus?authuser=0)からダウンロードしてください。

```
$ mkdir -p tmpdata/raw-data
$ mkdir -p tmpdata/data
$ mv ~/projectnextnlp-chat-dialogue-corpus.zip ./tmpdata/raw-data
$ unzip tmpdata/raw-data/projectnextnlp-chat-dialogue-corpus.zip -d tmpdata/raw-data
$ dataset_builder tmpdata/raw-data/json/*
$ mv tmpdata/raw-data/json/**/*.csv tmpdata/data
$ python exp/tools/split.py tmpdata/data
```

### コンテナイメージのビルド

```
$ mkdir output
$ make build-sif
```


### data augmentation

```
     make slurm-gpt-run GPT_EXPORT_NAME=n5 SUPERVISED_NUM=5
     make slurm-gpt-run GPT_EXPORT_NAME=n4 SUPERVISED_NUM=4
     make slurm-gpt-run GPT_EXPORT_NAME=n3 SUPERVISED_NUM=3
     make slurm-gpt-run GPT_EXPORT_NAME=n2 SUPERVISED_NUM=2
     make slurm-gpt-run GPT_EXPORT_NAME=n1 SUPERVISED_NUM=1
     make slurm-aug-and-train MODEL_DIR=n5
     make slurm-aug-and-train MODEL_DIR=n4
     make slurm-aug-and-train MODEL_DIR=n3
     make slurm-aug-and-train MODEL_DIR=n2
     make slurm-aug-and-train MODEL_DIR=n1
```
