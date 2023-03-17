import argparse
import os
import sys

from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    T5Tokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
)

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join("..", ".."))),
)
from ie_utils_tool import Mattermost

from exp.data.conf import (
    GPT_CHECK_DATA,
    GPT_MODEL_STORE_DIR,
    GPT_TRAIN_DATA,
    GPT_VAL_DATA,
)
from exp.data.dataset import GPTDataset
from exp.tools.conf import CHANNEL_ID, TOKEN, URL

assert os.path.exists(
    GPT_TRAIN_DATA
), f"GPT_TRAIN_DATA is not exist. Path is {GPT_TRAIN_DATA}"
assert os.path.exists(
    GPT_VAL_DATA
), f"GPT_VAL_DATA is not exist. Path is {GPT_VAL_DATA}"
assert os.path.exists(
    GPT_CHECK_DATA
), f"GPT_CHECK_DATA is not exist. Path is {GPT_CHECK_DATA}"

p = argparse.ArgumentParser(description="training bert model...")
p.add_argument("export_name")
p.add_argument("supervised_num")
args = p.parse_args()
export_name = args.export_name
supervised_num = int(args.supervised_num)

model_name = "rinna/japanese-gpt2-xsmall"
gpt_dataset = GPTDataset(GPT_TRAIN_DATA, GPT_VAL_DATA, model_name)
tokenizer = gpt_dataset.get_tokenizer()
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer.get_vocab()))

# train_dataset, val_dataset = gpt_dataset.load_dataset()
train_dataset, val_dataset = gpt_dataset.load_multiple_supervised_dataset(
    supervised_num
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
output_dir = os.path.join(GPT_MODEL_STORE_DIR, export_name)
training_args = TrainingArguments(
    output_dir=output_dir,  # The output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=100,  # number of training epochs
    per_device_train_batch_size=20,  # batch size for training
    per_device_eval_batch_size=20,  # batch size for evaluation
    eval_steps=50,  # Number of update steps between two evaluations.
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    logging_steps=1,
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    save_strategy="steps",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.add_callback(EarlyStoppingCallback)
trainer.train()
trainer.save_model()

mattermost = Mattermost(url=URL, token=TOKEN)
abs_output_dir = os.path.abspath(output_dir)
message = f"""Training is just finished.
To download the training results, you can execute the following command.
```
rsync -avhz "braun:{abs_output_dir}" ./
"""
mattermost.send_message(CHANNEL_ID, message)
