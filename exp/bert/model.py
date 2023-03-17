import pytorch_lightning as pl
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
)

from src.evalution import accuracy, jsd, mse


class SimpleExperimentModel(torch.nn.Module):
    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_state = True
        self.bert = AutoModel.from_pretrained(
            model_name,
        )

        # self.lstm = torch.nn.LSTM(self.config.hidden_size, self.config.hidden_size, batch_first=True)
        self.regressor = torch.nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids, attention_mask, token_type_ids, output_hidden_states=True
        )
        """
        out, _ = self.lstm(outputs["last_hidden_state"].detach(), None)
        sequence_output = out[:, -1, :]
        logits = self.regressor(sequence_output)
        """
        logits = self.regressor(outputs["last_hidden_state"][:, -1, :])
        return torch.nn.functional.softmax(logits, 1)


class SimpleExperimentLightningModel(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.bert = SimpleExperimentModel(model_name, num_labels, lr)
        self.save_hyperparameters()
        self.raw_html = ""
        self.row_num = 0

    def training_step(self, batch, batch_idx):
        output = self.bert(
            batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
        )
        labels = batch.pop("labels")
        pred = output.view(-1, self.hparams.num_labels)
        loss = jsd(pred, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.bert(
            batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
        )
        labels = batch.pop("labels")
        pred = output.view(-1, self.hparams.num_labels)
        loss = jsd(pred, labels)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        output = self.bert(
            batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
        )
        labels = batch.pop("labels")
        pred = output.view(-1, self.hparams.num_labels)
        loss = mse(pred, labels)
        self.log("mse", loss)

        loss = jsd(pred, labels)
        self.log("jsd", loss)

        acc = accuracy(pred, labels)
        self.log("accuracy", acc)

        self.raw_html += (
            f"<tr>"
            f"<td>{batch['text']}</td>"
            f"<td>{round(float(labels[0][0]), 2)}</td>"
            f"<td>{round(float(labels[0][1]), 2)}</td>"
            f"<td>{round(float(labels[0][2]), 2)}</td>"
            f"<td>{round(float(pred[0][0]), 2)}</td>"
            f"<td>{round(float(pred[0][1]), 2)}</td>"
            f"<td>{round(float(pred[0][2]), 2)}</td>"
            f"</tr>\n"
        )

        print(f"nb: pb : b, \npredict:{pred}")

    def forward(self, batch):
        output = self.bert(
            batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
        )
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def output_html(self):
        return self.raw_html
