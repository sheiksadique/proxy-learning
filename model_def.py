import torch
import sinabs
import torch.nn as nn
import torch.nn.functional as F
import sinabs.layers as sl
import pytorch_lightning as pl
from torchmetrics import Accuracy


class CNNModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.bias = False
        self.n_out = 10
        self.ann = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=(3, 3), stride=1, padding=0, bias=self.bias),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=0, bias=self.bias),
            nn.ReLU(),
            sl.SumPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=0, bias=self.bias),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=0, bias=self.bias),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=0, bias=self.bias),
            nn.ReLU(),
            sl.SumPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=1, padding=0, bias=self.bias),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 1024, bias=self.bias),
            nn.ReLU(),
            nn.Linear(1024, 512, bias=self.bias),
            nn.ReLU(),
            nn.Linear(512, 256, bias=self.bias),
            nn.ReLU(),
            nn.Linear(256, 10, bias=self.bias),
        )

        self.train_ann_accuracy = Accuracy()
        self.test_ann_accuracy = Accuracy()
        self.val_ann_accuracy = Accuracy()

    def forward(self, data):
        return self.ann(data)

    def configure_optimizers(self):
        return torch.optim.Adam(self.ann.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        out = self(data)
        loss = F.cross_entropy(out, labels)
        self.log("train_loss", loss)

        # Compute accuracy
        _, pred = torch.max(out, dim=-1)
        self.train_ann_accuracy(pred, labels)
        self.log("train_ann_accuracy", self.train_ann_accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        out = self(data)
        val_loss = F.cross_entropy(out, labels)
        self.log("val_loss", val_loss)

        # Compute accuracy
        _, pred = torch.max(out, dim=-1)
        self.val_ann_accuracy(pred, labels)
        self.log("val_ann_accuracy", self.val_ann_accuracy)
        return val_loss

    def test_step(self, batch, batch_idx):
        data, labels = batch
        out = self(data)
        test_loss = F.cross_entropy(out, labels)
        self.log("test_ann_loss", test_loss)
        # Compute accuracy
        _, pred = torch.max(out, dim=-1)
        self.test_ann_accuracy(pred, labels)
        self.log("test_ann_accuracy", self.test_ann_accuracy)
        return test_loss

    def predict_step(self, batch, batch_idx, **kwargs):
        data, labels = batch
        out = self(data)
        return out


class ProxyModel(CNNModel):
    def __init__(self):
        super().__init__()
        self.sinabs_network = sinabs.from_torch.from_model(self.ann)
        self.snn = self.sinabs_network.spiking_model

        # share same set of params across snn and ann
        for ((ann_param_name, ann_param), (snn_param_name, snn_param)) in zip(self.ann.named_parameters(), self.snn.named_parameters()):
            if (ann_param.shape != snn_param.shape):
                raise Exception(f"Parameter shapes don't match {ann_param.shape}, {snn_param.shape}")
            snn_param.data = ann_param.data

        # Metrics
        self.train_snn_accuracy = Accuracy()
        self.val_snn_accuracy = Accuracy()
        self.test_snn_accuracy = Accuracy()

    @staticmethod
    def compute_loss(out: torch.Tensor, labels: torch.Tensor):
        loss = F.cross_entropy(out, labels)
        return loss

    @staticmethod
    def compute_accuracies(out_ann: torch.Tensor, out_snn: torch.Tensor, labels: torch.Tensor, ann_accuracy: Accuracy, snn_accuracy: Accuracy):
        # ANN
        _, pred = torch.max(out_ann, dim=-1)
        ann_accuracy(pred, labels)
        # SNN
        _, pred = torch.max(out_snn.sum(1), dim=-1)
        snn_accuracy(pred, labels)

    def forward(self, data):
        img, spk = data
        (batch, time, channel, height, width) = spk.shape
        out_ann = self.ann(img)
        with torch.no_grad():
            # flatten batch time
            spk = spk.reshape((-1, channel, height, width))
            out_snn = self.snn(spk)
            # unwrap batch time
            out_snn = out_snn.reshape((batch, time, self.n_out))
        return out_ann, out_snn

    def training_step(self, batch, batch_idx):
        data, labels = batch
        # Reset snn states
        self.sinabs_network.reset_states()
        # Forward pass
        out_ann, out_snn = self(data)

        # Compute accuracy
        self.compute_accuracies(out_ann, out_snn, labels, self.train_ann_accuracy, self.train_snn_accuracy)
        self.log("train_snn_accuracy", self.train_snn_accuracy)
        self.log("train_ann_accuracy", self.train_ann_accuracy)

        # Loss computation
        # Replace ann output with snn output
        out_ann.data.copy_(out_snn.sum(1))
        loss = self.compute_loss(out_ann, labels)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        # Reset snn states
        self.sinabs_network.reset_states()
        # Forward pass
        out_ann, out_snn = self(data)
        val_loss = self.compute_loss(out_snn.sum(1), labels)
        self.log("val_loss", val_loss)

        # Compute accuracy
        self.compute_accuracies(out_ann, out_snn, labels, self.val_ann_accuracy, self.val_snn_accuracy)
        self.log("val_snn_accuracy", self.val_snn_accuracy)
        self.log("val_ann_accuracy", self.val_ann_accuracy)
        return val_loss

    def test_step(self, batch, batch_idx):
        data, labels = batch
        # Reset snn states
        self.sinabs_network.reset_states()
        # Forward pass
        out_ann, out_snn = self(data)
        test_loss = self.compute_loss(out_snn.sum(1), labels)
        self.log("test_loss", test_loss)

        # Compute accuracy
        self.compute_accuracies(out_ann, out_snn, labels, self.test_ann_accuracy, self.test_snn_accuracy)
        self.log("test_snn_accuracy", self.test_snn_accuracy)
        self.log("test_ann_accuracy", self.test_ann_accuracy)
        return test_loss

    def predict_step(self, batch, batch_idx, **kwargs):
        data, labels = batch
        out_ann, out_snn = self(data)
        return out_ann, out_snn
