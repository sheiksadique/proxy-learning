import torch
import sinabs
import torch.nn as nn
import torch.nn.functional as F
import sinabs.layers as sl
from sinabs.activation import SingleSpike, MembraneReset, MultiSpike, MembraneSubtract, SingleExponential, PeriodicExponential
import pytorch_lightning as pl
from torchmetrics import Accuracy


def build_cifar10_ann(bias=False):
    ann = nn.Sequential(
        nn.Conv2d(3, 256, kernel_size=(3, 3), stride=1, padding=0, bias=bias),
        nn.ReLU(),
        nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=0, bias=bias),
        nn.ReLU(),
        sl.SumPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=0, bias=bias),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=0, bias=bias),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=0, bias=bias),
        nn.ReLU(),
        sl.SumPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(512, 256, kernel_size=(3, 3), stride=1, padding=0, bias=bias),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1024, 1024, bias=bias),
        nn.ReLU(),
        nn.Linear(1024, 512, bias=bias),
        nn.ReLU(),
        nn.Linear(512, 256, bias=bias),
        nn.ReLU(),
        nn.Linear(256, 10, bias=bias),
    )
    return ann


def build_fashion_mnist(bias=False):
    ann = nn.Sequential(
        nn.Conv2d(1, 128, kernel_size=3, bias=bias),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, bias=bias),
        nn.ReLU(),
        sl.SumPool2d(kernel_size=2, stride=2),
        nn.Conv2d(128, 128, kernel_size=3, bias=bias),
        nn.ReLU(),
        sl.SumPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(3200, 1024, bias=bias),
        nn.ReLU(),
        nn.Linear(1024, 256, bias=bias),
        nn.ReLU(),
        nn.Linear(256, 10, bias=bias),
    )
    return ann


class CNNModel(pl.LightningModule):
    def __init__(self, bias=False, n_out=10, lr=1e-4, betas=(0.8, 0.99), eps=1e-07, weight_decay=1e-05, model="cifar10"):
        super().__init__()
        self.save_hyperparameters()
        if model == "cifar10":
            self.ann = build_cifar10_ann(bias=bias)
        elif model == "fashion_mnist":
            self.ann = build_fashion_mnist(bias=bias)

        self.train_ann_accuracy = Accuracy()
        self.test_ann_accuracy = Accuracy()
        self.val_ann_accuracy = Accuracy()

    def forward(self, data):
        return self.ann(data)

    def configure_optimizers(self):
        return torch.optim.Adam(self.ann.parameters(),
                                lr=self.hparams.lr,
                                betas=self.hparams.betas,
                                eps=self.hparams.eps,
                                weight_decay=self.hparams.weight_decay)

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
    def __init__(self, bias=False, n_out=10, lr=1e-4, betas=(0.8, 0.99), eps=1e-07, weight_decay=1e-05, spike_threshold=3.0, spike_fn="SingleSpike", reset_fn="MembraneReset", min_v_mem=-3.0, model="cifar10"):
        super().__init__(bias=bias, n_out=n_out, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, model=model)
        self.save_hyperparameters(ignore=["spike_fn", "reset_fn"])

        self.sinabs_network = sinabs.from_torch.from_model(
            self.ann,
            spike_threshold=spike_threshold,
            spike_fn=self.get_spike_fn(spike_fn),
            reset_fn=self.get_reset_fn(reset_fn),
            surrogate_grad_fn=self.get_surrogate_grad_fn(spike_fn),
            min_v_mem=min_v_mem)
        self.snn = self.sinabs_network.spiking_model

        self.share_weights()

        # Metrics
        self.train_snn_accuracy = Accuracy()
        self.val_snn_accuracy = Accuracy()
        self.test_snn_accuracy = Accuracy()

    def share_weights(self):
        # NOTE: While ideally this should only be required once, I am having to do this every run.
        # share same set of params across snn and ann
        for (param_name, ann_param) in self.ann.named_parameters():
            self.snn.get_parameter(param_name).data = ann_param.data

    def compute_loss(self, out: torch.Tensor, labels: torch.Tensor):
        # loss = F.cross_entropy(out, labels)  # CrossEntropy fails to learn anything meaningful so far.
        loss = F.mse_loss(out, F.one_hot(labels, self.hparams.n_out).float())
        return loss

    @staticmethod
    def get_spike_fn(name: str):
        if name == "SingleSpike":
            return SingleSpike()
        elif name == "MultiSpike":
            return MultiSpike()
        else:
            raise Exception("Unknown spike mechanism")

    @staticmethod
    def get_reset_fn(name: str):
        if name == "MembraneReset":
            return MembraneReset()
        elif name == "MembraneSubtract":
            return MembraneSubtract()
        else:
            raise Exception("Unknown reset mechanism")

    @staticmethod
    def get_surrogate_grad_fn(spike_fn: str):
        if spike_fn == "SingleSpike":
            return SingleExponential()
        elif spike_fn == "MultiSpike":
            return PeriodicExponential()
        else:
            raise Exception("Unknown gradient function")

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
            out_snn = out_snn.reshape((batch, time, self.hparams.n_out))
        return out_ann, out_snn

    def training_step(self, batch, batch_idx):
        self.share_weights()
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
        out_ann.data.copy_(out_snn.mean(1))
        loss = self.compute_loss(out_ann, labels)
        self.log("train_loss", loss)

        # Compute accuracies
        with torch.no_grad():
            mean_weight_ann, mean_weight_snn = self.compute_mean_weights()
            self.log("mean_weight_ann", mean_weight_ann.item())
            self.log("mean_weight_snn", mean_weight_snn.item())

        return loss

    def on_validation_start(self) -> None:
        self.share_weights()

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        # Reset snn states
        self.sinabs_network.reset_states()
        # Forward pass
        out_ann, out_snn = self(data)
        val_loss = self.compute_loss(out_snn.mean(1), labels)
        self.log("val_loss", val_loss)

        # Compute accuracy
        with torch.no_grad():
            self.compute_accuracies(out_ann, out_snn, labels, self.val_ann_accuracy, self.val_snn_accuracy)
            self.log("val_snn_accuracy", self.val_snn_accuracy)
            self.log("val_ann_accuracy", self.val_ann_accuracy)
        return val_loss

    def on_test_start(self) -> None:
        self.share_weights()

    def test_step(self, batch, batch_idx):
        data, labels = batch
        # Reset snn states
        self.sinabs_network.reset_states()
        # Forward pass
        out_ann, out_snn = self(data)
        test_loss = self.compute_loss(out_snn.mean(1), labels)
        self.log("test_loss", test_loss)

        # Compute accuracy
        with torch.no_grad():
            self.compute_accuracies(out_ann, out_snn, labels, self.test_ann_accuracy, self.test_snn_accuracy)
            self.log("test_snn_accuracy", self.test_snn_accuracy)
            self.log("test_ann_accuracy", self.test_ann_accuracy)
        return test_loss

    def predict_step(self, batch, batch_idx, **kwargs):
        data, labels = batch
        out_ann, out_snn = self(data)
        return out_ann, out_snn

    def compute_mean_weights(self):
        param_count = 0
        sum_all_weights_ann = 0
        sum_all_weights_snn = 0
        for p in self.ann.parameters():
            param_count += p.numel()
            sum_all_weights_ann += p.abs().sum()

        for p in self.snn.parameters():
            param_count += p.numel()
            sum_all_weights_snn += p.abs().sum()

        return sum_all_weights_ann/param_count, sum_all_weights_snn/param_count
