from model_def import CNNModel, ProxyModel
from cifar10 import Cifar10DataModule, ProxyCifar10
import pytorch_lightning as pl
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from sinabs.activation import SingleSpike, MembraneReset


def train_cnn_only():
    model = CNNModel()
    dm = Cifar10DataModule(batch_size=32, num_workers=10)
    mlf_logger = MLFlowLogger(experiment_name="ANN pure", tracking_uri="file:./ml-runs")
    trainer = pl.Trainer(gpus=1, logger=mlf_logger, max_steps=15)
    trainer.fit(model, datamodule=dm)


def train_snn_proxy():
    model = ProxyModel(bias=False, n_out=10, lr=1e-4, betas=(0.8, 0.99), eps=1e-07, weight_decay=1e-05, spike_threshold=3.0, spike_fn="SingleSpike", reset_fn="MembraneReset", min_v_mem=-3.0)
    dm = ProxyCifar10(batch_size=16, time_steps=60, num_workers=10)
    mlf_logger = MLFlowLogger(experiment_name="Proxy Learning", tracking_uri="file:./ml-runs")
    trainer = pl.Trainer(gpus=1, logger=mlf_logger, max_epochs=15)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train_snn_proxy()