from model_def import CNNModel, ProxyModel
from cifar10 import Cifar10DataModule, ProxyCifar10
import pytorch_lightning as pl
from pytorch_lightning.loggers.mlflow import MLFlowLogger


def train_cnn_only():
    model = CNNModel()
    dm = Cifar10DataModule(batch_size=32, num_workers=10)
    mlf_logger = MLFlowLogger(experiment_name="ANN pure", tracking_uri="file:./ml-runs")
    trainer = pl.Trainer(gpus=1, logger=mlf_logger, log_every_n_steps=1, max_steps=15)
    trainer.fit(model, datamodule=dm)


def train_snn_proxy():
    model = ProxyModel()
    dm = ProxyCifar10(batch_size=16, time_steps=10, num_workers=10)
    mlf_logger = MLFlowLogger(experiment_name="Proxy Learning", tracking_uri="file:./ml-runs")
    trainer = pl.Trainer(gpus=1, logger=mlf_logger, log_every_n_steps=1, max_epochs=15)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train_snn_proxy()