from torchvision.transforms import ToTensor, Compose
from torchvision.datasets import CIFAR10
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from transforms import ImageAndCurrent


class Cifar10DataModule(LightningDataModule):
    def __init__(self, batch_size=1, num_workers=6):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = ToTensor()

    def prepare_data(self) -> None:
        cifar10_train_dataset = CIFAR10(train=True, download=True, transform=self.transforms, root="./datasets/")
        # Split train dataset into train and validation dataset
        original_len_train_dataset = len(cifar10_train_dataset)
        len_val_dataset = int(0.10 * original_len_train_dataset)
        # Initialize datasets
        self.train_dataset, self.val_dataset = random_split(cifar10_train_dataset,
                                                            [(original_len_train_dataset - len_val_dataset),
                                                             len_val_dataset])
        print(
            f"Splitting the training dataset into train: {len(self.train_dataset)} and test: {len(self.val_dataset)} samples.")
        self.test_dataset = CIFAR10(train=False, download=True, transform=self.transforms, root="./datasets/")

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class ProxyCifar10(Cifar10DataModule):
    """
    Cifar 10 data in both image and current over time format
    """

    def __init__(self, batch_size=1, time_steps=60, num_workers=6):
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self.time_steps = time_steps
        self.transforms = Compose([ToTensor(), ImageAndCurrent(time_steps=self.time_steps)])


if __name__ == "__main__":
    data_module = ProxyCifar10(batch_size=5)
    data_module.prepare_data()

    dl_test = data_module.test_dataloader()
    for data, labels in dl_test:
        img, spks = data
        print(img.shape, spks.shape, len(labels))
        break

    dl_train = data_module.train_dataloader()
    for data, labels in dl_train:
        img, spks = data
        print(img.shape, spks.shape, len(labels))
        break

    dl_val = data_module.val_dataloader()
    for data, labels in dl_val:
        img, spks = data
        print(img.shape, spks.shape, len(labels))
        break
