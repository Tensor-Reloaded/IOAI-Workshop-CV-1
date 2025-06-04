from PIL import Image

import torch

from torch import nn, Tensor
from typing import List
from tqdm import trange, tqdm
from torchvision import datasets
from torchvision.transforms import v2


class ParityMNIST(torch.utils.data.Dataset):
    def __init__(self, train):
        self.dataset = datasets.MNIST(root="../data", train=train, download=True)
        self.transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.1307], [0.3081], inplace=True),
                torch.flatten,
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label = self.dataset[i]
        return {
            "data": self.transforms(image),
            "digit_label": label,
            "parity_label": 0 if label % 2 == 0 else 1,
        }


class GenericMultitaskModel(nn.Module):
    def __init__(self, backbone: nn.Module, task_heads: List[nn.Module]):
        super().__init__()
        self.backbone = backbone
        self.task_heads = nn.ModuleList(task_heads)

    def forward(self, x: Tensor) -> List[Tensor]:
        shared_representation = self.backbone(x)
        task_outputs = []
        for task_head in self.task_heads:
            # task_output: latent_shared_representation_space -> task_output
            task_output = task_head(shared_representation)
            task_outputs.append(task_output)
        return task_outputs


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            batch_size=64,
            val_batch_size=500,
            factor_1=1.0,
            factor_2=0.5,

    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.factor_1 = factor_1
        self.factor_2 = factor_2

        self.device = torch.accelerator.current_accelerator()
        self.train_loader = torch.utils.data.DataLoader(ParityMNIST(train=True), shuffle=True,
                                                        batch_size=self.batch_size, drop_last=True)
        self.val_loader = torch.utils.data.DataLoader(ParityMNIST(train=False), batch_size=self.val_batch_size)
        self.criterion_1 = nn.CrossEntropyLoss()
        self.criterion_2 = nn.CrossEntropyLoss()

        self.model = self.model.to(self.device)

    def train(self):
        self.model.train()

        correct_digits = 0
        correct_parity = 0
        total = 0
        total_loss = 0

        for batch in tqdm(self.train_loader, leave=False):
            data = batch["data"].to(self.device)
            digit_labels = batch["digit_label"].to(self.device)
            parity_labels = batch["parity_label"].to(self.device)
            digit_predicted, parity_predicted = self.model(data)
            loss_1 = self.criterion_1(digit_predicted, digit_labels) * self.factor_1
            loss_2 = self.criterion_2(parity_predicted, parity_labels) * self.factor_2
            loss = loss_1 + loss_2
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            correct_digits += digit_predicted.argmax(1).eq(digit_labels).sum().item()
            correct_parity += parity_predicted.argmax(1).eq(parity_labels).sum().item()
            total += digit_labels.size(0)
            total_loss += loss.item()

        return 100.0 * correct_digits / total, 100.0 * correct_parity / total, total_loss / len(self.train_loader)

    # @torch.no_grad()
    @torch.inference_mode()
    def val(self):
        self.model.eval()

        correct_digits = 0
        correct_parity = 0
        total = 0
        total_loss = 0

        for batch in tqdm(self.val_loader, leave=False):
            data = batch["data"].to(self.device)
            digit_labels = batch["digit_label"].to(self.device)
            parity_labels = batch["parity_label"].to(self.device)
            digit_predicted, parity_predicted = self.model(data)
            loss_1 = self.criterion_1(digit_predicted, digit_labels) * self.factor_1
            loss_2 = self.criterion_2(parity_predicted, parity_labels) * self.factor_2
            loss = loss_1 + loss_2

            correct_digits += digit_predicted.argmax(1).eq(digit_labels).sum().item()
            correct_parity += parity_predicted.argmax(1).eq(parity_labels).sum().item()
            total += digit_labels.size(0)
            total_loss += loss.item()

        return 100.0 * correct_digits / total, 100.0 * correct_parity / total, total_loss / len(self.val_loader)

    def run(self, epochs):
        with trange(epochs) as tbar:
            for _ in tbar:
                train_digits_acc, train_parity_acc, train_loss = self.train()
                val_digits_acc, val_parity_acc, val_loss = self.val()
                tbar.set_description(f"Train: {train_digits_acc:.1f}, {train_parity_acc:.1f}, {train_loss:.3f}, "
                                     f"Val: {val_digits_acc:.1f}, {val_parity_acc:.1f}, {val_loss:.3f}")


if __name__ == '__main__':
    model = GenericMultitaskModel(
        backbone=nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
        ),
        task_heads=[
            nn.Sequential(nn.Linear(100, 10)), nn.Sequential(nn.Linear(100, 2))
        ]
    )
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9, nesterov=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    Trainer(model, optimizer).run(25)
