import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import omnihub


# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Trainer:
    def __init__(self, custom_args, epochs=2):
        parser = ArgumentParser(
            description="Inference using a Hugging Face model",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "-o", "--output-dir", help="Path to store output", type=str, required=True
        )
        parser.add_argument(
            "-m", "--model-dir", help="Path to the model", type=str, default=None
        )

        self.args = parser.parse_args(args=custom_args)

        if not os.path.exists(self.args.output_dir) or not os.path.isdir(
            self.args.output_dir
        ):
            print("Output path does not exist")
            parser.print_help()
            sys.exit(1)

        model_path = f"{self.args.output_dir}/cifar_net.pth"
        print(f"Model will be saved to: {model_path}")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load CIFAR-10 dataset
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Load CIFAR-10 training dataset
        # root: Directory where the dataset will be saved
        # train: Boolean, if True creates dataset from training set, otherwise from test set
        # download: Boolean, if True downloads the dataset from the internet and puts it in root directory
        # transform: A function/transform that takes in an image and returns a transformed version
        trainset = torchvision.datasets.CIFAR10(
            root="~/.cache/pytorch/datasets",
            train=True,
            download=True,
            transform=transform,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=4, shuffle=True, num_workers=2
        )

        # Initialize the network, loss function and optimizer
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        self.model = model
        self.trainloader = trainloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.model_path = model_path

    @omnihub.tools.profile()
    def run(self):
        self.model.to(self.device)
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(
                tqdm(self.trainloader, desc=f"Epoch {epoch + 1}"), 0
            ):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(
                        f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 2000:.3f}"
                    )
                    running_loss = 0.0
        print("Finished Training")
        # Save the trained model
        print(f"Saving model to {self.model_path}")
        torch.save(self.model.state_dict(), self.model_path)

        return self.model


@omnihub.entrypoint
def run(args):
    Trainer(args).run()
