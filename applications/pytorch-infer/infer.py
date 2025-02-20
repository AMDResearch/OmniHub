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

import omnihub.run
import omnihub.tools
from omnihub.run.arguments import parse_config

# Workaround to set the CIFAR-10 dataset URL
torchvision.datasets.CIFAR10.url = (
    "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
)


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


class Inferencer:
    def __init__(self, custom_args, config=None):
        self._parse_config(config, custom_args)

        default_model_path = self.ModelArguments.pretrained_model_name_or_path
        omnihub_model_path = os.path.join(
            os.getenv("OMNIHUB_MODELS_DIR"), default_model_path
        )

        # Check if the provided argument/config is an existing directory with
        # an without the OMNIHUB_MODELS_DIR prefix. If no directory can be
        # found, assume it's a model name to be loaded from Huggingface.
        model_path = default_model_path
        if not os.path.isdir(default_model_path) and os.path.isdir(omnihub_model_path):
            model_path = omnihub_model_path

        if not os.path.exists(model_path) or not os.path.isdir(model_path):
            print("Model path does not exist")
            parser.print_help()
            sys.exit(1)

        model_path = f"{model_path}/cifar_net.pth"

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load the trained model for inference
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(model_path))

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
        testset = torchvision.datasets.CIFAR10(
            root="~/.cache/pytorch/datasets",
            train=False,
            download=True,
            transform=transform,
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=4, shuffle=False, num_workers=2
        )

        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

        self.model = model
        self.testloader = testloader
        self.classes = classes
        self.device = device

    def _parse_config(self, config: dict, custom_args: list):
        populated_dataclasses = parse_config(config, custom_args)

        for i in populated_dataclasses:
            setattr(self, i.__class__.__name__, i)

    @omnihub.tools.profile()
    def run(self):
        self.model.to(self.device)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # for i in range(len(predicted)):
                #     print(f'Predicted: {self.classes[predicted[i]]}, Actual: {self.classes[labels[i]]}')
        accuracy = 100 * correct / total
        print(f"Accuracy of the network on the 10000 test images: {accuracy:.2f}%")
        return accuracy


@omnihub.run.entrypoint
def run(*args, **kwargs):
    Inferencer(*args, **kwargs).run()
