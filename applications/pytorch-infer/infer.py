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
    def __init__(self, custom_args):
        parser = ArgumentParser(
            description="Inference using a PyTorch model",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "-m", "--model-dir", help="Path to the model", type=str, required=True
        )

        self.args = parser.parse_args(args=custom_args)

        if not os.path.exists(self.args.model_dir) or not os.path.isdir(
            self.args.model_dir
        ):
            print("Model path does not exist")
            parser.print_help()
            sys.exit(1)

        model_path = f"{self.args.model_dir}/cifar_net.pth"

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


@omnihub.entrypoint
def run(*args, **kwargs):
    Inferencer(*args, **kwargs).run()
