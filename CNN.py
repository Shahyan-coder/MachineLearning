import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Defining the architecture of the CNN model.
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Initializing two convolutional layers and two fully connected layers.
        # Using ReLU as the activation function and dropout to prevent overfitting.
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # First convolutional layer.
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # Second convolutional layer.
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer to reduce spatial dimensions.
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # First fully connected layer.
        self.fc2 = nn.Linear(
            512, 10
        )  # Second fully connected layer, outputting 10 classes.
        self.relu = nn.ReLU()  # Activation function.
        self.dropout = nn.Dropout(0.5)  # Dropout layer.

    def forward(self, x):
        # Defining the forward pass of the model.
        x = self.pool(
            self.relu(self.conv1(x))
        )  # Applying first conv layer, ReLU, and pooling.
        x = self.pool(
            self.relu(self.conv2(x))
        )  # Applying second conv layer, ReLU, and pooling.
        x = x.view(
            -1, 64 * 8 * 8
        )  # Flattening the output for the fully connected layers.
        x = self.relu(self.fc1(x))  # Applying first fully connected layer and ReLU.
        x = self.dropout(x)  # Applying dropout.
        x = self.fc2(x)  # Applying second fully connected layer.
        return x


def load_data():
    # Applying transforms to the dataset: converting to tensor and normalizing.
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    # Loading the training and testing sets of CIFAR-10.
    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=4, shuffle=False)
    return trainloader, testloader


def train_model(model, trainloader, criterion, optimizer, num_epochs=10):
    # Training the model for a specified number of epochs.
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data  # Getting inputs and labels from the data loader.
            optimizer.zero_grad()  # Zeroing the parameter gradients.
            outputs = model(inputs)  # Forward pass: predicting outputs with the model.
            loss = criterion(outputs, labels)  # Calculating loss.
            loss.backward()  # Backward pass: calculating gradients.
            optimizer.step()  # Updating model parameters.
            running_loss += loss.item()  # Accumulating loss.
            if i % 2000 == 1999:  # Print loss every 2000 mini-batches.
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
    print("Finished Training")


def test_model(model, testloader):
    # Testing the model and calculating its accuracy.
    correct = 0
    total = 0
    with torch.no_grad():  # No gradient calculation during testing.
        for data in testloader:
            images, labels = data  # Getting images and labels.
            outputs = model(images)  # Predicting outputs.
            _, predicted = torch.max(
                outputs.data, 1
            )  # Getting the class with the highest score.
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # Printing the accuracy of the model.
    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
    )


def main():
    trainloader, testloader = load_data()
    net = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train_model(net, trainloader, criterion, optimizer)
    test_model(net, testloader)


if __name__ == "__main__":
    main()
