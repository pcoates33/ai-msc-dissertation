# Based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# A basic example of an image classifier in pytorch
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time


def load_cifar10_data(data_path, batch_size):

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    

    # Code in tutorial has num_workers=2, but this gave an error on my mac, so switched to 0
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes


def imshow(img):
    # function to show an image
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    # Define the network for the classifier

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, trainloader, epochs=2, device=None):
    started = time.time()
    # Loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            if device is None:
                inputs, labels = data
            else:
                # send the tensors to the device
                inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    elapsed = time.time() - started
    print(f'Finished Training. Elapsed time = {elapsed:,.2f} seconds')
    # 91.41 seconds with gpu (mps)
    # 33.65 seconds with cpu


def run_a_test(net, testloader, classes, device):
    # Testing the classifer
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images
    #imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    outputs = net(images.to(device))

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(4)))

def run_bigger_test(net, testloader, device=None):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            if device is None:
                images, labels = data
            else:
                # send the tensors to the device
                images, labels = data[0].to(device), data[1].to(device)
            
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


def test_accuracy_by_class(net, testloader, classes, device=None):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            if device is None:
                images, labels = data
            else:
                # send the tensors to the device
                images, labels = data[0].to(device), data[1].to(device)
            
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

if __name__ == "__main__":
    # main function
    print(f'has_cuda, i.e. NVIDIA GPU : {torch.has_cuda}')
    print(f'has_mps, i.e. mac silicon gpu : {torch.has_mps}')
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps')
    device = torch.device('mps' if torch.has_mps else 'cpu')
    # device = torch.device('cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)

    data_path = './code/investigations/data'
    batch_size = 4

    trainloader, testloader, classes = load_cifar10_data(data_path, batch_size)

    # # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = next(dataiter)

    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # # define the network
    net = Net()
    # Try parallel code - didn't seem to achieve anything.
    # net = nn.DataParallel(net)
    # And use cpu or mps depending on device
    net.to(device)

    train(net, trainloader, epochs=2, device=device)

    # # Save all the parameters
    PATH = data_path + '/cifar_net_gpu.pth'
    # torch.save(net.state_dict(), PATH)

    # Reload the model
    # net = Net()
    # net.load_state_dict(torch.load(PATH))

    # net.to(device)

    run_a_test(net, testloader, classes, device)

    run_bigger_test(net, testloader, device)

    test_accuracy_by_class(net, testloader, classes, device)