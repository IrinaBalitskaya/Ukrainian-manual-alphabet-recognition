import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from GestureRecognitionDataset import GestureRecognitionDataset
from Net import Net


class Model:
    def __init__(self, directory, batch, epochs, size):
        self.directory = directory
        self.batch_size = batch
        self.num_epochs = epochs
        self.img_size = size

        self.train_transform = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.Grayscale(),
                                                   transforms.Resize((self.img_size, self.img_size)),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomRotation(degrees=(-30, 30)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.5], [0.5])])

        self.test_transform = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Grayscale(),
                                                  transforms.Resize((self.img_size, self.img_size)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.5], [0.5])])

        self.train_dataset = GestureRecognitionDataset(self.directory + 'train_labels.csv', self.directory + 'train',
                                                      transform=self.train_transform)
        self.test_dataset = GestureRecognitionDataset(directory + 'test_labels.csv', directory + 'test',
                                                     transform=self.test_transform)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)
        self.total_samples = len(self.train_dataset)
        self.net = Net()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.n_total_steps = len(self.train_loader)
        self.loss_values = []
        self.acc_values = []
        self.n_correct = 0
        self.n_samples = 0
        self.running_loss = 0.0

    def train(self):
        for epoch in range(self.num_epochs):
            self.running_loss = 0.0
            for i, (images, labels) in enumerate(self.train_loader):
                outputs = self.net(images)
                loss = self.loss_function(outputs, labels)
                _, predictions = torch.max(outputs, 1)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.running_loss += loss.item() * images.size(0)

                self.n_samples += labels.size(0)
                self.n_correct += (predictions == labels).sum().item()

                if (i + 1) % 10 == 0:
                    print(f'epoch {epoch + 1}/{self.num_epochs}, step {i + 1}/{self.n_total_steps}, loss = {loss.item():.4f}')
            acc = 100.0 * self.n_correct / self.n_samples
            self.loss_values.append(self.running_loss / len(self.train_dataset))
            self.acc_values.append(acc)

    def test(self):
        with torch.no_grad():
            self.n_correct = 0
            self.n_samples = 0
            for images, labels in self.test_loader:
                outputs = self.net(images)
                _, predictions = torch.max(outputs, 1)
                self.n_samples += labels.size(0)
                self.n_correct += (predictions == labels).sum().item()

            acc = 100.0 * self.n_correct / self.n_samples
            print(f'accuracy = {acc:.4f}')

    def saveModel(self):
        torch.save(self.net.state_dict(), f'./Net_{self.num_epochs}.pth')

    def buidPlot(self):
        ax1 = plt.subplot2grid((2, 1), (0, 0))
        ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

        ax1.plot(self.acc_values, label="accuracies")
        ax1.legend(loc=2)
        ax2.plot(self.loss_values, label="losses")
        ax2.legend(loc=2)
        plt.savefig(f'plot_{self.num_epochs}_epochs.png', dpi=300)
        plt.show()


if __name__ == '__main__':
    directory = './images/'
    img_size = 60
    batch_size = 4
    num_epochs = 100
    model = Model(directory, batch_size, num_epochs, img_size)
    model.train()
    model.test()
    model.saveModel()
    model.buidPlot()
