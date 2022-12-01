import torch.nn
import torch.optim
import nn

from LoadDataset                 import PyTorchDataset
from TeacherModels.Resnet.Resnet import ResNet18


def TrainResNet(dataset, device, learning_rates=[0.001, 0.0001], iters=[50, 50]):

    train_loader, test_loader, _ = PyTorchDataset.GetDataset(dataset, batch_size=128)
    criterion = torch.nn.CrossEntropyLoss()

    for lr, iter in zip(learning_rates, iters):
        print("Training with lr=%f for %d iters" % (lr, iter))
        optimizer = torch.optim.Adam(ResNet18.parameters(), lr=lr)
        model = ResNet18().to(device)
        nn.train_model(model, optimizer, criterion, train_loader, epochs=iter)
        nn.save_model( model, output_file='resnet18_cifar10.model')
