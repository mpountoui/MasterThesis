import torch

from nn.retrieval_evaluation    import evaluate_model_retrieval
from TeacherModels.TrainResnet  import ResNet18
from LoadDataset.PyTorchDataset import GetDataset

import torchvision.datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    evaluate_model_retrieval(dataset_loader=GetDataset(torchvision.datasets.CIFAR10, batch_size=64),
                             net=ResNet18(num_classes=10),
                             device=device,
                             path='TeacherModels/ModelsTraining/Cifar10/resnet18_cifar10.model',
                             result_path='TeacherModels/ModelsTraining/Cifar10/Results/resnet18_cifar10_baseline_retrieval.pickle',
                             layer=3)
