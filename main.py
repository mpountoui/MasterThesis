import torch

from nn.retrieval_evaluation    import evaluate_model_retrieval
from TeacherModels.TrainResnet  import ResNet18
from LoadDataset.PyTorchDataset import GetDataset

import torchvision.datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    print("main")
