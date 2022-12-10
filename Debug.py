import sys

import torch
import torchvision

from torch.utils.data           import Subset
from torch.utils.data           import DataLoader
from TeacherModels.TrainResnet  import ResNet18

import LoadDataset.Tools.Tools as Tools

from nn.nn_utils import extract_features, extract_features_raw

from nn.retrieval_evaluation import Database


def LoadDataset(dataset, batch_size=1):

    """
        dataset should be torchvision.datasets
    """

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std  = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = Tools.TrainImageTransformer(Tools.ImageData(32, mean, std))
    test_transform  = Tools.TestImageTransformer( Tools.ImageData(32, mean, std))

    train_data          = dataset('LoadDataset/Data', train=True , transform=train_transform, download=True)
    train_data_original = dataset('LoadDataset/Data', train=True , transform=test_transform , download=True)
    test_data           = dataset('LoadDataset/Data', train=False, transform=test_transform , download=True)

    indices1 = torch.randperm(len(train_data))[:10]
    indices2 = torch.randperm(len(test_data))[:4]

    print(train_data)
    print(train_data_original)
    print(test_data)

    train_data          = Subset(train_data, indices1)
    train_data_original = Subset(train_data_original, indices1)
    test_data           = Subset(test_data, indices2)

    train_loader          = DataLoader(train_data         , batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader           = DataLoader(test_data          , batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    train_loader_original = DataLoader(train_data_original, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    return train_loader, test_loader, train_loader_original


if __name__ == '__main__':
    net = ResNet18(num_classes=10)
    train_loader, test_loader, train_loader_original = LoadDataset(torchvision.datasets.CIFAR10)
    train_features, train_labels = extract_features(ResNet18(num_classes=10), 'cpu', train_loader_original, layer=3)
    test_features , test_labels  = extract_features(ResNet18(num_classes=10), 'cpu', test_loader          , layer=3)

    print('----------------------------------------------------------------------------------')
    print(train_features.shape)
    print(train_labels.shape  )
    print(test_features.shape )
    print(test_labels.shape   )

    print('----------------------------------------------------------------------------------')
    database = Database(train_features, train_labels, metric='cosine')
    relevant_vectors = database.GetBinaryRelevance(test_features, test_labels)
    print(f"relevant vectors {relevant_vectors.shape}")
    print(relevant_vectors)

    print('----------------------------------------------------------------------------------')
    precisions = database.CalculatePrecision(relevant_vectors)
    print(f"Precisions {len(precisions)}")
    print(precisions)

    sys.exit()

    print('----------------------------------------------------------------------------------')
    metrics = database.get_metrics(relevant_vectors, test_labels)
    print(f"get metrics {len(metrics)}")
    print(metrics[0])
    print(metrics[1].shape)
    print(metrics[2].shape)
    print(metrics[3].shape)

    print('----------------------------------------------------------------------------------')

#    results = database.evaluate(test_features, test_labels, batch_size=128)

