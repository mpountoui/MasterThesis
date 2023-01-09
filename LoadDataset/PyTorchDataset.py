try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

Path = '/Users/ioannisbountouris/PycharmProjects/MasterThesis'

if IN_COLAB :
    import sys
    sys.path.append('/content/MasterThesis')
    Path = '/content/MasterThesis'

from torch.utils.data import DataLoader

import LoadDataset.Tools.Tools as Tools


def GetDataset(dataset, batch_size=128):

    """
        dataset should be torchvision.datasets
    """

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std  = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = Tools.TrainImageTransformer(Tools.ImageData({'height': 32, 'width' : 32}, mean, std))
    test_transform  = Tools.TestImageTransformer( Tools.ImageData({'height': 32, 'width' : 32}, mean, std))

    train_data          = dataset(Path + '/LoadDataset/Data', train=True , transform=train_transform, download=True)
    train_data_original = dataset(Path + '/LoadDataset/Data', train=True , transform=test_transform , download=True)
    test_data           = dataset(Path + '/LoadDataset/Data', train=False, transform=test_transform , download=True)

    train_loader          = DataLoader(train_data         , batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader           = DataLoader(test_data          , batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    train_loader_original = DataLoader(train_data_original, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    return train_loader, test_loader, train_loader_original
