from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

'----------------------------------------------------------------------------------------------------------------------'


class ImageData:
    def __init__(self, newSize, imageMean, imageStd):
        self.newSize   = newSize
        self.imageMean = imageMean
        self.imageStd  = imageStd


'----------------------------------------------------------------------------------------------------------------------'


def TrainImageTransformer(image_data):
    return Compose([
                RandomResizedCrop(image_data.newSize),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=image_data.imageMean, std=image_data.imageStd),
            ])


'----------------------------------------------------------------------------------------------------------------------'


def TestImageTransformer(image_data):
    return Compose([
                Resize(image_data.newSize),
                CenterCrop(image_data.newSize),
                ToTensor(),
                Normalize(mean=image_data.imageMean, std=image_data.imageStd),
            ])