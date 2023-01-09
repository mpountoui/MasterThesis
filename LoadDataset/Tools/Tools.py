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
                RandomResizedCrop(size=(image_data.newSize['height'], image_data.newSize['width'])),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=image_data.imageMean, std=image_data.imageStd),
            ])


'----------------------------------------------------------------------------------------------------------------------'


def TestImageTransformer(image_data):
    return Compose([
                Resize(size=(image_data.newSize['height'], image_data.newSize['width'])),
                CenterCrop(size=(image_data.newSize['height'], image_data.newSize['width'])),
                ToTensor(),
                Normalize(mean=image_data.imageMean, std=image_data.imageStd),
            ])
