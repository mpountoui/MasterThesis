from datasets               import load_dataset
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


def TransformTrainImage(image, image_data):
    return Compose(
            [
                RandomResizedCrop(image_data.newSize),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=image_data.imageMean, std=image_data.imageStd),
            ]
            )(image)


'----------------------------------------------------------------------------------------------------------------------'


def TransformValidationImage(image, image_data):
    return Compose(
            [
                Resize(image_data.newSize),
                CenterCrop(image_data.newSize),
                ToTensor(),
                Normalize(mean=image_data.imageMean, std=image_data.imageStd),
            ]
        )(image)


'----------------------------------------------------------------------------------------------------------------------'


class TransformTrainImages:
    def __init__(self, image_data):
        self.image_data = image_data

    def __call__(self, examples):
        examples['pixel_values'] = [TransformTrainImage(image.convert("RGB"), self.image_data) for image in examples['img']]
        return examples


'----------------------------------------------------------------------------------------------------------------------'


class TransformValidationImages:
    def __init__(self, image_data):
        self.image_data = image_data

    def __call__(self, examples):
        examples['pixel_values'] = [TransformValidationImage(image.convert("RGB"), self.image_data) for image in examples['img']]
        return examples


'----------------------------------------------------------------------------------------------------------------------'


def GetDataset(dataset, image_data):

    TrainDS, TestDS    = load_dataset(dataset, split=['train[:500]', 'test[:200]'])
    TrainAndValidation = TrainDS.train_test_split(test_size=0.1)
    TrainDS            = TrainAndValidation['train']
    ValidationDS       = TrainAndValidation['test' ]
    TrainDS.set_transform(     TransformTrainImages(     image_data))
    ValidationDS.set_transform(TransformValidationImages(image_data))
    TestDS.set_transform(      TransformValidationImages(image_data))

    return TrainDS, ValidationDS, TestDS
