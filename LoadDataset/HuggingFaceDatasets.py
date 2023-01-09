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
else:
    import sys

import LoadDataset.Tools.Tools as Tools
from datasets import load_dataset
from transformers import ViTFeatureExtractor, ViTConfig
from torchvision.transforms import RandomResizedCrop


'----------------------------------------------------------------------------------------------------------------------'


class TransformTrainImages:
    def __init__(self, image_data):
        self.image_data = image_data

    def __call__(self, examples):
        examples['pixel_values'] = [Tools.TrainImageTransformer(self.image_data)(image.convert("RGB")) for image in examples['img']]
        return examples


'----------------------------------------------------------------------------------------------------------------------'


class TransformTestImages:
    def __init__(self, image_data):
        self.image_data = image_data

    def __call__(self, examples):
        examples['pixel_values'] = [Tools.TestImageTransformer(self.image_data)(image.convert("RGB")) for image in examples['img']]
        return examples


'----------------------------------------------------------------------------------------------------------------------'


def GetDataset(dataset, image_data):

    TrainDS, TestDS      = load_dataset(dataset, split=['train[:500]', 'test[:100]'])
    TrainAndValidationDS = TrainDS.train_test_split(test_size=0.1)
    TrainDS              = TrainAndValidationDS['train']
    ValidationDS         = TrainAndValidationDS['test' ]

    TrainDS.set_transform(     TransformTrainImages(image_data))
    ValidationDS.set_transform(TransformTestImages( image_data))
    TestDS.set_transform(      TransformTestImages( image_data))

    return TrainDS, ValidationDS, TestDS


'----------------------------------------------------------------------------------------------------------------------'


def DebugFun():
    feature_extractor  = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    TrainDS, TestDS = load_dataset('cifar10', split=['train[:500]', 'test[:100]'])
    image_data = Tools.ImageData(feature_extractor.size, feature_extractor.image_mean, feature_extractor.image_std)
    image = TrainDS[0]['img'].convert("RGB")
    RandomResizedCrop(size=(image_data.newSize['height'], image_data.newSize['width']))(image)


'----------------------------------------------------------------------------------------------------------------------'


if __name__ == '__main__':
    DebugFun()
