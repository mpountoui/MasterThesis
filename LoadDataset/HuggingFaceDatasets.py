import Tools.Tools as Tools
from datasets import load_dataset

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
