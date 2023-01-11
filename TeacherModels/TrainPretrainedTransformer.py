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
    sys.path.append('/Users/ioannisbountouris/PycharmProjects/MasterThesis')
    sys.path.append('/Users/ioannisbountouris/PycharmProjects/MasterThesis/LoadDataset')

'----------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'

import matplotlib.pyplot       as plt
import numpy                   as np
import torch

from datasets                        import load_metric
from transformers                    import ViTFeatureExtractor, ViTConfig
from transformers                    import ViTForImageClassification
from transformers                    import TrainingArguments
from transformers                    import Trainer
from sklearn.metrics                 import confusion_matrix
from sklearn.metrics                 import ConfusionMatrixDisplay
from torchvision.transforms          import ToTensor
from torchvision.transforms          import ToPILImage

from LoadDataset                     import HuggingFaceDatasets
from LoadDataset.HuggingFaceDatasets import GetDataset
from LoadDataset.Tools               import Tools

from torchvision.datasets.cifar      import CIFAR10

from torch.utils.data                import DataLoader
from nn.nn_utils                     import train_model, save_model

from transformers import ViTModel
from datasets import load_dataset


'----------------------------------------------------------------------------------------------------------------------'


def ArgumentsForTraining():
    return TrainingArguments(
        f"../test-cifar-10",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='logs',
        remove_unused_columns=False,
    )


'----------------------------------------------------------------------------------------------------------------------'


def ComputeMetrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return load_metric("accuracy").compute(predictions=predictions, references=labels)


'----------------------------------------------------------------------------------------------------------------------'


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


'----------------------------------------------------------------------------------------------------------------------'


def CreateTrainer(model, args, train_ds, val_ds, feature_extractor):
    return Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=ComputeMetrics,
        tokenizer=feature_extractor,
    )


'----------------------------------------------------------------------------------------------------------------------'


def ShowConfusionMatrix(outputs, TrainDS):
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)

    labels = TrainDS.features['label'].names
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45)
    plt.savefig('ConfurionMatrix.png')


'----------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------ ViT -----------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'


class MyViTForImageClassification(ViTForImageClassification):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    '------------------------------------------------------------------------------------------------------------------'

    def get_hidden_state(self, tensors, layers=[-1]):

        image_data = Tools.ImageData(self.feature_extractor.size,
                                     self.feature_extractor.image_mean,
                                     self.feature_extractor.image_std)

        images = [ToPILImage()(tensor) for tensor in tensors]

        train_features = [Tools.TestImageTransformer(image_data)(image.convert("RGB")) for image in images]
        train_features = torch.stack(train_features)

        if next(self.parameters()).is_cuda:
            train_features = train_features.to('cuda')

        train_features = {'pixel_values': train_features}

        with torch.no_grad():
            outputs = self.forward(**train_features, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        features = [None] * len(layers)

        counter = 0
        for layer in layers:
            features[counter] = hidden_states[layer]
            counter += 1

        return hidden_states[-1], features

    '----------------------------------------------------------------------------------------------------------------------'

    def get_features(self, images, layers=[-1]):
        _, features = self.get_hidden_state(images, layers)
        featuresReshape = []
        for f in features:
            if f is not None:
                featuresReshape.append(f.view(f.shape[0], -1))

        if len(featuresReshape):
            return featuresReshape
        else:
            raise Exception("wrong input")


'----------------------------------------------------------------------------------------------------------------------'


def PerformTraining():
    feature_extractor  = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    TrainDS, ValidationDS, TestDS = GetDataset('cifar10', Tools.ImageData(feature_extractor.size,
                                                                          feature_extractor.image_mean,
                                                                          feature_extractor.image_std) )

    id2label           = {id: label for id, label in enumerate(TrainDS.features['label'].names)}
    label2id           = {label: id for id, label in id2label.items()}

    ViT                = MyViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                                     num_labels=10, id2label=id2label, label2id=label2id)

    trainer = CreateTrainer(ViT, ArgumentsForTraining(), TrainDS, ValidationDS, feature_extractor)
    trainer.train()
    outputs = trainer.predict(TestDS)
    print(outputs.metrics)
    ShowConfusionMatrix(outputs, TrainDS)

    torch.save(ViT, Path + '/TeacherModels/TrainedModels/Cifar10/ViT_Teacher_cifar10.pth')
    save_model(ViT, Path + '/TeacherModels/TrainedModels/Cifar10/ViT_Teacher_cifar10.model')
    plt.show()


'----------------------------------------------------------------------------------------------------------------------'


def DebugFun():
    feature_extractor  = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    transform = ToTensor()

    train_set = CIFAR10(root='./../LoadDataset/Data', train=True , download=True, transform=transform)
    test_set  = CIFAR10(root='./../LoadDataset/Data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True , batch_size=4)
    test_loader  = DataLoader(test_set , shuffle=False, batch_size=4)

    train_features, train_labels = next(iter(test_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    model = torch.load(Path + '/TeacherModels/TrainedModels/Cifar10/ViT_Teacher_cifar10.pth')
    model.eval()

    image_data = Tools.ImageData(feature_extractor.size, feature_extractor.image_mean, feature_extractor.image_std)

    images = [ ToPILImage()(tensor) for tensor in train_features ]

    train_features = [Tools.TestImageTransformer(image_data)(image.convert("RGB")) for image in images]
    train_features = torch.stack(train_features)

    train_features = {'pixel_values': train_features}

    with torch.no_grad():
        outputs = model(**train_features, output_hidden_states=True)

    hidden_states = outputs.hidden_states


if __name__ == '__main__':
    PerformTraining()
