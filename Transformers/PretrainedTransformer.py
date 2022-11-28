from datasets               import load_dataset
from datasets               import load_metric
from transformers           import ViTFeatureExtractor
from transformers           import ViTForImageClassification
from transformers           import TrainingArguments
from transformers           import Trainer
from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from sklearn.metrics        import confusion_matrix
from sklearn.metrics        import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import numpy             as np
import torch

'----------------------------------------------------------------------------------------------------------------------'


def TransformTrainImage(image, feature_extractor):
    return Compose(
            [
                RandomResizedCrop(feature_extractor.size),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
            ]
            )(image)


'----------------------------------------------------------------------------------------------------------------------'


def TransformValidationImage(image, feature_extractor):
    return Compose(
            [
                Resize(feature_extractor.size),
                CenterCrop(feature_extractor.size),
                ToTensor(),
                Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
            ]
        )(image)


'----------------------------------------------------------------------------------------------------------------------'


class TransformTrainImages:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, examples):
        examples['pixel_values'] = [TransformTrainImage(image.convert("RGB"), self.feature_extractor) for image in examples['img']]
        return examples


'----------------------------------------------------------------------------------------------------------------------'


class TransformValidationImages:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, examples):
        examples['pixel_values'] = [TransformValidationImage(image.convert("RGB"), self.feature_extractor) for image in examples['img']]
        return examples


'----------------------------------------------------------------------------------------------------------------------'


def ArgumentsForTraining():
    return TrainingArguments(
        f"../test-cifar-10",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
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
    plt.savefig('foo.png')


'----------------------------------------------------------------------------------------------------------------------'


TrainDS_Size = 5000
TrainDS_Size = 2000

if __name__ == '__main__':
    feature_extractor  = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    TrainDS, TestDS    = load_dataset('../cifar10', split=['train[:500]', 'test[:200]'])
    TrainAndValidation = TrainDS.train_test_split(test_size=0.1)
    TrainDS            = TrainAndValidation['train']
    ValidationDS       = TrainAndValidation['test' ]
    TrainDS.set_transform(     TransformTrainImages(     feature_extractor))
    ValidationDS.set_transform(TransformValidationImages(feature_extractor))
    TestDS.set_transform(      TransformValidationImages(feature_extractor))

    id2label           = {id: label for id, label in enumerate(TrainDS.features['label'].names)}
    label2id           = {label: id for id, label in id2label.items()}

    ViT                = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                                   num_labels=10, id2label=id2label, label2id=label2id)
    trainer = CreateTrainer(ViT, ArgumentsForTraining(), TrainDS, ValidationDS, feature_extractor)
    trainer.train()
    outputs = trainer.predict(TestDS)
    print(outputs.metrics)
    ShowConfusionMatrix(outputs, TrainDS)

    torch.save(ViT, 'checkpoint.pth')
    plt.show()
