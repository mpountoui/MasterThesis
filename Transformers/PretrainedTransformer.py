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

'----------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'

import matplotlib.pyplot as plt
import numpy             as np
import torch

from datasets                        import load_metric
from transformers                    import ViTFeatureExtractor
from transformers                    import ViTForImageClassification
from transformers                    import TrainingArguments
from transformers                    import Trainer
from sklearn.metrics                 import confusion_matrix
from sklearn.metrics                 import ConfusionMatrixDisplay
from LoadDataset.HuggingFaceDatasets import GetDataset
from LoadDataset.HuggingFaceDatasets import ImageData


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


if __name__ == '__main__':
    feature_extractor  = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    TrainDS, ValidationDS, TestDS = GetDataset('cifar10', ImageData(feature_extractor.size,
                                                                    feature_extractor.image_mean,
                                                                    feature_extractor.image_std))

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
