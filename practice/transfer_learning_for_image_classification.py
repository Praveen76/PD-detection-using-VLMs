import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import io
import os

from huggingface_hub import notebook_login

import datasets
from datasets import load_dataset, DatasetDict

from transformers import AutoImageProcessor, ViTForImageClassification

from transformers import Trainer, TrainingArguments

import evaluate

from PIL import Image

from pathlib import Path
datasets.config.DOWNLOADED_DATASETS_PATH = Path("/localdisk1/.practice_datasets/")
os.environ["HUGGINGFACE_HUB_CACHE"] = "/localdisk1/.huggingface/"
os.environ["HF_HOME"] = "/localdisk1/.huggingface/"

dataset = load_dataset('pcuenq/oxford-pets')
# print(dataset['train'][0].keys()) # ['path', 'label', 'dog', 'image']

labels = dataset['train'].unique('label')

def show_samples(ds,rows,cols):
    samples = ds.shuffle().select(np.arange(rows*cols)) # selecting random images
    fig = plt.figure(figsize=(cols*4,rows*4))
    # plotting
    for i in range(rows*cols):
        img = Image.open(io.BytesIO(samples[i]['image']['bytes']))
        label = samples[i]['label']
        fig.add_subplot(rows,cols,i+1)
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')

    plt.savefig("temporary.png", dpi=300)
            
# show_samples(dataset['train'],rows=3,cols=5)

train_eval_dataset = dataset["train"].train_test_split(test_size=0.2)
eval_dataset = train_eval_dataset["test"]
eval_splits = eval_dataset.train_test_split(test_size=0.5)

train_dataset = train_eval_dataset["train"]
test_dataset = eval_splits["test"]
validation_dataset = eval_splits["train"]

our_dataset = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})

label2id = {label:index for index, label in enumerate(labels)}
id2label = {index:label for index, label in enumerate(labels)}

processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')

def transforms(batch):
    batch['image'] = [Image.open(io.BytesIO(x['bytes'])).convert('RGB') for x in batch['image']]
    inputs = processor(batch['image'],return_tensors='pt')
    inputs['labels']=[label2id[y] for y in batch['label']]
    return inputs

processed_dataset = our_dataset.with_transform(transforms)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

accuracy = evaluate.load('accuracy')
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits,axis=1)
    score = accuracy.compute(predictions=predictions, references=labels)
    return score

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels = len(labels),
    id2label = id2label,
    label2id = label2id,
    ignore_mismatched_sizes = True
)

# print(model)

for name,p in model.named_parameters():
    if not name.startswith('classifier'):
        p.requires_grad = False

num_params = sum([p.numel() for p in model.parameters()])
trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])

print(f"{num_params = :,} | {trainable_params = :,}")

training_args = TrainingArguments(
    output_dir="/localdisk1/.huggingface/vit-base-oxford-iiit-pets",
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    num_train_epochs=5,
    learning_rate=3e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    tokenizer=processor
)

trainer.train()

trainer.evaluate(processed_dataset['test'])

def show_predictions(rows,cols):
    samples = our_dataset['test'].shuffle().select(np.arange(rows*cols))
    processed_samples = samples.with_transform(transforms)
    predictions = trainer.predict(processed_samples).predictions.argmax(axis=1) # predicted labels from logits
    fig = plt.figure(figsize=(cols*4,rows*4))
    for i in range(rows*cols):
        img = samples[i]['image']
        prediction = predictions[i]
        label = f"label: {samples[i]['label']}\npredicted: {id2label[prediction]}"
        fig.add_subplot(rows,cols,i+1)
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
            
show_predictions(rows=5,cols=5)