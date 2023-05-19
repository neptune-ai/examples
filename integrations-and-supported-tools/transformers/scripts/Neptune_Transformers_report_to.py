import neptune
from datasets import load_dataset
from evaluate import load
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

task = "cola"
model_checkpoint = "prajjwal1/bert-tiny"
batch_size = 16
dataset = load_dataset("glue", task)
metric = load("glue", task)
num_labels = 2

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True)


encoded_dataset = dataset.map(preprocess_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    report_to="neptune",
)

validation_key = "validation"

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
)

trainer.train()
