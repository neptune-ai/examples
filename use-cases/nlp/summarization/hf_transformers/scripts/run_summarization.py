# File inspired from
# https://github.com/huggingface/transformers/blob/21f6f58721dd9154357576be6de54eefef1f1818/examples/pytorch/summarization/run_summarization.py
# Date Accessed: 31st August 2022
"""
Fine-tuning the library models for sequence to sequence.
"""

import io

import evaluate
import neptune.new as neptune
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import pandas as pd
import transformers
from arg_parsers import DataTrainingArguments, ModelArguments
from datasets import load_dataset
from filelock import FileLock
from neptune.new.types import File
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.utils import check_min_version, is_offline_mode
from transformers.utils.versions import require_version
from utils import get_dataset

metric = evaluate.load("rouge")

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.22.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/summarization/requirements.txt",
)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
]


class EvalLogger:
    def __init__(self, run, tokenizer, ignore_pad_token_for_loss):
        self.run = run
        self.cnt = 0
        self.tokenizer = tokenizer
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.examples_df = {}

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def log_at_eval(self, eval_preds):
        tokenizer = self.tokenizer
        preds = eval_preds.predictions
        labels = eval_preds.label_ids
        inputs = eval_preds.inputs
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        preds = np.argmax(preds, axis=-1)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if inputs is not None:
                inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

        if inputs is not None:
            decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

        for idx, (pred, label, input) in enumerate(
            zip(decoded_preds, decoded_labels, decoded_inputs)
        ):
            result = metric.compute(predictions=[pred], references=[label], use_stemmer=True)
            if self.cnt == 0:
                example_dict = {}
                example_dict["input"] = input
                example_dict["label"] = label
                self.run[f"finetuning/eval_predictions/example_{idx}"] = example_dict

            if idx in self.examples_df:
                # Append the new prediction to existing df
                df = self.examples_df[idx]
                new_row = pd.DataFrame(
                    {
                        "eval_step": [self.cnt],
                        "summarized prediction": [pred],
                        "metric": [result["rouge1"]],
                    }
                )
                df = pd.concat([df, new_row])
                df.reset_index(drop=True)
                self.examples_df[idx] = df
            else:
                # Create a new df for the example
                df = pd.DataFrame(
                    {
                        "eval_step": [self.cnt],
                        "summarized prediction": [pred],
                        "metric": [result["rouge1"]],
                    }
                )
                self.examples_df[idx] = df

            # (neptune) Upload the dataframe as csv
            buffer = io.StringIO()
            self.examples_df[idx].to_csv(buffer, index=False)
            self.run[f"finetuning/eval_predictions/example_{idx}/predictions"].upload(
                File.from_stream(buffer, extension="csv")
            )

        self.cnt += 1
        return {}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # (neptune) Initialize Neptune run
    run = neptune.init_run()

    # (neptune) Track S3 data
    DATA_DIR = "../data/"
    run["data"].track_files(data_args.s3_path)
    run.wait()
    run["data"].download(DATA_DIR)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Set up dataset
    data_files = {}
    data_files["train"] = DATA_DIR + "train.jsonl"
    extension = "json"
    data_files["validation"] = DATA_DIR + "val.jsonl"
    extension = "json"
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    # We need to tokenize inputs and targets.
    column_names = raw_datasets["train"].column_names
    text_column = "dialogue"
    summary_column = "summary"

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    # Get dataset for Trainer
    train_dataset, eval_dataset = get_dataset(
        data_args=data_args,
        training_args=training_args,
        tokenizer=tokenizer,
        text_column=text_column,
        summary_column=summary_column,
        padding=padding,
        max_target_length=max_target_length,
        raw_datasets=raw_datasets,
        column_names=column_names,
    )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # (neptune) NeptuneCallback takes care of logging trainer_params, model_params, etc
    neptune_callback = transformers.integrations.NeptuneCallback(run=run, log_checkpoints="best")

    # (neptune) Helper to log predictions of the evaluation data.
    eval_logger = EvalLogger(run, tokenizer, data_args.ignore_pad_token_for_loss).log_at_eval

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=eval_logger,  # (neptune) pass the eval logger to the trainer
        callbacks=[neptune_callback],  # (neptune) pass the callback to the trainer
    )

    # Training
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload


if __name__ == "__main__":
    main()
