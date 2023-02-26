from functools import partial


def preprocess_function(
    examples, *, text_column, summary_column, tokenizer, data_args, padding, max_target_length
):
    prefix = "summarize: "

    # remove pairs where at least one record is None

    inputs, targets = [], []
    for i in range(len(examples[text_column])):
        if examples[text_column][i] and examples[summary_column][i]:
            inputs.append(examples[text_column][i])
            targets.append(examples[summary_column][i])

    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(
        inputs, max_length=data_args.max_source_length, padding=padding, truncation=True
    )

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        padding=padding,
        truncation=True,
    )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def get_dataset(
    *,
    tokenizer,
    text_column,
    summary_column,
    data_args,
    training_args,
    padding,
    max_target_length,
    raw_datasets,
    column_names
):
    preprocess_fn = partial(
        preprocess_function,
        tokenizer=tokenizer,
        text_column=text_column,
        summary_column=summary_column,
        data_args=data_args,
        padding=padding,
        max_target_length=max_target_length,
    )

    train_dataset = raw_datasets["train"]
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_fn,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    max_target_length = data_args.val_max_target_length
    eval_dataset = raw_datasets["validation"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            preprocess_fn,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    return train_dataset, eval_dataset
