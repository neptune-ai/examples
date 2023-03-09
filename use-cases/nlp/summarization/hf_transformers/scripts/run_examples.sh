set -e

echo "Installing requirements..."
pip install -U -r requirements.txt --user

echo "Running run_summarization.py..."
python run_summarization.py \
    --s3_path "s3://neptune-examples/data/samsum/data_v1/" \
    --learning_rate 5e-4 \
    --num_train_epochs 2 \
    --max_target_length 100 \
    --max_train_samples 2000 \
    --max_eval_samples 100 \
    --model_name_or_path google/t5-efficient-tiny \
    --dataset_name samsum \
    --dataset_config samsum \
    --report_to "none" \
    --output_dir "models/" \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=2 \
    --overwrite_output_dir \
    --load_best_model_at_end true \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --include_inputs_for_metrics true \
    --neptune_project "common/project-text-summarization-hf"
