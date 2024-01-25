# qsub -q "gpu.q" -pe smp 1 -l h_rt=336:00:00 -cwd -j yes -o logs/ucsf_bert_single_label.log run_single_label_tasks.sh

export PYTHONPATH=~/SequenceInfoInBERT/src/:$PYTHONPATH
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bc_path_extr
export PYTHONPATH=~/BreastCancerPathExtr/src/:$PYTHONPATH

tasks=('path_type' 'grade' 'pr' 'her2' 'biopsy' 'er' 'lvi' 'dcis_margins' 'margins' 'ln_involvement')
n_epochs=40

for task in "${tasks[@]}"
do
  # CUDA_VISIBLE_DEVICES=$SGE_GPU python3 BaseLineBERTSingle.py -task_name $task -ftrain train_$task.csv -fdev dev_$task.csv -ftest test_$task.csv
  out_dir="../../output/ucsf_bert/${task}/"

  CUDA_VISIBLE_DEVICES=$SGE_GPU \
  python3 run_transformers_classification.py \
  --task_name $task \
  --output_dir $out_dir \
  --num_train_epochs $n_epochs \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --overwrite_cache \
  --overwrite_predictions \
  --save_steps 500

  CUDA_VISIBLE_DEVICES=$SGE_GPU \
  python3 run_transformers_classification.py \
  --task_name $task \
  --output_dir $out_dir \
  --do_test \
  --per_gpu_eval_batch_size 16 \
  --overwrite_output_dir \
  --overwrite_cache \
  --overwrite_predictions
done