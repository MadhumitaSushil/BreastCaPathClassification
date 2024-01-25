# qsub -q "gpu.q" -pe smp 2 -l h_rt=336:00:00 -l gpu_mem=40G -cwd -j yes -o h_ucsf_bert_single_label.log run_single_label_hier_tasks.sh

export PYTHONPATH=~/SequenceInfoInBERT/src/:$PYTHONPATH
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bc_path_extr
export PYTHONPATH=~/BreastCancerPathExtr/src/:$PYTHONPATH

tasks=('path_type' 'grade' 'pr' 'her2' 'biopsy' 'er' 'lvi' 'dcis_margins' 'margins' 'ln_involvement')
#tasks=('histo' 'site_disease' 'site_examined')
max_seq_len=4096

for task in "${tasks[@]}"
do
#  CUDA_VISIBLE_DEVICES=$SGE_GPU python3 HiBERTSingle.py -task_name $task -ftrain train_$task.csv -fdev dev_$task.csv -ftest test_$task.csv
  out_dir="../../output/h_ucsf_bert/${task}/"

  CUDA_VISIBLE_DEVICES=$SGE_GPU \
  python3 run_transformers_classification.py \
  --task_name $task \
  --output_dir $out_dir \
  --max_seq_length $max_seq_len \
  --hierarchical \
  --num_train_epochs 40 \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --overwrite_cache \
  --overwrite_predictions \

  CUDA_VISIBLE_DEVICES=$SGE_GPU \
  python3 run_transformers_classification.py \
  --task_name $task \
  --output_dir $out_dir \
  --max_seq_length $max_seq_len \
  --hierarchical \
  --do_test \
  --per_gpu_eval_batch_size 4 \
  --overwrite_output_dir \
  --overwrite_cache \
  --overwrite_predictions

done