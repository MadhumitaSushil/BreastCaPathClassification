# qsub -q "gpu.q" -pe smp 1 -l h_rt=336:00:00 -cwd -j yes -o lstm.log run_lstm_path.sh

export PYTHONPATH=~/SequenceInfoInBERT/src/:$PYTHONPATH
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bc_path_extr
export PYTHONPATH=~/BreastCancerPathExtr/src/:$PYTHONPATH

CUDA_VISIBLE_DEVICES=$SGE_GPU python lstm/lstm_path_classification.py