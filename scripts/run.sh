#!/bin/bash
#SBATCH --job-name audio-caption
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclude=gqxx-01-016
#SBATCH --output=logs/audio-caption-%j.log
#SBATCH --error=logs/audio-caption-%j.err

module load java/1.8.0.162

run_script=$1
run_config=$2
kaldi_stream=$3
kaldi_scp=$4
eval_caption_file=$5
eval_embedding_file=$6

seed=1
if [ $# -eq 7 ]; then
    seed=$7
fi

# stage 1, train the audio caption model
if [ ! $experiment_path ]; then
    experiment_path=`python ${run_script} \
                            train \
                            ${run_config} \
                            --seed $seed`
fi

# stage 2, evaluate by several metrics

if [ ! $experiment_path ]; then
    echo "invalid experiment path, maybe the training ended abnormally"
    exit 1
fi

python ${run_script} \
       evaluate \
       ${experiment_path} \
       "${kaldi_stream}" \
       ${kaldi_scp} \
       ${eval_caption_file} \
       --caption-embedding-path ${eval_embedding_file}

#python ${run_script} \
       #sample \
       #${experiment_path} \
       #"${kaldi_stream}" \
       #$kaldi_scp
