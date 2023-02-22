#!/bin/bash
#SBATCH --gres=gpu:v100l:1      # Request GPU "generic resources"
#SBATCH --cpus-per-task=6      # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64000M            # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=24:00:00
#SBATCH --output=commonsense_qa_random.out

module load python
export HF_HOME="/project/def-carenini/liraymo6"
source /home/liraymo6/virtualenvs/metaicl/bin/activate
export MASTER_PORT=44145

python test.py --dataset "commonsense_qa_random_english_words_gold_labels_seed=100" --gpt2 channel-metaicl --method channel --out_dir out/channel-metaicl-random --do_zeroshot \
       --test_batch_size 1 --log_file commonsenseqa-channel-metaicl-demo-random.txt --use_demonstrations --k 16 --seed 100 --interpret --zero_baseline
