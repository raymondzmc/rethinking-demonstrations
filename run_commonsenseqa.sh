python test.py --dataset commonsense_qa --gpt2 channel-metaicl --method channel --out_dir out/channel-metaicl --do_zeroshot \
       --test_batch_size 1 --log_file commonsenseqa-channel-metaicl-demo-gold.txt --use_demonstrations --k 16 --seed 100 --interpret --device cpu --zero_baseline

# python test.py --dataset "commonsense_qa_random_english_words_gold_labels_seed=100" --gpt2 channel-metaicl --method channel --out_dir out/channel-metaicl --do_zeroshot \
#        --test_batch_size 1 --log_file commonsenseqa-channel-metaicl-demo-random.txt --use_demonstrations --k 16 --seed 100 --interpret --device cpu --zero_baseline
