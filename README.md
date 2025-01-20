# STEPS #
1. Modify Annotation File using modify_data_annotation.ipynb
2. Preprocessing Image Features using prepro_feats.py
3. Preprocessing Label Using scripts/prepro_labels.py
4. Create tokenizer bpe using scripts/bpe_encoding.py
5. Create vocab with bpe tokenizer using bpe_encoding.py
6. Create ngrams vocab with bpe tokenizer using scripts/prepro_ngrams.py

# TRAIN #
1. train.py

# EVAL #
1. eval.py
