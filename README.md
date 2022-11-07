# <p align=center>`MBart with Longformer windowed attention`</p>

Pretrained mBART model from huggingface with `Longformer` windowed attention in encoder (decoder has standard attention).

### Changelog
This is a fork of [a-rios/longmbart](https://github.com/a-rios/longmbart) which itself is highly inspired by [allenai/longformer](https://github.com/allenai/longformer). We added a few things:
* added more information to the readme
* changed the gitignore
* changed the requirements

This project has a chain of dependencies which are a bit tricky:
1. Your GPU determines which local cuda version you need 
1. Your local cuda version determindes which cudatoolkit you need to install
1. Your cudatoolkit determines which torch version you need
1. You torch version determines which pytorch-lightning version you need to use

When you are troubleshooting start at the top of this list and work your way through. This specific runs with an NVIDIA GeForce RTX 3090 24GB and Ubuntu 20.04. 

### Installation

Check your cuda version by running: `nvcc --version`. Please stick to this Installation procedure. Especially the transformer part in `requirements.txt` is important.

```
    bash
    conda create --name longmbart python=3.8.5 
    conda activate longmbart
    git clone https://github.com/ZurichNLP/longformer.git longmbart
    cd longmbart
    git checkout longmbart_hf4
    # replace this with your cuda version
    conda install cudatoolkit=11
    # change this version according to the cudatoolkit  
    conda install pytorch=1.12.1
    pip install .
    pip install -r requirements.txt
  ```
    
   To convert the huggingface mBART model, use scripts/convert_mbart_to_longformerencoderdecoder.py, for example:
   
   ```
   conda activate longmbart
   ```
   
   ``` 
   python $longformer_dir/scripts/convert_mbart_to_longformerencoderdecoder.py \
   --save_model_to path-to-save-new-model \
   --attention_window 512 \
   --reduce-to-vocab list-of-spm-pieces \
   --cache_dir path-to-huggingface-mbart \
   --add_language_tags de_A1 de_A2 de_B1 \
   --initialize_tags de_DE de_DE de_DE
   ```
   Convert the base model (without domain adaptation)
   ```
   python ./scripts/convert_mbart_to_longformerencoderdecoder.py \
   --save_model_to ./output/longmbart-large-cc25-base \
   --attention_window 512 \
   --cache_dir ./output/mbart-large-cc25 \
   --base_model facebook/mbart-large-cc25 \
   --tokenizer_name_or_path facebook/mbart-large-cc25\
   --add_language_tags de_OR de_SI \
   --initialize_tags de_DE de_DE \
   --max_pos 1024 \
   --verbose 1
   ```
   
   Convert the base model (without domain adaptation)
   ```
   python ./scripts/convert_mbart_to_longformerencoderdecoder.py \
   --save_model_to ./output/longmbart-large-cc25-german-literature \
   --attention_window 512 \
   --cache_dir ./output/domain-adaptation/cache \
   --base_model ./output/domain-adaptation \
   --tokenizer_name_or_path facebook/mbart-large-cc25\
   --add_language_tags de_OR de_SI \
   --initialize_tags de_DE de_DE \
   --max_pos 1024 \
   --verbose 1
   ```
   
   It is possible to change the `--base_model` parameter, which is default `facebook/mbart-large-cc25`, to any huggingface MBartForConditionalGeneration model. But in keep in mind to change `--max_pos` accordingly to avoid a tensor mismatch.
   
   After fiddeling with the source code it is recommended to run the script with `--verbose 2` which runs a test after the conversion. When you changed the added language tags, chnage line ~369 in `$longformer_dir/scripts/convert_mbart_to_longformerencoderdecoder.py` accordinly.
    
   `--reduce-vocab-to-list` will resize the orginal pretrained model's vocabulary to the the pieces given in the list (text file, one piece per line). Pieces must be part of the pretrained sentencepiece model. 
   `--add_language_tags` will add new language tags, use `--initialize_tags` to specify which embeddings they should be initialized with, e.g. for German language levels, start with the German embeddings.
   
   To fine-tune the converted model, use `longformer/simplification.py`. If training on multilingual data, preprocess your data to contain the language tags and </s> like this:
   * source text: `src_lang source_sequene` (actual sequence in the model will be `source_sequence </s> src_lang`, reordering happens internally). E.g.,: `de_DE Ein Satz zum testen.`
   * target text: `trg_lang target_sequence` 
   
 Example for fine-tuning (see `longformer/simplification.py` for all options):
   
```
python -m longformer.simplification \
--from_pretrained path-to-converted-model \
--tokenizer path-to-converted-model \
--save_dir path-to-save-fine-tuned-model \
--save_prefix "w512" \
--train_source path-to-source-train \
--train_target path-to-target-train \
--val_source path-to-source-dev \
--val_target path-to-target-dev \
--test_source path-to-source-test \
--test_target path-to-target-test \
--max_output_len max_target_length \
--max_input_len max_source_length \
--batch_size 1 \
--grad_accum 60 \
--num_workers 5 \
--gpus 1 \
--seed 222 \
--attention_dropout 0.1 \
--dropout 0.3 \
--attention_mode sliding_chunks \
--attention_window 512 \
--label_smoothing 0.2 \
--lr 0.00003 \
--val_every 1.0 \
--val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric 'rougeL' \
--patience 10 \
--lr_reduce_patience 8 \
--lr_reduce_factor 0.5 \
--grad_ckpt \
--progress_bar_refresh_rate 10
```

```
python -m longformer.simplification \
--from_pretrained ./output/longmbart-large-cc25-german-literature \
--tokenizer ./output/longmbart-large-cc25-german-literature \
--save_dir ./output/longmbart-large-cc25-german-literature-simplification \
--save_prefix "w512" \
--train_source ./data/gnats/train-source.txt \
--train_target ./data/gnats/train-target.txt \
--val_source ./data/gnats/val-source.txt \
--val_target ./data/gnats/val-target.txt \
--test_source ./data/gnats/test-source.txt \
--test_target ./data/gnats/test-target.txt \
--max_output_len 1024 \
--max_input_len 1024 \
--batch_size 1 \
--grad_accum 60 \
--num_workers 5 \
--gpus 1 \
--seed 222 \
--attention_dropout 0.1 \
--dropout 0.3 \
--attention_mode sliding_chunks \
--attention_window 512 \
--label_smoothing 0.2 \
--lr 0.00003 \
--val_every 1.0 \
--val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric 'f_bert' \
--patience 50 \
--lr_reduce_patience 8 \
--lr_reduce_factor 0.5 \
--grad_ckpt \
--progress_bar_refresh_rate 10 \
--wandb longmbart-gnats \
--tags_included
```

```
python -m longformer.simplification \
--from_pretrained ./output/converted-longmbart \
--tokenizer ./output/converted-longmbart \
--save_dir ./output/longmbart-large-cc25-german-literature-masked \
--save_prefix "w512" \
--train_source ./data/textgrid/train_source.txt \
--train_target ./data/textgrid/train_source.txt \
--val_source ./data/textgrid/val_source.txt \
--val_target ./data/textgrid/val_target.txt \
--test_source ./data/gnats/test-source.txt \
--test_target ./data/gnats/test-target.txt \
--max_output_len 20 \
--max_input_len 20 \
--batch_size 12 \
--grad_accum 60 \
--num_workers 5 \
--gpus 1 \
--seed 222 \
--attention_dropout 0.1 \
--dropout 0.3 \
--attention_mode sliding_chunks \
--attention_window 512 \
--label_smoothing 0.2 \
--lr 3e-8 \
--val_every 1.0 \
--val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric 'rougeL' \
--max_epochs 1 \
--patience 1 \
--lr_reduce_patience 8 \
--lr_reduce_factor 0.5 \
--grad_ckpt \
--progress_bar_refresh_rate 10 \
--wandb longmbart-domain-adaptation \
--tags_included
```

```
python -m longformer.simplification \
--from_pretrained ./output/converted-longmbart \
--tokenizer ./output/converted-longmbart \
--save_dir ./output/longmbart-large-cc25-german-literature-masked \
--save_prefix "w512" \
--train_source ./data/train-source.txt \
--train_target ./data/train-target.txt \
--val_source ./data/val-source.txt \
--val_target ./data/val-target.txt \
--test_source ./data/test-source.txt \
--test_target ./data/test-target.txt \
--max_output_len 20 \
--max_input_len 20 \
--batch_size 12 \
--grad_accum 60 \
--num_workers 5 \
--gpus 1 \
--seed 222 \
--attention_dropout 0.1 \
--dropout 0.3 \
--attention_mode sliding_chunks \
--attention_window 512 \
--label_smoothing 0.2 \
--lr 0.00003 \
--val_every 1.0 \
--val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric 'rougeL' \
--max_epochs 50 \
--patience 3 \
--lr_reduce_patience 8 \
--lr_reduce_factor 0.5 \
--grad_ckpt \
--progress_bar_refresh_rate 10 \
--wandb longmbart-domain-adaptation \
--tags_included
```

```
python -m longformer.simplification \
--from_pretrained ./output/converted-longmbart \
--tokenizer ./output/converted-longmbart \
--save_dir ./output/longmbart-large-cc25-simplification \
--save_prefix "w512" \
--train_source ./data/gnats/train-source.txt \
--train_target ./data/gnats/train-target.txt \
--val_source ./data/gnats/val-source.txt \
--val_target ./data/gnats/val-target.txt \
--test_source ./data/gnats/test-source.txt \
--test_target ./data/gnats/test-target.txt \
--max_output_len 1024 \
--max_input_len 1024 \
--batch_size 1 \
--grad_accum 60 \
--num_workers 5 \
--gpus 1 \
--seed 222 \
--attention_dropout 0.1 \
--dropout 0.3 \
--attention_mode sliding_chunks \
--attention_window 512 \
--label_smoothing 0.2 \
--lr 0.00003 \
--val_every 1.0 \
--val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric 'f_bert' \
--patience 3 \
--lr_reduce_patience 8 \
--lr_reduce_factor 0.5 \
--grad_ckpt \
--progress_bar_refresh_rate 10 \
--wandb longmbart-gnats \
--tags_included
```
```
python -m longformer.simplification \
--from_pretrained ./output/longmbart-large-cc25-german-literature-masked/w512 \
--resume_ckpt checkpoint{epoch:02d}_{rougeL:.5f}/epoch=1-step=38.ckpt \
--tokenizer ./output/longmbart-large-cc25-german-literature-masked/w512 \
--save_dir ./output/longmbart-large-cc25-german-literature-masked-simplification \
--save_prefix "w512" \
--train_source ./data/gnats/train-source.txt \
--train_target ./data/gnats/train-target.txt \
--val_source ./data/gnats/val-source.txt \
--val_target ./data/gnats/val-target.txt \
--test_source ./data/gnats/test-source.txt \
--test_target ./data/gnats/test-target.txt \
--max_output_len 1024 \
--max_input_len 1024 \
--batch_size 1 \
--grad_accum 60 \
--num_workers 5 \
--gpus 1 \
--seed 222 \
--attention_dropout 0.1 \
--dropout 0.3 \
--attention_mode sliding_chunks \
--attention_window 512 \
--label_smoothing 0.2 \
--lr 0.00003 \
--val_every 1.0 \
--val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric 'f_bert' \
--patience 3 \
--lr_reduce_patience 8 \
--lr_reduce_factor 0.5 \
--grad_ckpt \
--progress_bar_refresh_rate 10 \
--wandb longmbart-gnats \
--tags_included
```
```
python -m longformer.simplification \
--from_pretrained ./output/longmbart-large-cc25-german-literature-masked/w512/checkpoint{epoch:02d}_{rougeL:.5f}/epoch=0-step=38.ckpt \
--tokenizer ./output/longmbart-large-cc25-german-literature-masked/w512 \
--save_dir ./output/longmbart-large-cc25-german-literature-masked-simplification \
--save_prefix "w512" \
--train_source ./data/gnats/train-source.txt \
--train_target ./data/gnats/train-target.txt \
--val_source ./data/gnats/val-source.txt \
--val_target ./data/gnats/val-target.txt \
--test_source ./data/gnats/test-source.txt \
--test_target ./data/gnats/test-target.txt \
--max_output_len 1024 \
--max_input_len 1024 \
--batch_size 1 \
--grad_accum 60 \
--num_workers 5 \
--gpus 1 \
--seed 222 \
--attention_dropout 0.1 \
--dropout 0.3 \
--attention_mode sliding_chunks \
--attention_window 512 \
--label_smoothing 0.2 \
--lr 0.00003 \
--val_every 1.0 \
--val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric 'f_bert' \
--patience 3 \
--lr_reduce_patience 8 \
--lr_reduce_factor 0.5 \
--grad_ckpt \
--progress_bar_refresh_rate 10 \
--wandb longmbart-gnats \
--tags_included
```

Early stopping on one of these metrics: vloss, rouge1, rouge2, rougeL, rougeLsum, bleu (requires rouge_score and sacrebleu to be installed).
In a setting where translating from A to B, set `--src_lang A` and `--tgt_lang B` (input has no language tags), in a multilingual setting where source and target text already have language tags, use `--tags_included`. One of the options have to be selected, otherwise a KeyError will occur. 

To translate with a fine-tuned model, use `longformer/simplify.py`, for example like this:
```
python -m longformer.simplify \
--model_path path-to-fine-tuned-model \
--checkpoint "checkpointepoch=name-of-checkpoint" \
--tokenizer path-to-fine-tuned-model \
--translation output-file \
--test_source path-to-source \
--test_target path-to-reference \
--max_output_len max_target_length \
--max_input_len max_source_length \
--batch_size 2 \
--num_workers 5 \
--gpus 1 \
--beam_size 6 \
--progress_bar_refresh_rate 1 \
--tags_included
```


```
python -m longformer.simplify --model_path ./output/longmbart-large-cc25-simplification/w512 --checkpoint "checkpoint{epoch:02d}_{f_bert:.5f}/epoch=3-step=3-v1.ckpt" \
--tokenizer ./output/longmbart-large-cc25-simplification/w512 --translation ./output/generated/longmbart-large-cc25-simplification-epoch-3-v1.txt --test_source ./data/gnats/test-source.txt --test_target ./data/gnats/test-target.txt --max_output_len 1024 --max_input_len 1024 --batch 1 \
--num_workers 5 \
--gpus 1 \
--beam_size 6 \
--progress_bar_refresh_rate 1 \
--tags_included 
```

```
python -m longformer.simplify --model_path ./output/longmbart-large-cc25-german-literature-masked/w512 --checkpoint "checkpoint{epoch:02d}_{rougeL:.5f}/epoch=0-step=38.ckpt" \
--tokenizer ./output/longmbart-large-cc25-german-literature-masked/w512 --translation ./output/generated/tlongmbart-large-cc25-german-literature-masked-epoch-0-step38.txt --test_source ./data/gnats/test-source.txt --test_target ./data/gnats/test-target.txt --max_output_len 1024 --max_input_len 1024 --batch 1 \
--num_workers 5 \
--gpus 1 \
--beam_size 6 \
--progress_bar_refresh_rate 1 \
--tags_included 
```


```
python -m longformer.simplify --model_path ./output/converted-longmbart \
--tokenizer ./output/converted-longmbart --translation ./output/generated/converted-longmbart.txt --test_source ./data/gnats/test-source.txt --test_target ./data/gnats/test-target.txt --max_output_len 1024 --max_input_len 1024 --batch 1 \
--num_workers 5 \
--gpus 1 \
--beam_size 6 \
--progress_bar_refresh_rate 1 \
--tags_included 
```

```
python -m longformer.simplify --model_path ./output/longmbart-large-cc25-german-literature-masked-simplification/w512 --checkpoint "checkpoint{epoch:02d}_{f_bert:.5f}/epoch=3-step=3-v8.ckpt" \
--tokenizer ./output/longmbart-large-cc25-german-literature-masked-simplification/w512 --translation ./output/generated/longmbart-large-cc25-german-literature-masked-simplification-epoch-0-v8.txt --test_source ./data/gnats/test-source.txt --test_target ./data/gnats/test-target.txt --max_output_len 1024 --max_input_len 1024 --batch 1 \
--num_workers 5 \
--gpus 1 \
--beam_size 6 \
--progress_bar_refresh_rate 1 \
--tags_included 
```

```
python -m longformer.simplify --model_path ./output/longmbart-large-cc25-simplification/w512 --checkpoint "checkpoint{epoch:02d}_{f_bert:.5f}/epoch=3-step=3.ckpt" \
--tokenizer ./output/longmbart-large-cc25-simplification/w512 --translation ./output/generated/longmbart-large-cc25-simplification-epoch-3.txt --test_source ./data/gnats/test-source.txt --test_target ./data/gnats/test-target.txt --max_output_len 1024 --max_input_len 1024 --batch 1 \
--num_workers 5 \
--gpus 1 \
--beam_size 6 \
--progress_bar_refresh_rate 1 \
--tags_included 
```


Reference file is optional, if given, will print evaluation metrics (rouge1, rouge2, rougeL, rougeLsum, bleu). 
If only one target language, use `--tgt_lang` to set, if multiple languages, either give a reference file with tags (`tgt_lang target_sequence`) with `--tags_included` or just a list of target tags with `--target_tags` (one tag per line for each sample in `--test_source`).

## Install EASSE
```
git clone https://github.com/feralvam/easse.git
cd easse
pip install -e .
```

## Masterthesis Commands
### 1. ...
### 2. Domain Adaptation
#### 2.0 Debug F-BERT for batch size > 1
```
python -m longformer.simplification \
--from_pretrained ./output/converted-longmbart \
--tokenizer ./output/converted-longmbart \
--save_dir ./output/longmbart-large-cc25-german-literature-masked \
--save_prefix "5" \
--train_source ./data/textgrid/5_train_source.txt \
--train_target ./data/textgrid/5_train_target.txt \
--val_source ./data/textgrid/5_val_source.txt \
--val_target ./data/textgrid/5_val_target.txt \
--test_source ./data/textgrid/5_test_source.txt \
--test_target ./data/textgrid/5_test_target.txt \
--max_output_len 20 \
--max_input_len 20 \
--batch_size 12 \
--grad_accum 60 --num_workers 5 --gpus 1 --seed 222 --attention_dropout 0.1 --dropout 0.3 \
--attention_mode sliding_chunks --attention_window 512 --label_smoothing 0.2 \
--lr 3e-8 \
--val_every 1.0 --val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric 'rougeL' \
--max_epochs 1 \
--patience 1 \
--lr_reduce_patience 8 --lr_reduce_factor 0.5 --grad_ckpt --progress_bar_refresh_rate 10 \
--wandb longmbart-large-cc25-german-literature-masked \
--tags_included
```
#### 2.1. Bucket Size 50
```
python -m longformer.simplification \
--from_pretrained ./output/converted-longmbart \
--tokenizer ./output/converted-longmbart \
--save_dir ./output/longmbart-large-cc25-german-literature-masked \
--save_prefix "50" \
--train_source ./data/textgrid/50_train_source.txt \
--train_target ./data/textgrid/50_train_source.txt \
--val_source ./data/textgrid/50_val_source.txt \
--val_target ./data/textgrid/50_val_target.txt \
--test_source ./data/textgrid/50_test_source.txt \
--test_target ./data/textgrid/50_test_target.txt \
--max_output_len 70 \
--max_input_len 70 \
--batch_size 12 \
--grad_accum 60 --num_workers 5 --gpus 1 --seed 222 --attention_dropout 0.1 --dropout 0.3 \
--attention_mode sliding_chunks --attention_window 512 --label_smoothing 0.2 \
--lr 3e-8 \
--val_every 1.0 --val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric 'rougeL' \
--max_epochs 1 \
--patience 1 \
--lr_reduce_patience 8 --lr_reduce_factor 0.5 --grad_ckpt --progress_bar_refresh_rate 10 \
--wandb longmbart-large-cc25-german-literature-masked \
--tags_included
```

#### 2.2. Bucket Size 150
```
python -m longformer.simplification \
--from_pretrained ./output/converted-longmbart \
--tokenizer ./output/converted-longmbart \
--save_dir ./output/longmbart-large-cc25-german-literature-masked \
--save_prefix "100" \
--train_source ./data/textgrid/100_train_source.txt \
--train_target ./data/textgrid/100_train_source.txt \
--val_source ./data/textgrid/100_val_source.txt \
--val_target ./data/textgrid/100_val_target.txt \
--test_source ./data/textgrid/100_test_source.txt \
--test_target ./data/textgrid/100_test_target.txt \
--max_output_len 70 \
--max_input_len 70 \
--batch_size 12 \
--grad_accum 60 --num_workers 5 --gpus 1 --seed 222 --attention_dropout 0.1 --dropout 0.3 \
--attention_mode sliding_chunks --attention_window 512 --label_smoothing 0.2 \
--lr 3e-10 \
--val_every 1.0 --val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric 'rougeL' \
--max_epochs 1 \
--patience 1 \
--lr_reduce_patience 8 --lr_reduce_factor 0.5 --grad_ckpt --progress_bar_refresh_rate 10 \
--wandb longmbart-large-cc25-german-literature-masked \
--tags_included
```
#### 2.2. Bucket Size 100
```
python -m longformer.simplification \
--from_pretrained ./output/converted-longmbart \
--tokenizer ./output/converted-longmbart \
--save_dir ./output/longmbart-large-cc25-german-literature-masked \
--save_prefix "150" \
--train_source ./data/textgrid/150_train_source.txt \
--train_target ./data/textgrid/150_train_source.txt \
--val_source ./data/textgrid/150_val_source.txt \
--val_target ./data/textgrid/150_val_target.txt \
--test_source ./data/textgrid/150_test_source.txt \
--test_target ./data/textgrid/150_test_target.txt \
--max_output_len 70 \
--max_input_len 70 \
--batch_size 12 \
--grad_accum 60 --num_workers 5 --gpus 1 --seed 222 --attention_dropout 0.1 --dropout 0.3 \
--attention_mode sliding_chunks --attention_window 512 --label_smoothing 0.2 \
--lr 3e-10 \
--val_every 1.0 --val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric 'rougeL' \
--max_epochs 1 \
--patience 1 \
--lr_reduce_patience 8 --lr_reduce_factor 0.5 --grad_ckpt --progress_bar_refresh_rate 10 \
--wandb longmbart-large-cc25-german-literature-masked \
--tags_included
```