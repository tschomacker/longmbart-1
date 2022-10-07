# https://colab.research.google.com/github/georgianpartners/Transformers-Domain-Adaptation/blob/master/notebooks/GuideToTransformersDomainAdaptation.ipynb#scrollTo=-Jt-zg1V4i3g

import itertools as it
from pathlib import Path
from typing import Sequence, Union, Generator

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import MBartForConditionalGeneration, MBartTokenizer
import argparse

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def main():
    parser = argparse.ArgumentParser(description="Convert BART to LongBART. Replaces BART encoder's SelfAttnetion with LongformerSelfAttention")
    parser.add_argument(
        '--base_model',
        type=str,
        default='facebook/mbart-large-cc25',
        help='The name or path of the base model you want to convert'
    )
    
    args = parser.parse_args()

    model_card = 'facebook/mbart-large-cc25'#args.base_model

    # this code is mostly based on this discussion: https://github.com/huggingface/transformers/issues/5096

    model = MBartForConditionalGeneration.from_pretrained(model_card)
    tok = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')

    input_batch = ["My dog is <mask></s>", "It loves to play in the <mask></s>"]
    decoder_input_batch = ["<s>My dog is cute", "<s>It loves to play in the park"]
    labels_batch = ["My dog is cute</s>", "It loves to play in the park</s>"]

    input_ids = tok.batch_encode_plus(input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
    decoder_input_ids = tok.batch_encode_plus(decoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
    labels = tok.batch_encode_plus(labels_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids

    loss = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)[0]
    print(loss)

    # print('Domain Adaptation for',model_card,'vocab size:',model.config.vocab_size 
    # ,'and a tokenizer with a vocab_size:', len(tokenizer))

    # datasets = load_dataset(
    #     'text', 
    #     data_files={
    #         "train": '../data/domain-adaptation-train.txt', 
    #         "val": '../data/domain-adaptation-train.txt'
    #     }
    # )

    # print(datasets)


    # tokenized_datasets = datasets.map(
    #     lambda examples: tokenizer(examples['text'], truncation=True, max_length=model.config.max_position_embeddings), 
    #     batched=True
    # )

    # print(tokenized_datasets)
    # print('train',tokenized_datasets['train'])

    # for sample in tokenized_datasets['train']:
    #     print('sample',sample)



    # training_args = TrainingArguments(
    #     output_dir="../output/domain_pre_training",
    #     overwrite_output_dir=True,
    #     max_steps=100,
    #     per_device_train_batch_size=1,
    #     per_device_eval_batch_size=1,
    #     evaluation_strategy="steps",
    #     save_steps=50,
    #     save_total_limit=2,
    #     logging_steps=50,
    #     seed=42,
    #     # fp16=True,
    #     dataloader_num_workers=2,
    #     disable_tqdm=False,
    #     no_cuda=True,
    # )

    # # DataCollatorForLanguageModeling do NOT work with mBART -> https://github.com/huggingface/transformers/issues/11451


    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    # )
    # model.resize_token_embeddings(len(tokenizer))

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_datasets['train'],
    #     eval_dataset=tokenized_datasets['val'],
    #     data_collator=data_collator,
    #     tokenizer=tokenizer,  # This tokenizer has new tokens
    # )

    # trainer.train()

if __name__ == "__main__":
    main()