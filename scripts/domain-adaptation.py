# https://colab.research.google.com/github/georgianpartners/Transformers-Domain-Adaptation/blob/master/notebooks/GuideToTransformersDomainAdaptation.ipynb#scrollTo=-Jt-zg1V4i3g

import itertools as it
from pathlib import Path
from typing import List, Sequence, Union, Generator

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import MBartForConditionalGeneration, MBartTokenizer
import argparse

import random
from tqdm.auto import tqdm

import torch


def create_batches_from_file(filepath:str,batch_size=8,mask_percentage=15):
    with open(filepath) as f:
        batches = []
        sentences_batch=[]
        for line in f:
            # append and remove line break
            sentences_batch.append(line.strip())
            # add current batch to batch list
            if len(sentences_batch) == batch_size:
                batches.append(create_batch(sentences_batch, mask_percentage))
                sentences_batch = []
        # add the last batch, which has size < batch_size
        if len(sentences_batch) > 0:
            batches.append(create_batch(sentences_batch, mask_percentage))
    return batches


def create_batch(sentences:List, mask_percentage=15):
    
    input_batch = []
    decoder_input_batch = []
    labels_batch = []
    for sentence in sentences:
        masked_words = []
        for word in sentence.split(' '):
            if random.randint(0, 100)  > mask_percentage:
                masked_words.append(word)
            else:
                # masking
                masked_words.append('<mask>')
        input_batch.append(' '.join(masked_words)+'</s>')
        decoder_input_batch.append('<s>'+sentence)
        labels_batch.append(sentence+'</s>')

    return input_batch, decoder_input_batch, labels_batch


def main():
    parser = argparse.ArgumentParser(description="Convert BART to LongBART. Replaces BART encoder's SelfAttnetion with LongformerSelfAttention")
    parser.add_argument(
        '--base_model',
        type=str,
        default='facebook/mbart-large-cc25',
        help='The name or path of the base model you want to convert'
    )
    
    args = parser.parse_args()

    model_card = args.base_model

    # this code is mostly based on this discussion: https://github.com/huggingface/transformers/issues/5096
    # https://moon-ci-docs.huggingface.co/docs/transformers/pr_18904/en/model_doc/mbart#training-of-mbart
    
    model = MBartForConditionalGeneration.from_pretrained(model_card).to("cuda")
    tok = MBartTokenizer.from_pretrained(model_card)
    
    #print(input_batch)
    #print(decoder_input_batch)
    #print(labels_batch)

    input_file = '../data/domain_adaptation_sentences.txt'
    print('start creating the batches')
    batches = create_batches_from_file(input_file,batch_size=8,mask_percentage=15)
    max_sentence_length = 30

    for input_batch, decoder_input_batch, labels_batch in tqdm(batches, desc="Training Batch(es)"):

        input_ids = tok.batch_encode_plus(input_batch, 
                                            add_special_tokens=False, 
                                            return_tensors="pt", 
                                            padding=True,
                                            truncation=True,
                                            max_length=max_sentence_length).input_ids.cuda()

        decoder_input_ids = tok.batch_encode_plus(decoder_input_batch, 
                                                    add_special_tokens=False, 
                                                    return_tensors="pt", 
                                                    padding=True,
                                                    truncation=True,
                                                    max_length=max_sentence_length).input_ids.cuda()

        labels = tok.batch_encode_plus(labels_batch, 
                                        add_special_tokens=False, 
                                        return_tensors="pt", 
                                        padding=True,
                                        truncation=True,
                                        max_length=max_sentence_length).input_ids.cuda()

        loss = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)[0]
        #del input_ids
        #del decoder_input_ids
        #del labels
        #torch.cuda.empty_cache()
    print(loss)
    #https://huggingface.co/docs/transformers/serialization?highlight=save%20model#onnx
    model.save_pretrained("../output/spt-domain-adaptation")

if __name__ == "__main__":
    main()