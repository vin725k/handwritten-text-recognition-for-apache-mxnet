# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:52:30 2019

@author: 703222761
"""

from collections import defaultdict
import json
import os
import random
import string

#%%

from gluoncv.data.batchify import Tuple, Stack, Append, Pad
import gluonnlp as nlp
import hnswlib # https://github.com/nmslib/hnswlib
import mxboard
import mxnet as mx
from mxnet import gluon, autograd
import numpy as np
import re
from tqdm import tqdm

#%%

from ocr.utils.encoder_decoder import get_transformer_encoder_decoder, Denoiser, encode_char, decode_char, LabelSmoothing, SoftmaxCEMaskedLoss

#%%

ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()

#%%
# See get_mode.py
text_filepath = 'dataset/typo/all.txt'

#%%
ALPHABET = ['<UNK>', '<PAD>', '<BOS>', '<EOS>']+list(' ' + string.ascii_letters + string.digits + string.punctuation)
ALPHABET_INDEX = {letter: index for index, letter in enumerate(ALPHABET)} # { a: 0, b: 1, etc}
FEATURE_LEN = 150 # max-length in characters for one document
NUM_WORKERS = 8 # number of workers used in the data loading
BATCH_SIZE = 64 # number of documents per batch
MAX_LEN_SENTENCE = 150
PAD = 1
BOS = 2
EOS = 3
UNK = 0
max_len_vocab = 500000

moses_detokenizer = nlp.data.SacreMosesDetokenizer()
moses_tokenizer = nlp.data.SacreMosesTokenizer()


#%%


def get_knn_index():
    model, vocab = nlp.model.big_rnn_lm_2048_512(dataset_name='gbw', pretrained=True, ctx=mx.cpu())

    step = 1024
    dim = 512
    num_elements = max_len_vocab+step
    data = np.zeros((num_elements, dim), dtype='float32')
    data_labels = np.arange(max_len_vocab)
    for i in tqdm(range(1, max_len_vocab, step)):
        data[i:i+step,:] = model.embedding(mx.nd.arange(i,i+step)).asnumpy()
    # Declaring index
    p = hnswlib.Index(space = 'cosine', dim = dim) # possible options are l2, cosine or ip

    # Initing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements = max_len_vocab, ef_construction = 200, M = 16)

    # Element insertion (can be called several times):
    p.add_items(data[:max_len_vocab], data_labels)
    # Controlling the recall by setting ef:
    p.set_ef(50) # ef should always be > k
    return p, data, vocab

#%%
    
class NoisyTextDataset(mx.gluon.data.Dataset):
    def __init__(self, 
                 text_filepath=None, 
                 substitute_costs_filepath='models/substitute_probs.json', 
                 insert_weight=1, 
                 delete_weight=1, 
                 glue_prob=0.05, 
                 substitute_weight=2,
                 max_replace=0.3,
                 is_train=True, 
                 split=0.9, 
                 data_type='corpus', 
                 gbw_corpus=None,
                 knn_index=knn_index,
                 knn_data=knn_data,
                 knn_vocab=knn_vocab,
                 proba_synonym=0.1
                ):
        self.max_replace = max_replace
        self.replace_weight = 0 #replace_prob  #Ignore typo dataset
        self.substitute_threshold = float(substitute_weight) / (insert_weight + delete_weight + substitute_weight)
        self.insert_threshold = self.substitute_threshold + float(insert_weight) / (insert_weight + delete_weight + substitute_weight)
        self.delete_threshold = self.insert_threshold + float(delete_weight) / (insert_weight + delete_weight + substitute_weight)
        self.glue_prob = glue_prob
        self.substitute_dict = json.load(open(substitute_costs_filepath,'r'))
        self.split = split
        self.data_type = data_type
        if self.data_type == 'corpus':
            self.text = self._process_text(text_filepath, is_train)
        elif self.data_type == 'GBW':
            self.gbw_corpus = gbw_corpus
        self.knn_index = knn_index
        self.knn_data = knn_data
        self.knn_vocab = knn_vocab
        self.proba_synonym = proba_synonym
    
    def _process_text(self, filename, is_train):
        with open(filename, 'r', encoding='Latin-1') as f:
            text = []
            for line in f.readlines():
                if line != '':
                    text.append(line.strip())
            
            split_index = int(self.split*len(text))
            if is_train:
                text = text[:split_index]
            else:
                text = text[split_index:]
        return text
    
    def _replace_synonym(self, line):
        processed_line = self._pre_process_line(line)
        words = []
        num_words = 100
        for i, word in enumerate(processed_line):
            draw = random.random()
            if word in self.knn_vocab and self.knn_vocab[word] < max_len_vocab and draw < self.proba_synonym and word not in string.punctuation :
                index_list = self.knn_index.knn_query(self.knn_data[self.knn_vocab[word]], k=num_words)[0][0]
                word = self.knn_vocab.idx_to_token[index_list[random.randint(0,num_words-1)]]
            words.append(word)
        return self._post_process_line(words)
    
    def _transform_line(self, line):
        """
        replace words that are in the typo dataset with a typo
        with a probability `self.replace_proba`
        """
        output = []
        
        processed_line = self._pre_process_line(line)
        
        # We get randomly the index of the modifications
        num_chars = len(''.join(processed_line))
        if num_chars:
            index_modifications = np.random.choice(num_chars, random.randint(0, int(self.max_replace*num_chars)), replace=False)
            substitute_letters = []
            insert_letters = []
            delete_letters = []
            # We randomly assign these indices to modifications based on precalculated thresholds
            for index in index_modifications:
                draw = random.random()
                if draw < self.substitute_threshold:
                    substitute_letters.append(index)
                    continue
                if draw < self.insert_threshold:
                    insert_letters.append(index)
                    continue
                else:
                    delete_letters.append(index)
                            
        
        j = 0
        for i, word in enumerate(processed_line):
            
            if word != '' and word not in string.punctuation:
                
                len_word = len(word)
                word_ = []
                k = j
                for letter in word:
                    if k in substitute_letters and letter in self.substitute_dict:
                        draw = random.random()
                        for replace, prob in self.substitute_dict[letter].items():
                            if draw < prob:
                                letter = replace
                                break
                    word_.append(letter)
                    k += 1
                word = ''.join(word_)
                                
                # Insert random letter
                k = j
                word_ = []
                for letter in word:
                    if k in insert_letters:
                        word_.append(ALPHABET[random.randint(4, len(ALPHABET)-1)])
                    word_.append(letter)
                    k += 1
                word = ''.join(word_)
                
                # Delete random letter
                k = j
                word_ = []
                for letter in word:
                    if k not in delete_letters:
                        word_.append(letter)
                    k += 1
                word = ''.join(word_)
                    
                output.append(word)
            else:
                output.append(word)
            j += len(word)

        output_ = [""]*len(output)
        j = 0
        for i, word in enumerate(output):
            output_[j] += word
            if random.random() > self.glue_prob:
                j += 1
        
        line = self._post_process_line(output_)
        return line.strip()
    
    def _pre_process_line(self, line):
        line = line.replace('\n','').replace('`',"'").replace('--',' -- ')
        return moses_tokenizer(line)
        
    def _post_process_line(self, words):
        output = ' '.join(moses_detokenizer(words))
        return output
    
    def _match_caps(self, original, typo):
        if original.isupper():
            return typo.upper()
        elif original.istitle():
            return typo.capitalize()
        else:
            return typo
    
    def __getitem__(self, idx):
        if self.data_type == 'GBW':
            tokens = moses_detokenizer(self.gbw_corpus[idx][:-1])
            if len(tokens) > 6:
                start = random.randint(0, len(tokens)-3)
                end = random.randint(start, len(tokens))
                tokens = tokens[start:end]
            line = ' '.join(tokens)
        else:
            line = self.text[idx]
        line = self._replace_synonym(line)
        line_typo = self._transform_line(line)
        return line_typo, line

    def __len__(self):
        if self.data_type == 'GBW':
            return len(self.gbw_corpus)
        else:
            return len(self.text)
        
#%%
            
def encode_char(text, src=True):
    encoded = np.ones(FEATURE_LEN, dtype='float32') * PAD
    text = text[:FEATURE_LEN-2]
    i = 0
    if not src:
        encoded[0] = BOS
        i = 1
    for letter in text:
        if letter in ALPHABET_INDEX:
            encoded[i] = ALPHABET_INDEX[letter]
        i += 1
    encoded[i] = EOS
    return encoded, np.array([i+1]).astype('float32')

def encode_word(text, src=True):
    tokens = tokenizer(text)
    indices = vocab[tokens]
    indices += [vocab['<EOS>']]
    indices = [vocab['<BOS>']]+indices
    return indices, np.array([len(indices)]).astype('float32')

def transform(data, label):
    src, src_valid_length = encode_char(data, src=True)
    tgt, tgt_valid_length = encode_char(label, src=False)
    return src, src_valid_length, tgt, tgt_valid_length, data, label

def decode_char(text):
    output = []
    for val in text:
        if val == EOS:
            break
        elif val == PAD or val == BOS:
            continue
        output.append(ALPHABET[int(val)])
    return "".join(output)


def decode_word(indices):
    return detokenizer([vocab.idx_to_token[int(i)] for i in indices], return_str=True).replace('<PAD>','')


#%%
    

%%time
# We get a knn index to substitute words
knn_index, knn_data, knn_vocab  = get_knn_index()

#%%


word = "test"
num_words = 200
index_list = knn_index.knn_query(knn_data[knn_vocab[word]], k=1000)[0][0]
knn_vocab.idx_to_token[index_list[random.randint(0,num_words-1)]]

#%%


dataset_train = NoisyTextDataset(text_filepath=text_filepath, glue_prob=0.2, is_train=True).transform(transform)
dataset_test = NoisyTextDataset(text_filepath=text_filepath, glue_prob=0.2, is_train=False).transform(transform)

# Finetuning on the text from the IAM dataset
dataset_train_ft = NoisyTextDataset(text_filepath='dataset/typo/text_train.txt', is_train=True, split=1.0, knn_index=knn_index).transform(transform)

#%%

dataset_train[random.randint(0, len(dataset_train)-1)]


#%%

for i in range(10):
    print(dataset_train[42][5])

#%%
    
data = json.load(open('dataset/typo/validating.json','r'))
data_ = []
for label, modified in data:
    if label.strip() != modified.strip():
        data_.append([label, modified])
val_dataset_ft = gluon.data.ArrayDataset(list(list(zip(*data_))[1]), list(list(zip(*data_))[0])).transform(transform)

#%%

val_dataset_ft[random.randint(0, len(val_dataset_ft)-1)]

#%%

gbw_stream = nlp.data.GBWStream(segment='train', skip_empty=True, bos=None, eos='<EOS>')

#%%

for e, corpus in enumerate(gbw_stream):
    dataset_gbw = NoisyTextDataset(gbw_corpus=corpus, data_type='GBW').transform(transform)
    break

#%%
    
dataset_gbw[7]

#%%

def batchify_list(elem):
    output = []
    for e in elem:
        output.append(elem)
    return output
    
batchify = Tuple(Stack(), Stack(), Stack(), Stack(), batchify_list, batchify_list)
batchify_word = Tuple(Stack(), Stack(), Pad(), Stack(), batchify_list, batchify_list)

#%%

train_data = gluon.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, last_batch='rollover', batchify_fn=batchify, num_workers=5)
test_data = gluon.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, last_batch='keep', batchify_fn=batchify, num_workers=5)
val_data_ft = gluon.data.DataLoader(val_dataset_ft, batch_size=BATCH_SIZE, shuffle=True, last_batch='keep', batchify_fn=batchify, num_workers=0)
train_data_ft = gluon.data.DataLoader(dataset_train_ft, batch_size=BATCH_SIZE, shuffle=True, last_batch='rollover', batchify_fn=batchify, num_workers=5)

#%%

