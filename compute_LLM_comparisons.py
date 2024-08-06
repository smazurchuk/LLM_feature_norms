#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:26:05 2024

This script does pairwise comparisons for a given feature (indexed
by 'featID' variable). 

To not run in parallel, just loop over FeatID

Note: Both ends of a feature are used (e.g., Larger and Smaller)

@author: smazurchuk
"""
import os, sys
featID = int(args[1])
gpuN   = str(args[2]) # can pass list e.g., ['1','2']
print(f'I have gpus: {gpuN} and featID: {featID}')
os.environ['CUDA_VISIBLE_DEVICES'] = gpuN
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
import gc
import torch
import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, FlaxGemmaForCausalLM
from glob import glob


# # Load model
model_id = "google/gemma-1.1-7b-it"
dtype    = torch.float16
tokenizer = AutoTokenizer.from_pretrained(model_id)

model, params = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto")

# # Create prompts for model
word_df = pd.read_excel('data/categories.xlsx')
hRatings = {}
fnames = sorted(glob('data/data_tables/table_*.csv'))
for name in fnames:
    tmp   = pd.read_csv(name).mean(1)
    feat  = name.split('_')[-1].split('.csv')[0] 
    cat   = name.split('_')[-2] 
    words =  word_df[cat].dropna()
    tmp.index = words
    hRatings[f'{cat}_{feat}'] = tmp
featDict = {
    'danger': ['more dangerous, deadly, or threatening','more safe, harmless, or calm'],
    'gender': ['more masculine','more feminine'],
    'intelligence': ['more intelligent, smart, or wise','more stupid, dumb, idiotic'],
    'loudness': ['louder, more deafening, or noisy','quieter or more soft'],
    'size': ['larger or bigger','smaller or tiny'],
    'speed': ['faster, speedier, or more quick','slower or more sluggish'],
    'weight': ['heavier, fatter, or thicker','lighter, skinnier, or more thin'],
    'wetness': [ 'more wet or wetter','more dry, or land associated'],
    'arousal': ['more interesting, exciting, or fun','more boring, unexciting, or dull'],
    'cost': ['more expensive, costly, or fancy','more inexpensive, cheap, or budget'],
    'religiosity': ['more religious, spiritual, or orthodox','more atheist, secular, or agnostic'],
    'wealth': ['richer, wealthier, or more privileged','poorer, impoverished, or underpriviledged'],
    'age': ['older or more elderly','younger, more youthful, or child-like'],
    'location': ['more indoors or inside','more outdoors or outside'],
    'political': ['more democrat, liberal, or progressive','more republican, conservative, or redneck'],  
    'valence': ['more pleasant, good, or happy','more bad, awful, sad'],
    'temperature': ['more hot, warm, or tropical','colder, cooler, or more frigid']}
entries = sorted(list(hRatings.keys()))


# To not run in parallel  put everything below this is a loop
#for featID in range(len(entries)):
entry = entries[featID]
if not os.path.exists(f'output_rdms/{entry}_v4.npy'):
    entry = entries[featID]
    print(f'Working on: {entry}')

    # Make new prompts
    cat, feat = entry.split('_')
    if cat == 'clothing':
        cat = 'pieces of clothing'
    if cat == 'myth':
        cat = 'myths'
    if cat == 'weather':
        cat = 'weather events'
    series = hRatings[entry].sort_index(); 
    words = series.index.to_list()
    nPrompts1 = []; nPrompts2 = []
    for i in range(len(words)):
        for j in range(len(words)):
            if i != j:
                nPrompts1.append(f'Which of the following {cat} is {featDict[feat][0]}? \nA) {words[i]} \nB) {words[j]}')
                nPrompts2.append(f'Which of the following {cat} is {featDict[feat][1]}? \nA) {words[i]} \nB) {words[j]}')

    # Run model!
    outputs1 = []; outputs2 = []
    for idx in tqdm(range(len(nPrompts1))):    
        chat1 = [
            { "role": "user", "content": nPrompts1[idx]},
        ]
        chat2 = [
            { "role": "user", "content": nPrompts2[idx]},
        ]
        prompt1 = tokenizer.apply_chat_template(chat1, tokenize=False, add_generation_prompt=True)
        prompt2 = tokenizer.apply_chat_template(chat2, tokenize=False, add_generation_prompt=True)
        # Process 1
        inputs = tokenizer(prompt1,return_tensors="np",padding=True)
        output = model.generate(**inputs, params=params, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        
        inputs = tokenizer.encode(prompt1, add_special_tokens=False, return_tensors="pt")
        out = model.generate(input_ids=inputs.to(model.device), max_new_tokens=10)
        out2 = tokenizer.decode(out[0],skip_special_tokens=True)
        outputs1.append(out2)
        # Process 2
        inputs = tokenizer.encode(prompt2, add_special_tokens=False, return_tensors="pt")
        out = model.generate(input_ids=inputs.to(model.device), max_new_tokens=10)
        out2 = tokenizer.decode(out[0],skip_special_tokens=True)
        outputs2.append(out2)

    # Process model outputs
    rdm1 = np.zeros((len(words),len(words))); c=0
    exceptions = []
    for i in range(len(words)):
        for j in range(len(words)):
            if i != j:
                w1 = words[i]; w2 = words[j]   
                out = outputs1[c]; comp = 10
                out = out.strip().lower().split('\nmodel\n')[-1]
                id1 = out.find(w1); id2 = out.find(w2)
                probWord = 'NA'
                if ((id1 < id2) or id2 ==-1) and id1 != -1 :
                    probWord = w1
                if ((id2 < id1) or id1 == -1) and id2 != -1:
                    probWord = w2
                if (' a) ' in out) or (w1 == probWord):
                    comp = 1; 
                if (' b) ' in out) or (w2 == probWord):
                    if comp == 1:
                        comp = 10
                    else:
                        comp = -1;
                if comp not in [1,-1]:
                    comp = 0
                    exceptions.append([w1,w2,nPrompts1[c],out])
                rdm1[i,j] = comp; c+=1
    rdm2 = np.zeros((len(words),len(words))); c=0
    for i in range(len(words)):
        for j in range(len(words)):
            if i != j:
                w1 = words[i]; w2 = words[j]   
                out = outputs2[c]; comp = 10
                out = out.strip().lower().split('\nmodel\n')[-1]
                id1 = out.find(w1); id2 = out.find(w2)
                probWord = 'NA'
                if ((id1 < id2) or id2 ==-1) and id1 != -1 :
                    probWord = w1
                if ((id2 < id1) or id1 == -1) and id2 != -1:
                    probWord = w2
                if (' a) ' in out) or (w1 == probWord):
                    comp = 1; 
                if (' b) ' in out) or (w2 == probWord):
                    if comp == 1:
                        comp = 10
                    else:
                        comp = -1;
                if comp not in [1,-1]:
                    comp = 0
                    exceptions.append([w1,w2,nPrompts2[c],out])
                rdm1[i,j] = comp; c+=1

    # Quick Check!
    pred1 = rdm1.mean(1) - rdm1.mean(0)
    act = series
    c1 = np.corrcoef(pred1,act)[0,1]
    pred2 = -(rdm2.mean(1) - rdm2.mean(0))
    c2 = np.corrcoef(pred2,act)[0,1]
    rdm = rdm1 - rdm2
    pred = rdm.mean(1) - rdm.mean(0)
    c = np.corrcoef(pred,act)[0,1]
    
    
    print(f'There were {len(exceptions)} exceptions')
    print(f'Corr {entry}: \n\tpred1: {c1}  \n\tpred2: {c2} \n\tpredB: {c}')
    
    # Save!
    np.save(f'output_rdms/{entry}_v4.npy',rdm)

# # Long check
# data = []
# for i in range(rdm.shape[0]):
#     for j in range(rdm.shape[0]):
#         outcome = rdm[i,j]
#         if outcome > 0:
#             data.append((i,j))
#         if outcome < 0:
#             data.append((j,i))
# f_param = choix.ilsr_pairwise(rdm.shape[0],data,alpha=10)
# c = np.corrcoef(f_param,series)[0,1]
# print(f'Corr {entry}: {c}')
