#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:22:59 2024

This processes the outputs of pair-wise comparisons and generates figures
for poster

@author: smazurchuk
"""
import os
import choix
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from scipy.spatial.distance import pdist, cdist
from scipy.stats import kendalltau


# # # 
# Load human ratings from Grand et al, calculate agreement (icc)
# # #
word_df = pd.read_excel('data/categories.xlsx')
hRatings = {}; iccs ={}; pcorrs = {}
fnames = sorted(glob('data/data_tables/table_*.csv'))

# Calculate average pairwise correlation (using Fischer-Z transform)
def calcPcorr(a,b=None):
    if b is None:
        corrs = np.tanh(np.mean(np.arctanh(1-pdist(a,'correlation'))))
    else:
        corrs = np.tanh(np.mean(np.arctanh(1-cdist(a,b,'correlation'))))
    return corrs

for name in fnames:
    tmp    = pd.read_csv(name)
    tmp['word_idx'] = tmp.index
    longDF = pd.melt(tmp,id_vars='word_idx')
    icc = pg.intraclass_corr(longDF,targets='word_idx',raters='variable',ratings='value')
    feat   = name.split('_')[-1].split('.csv')[0] 
    cat    = name.split('_')[-2] 
    words  =  word_df[cat].dropna()
    tmp = tmp.drop('word_idx',axis=1)
    tmp.index = words
    hRatings[f'{cat}_{feat}'] = tmp
    iccs[f'{cat}_{feat}'] = icc.iloc[1].ICC
    pcorrs[f'{cat}_{feat}'] = calcPcorr(tmp.to_numpy().T)

# Load LLM pair-wise comparisons
entries = sorted(list(hRatings.keys()))
llm_rdms = {}
for entry in entries:
    fname = f'output_rdms/{entry}_v4.npy'
    if os.path.exists(fname) and 'states_intelligence' not in fname:
        llm_rdms[entry] = np.load(fname)

# Load glove vectors
glove_ratings = []
for entry in entries:
    df = pd.read_csv(f'data/data_tables/glove_{entry}.csv',header=None).to_numpy().squeeze()
    glove_ratings.append(df)
        
# Generate correlations!
corr_llm = []; agreement = []; fEntries = []; hPcorr=[]; symmetry = []
corr_glove = []
for idx, entry in enumerate(entries):
    if entry in llm_rdms:
        act  = hRatings[entry].mean(1) 
        fEntries.append(entry)
        pred1 = (llm_rdms[entry]).mean(1) - (llm_rdms[entry]).mean(0) # Sum columns minus sum rows
        
        # Are comparisons symmetric?
        symmetry.append(kendalltau(k.sum(0),-k.sum(1))[0])
        c = np.corrcoef(act,pred1)[0,1]
        corr_llm.append(c)

        # Correlate human with Glove
        c = np.corrcoef(act,glove_ratings[idx])[0,1]
        corr_glove.append(c)
        agreement.append(iccs[entry])
        hPcorr.append(pcorrs[entry])
df = pd.DataFrame({'cat_feat':fEntries,'Semantic Projection':corr_glove,'LLM Projection':corr_llm})

# Figure 2 of poster
plt.scatter(agreement,corr_llm,c='tab:orange')
plt.xlabel('Human ICC'); plt.ylabel('LLM Corr with Mean Human')
plt.grid()
plt.xlim([-.1,1]);plt.ylim([-.2,1])
plt.savefig('poster_figs/llm_corr_hICC.png',dpi=300,bbox_inches='tight')

# Figure 3a of poster
plt.scatter(agreement,corr_glove,c='tab:blue')
plt.xlabel('Human ICC'); plt.ylabel('GLOVE Corr with Mean Human')
plt.xlim([-.1,1]);plt.ylim([-.2,1])
plt.grid();
plt.savefig('poster_figs/glove_corr_hICC.png',dpi=300,bbox_inches='tight')

# Figure 3b of poster
plt.scatter(symmetry,corr_llm,c='tab:orange')
plt.xlabel('Kendall tau-b Similarity of UpperTriangular and LowerTriangular');plt.ylabel('LLM Corr with Mean Human')
plt.title('Predicting LLM Performance')
plt.grid(); plt.ylim([-.2,1])
plt.savefig('poster_figs/pred_performance.png',dpi=300,bbox_inches='tight')

# Figure 1b of poster
df2 = df.melt('cat_feat')
plt.clf()
ax1 = sns.barplot(x='variable',y='value',data=df2,alpha=.2,capsize=.1)
ax1 = sns.stripplot(x='variable',y='value',jitter=True,data=df2, ax=ax1)
# Extract x and y coordinates
stripplot_data = [point.get_offsets() for point in ax1.collections]
x_coords = [point[0] for points in stripplot_data for point in points]
y_coords = [point[1] for points in stripplot_data for point in points]
numS = int(len(x_coords)/2)
for i in range(numS):
    x1 = x_coords[i]
    x2 = x_coords[i+numS]
    y1 = y_coords[i]
    y2 = y_coords[i+numS]
    plt.plot([x1,x2],[y1,y2],c='blue',alpha=.15)
plt.xlabel('Method'); plt.ylabel('Correlation with Human Ratings'); plt.title('Method Comparison')
#plt.ylim([-.01,.08])
plt.savefig('poster_figs/method_comparison.png',dpi=300, bbox_inches='tight')
plt.show()

# Figure 1a of poster
cats = sorted(set(cats)); feats = sorted(set(feats))
diffs = np.zeros((len(cats),len(feats)))*np.nan
for i in range(len(cats)):
    for j in range(len(feats)):
        entry = cats[i] +'_'+ feats[j]
        if entry in df.cat_feat.to_list():
            diffs[i,j] = df[df.cat_feat==entry]['LLM Projection'].item() - df[df.cat_feat==entry]['Semantic Projection'].item()     
ax = sns.heatmap(diffs,
            cmap='seismic',
            center=0,
            xticklabels=feats,
            yticklabels=cats,
            square=True)
ax.xaxis.tick_top()
ax.set_facecolor((238/255,238/255,238/255))
plt.xticks(rotation=80)
plt.savefig('poster_figs/comparison.png',dpi=300,bbox_inches='tight')