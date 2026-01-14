import os
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from tqdm import tqdm
import pandas as pd
import sys
from typing import List
import numpy as np
import joblib
from pprint import pprint
import imodelsx.util
# import sasc.viz
import pickle as pkl
import json
from copy import deepcopy
from numpy.linalg import norm
import os
import pandas as pd
import neuro.sasc.modules.fmri_module
import imodelsx.util
from functools import partial
import numpy as np
from neuro.config import REGION_IDXS_DIR
subject = 'S02'

rois_dict = joblib.load(join(REGION_IDXS_DIR, f'rois_{subject}.jbl'))

# out_dir = 'repr_steering/greedy'
out_dir = f'repr_steering/{subject}/sampling_p=0.9_t=0.8'
rois = os.listdir(out_dir)
roi = 'RSC'
roi_dir = os.path.join(out_dir, roi)

dfs = []
for seed_dir in os.listdir(out_dir):
    for roi_dir in os.listdir(os.path.join(out_dir, seed_dir)):
        full_dir = os.path.join(out_dir, seed_dir, roi_dir)
        fnames = os.listdir(full_dir)
        
# fnames = os.listdir(roi_dir)
# fnames = sorted(fnames, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        # read the string in each file
        generations = []
        for fname in fnames:
            with open(os.path.join(full_dir, fname), 'r') as f:
                generations.append(f.read())
        # remove the longest common prefix from all generations
        longest_prefix = os.path.commonprefix(generations)
        generations = [gen[len(longest_prefix):] for gen in generations]
        # remove quote at the start
        generations = [gen[1:] if gen.startswith('"') else gen for gen in generations]

        df = pd.DataFrame({'generation': generations})
        df['fname'] = fnames
        df['ngrams_lists'] = df['generation'].apply(partial(imodelsx.util.generate_ngrams_list, ngrams=10, pad_starting_ngrams=False))
        df['seed'] = seed_dir
        df['roi'] = roi_dir
        dfs.append(df)
df = pd.concat(dfs, ignore_index=True)


mod = neuro.sasc.modules.fmri_module.fMRIModule(
    subject=f"UT{subject}",
    # checkpoint="facebook/opt-30b",
    checkpoint="huggyllama/llama-30b",
    init_model=True,
    restrict_weights=False,
)

driving_vals = []
for i in tqdm(range(len(df))):
    ngrams_list = df['ngrams_lists'].iloc[i]
    if len(ngrams_list) == 0:
        driving_vals.append(np.array([np.nan]))
        continue
    voxel_preds = mod(ngrams_list, return_all=True)
    vals_par = voxel_preds[:, np.array(rois_dict[df['roi'].iloc[i]])].mean(axis=1)
    driving_vals.append(vals_par)
df['driving_vals'] = driving_vals
joblib.dump(df, f'repr_steering/{subject}/sampling_p=0.9_t=0.8/roi_driving_vals.jbl')