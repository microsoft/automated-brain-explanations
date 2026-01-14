import sys
sys.path.append('..')
from neuro import config
from os.path import join
sys.path.append(join(config.REPO_DIR, 'experiments'))

import dvu
import seaborn as sns
import os
import pandas as pd
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
from neuro import config
import imodelsx.process_results
import neuro.features.qa_questions as qa_questions
import joblib
from tqdm import tqdm
import neuro.viz
from neuro import analyze_helper, viz
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuro.config import REGION_IDXS_DIR
import torch
from transformers import set_seed
from neuro.repr_steering import generate_with_steering, get_avg_weight_llama70, select_weight_for_roi



if __name__ == "__main__":
    # Load weight
    # subject = 'S02'
    subject = 'S03'
    max_new_tokens = 100
    rois_dict = {
        'S02': ['RSC', 'OPA', 'PPA', 'IPS', 'pSTS', 'sPMv', 'EBA', 'OFA'],
        'S03': ['RSC', 'OPA', 'PPA', 'IPS', 'sPMv', 'EBA', 'OFA'],
    }
    rois = rois_dict[subject]
    sampling_params = {
        'do_sample': True,
        'top_p': 0.9,
        'temperature': 0.8,
    }
    vec_multipliers = np.logspace(2, 5, 40)
    seeds = range(5)


    # load weights and model
    print('selecting & loading weights...')
    wt, emb_layer = get_avg_weight_llama70(subject)
    # print('wt.shape', wt.shape, 'emb_layer', emb_layer)

    print('loading model...')
    model_name = "meta-llama/Meta-Llama-3-70B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()

    print('generating...')
    for seed in tqdm(seeds):
        for roi in rois:
            if sampling_params['do_sample']:
                save_dir = f'./repr_steering/{subject}/sampling_p={sampling_params["top_p"]}_t={sampling_params["temperature"]}/seed={seed}/{roi}'
            else:
                save_dir = f'./repr_steering/{subject}/greedy/{roi}'

            os.makedirs(save_dir, exist_ok=True)
            for i in tqdm(range(len(vec_multipliers))):
                fname = join(save_dir, f'generated_paragraph_steered_{vec_multipliers[i]:.2f}.txt')
                if os.path.exists(fname):  
                    print(f'Skipping existing file: {fname}')
                    continue
                wt_vec = select_weight_for_roi(wt, subject, roi)
                vec_multiplier = vec_multipliers[i]
                print(f'Generating with vec_multiplier={vec_multiplier:.2f} ({i+1}/{len(vec_multipliers)})')
                paragraph_steered = generate_with_steering(
                        model,
                        tokenizer,
                        emb_layer,
                        wt_vec,
                        prompt = "Here is the first paragraph of a story:",
                        vec_multiplier = vec_multiplier,
                        max_new_tokens = max_new_tokens,
                        sampling_params = sampling_params,
                        seed = seed,
                )
                print(paragraph_steered)
                with open(fname, 'w') as f:
                    f.write(paragraph_steered)