import sys
sys.path.append('..')
from neuro import config
from os.path import join

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

def get_avg_weight_llama70(subject):
    data = joblib.load(join(config.RESULTS_DIR_LOCAL, 'results_best_ensemble.pkl'))
    rr, cols_varied, mets = data['r'], data['cols_varied'], data['mets']
    metric_sort = 'corrs_tune_pc_weighted_mean'
    r = rr[rr.subject.isin([subject])]
    r = r[r.feature_space == 'meta-llama/Meta-Llama-3-70B']
    r = r[r.num_stories == -1]
    metric_sort = 'corrs_tune_pc_weighted_mean'
    args = r.sort_values(metric_sort, ascending=False).iloc[0]
    model_params = joblib.load(
        join(args.save_dir_unique, 'model_params.pkl'))
    print(args.feature_space, args.pc_components, args.ndelays)
    wt = model_params['weights'] # wt is (n_delays x n_features) x n_voxels
    n_features = wt.shape[0] / args.ndelays
    wt = wt.reshape(args.ndelays, int(n_features), -1)
    wt = wt.mean(axis=0) # average over delays
    # preds_test = stim_test_delayed @ wt
    # # stim_test_delayed: np.ndarray
    #         n_time_points x(n_delays x n_features)
    return wt, args.embedding_layer

def select_weight_for_roi(wt, subject, roi_name):
    # select weight for voxels
    # if '_only' in 
    rois_dict = joblib.load(join(REGION_IDXS_DIR, f'rois_{subject}.jbl'))
    wt_roi = wt[:, rois_dict[roi_name]] # n_features x n_voxels_in_roi
    wt_roi_mean = wt_roi.mean(axis=1) # n_features
    return wt_roi_mean

def _add_vector_to_last_token_hook(vec: torch.Tensor, pbar):
    """
    Returns a forward hook that adds `vec` to hidden_states[:, -1, :]
    Works whether module output is a Tensor or a tuple whose first item is Tensor.
    """
    def hook(module, inputs, output):
        # `output` for LlamaDecoderLayer is typically a tuple:
        # (hidden_states, present_key_value, ...) depending on config/use_cache
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # hidden: [batch, seq_len, hidden_size]
        # Add only to the last token position.
        hidden[:, -1, :] += vec
        # print(hidden[:, -1, :20])
        # print('hook!')
        pbar.update(1)
        # Return in the same structure HF expects
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        else:
            return hidden

    return hook

def generate_with_steering(
        model,
        tokenizer,
        emb_layer,
        wt_vec,
        prompt = "Here is the first paragraph of a story:",
        vec_multiplier = 0,
        max_new_tokens = 10,
        sampling_params = {},
        seed = 42,
):
    
    device = next(model.model.layers[emb_layer].parameters()).device
    dtype = next(model.parameters()).dtype
    intervention_vec = torch.tensor(
        wt_vec, device=device, dtype=dtype
    ) * vec_multiplier

    inputs = tokenizer(prompt, return_tensors="pt")
    embed_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(embed_device) for k, v in inputs.items()}

    pbar = tqdm(desc="Generating", unit="tok", total=max_new_tokens)
    layer = model.model.layers[emb_layer]
    handle = layer.register_forward_hook(_add_vector_to_last_token_hook(intervention_vec.view(1, -1), pbar))


    try:
        set_seed(seed)
        out = model.generate(
            **inputs,
            **sampling_params,
            # do_sample=True,
            # temperature=0.8,
            # top_p=0.95,
            max_new_tokens=max_new_tokens,
            use_cache=True,  # hook will fire on prefill + each decode step
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        handle.remove()
        pbar.close()


    return tokenizer.decode(out[0], skip_special_tokens=True)