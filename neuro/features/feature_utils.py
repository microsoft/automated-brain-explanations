from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
import torch

import neuro.features.feature_spaces as feature_spaces
import neuro.features.qa_questions as qa_questions
from neuro.data.npp import zscore
from neuro.features.questions.merge_v3_boostexamples import DICT_MERGE_V3_BOOSTEXAMPLES


def get_features_full(
        args, feature_space, qa_embedding_model, story_names,
        extract_only=False, use_brain_drive=False, use_added_wordrate_feature=False,
        use_added_delays=True):
    '''
    Params
    - -----
    extract_only: bool
        if True, just run feature extraction and return

    Returns
    - ------
    features_delayed: np.ndarray
        n_time_points x(n_delays x n_features)
    '''
    # for ensemble, recursively call this function and average the features
    ensemble_dict = {
        'ensemble1': ['mistralai/Mistral-7B-Instruct-v0.2', 'meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct-fewshot'],
        'ensemble2': ['mistralai/Mistral-7B-Instruct-v0.2', 'meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct-fewshot', 'meta-llama/Meta-Llama-3-70B-Instruct'],
    }
    if qa_embedding_model in ensemble_dict.keys():
        features_delayed_list = []
        for qa_embedding_model in ensemble_dict[qa_embedding_model]:
            features_delayed = get_features_full(
                args, feature_space,
                qa_embedding_model,
                story_names,
                extract_only=extract_only,
                use_brain_drive=use_brain_drive,
                use_added_wordrate_feature=use_added_wordrate_feature,
                use_added_delays=use_added_delays,
            )
            features_delayed_list.append(features_delayed)
        features_delayed_avg = np.mean(features_delayed_list, axis=0)
        # features_delayed_avg = features_delayed_avg / \
        # np.std(features_delayed_avg, axis=0)
        return features_delayed_avg

    # for qa versions, we extract features multiple times and concatenate them
    # this helps with caching
    if 'qa_embedder' in feature_space:
        kwargs_list = qa_questions.get_kwargs_list_for_version_str(
            args.qa_questions_version)
    elif 'qa_agent' in feature_space:
        kwargs_list = [{'qa_questions_version': args.qa_questions_version}]
    else:
        kwargs_list = [{}]

    features_downsampled_list = []
    for kwargs in kwargs_list:
        features_downsampled_dict = feature_spaces.get_features(
            args=args,
            feature_space=feature_space,
            story_names=story_names,
            qa_embedding_model=qa_embedding_model,
            # use_huge=args.use_huge,
            # always use_huge, since it's just faster (for loading feats)
            use_huge=True,
            use_brain_drive=use_brain_drive,
            # use_cache=False,
            **kwargs)
        # n_time_points x n_features
        features_downsampled = trim_and_normalize_features(
            features_downsampled_dict, trim=5, normalize=True
        )
        features_downsampled_list.append(deepcopy(features_downsampled))
    torch.cuda.empty_cache()
    if extract_only:
        return

    features_downsampled = np.hstack(features_downsampled_list)

    # subselect questions
    if args.num_questions_restrict > 0:
        features_downsampled = features_downsampled[:, :args.num_questions_restrict]

    # apply averaging over answers if relevant (and drop some questions)
    if feature_space == 'qa_embedder' and '_merged' in args.qa_questions_version:
        assert args.qa_questions_version == 'v3_boostexamples_merged', f'Only v3_boostexamples_merged is supported but got {args.qa_questions_version}'
        # apply averaging over stim
        questions = np.array(qa_questions.get_questions(
            args.qa_questions_version.replace('_merged', ''), full=True))
        for k, v in DICT_MERGE_V3_BOOSTEXAMPLES.items():
            idx_k = np.where(questions == k)[0][0]
            idxs_v = np.where(pd.Series(questions).isin(v))[0]
            features_downsampled[:, idx_k] = features_downsampled[:, idxs_v].mean(
                axis=1)
        # keep only cols corresponding for non-drop features
        idxs_to_keep = qa_questions._get_merged_keep_indices_v3_boostexamples()
        features_downsampled = features_downsampled[:, idxs_to_keep]

    if use_added_delays:
        features_delayed = make_delayed(features_downsampled,
                                        delays=range(1, args.ndelays+1))
    else:
        features_delayed = features_downsampled

    if use_added_wordrate_feature:
        features_delayed_wordrate = get_features_full(
            args,
            feature_space='wordrate',
            qa_embedding_model=qa_embedding_model,
            story_names=story_names,
            extract_only=extract_only,
            use_brain_drive=use_brain_drive,
            use_added_wordrate_feature=False
        )
        features_delayed = np.hstack(
            [features_delayed, features_delayed_wordrate])

    return features_delayed


def trim_and_normalize_features(downsampled_feat, trim=5, normalize=True):
    """Trim and normalize the downsampled stimulus for train and test stories.

    Params
    ------
    stories
            List of stimuli stories.
    downsampled_feat (dict)
            Downsampled feature vectors for all stories.
    trim
            Trim downsampled stimulus matrix.
    """
    feat = [downsampled_feat[s][5+trim:-trim] for s in downsampled_feat]
    if normalize:
        feat = [zscore(f) for f in feat]
    feat = np.vstack(feat)
    return feat


def make_delayed(stim, delays: List[int], circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays]
    (in samples).

    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.

    Returns
    -------
    np.ndarray
        n_time_points x (n_delays x n_features)

    """
    num_trs, dim_emb = stim.shape
    dstims = []
    for di, d in enumerate(delays):
        dstim = np.zeros((num_trs, dim_emb))
        if d < 0:  # negative delay
            dstim[:d, :] = stim[-d:, :]
            if circpad:
                dstim[d:, :] = stim[:-d, :]
        elif d > 0:
            dstim[d:, :] = stim[:-d, :]
            if circpad:
                dstim[:d, :] = stim[-d:, :]
        else:  # d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)

# def add_delays(stim, ndelays):
#     """Get delayed stimulus matrix.
#     The stimulus matrix is delayed (typically by 2, 4, 6, 8 secs) to estimate the
#     hemodynamic response function with a Finite Impulse Response model.
#     """
#     # List of delays for Finite Impulse Response (FIR) model.
#     delays = range(1, ndelays+1)
#     delstim = make_delayed(stim, delays)
#     return delstim
