# from neuro.data.npp import mcorr
# from typing import List
# import json
import logging
import os

# import time
import os.path
import pathlib
from os.path import join

# from multiprocessing.pool import ThreadPool
import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import neuro.config as config
import neuro.features
import neuro.flatmaps_helper

# import random


def load_response(stories, subject):
    """Get the subject"s fMRI response for stories."""
    main_path = pathlib.Path(__file__).parent.parent.resolve()
    subject_dir = join(
        config.FMRI_DIR_BLOB, 'data', f"ds003020/derivative/preprocessed_data/{subject}")
    base = os.path.join(main_path, subject_dir)
    resp = []
    for story in stories:
        resp_path = os.path.join(base, f"{story}.hf5")
        hf = h5py.File(resp_path, "r")
        resp.extend(hf["data"][:])
        hf.close()
    return np.array(resp)


def load_response_huge(stories, subject):
    resps = joblib.load(join(config.FMRI_DIR_BLOB, 'data',
                             'huge_data', f'{subject}_responses.jbl'))
    return np.vstack([resps[story] for story in stories])


def load_response_brain_drive(stories):
    df = joblib.load(join(config.GCT_RESPS_DIR, 'metadata.pkl'))
    df_filt = df.loc[stories]
    resps = []
    file_paths = (config.GCT_RESPS_DIR + '/' +
                  df_filt['session'] + '/' + df_filt['resp_file']).to_list()
    for file_path in file_paths:
        resp = np.load(file_path)
        resps.append(resp)
    return np.vstack(resps)


def load_response_wrapper(args, stories, subject, use_brain_drive=False):
    if len(stories) == 0:
        return []
    if use_brain_drive or all([s.startswith('GenStory') for s in stories]):
        return load_response_brain_drive(stories)
    if args.use_huge and subject in ['UTS01', 'UTS02', 'UTS03']:
        return load_response_huge(stories, subject)
    else:
        return load_response(stories, subject)


def load_pca(subject, pc_components=None):
    if pc_components == 100:
        pca_filename = join(config.RESP_PROCESSING_DIR,
                            subject, 'resps_pca_100.pkl')
        return joblib.load(pca_filename)
    else:
        pca_filename = join(config.RESP_PROCESSING_DIR,
                            subject, 'resps_pca.pkl')
        pca = joblib.load(pca_filename)
        pca.components_ = pca.components_[
            :pc_components]
        return pca


def get_resps_full(
    args, subject, story_names_train, story_names_test
):
    '''
    resp_train: np.ndarray
        n_time_points x n_voxels
    '''
    if subject == 'shared':
        resp_train = np.hstack([
            get_resps_full(args, s, story_names_train, story_names_test)[0]
            for s in ['UTS01', 'UTS02', 'UTS03']])
        return resp_train
    resp_test = load_response_wrapper(
        args, story_names_test, subject)
    resp_train = load_response_wrapper(
        args, story_names_train, subject)

    if args.pc_components <= 0:

        # actually remove non-masked if PCA is not used
        if not args.predict_subset == 'all':
            idxs_mask = neuro.flatmaps_helper.load_custom_rois(
                subject.replace('UT', ''), '_lobes')[args.predict_subset]
            resp_train = resp_train[:, idxs_mask]
            resp_test = resp_test[:, idxs_mask]
            logging.info(
                f'resp_train.shape (no pca) {resp_train.shape} vox0 nunique {len(np.unique(resp_train[0]))}')
        return resp_train, resp_test
    else:
        # if PCA is used, just set the non-masked to 0
        if not args.predict_subset == 'all':
            idxs_mask = neuro.flatmaps_helper.load_custom_rois(
                subject.replace('UT', ''), '_lobes')[args.predict_subset]
            resp_train[:, idxs_mask] = 0
            resp_test[:, idxs_mask] = 0

        logging.info('pc transforming resps...')

        # pca.components_ is (n_components, n_voxels)
        pca = load_pca(subject, args.pc_components)
        scaler_train = None
        scaler_test = None

        resp_train[np.isnan(resp_train)] = np.nanmean(resp_train)
        resp_train = pca.transform(resp_train)
        scaler_train = StandardScaler().fit(resp_train)
        resp_train = scaler_train.transform(resp_train)
        logging.info(f'resp_train.shape (after pca) {resp_train.shape}')

        resp_test[np.isnan(resp_test)] = np.nanmean(resp_test)
        resp_test = pca.transform(resp_test)
        scaler_test = StandardScaler().fit(resp_test)
        resp_test = scaler_test.transform(resp_test)
        return resp_train, resp_test, pca, scaler_train, scaler_test


def get_resp_distilled(args, story_names):
    logging.info('loading distill model...')
    args_distill = pd.Series(joblib.load(
        join(args.distill_model_path, 'results.pkl')))
    for k in ['subject', 'pc_components']:
        assert args_distill[k] == vars(args)[k], f'{k} mismatch'
    assert args_distill.pc_components > 0, 'distill only supported for pc_components > 0'

    model_params = joblib.load(
        join(args.distill_model_path, 'model_params.pkl'))
    features_delayed_distill = neuro.features.get_features_full(
        args_distill, args_distill.qa_embedding_model, story_names)
    preds_distilled = features_delayed_distill @ model_params['weights_pc']
    return preds_distilled


# def get_permuted_corrs(true, pred, blocklen):
#     nblocks = int(true.shape[0] / blocklen)
#     true = true[:blocklen*nblocks]
#     block_index = np.random.choice(range(nblocks), nblocks)
#     index = []
#     for i in block_index:
#         start, end = i*blocklen, (i+1)*blocklen
#         index.extend(range(start, end))
#     pred_perm = pred[index]
#     nvox = true.shape[1]
#     corrs = np.nan_to_num(mcorr(true, pred_perm))
#     return corrs


# def permutation_test(true, pred, blocklen, nperms):
#     start_time = time.time()
#     pool = ThreadPool(processes=10)
#     perm_rsqs = pool.map(
#         lambda perm: get_permuted_corrs(true, pred, blocklen), range(nperms))
#     pool.close()
#     end_time = time.time()
#     print((end_time - start_time) / 60)
#     perm_rsqs = np.array(perm_rsqs).astype(np.float32)
#     real_rsqs = np.nan_to_num(mcorr(true, pred))
#     pvals = (real_rsqs <= perm_rsqs).mean(0)
#     return np.array(pvals), perm_rsqs, real_rsqs
