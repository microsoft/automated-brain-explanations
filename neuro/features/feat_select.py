import logging
import os
import os.path
from os.path import dirname, join

import joblib
import numpy as np
from sklearn.linear_model import MultiTaskElasticNet

import neuro.config as config
from neuro.data import response_utils


def get_alphas(feature_space: str):
    if feature_space == 'qa_embedder':
        return sorted(np.logspace(0, -3, 20).tolist() + [0.40] + [0.28], reverse=True)[2: 11]
    elif feature_space == 'eng1000':
        return sorted(np.logspace(0, -3, 20).tolist() + [0.19], reverse=True)[3: 12]
    else:
        return sorted(np.logspace(0, -3, 20).tolist(), reverse=True)


def get_selected_coef(args, feature_selection_stability_seeds: int, seed: int,
                      stim_train_delayed, story_names_train, story_names_test):
    '''
    Returns
    -------
    coef_nonzero: (n_features,)
    '''
    if feature_selection_stability_seeds > 0:
        coef_nonzero_arr = np.vstack([
            get_selected_coef(args, -1, seed,
                              stim_train_delayed, story_names_train, story_names_test)
            for seed in range(feature_selection_stability_seeds)
        ]).astype(int)
        coef_nonzero = coef_nonzero_arr.min(axis=0).astype(bool)
        print('nonzero stable', coef_nonzero.sum())
        return coef_nonzero


    suffix_pc = ''
    if args.feature_selection_pc_components > 0:
        suffix_pc = f'_pc={args.feature_selection_pc_components}'
    suffix_iter = ''
    if args.feature_selection_max_iter != 5000:
        suffix_iter = f'_iter={args.feature_selection_max_iter}'

    cache_dir = join(config.FMRI_DIR_BLOB, 'qa', f'sparse_feats_shared{suffix_pc}{suffix_iter}')

    cache_file = join(
        cache_dir,
        args.predict_subset,
        f"{args.feature_space.replace('/', '-')}___qa_questions_version={args.qa_questions_version}___{args.qa_embedding_model.replace('/', '-')}",
        f'seed={seed}___feature_selection_frac={args.feature_selection_frac:.2f}___feature_selection_alpha={args.feature_selection_alpha:.2e}.joblib'
    )
    os.makedirs(dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        coef_enet = joblib.load(cache_file)
        print('\tLoaded from cache:', cache_file)
    else:
        print('\tcouldn\'t find cache file:', cache_file)
        print('\tloading shared responses to fit multitaskelasticnet...')
        # remove delays from stim
        stim_train = stim_train_delayed[:,
                                        :stim_train_delayed.shape[1] // args.ndelays]
        
        # get special resps by concatenating across subjects
        resp_train_shared = response_utils.get_resps_full(
            args, 'shared', story_names_train, story_names_test)
        

        if args.feature_selection_pc_components > 0:
            # take the first args.feature_selection_pc_components of each subject's responses
            num_pcs_per_subj = resp_train_shared.shape[1] // 3
            resp_train_shared = np.concatenate([
                resp_train_shared[:, 
                                  (i * num_pcs_per_subj):
                                  (i * num_pcs_per_subj) + args.feature_selection_pc_components]
                for i in range(3)
            ], axis=1)
            print('resp_train_shared shape after selecting feature_selection_pc_components:', resp_train_shared.shape)

        # randomly subsample based on seed
        if 0 < args.feature_selection_frac < 1:
            rng = np.random.RandomState(seed)
            idxs_subsample = rng.choice(
                np.arange(stim_train.shape[0]), int(args.feature_selection_frac * stim_train.shape[0]), replace=False)
            stim_train = stim_train[idxs_subsample]
            resp_train_shared = resp_train_shared[idxs_subsample]

        print('\tfitting multitaskelasticnet...')
        m = MultiTaskElasticNet(
            alpha=args.feature_selection_alpha,
            l1_ratio=0.9,
            max_iter=args.feature_selection_max_iter,
            random_state=seed,
        ).fit(stim_train, resp_train_shared)
        coef_enet = m.coef_
        joblib.dump(coef_enet, cache_file)
        logging.info(
            f"Succesfully completed feature selection (frac nonzero: {np.any(np.abs(coef_enet) > 0, axis=0).mean()}). Saved to {cache_file}")
    if args.subject == 'shared':
        print(
            f"Stopping because {args.subject=}")
        exit(0)

    # coef_enet: (n_targets, n_features)
    coef_nonzero = np.any(np.abs(coef_enet) > 0, axis=0).squeeze()
    print('nonzero', coef_nonzero.sum(), cache_file)

    return coef_nonzero


def select_features(args, r, stim_train_delayed, stim_test_delayed, story_names_train, story_names_test):
    coef_nonzero = get_selected_coef(
        args, args.feature_selection_stability_seeds, args.seed,
        stim_train_delayed, story_names_train, story_names_test)

    # pick the coefs
    # coef_enet = coefs_enet[:, :, args.feature_selection_alpha_index]

    # add a 1 to the end of coef_nonzero to account for the wordrate feature
    if args.use_added_wordrate_feature:
        coef_nonzero = np.concatenate([coef_nonzero, [1]]).astype(bool)

    # r['alpha'] = alphas_enet[args.feature_selection_alpha_index]
    # r['weights_enet'] = coef_enet
    r['weight_enet_mask'] = coef_nonzero
    r['weight_enet_mask_num_nonzero'] = coef_nonzero.sum()

    # mask stim_delayed based on nonzero coefs (need to repeat by args.ndelays)
    coef_nonzero_rep = np.tile(
        coef_nonzero.flatten(), args.ndelays).astype(bool)
    stim_train_delayed = stim_train_delayed[:, coef_nonzero_rep]
    stim_test_delayed = stim_test_delayed[:, coef_nonzero_rep]

    return r, stim_train_delayed, stim_test_delayed


def select_random_feature_subset(args, r, stim_train_delayed, stim_test_delayed):
    rng = np.random.default_rng(args.seed)
    r['weight_random_mask'] = np.tile(
        rng.choice([0, 1], stim_train_delayed.shape[1] // args.ndelays),
        args.ndelays
    ).astype(bool)
    stim_train_delayed = stim_train_delayed[:, r['weight_random_mask']]
    stim_test_delayed = stim_test_delayed[:, r['weight_random_mask']]
    return r, stim_train_delayed, stim_test_delayed


def select_single_feature(args, r, stim_train_delayed, stim_test_delayed):
    idx_select = np.zeros(stim_train_delayed.shape[1] // args.ndelays)
    idx_select[args.single_question_idx] = 1
    r['weight_random_mask'] = np.tile(
        idx_select,
        args.ndelays
    ).astype(bool)
    stim_train_delayed = stim_train_delayed[:, r['weight_random_mask']]
    stim_test_delayed = stim_test_delayed[:, r['weight_random_mask']]
    return r, stim_train_delayed, stim_test_delayed
