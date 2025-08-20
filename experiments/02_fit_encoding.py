import argparse
import logging
import os
import os.path
import random
from os.path import expanduser
import time
from collections import defaultdict
from copy import deepcopy
from os.path import join

import imodelsx.cache_save_utils
import joblib
import numpy as np
import torch
from tqdm import tqdm

import neuro.agent
import neuro.data.story_names as story_names
from neuro.data import response_utils
from neuro.encoding.eval import (
    add_summary_stats,
    evaluate_pc_model_on_each_voxel,
    explained_var_over_targets_and_delays,
    get_ngrams_top_errors_df,
    nancorr,
)
from neuro.encoding.fit import fit_regression
from neuro.features import feat_select, feature_utils
from neuro.features.questions.gpt4 import QS_HYPOTHESES_COMPUTED

# get path to current file
path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_to_file = os.path.dirname(os.path.abspath(__file__))

def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """
    # data arguments
    parser.add_argument("--subject", type=str, default='UTS03',
                        choices=[f'UTS0{k}' for k in range(1, 9)] + ['shared'],
                        help='shared concatenates responses for S01-S03 (and only load shared stories), useful for feature selection but will not actually run regression')
    parser.add_argument('--pc_components', type=int, default=-1,
                        help='''number of principal components to use for reducing output (-1 doesnt use PCA at all).
                        Note, use_test_setup alters this to 100.''')
    parser.add_argument('--use_huge', type=int, default=1,
                        help='''Whether to use huge list of stories
                        (if use_test_setup or not UTS01-03, this will automatically be set to 0)
                        ''')
    parser.add_argument('--num_stories', type=int, default=-1,
                        help='''number of stories to use for training (-1 for all).
                        Stories are selected from huge list unless use_test_setup''')
    parser.add_argument("--distill_model_path", type=str,
                        default=None,
                        # default='/home/chansingh/mntv1/deep-fMRI/encoding/results_apr7/68936a10a548e2b4ce895d14047ac49e7a56c3217e50365134f78f990036c5f7',
                        help='Path to saved pickles for distillation. Instead of fitting responses, fit the predictions of this model.')
    parser.add_argument("--use_eval_brain_drive", type=int, default=0,
                        help='Whether to evaluate fitted model on brain drive stories')
    parser.add_argument('--predict_subset', type=str, default='all', choices=[
                        'all', 'prefrontal', 'parietal', 'temporal', 'occipital', 'sensorimotor', 'cingulate', 'insula'])

    # encoding
    parser.add_argument("--feature_space", type=str,
                        default='qa_embedder',
                        choices=['qa_embedder', 'qa_agent', 'eng1000', 'wordrate', 'finetune_roberta-base', 'finetune_roberta-base_binary',
                                 'bert-base-uncased', 'distilbert-base-uncased',  'roberta-base',
                                 'meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3-70B'],
                        help='''Passing a standard HF model name will compute embeddings from that model.
                        Models starting with "finetune_" load custom models
                        qa_embedder computes qa embeddings with the checkpoint in args.qa_embedding_model
                        ''')
    parser.add_argument('--embedding_layer', type=int, default=-1,
                        help='''If args.feature_space is a HF model, which layer to use for embeddings (-1 for default layer)''')
    parser.add_argument('--input_chunking_type', type=str, default='ngram',
                        choices=['ngram', 'tr', 'sec'],
                        help='''Type of chunking to use for input features.
                        ngram chunks are number of words
                        tr chunks by TRs (and does not compute features per-word, so is faster but less accurate)
                        sec chunks by seconds leading up to each word''')
    parser.add_argument('--input_chunking_size', type=int, default=10,
                        help='Number of input chunks (corresponding to input_chunking_type)')
    parser.add_argument("--feature_selection_alpha", type=float,
                        default=-1,
                        help='Alpha to use when running feature selection (if >= 0). Alpha to use for feature selection.')
    parser.add_argument("--feature_selection_frac", type=float,
                        default=0.5,
                        help='''Randomly bootstraps data to this fraction of examples.
                        Applies if feature_selection_alpha >= 0.''')
    parser.add_argument("--feature_selection_max_iter", type=int,
                        default=5000,
                        help='''Number of iterations to use for elasticnet feateure selection
                        Applies if feature_selection_alpha >= 0.
                        ''')
    parser.add_argument("--feature_selection_stability_seeds", type=int,
                        default=-1,
                        help='''Number of seeds to use for stability selection (only keeps a feature if it was selected in all seeds).
                        Applies if feature_selection_alpha >= 0.
                        Note: needs to run feature selection with this many different seeds (slow, good to run in parallel before calling this)
                        ''')
    parser.add_argument("--feature_selection_pc_components", type=int,
                        default=-1,
                        help='''Number of principal components of the response to use for feature selection, when using feature selection.
                        If -1, uses pc_components arg above
                        ''')
    parser.add_argument("--use_added_wordrate_feature", type=int, default=0,
                        choices=[0, 1], help='Whether to add the wordrate feature')

    # qa features
    parser.add_argument("--qa_embedding_model", type=str,
                        default='mistralai/Mistral-7B-Instruct-v0.2', # commonly use 'meta-llama/Meta-Llama-3-8B-Instruct'
                        # default='gpt4',
                        help='Model to use for QA embedding, if feature_space is qa_embedder',
                        )
    parser.add_argument("--qa_questions_version", type=str,
                        default='v1',
                        choices=['v1', 'v2', 'v3', 'v3_boostexamples',
                                 'v4_boostexamples', 'v4', 'v5', 'v3_boostexamples_merged'] +
                        ['v1neurosynth', 'qs_35', 'hypothesae'] + QS_HYPOTHESES_COMPUTED,
                        help='''Only when feature_space is qa_embedder
                        If one of the sets (e.g. v1, v2) - which set of QA questions to use.
                            v1neurosynth: will use the set of GPT-4 hypotheses that were not computed with GPT-4
                            qs_35: the set of 35 stable questions
                            hypothesae: loads from the questions generated by running hypothesae (384 questions) from 06_fit_hypothesaes_and_extract_qs.ipynb
                        If single question, will use only that question (only supported with GPT-4).
                        ''')
    parser.add_argument("--use_random_subset_features", type=int, default=0,
                        help='Whether to use a random subset of features')
    parser.add_argument("--single_question_idx", type=int, default=-1,
                        help='If passed, only use this question index for QA features')
    parser.add_argument("--num_questions_restrict", type=int,
                        default=-1,
                        help='''Restricts the number of features to use (takes them in sorted order).''')

    # agent features
    parser.add_argument("--agent_checkpoint", type=str, default='gpt-4o', choices=['gpt-4o', 'o4-mini', 'gpt-4.1', 'gpt-5', 'gpt-5-chat'],
                        help='Checkpoint to use for agent (if feature_space is qa_agent)')
    parser.add_argument("--num_agent_epochs", type=int, default=5,
                        help='Number of epochs to train the agent for (if feature_space is qa_agent)')
    parser.add_argument('--topk_agent_errors', type=int, default=100,
                        help='Number of top errors to use for agent revisions (if feature_space is qa_agent')
    parser.add_argument('--topk_agent_correlated_questions', type=int, default=15,
                        help='Number of top correlated questions to use for agent revisions (if feature_space is qa_agent)')
    parser.add_argument('--num_questions_brainstorm_target', type=int, default=25,
                        help='Number of questions to target for initial brainstorming (if feature_space is qa_agent)')
    parser.add_argument('--num_questions_increment_target', type=int, default=25,
                        help='Number of questions to target for incrementing the list when revising (if feature_space is qa_agent)')

    # linear modeling
    parser.add_argument("--encoding_model", type=str,
                        default='ridge',
                        # default='randomforest'
                        )
    parser.add_argument("--ndelays", type=int, default=4)
    parser.add_argument("--nboots", type=int, default=5)
    parser.add_argument("--chunklen", type=int, default=40,
                        help='try to get nchunks * chunklen to ~20% of training data')
    parser.add_argument("--nchunks", type=int, default=125)
    parser.add_argument("--singcutoff", type=float, default=1e-10)
    parser.add_argument("-single_alpha", action="store_true")
    # parser.add_argument("--trim", type=int, default=5) # always end up using 5
    # parser.add_argument("--l1_ratio", type=float,
    # default=0.5, help='l1 ratio for elasticnet (ignored if encoding_model is not elasticnet)')
    # parser.add_argument("--min_alpha", type=float,
    # default=-1, help='min alpha, useful for forcing sparse coefs in elasticnet. Note: if too large, we arent really doing CV at all.')
    # parser.add_argument('--pc_components_input', type=int, default=-1,
    # help='number of principal components to use to transform features (-1 doesnt use PCA at all)')
    # parser.add_argument("--mlp_dim_hidden", type=int,
    # help="hidden dim for MLP", default=512)

    # basic params
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_test_setup', type=int, default=1,
                        help='For fast testing - train/test on a couple stories with few nboots. Bypasses overall caching.')
    return parser


def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldnt use when checking cache)"""
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join(path_to_repo, 'results'))
    parser.add_argument(
        "--use_cache",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to check for cache",
    )
    parser.add_argument(
        "--use_save_features",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to save the constructed features",
    )
    parser.add_argument(
        "--use_extract_only",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to jointly extract train/test (speeds things up if running over many seeds)",
    )
    parser.add_argument(
        '--seed_stories',
        type=int,
        default=1,
        help='seed for order that stories are processed in',
    )
    parser.add_argument(
        '--qa_batch_size',
        type=int,
        default=16,
        help='batch size for QA embedding extraction',
    )
    return parser


def _check_args(args, parser):
    if args.subject not in ['UTS01', 'UTS02', 'UTS03', 'shared'] and args.use_huge:
        args.use_huge = 0
        # warnings.warn(
        # f'Not using huge list of stories for subject {args.subject}')

    if args.embedding_layer >= 0:
        assert args.feature_space not in ['qa_agent', 'qa_embedder', 'eng1000', 'finetune_roberta-base',
                                          'finetune_roberta-base_binary'], f'embedding_layer only used for HF models but {args.feature_space} passed'
        assert args.qa_questions_version == parser.get_default('qa_questions_version'), 'embedding_layer only used with v1'
        assert args.qa_embedding_model == parser.get_default('qa_embedding_model'), 'embedding_layer only used with default (mistral) qa_embedding_model'

    return args

def run_pipeline(args, r):

    # get data
    story_names_train, story_names_test = story_names.get_story_names_from_args(args)
    if args.use_extract_only:
        # extract braindrive
        # if args.use_eval_brain_drive:
        #     story_names_brain_drive = story_names.get_story_names(
        #         use_brain_drive=True, all=True)
        #     stim_brain_drive_delayed = feature_utils.get_features_full(
        #         args, args.feature_space,  args.qa_embedding_model, story_names_brain_drive,
        #         use_brain_drive=True, use_added_wordrate_feature=args.use_added_wordrate_feature)

        all_stories = story_names.get_story_names(all=True)
        random.shuffle(all_stories)
        feature_utils.get_features_full(
            args, args.feature_space, args.qa_embedding_model,
            all_stories, extract_only=True, use_added_wordrate_feature=args.use_added_wordrate_feature)

    print('loading features...')
    stim_test_delayed = feature_utils.get_features_full(
        args, args.feature_space, args.qa_embedding_model, story_names_test, use_added_wordrate_feature=args.use_added_wordrate_feature)
    stim_train_delayed = feature_utils.get_features_full(
        args, args.feature_space, args.qa_embedding_model, story_names_train, use_added_wordrate_feature=args.use_added_wordrate_feature)
    print('feature shapes before selection',
          stim_train_delayed.shape, stim_test_delayed.shape)

    # select features
    if args.feature_selection_alpha >= 0:
        print('selecting features...')
        r, stim_train_delayed, stim_test_delayed = feat_select.select_features(
            args, r, stim_train_delayed, stim_test_delayed,
            story_names_train, story_names_test)
        if stim_train_delayed.shape[1] == 0:
            logging.error(f'No features exist after selection!!!! {stim_train_delayed.shape=} {stim_test_delayed.shape=}')
            exit(0)
    if args.use_random_subset_features:
        r, stim_train_delayed, stim_test_delayed = feat_select.select_random_feature_subset(
            args, r, stim_train_delayed, stim_test_delayed)
    elif args.single_question_idx >= 0:
        r, stim_train_delayed, stim_test_delayed = feat_select.select_single_feature(
            args, r, stim_train_delayed, stim_test_delayed)
    print('feature shapes after selection',
          stim_train_delayed.shape, stim_test_delayed.shape)

    print('loading resps...')
    if args.pc_components <= 0:
        resp_train, resp_test = response_utils.get_resps_full(
            args, args.subject, story_names_train, story_names_test)
    else:
        resp_train, resp_test, pca, scaler_train, scaler_test = response_utils.get_resps_full(
            args, args.subject, story_names_train, story_names_test)

    # overwrite resp_train with distill model predictions
    if args.distill_model_path is not None:
        resp_train = response_utils.get_resp_distilled(
            args, story_names_train)

    # fit model
    print('fitting regression...')
    r, model_params_to_save = fit_regression(
        args, r, stim_train_delayed, resp_train, stim_test_delayed, resp_test)

    # evaluate per voxel
    if args.pc_components > 0:
        resp_test_voxel = response_utils.load_response_wrapper(
            args, story_names_test, args.subject)
        r['corrs_test'] = evaluate_pc_model_on_each_voxel(
            args, stim_test_delayed, resp_test_voxel,
            model_params_to_save, pca, scaler_test, args.predict_subset)
        r['corrs_tune_pc_mean_weighted_by_explained_var'] = np.sum(
            pca.explained_variance_ratio_[:args.pc_components] * r['corrs_tune_pc'])
        model_params_to_save['scaler_test'] = scaler_test
        model_params_to_save['scaler_train'] = scaler_train

        if args.feature_space == 'qa_agent':
            resp_train_voxel = response_utils.load_response_wrapper(
                args, story_names_train, args.subject)
            r['feature_importances_var_explained'] = explained_var_over_targets_and_delays(
                
                args, stim_train_delayed, resp_train_voxel, model_params_to_save)
            r['feature_importances_var_explained_norm'] = r['feature_importances_var_explained'] / \
                np.sum(np.abs(r['feature_importances_var_explained']))
            stim_train_mini = stim_train_delayed[:10000, :len(args.qa_questions_version)]
            r['feature_correlations'] = np.corrcoef(stim_train_mini, rowvar=False)

            r['error_ngrams_df'] = get_ngrams_top_errors_df(story_names_train, stim_train_delayed, resp_train_voxel, model_params_to_save)

        # compute weighted corrs_tune_pc
        # explained_var_weight = pca.explained_variance_[:args.pc_components]
        # explained_var_weight = explained_var_weight / \
        #     explained_var_weight.sum() * len(explained_var_weight)
        # r['corrs_tune_pc_weighted_mean'] = np.mean(
        #     explained_var_weight * r['corrs_tune_pc'])

    
    # if args.use_eval_brain_drive and args.subject in story_names.TEST_BRAINDRIVE.keys():
    #     story_names_brain_drive = story_names.get_story_names(
    #         subject=args.subject, use_brain_drive=True)
    #     stim_brain_drive_delayed = feature_utils.get_features_full(
    #         args, args.feature_space, args.qa_embedding_model, story_names_brain_drive, use_brain_drive=True, use_added_wordrate_feature=args.use_added_wordrate_feature)
    #     resp_brain_drive = response_utils.load_response_wrapper(
    #         args, story_names_brain_drive, args.subject, use_brain_drive=True)
    #     r['corrs_brain_drive'] = evaluate_pc_model_on_each_voxel(
    #         args, stim_brain_drive_delayed, resp_brain_drive,
    #         model_params_to_save, pca, scaler_test)
    

    # add extra stats
    r = add_summary_stats(r, verbose=True)
    return r, model_params_to_save


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(
        deepcopy(parser_without_computational_args))
    args = parser.parse_args()
    args = _check_args(args, parser)

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = imodelsx.cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir
    )
    if args.use_cache and already_cached and not args.use_test_setup:
        if args.feature_space == 'qa_agent':
            # try loading and continuining
            r = joblib.load(join(save_dir_unique, "results.pkl"))
            n_epochs_run = len(r)
        else:
            print("cached version exists! Successfully skipping :)\n\n\n")
            exit(0)
    for k in sorted(vars(args)):
        print("\t" + k + " " + str(vars(args)[k]))
    logging.info("\n\n\tsaving to " + save_dir_unique + "\n")

    # set seed
    t0 = time.time()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # intialize object for saving
    if 'r' in locals() or 'r' in globals():
        # if r already exists, we are continuing an agent run
        pass
    else:
        r = defaultdict(list)
        r.update(vars(args))
        r["git_commit_id"] = imodelsx.cache_save_utils.get_git_commit_id()
        r["save_dir_unique"] = save_dir_unique

    if args.feature_space != 'qa_agent':
        r, model_params_to_save = run_pipeline(args, r)
        os.makedirs(save_dir_unique, exist_ok=True)
        joblib.dump(r, join(save_dir_unique, "results.pkl"))
        if args.encoding_model == 'ridge':
            joblib.dump(model_params_to_save, join(
                save_dir_unique, "model_params.pkl"))
        print(
            f"Succesfully completed in {(time.time() - t0)/60:0.1f} minutes, saved to {save_dir_unique}")

    elif args.feature_space == 'qa_agent':
        lm = imodelsx.llm.get_llm(
            args.agent_checkpoint,
            CACHE_DIR=neuro.config.CACHE_OPENAI_AGENT_DIR,
        )

        # first time running agent, initialize questions
        if isinstance(r, defaultdict):
            r_epoch = deepcopy(r)
            r_epoch['epoch'] = 0
            r = []
            questions_list_init = neuro.agent.brainstorm_init_questions(lm, args)
            args.qa_questions_version = questions_list_init
            r_epoch['questions_list'] = questions_list_init
        elif isinstance(r, list):
            r_epoch = r[-1]
            args.qa_questions_version = r_epoch['questions_list']

        while r_epoch['epoch'] < args.num_agent_epochs:

            # run with questions
            logging.info(f"Running agent epoch {r_epoch['epoch'] + 1}/{args.num_agent_epochs}")                        
            logging.info('questions_list '  + '\n'.join(r_epoch['questions_list']))
            r_epoch, model_params_to_save = run_pipeline(args, r_epoch)

            # save results
            os.makedirs(save_dir_unique, exist_ok=True)
            r.append(deepcopy(r_epoch))
            joblib.dump(r, join(save_dir_unique, "results.pkl"))
            if args.encoding_model == 'ridge':
                joblib.dump(model_params_to_save, join(
                    save_dir_unique, "model_params.pkl"))
            print(
                f"Succesfully completed epoch {r_epoch['epoch'] + 1}/{args.num_agent_epochs} in {(time.time() - t0)/60:0.1f} minutes, saved to {save_dir_unique}")

            # update questions for the next epoch
            questions_list_revised = neuro.agent.update_questions(lm, args, args.qa_questions_version, r_epoch)
            args.qa_questions_version = questions_list_revised
            r_epoch['questions_list'] = questions_list_revised
            r_epoch['epoch'] += 1

