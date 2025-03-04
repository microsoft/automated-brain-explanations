import argparse
import logging
import os.path
from collections import defaultdict
from copy import deepcopy
from os.path import join

import imodelsx.cache_save_utils
import joblib
import numpy as np

import neuro.data.story_names
import neuro.features.qa_questions
from neuro.data.response_utils import get_resps_full
from neuro.features import feature_utils

path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# initialize args


def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """
    parser.add_argument("--subject", type=str, default='UTS03',
                        choices=['UTS01', 'UTS02', 'UTS03'],
                        help='top3 concatenates responses for S01-S03, useful for feature selection')
    parser.add_argument("--feature_space", type=str,
                        default='qa_embedder',
                        choices=['qa_embedder', 'eng1000', 'finetune_roberta-base', 'finetune_roberta-base_binary',
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

    parser.add_argument("--qa_embedding_model", type=str,
                        default='mistralai/Mistral-7B-Instruct-v0.2',
                        help='Model to use for QA embedding, if feature_space is qa_embedder',
                        )
    parser.add_argument("--qa_questions_version", type=str,
                        default='v1',
                        help='Which set of QA questions to use, if feature_space is qa_embedder')

    parser.add_argument("--ndelays", type=int, default=4)
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
    parser.add_argument('--use_huge', type=int, default=1,
                        help='''Whether to use huge list of stories
                        (if use_test_setup or not UTS01-03, this will automatically be set to 0)''')
    return parser


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(
        deepcopy(parser_without_computational_args))
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = imodelsx.cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir
    )

    if args.use_cache and already_cached:
        logging.info("cached version exists! Successfully skipping :)\n\n\n")
        exit(0)
    for k in sorted(vars(args)):
        logger.info("\t" + k + " " + str(vars(args)[k]))
    logging.info("\n\n\tsaving to " + save_dir_unique + "\n")

    story_names_train = neuro.data.story_names.get_story_names(
        args.subject, train_or_test='train', use_huge=args.use_huge)
    story_names_test = neuro.data.story_names.get_story_names(
        args.subject, train_or_test='test', use_huge=args.use_huge)

    def _normalize_columns_for_corr(mat):
        # normalize so each col has mean 0 and norm 1
        mat = mat - mat.mean(axis=0)
        mat = mat / (np.linalg.norm(mat, axis=0) + 1e-6)
        return mat

    stim_train_delayed = feature_utils.get_features_full(
        args, args.qa_embedding_model, story_names=story_names_train)
    stim_test_delayed = feature_utils.get_features_full(
        args, args.qa_embedding_model, story_names=story_names_test)
    stim_train_delayed = _normalize_columns_for_corr(stim_train_delayed)
    stim_test_delayed = _normalize_columns_for_corr(stim_test_delayed)
    # duplicate simulus with negative sign and concatenate
    stim_train_delayed = np.concatenate(
        [stim_train_delayed, -stim_train_delayed], axis=1)
    stim_test_delayed = np.concatenate(
        [stim_test_delayed, -stim_test_delayed], axis=1)
    print('num_trs_train',
          stim_train_delayed.shape[0], 'num_trs_test', stim_test_delayed.shape[0])

    args.pc_components = 0
    resp_train, resp_test = get_resps_full(
        args, args.subject, story_names_train, story_names_test)
    resp_train = _normalize_columns_for_corr(resp_train)
    resp_test = _normalize_columns_for_corr(resp_test)

    # stim is (trs x questions) and resp is (trs x voxels)
    # corrs becomes questions x voxels
    corrs_test = stim_test_delayed.T @ resp_test
    corrs_train = stim_train_delayed.T @ resp_train

    # select best question for each voxel
    qs_selected = corrs_train.argmax(axis=0)
    corrs_train_selected = corrs_train[
        qs_selected, np.arange(corrs_train.shape[1])]
    corrs_test_selected = corrs_test[
        qs_selected, np.arange(corrs_test.shape[1])]

    r = defaultdict(list)
    r.update(vars(args))
    r.update({
        'qs_selected': qs_selected,
        'corrs_train_selected': corrs_train_selected,
        'corrs_test_selected': corrs_test_selected,
        'corrs_train_selected_mean': np.nanmean(corrs_train_selected),
        'corrs_test_selected_mean': np.nanmean(corrs_test_selected),
        'corrs_test_mean_baseline': np.nanmean(corrs_test),
    })
    print('corrs_test_selected_mean', r['corrs_test_selected_mean'])

    # save results
    os.makedirs(save_dir_unique, exist_ok=True)
    joblib.dump(
        r, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    logging.info("Succesfully completed :)\n\n")
