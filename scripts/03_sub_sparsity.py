
import os
import sys
from os.path import dirname, expanduser, join

from imodelsx import submit_utils

from neuro.features.feat_select import get_alphas

path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

params_shared_dict = {
    # things to average over
    'use_cache': [1],
    'nboots': [5],
    'use_test_setup': [0],
    'use_extract_only': [0],
    'use_huge': [1],
    # 'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/may7'],
    # 'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/may27'],
    # this dir contains results for non-full cortex
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/mar20_2025'],
    'pc_components': [100],

    # first run to perform and save feature selection #######################################
    # run with a single subject, which will do feature selection across UTS01-UTS03 automatically when feature_selection_alpha > 0
    'subject': ['shared'],

    # full
    # 'predict_subset': ['all'],
    # 'seed': range(5),
    # 'feature_selection_frac': [0.5],
    # 'feature_selection_max_iter': [5000],


    # subsets for agentic single
    'predict_subset': ['all', 'prefrontal', 'occipital', 'sensorimotor', 'cingulate', 'insula', 'parietal', 'temporal'],
    'seed': range(1),
    'feature_selection_frac': [1],
    'feature_selection_max_iter': [100],
    'feature_selection_pc_components': [10],

    # subsets for agentic stab selection (all was already run originally, but with higher feature_selection_frac)
    # 'predict_subset': ['prefrontal', 'occipital', 'sensorimotor', 'cingulate', 'insula', 'parietal', 'temporal'],
    # 'seed': range(5),
    # 'feature_selection_frac': [0.5],
    # 'feature_selection_max_iter': [1000],
    # 'feature_selection_pc_components': [10],




    # second, we can use selected features to fit ridge #######################################
    # 'ndelays': [4, 8],
    # 'ndelays': [8],
    # 'subject': ['UTS01', 'UTS02', 'UTS03'],
    # 'feature_selection_stability_seeds': [5],

    # third, we can also use selected features with subsampling #######################################
    # 'ndelays': [8],
    # 'subject': [f'UTS0{k}' for k in range(1, 9)],
    # 'feature_selection_stability_seeds': [5],
    # 'num_stories': [-1, 5, 10, 20],

}

params_coupled_dict = {
    ('feature_space', 'qa_questions_version', 'qa_embedding_model', 'feature_selection_alpha'):
    [
        # ('qa_embedder', 'v3_boostexamples_merged', 'ensemble2', alpha)
        # note, would run all of them when not picking subset
        # for alpha in get_alphas('qa_embedder')[1:-3]
    ]
    +
    [
        ('eng1000', None, None, alpha)
        for alpha in get_alphas('eng1000')
    ]
    +
    [
        # agent setting uses just llama instead of ensemble and v3 instead of v3_boostexamples_merged
        ('qa_embedder', 'v3', 'meta-llama/Meta-Llama-3-8B-Instruct', alpha)
        # note, would run all of them when not picking subset
        for alpha in get_alphas('qa_embedder')[1:-3]
    ],
}
# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
script_name = join(repo_dir, 'experiments', '02_fit_encoding.py')
amlt_kwargs = {
    'amlt_file': join(repo_dir, 'scripts', 'launch.yaml'),
    'sku': '8C7',
    # 'sku': '8C15',
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
    'target___name': 'msrresrchvc',
}
    
submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    amlt_kwargs=amlt_kwargs,
    # n_cpus=4,
    # n_cpus=1,
    # gpu_ids=[0, 1, 2, 3],
    # actually_run=False,
    repeat_failed_jobs=True,
    # shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; .venv/bin/python',
)
