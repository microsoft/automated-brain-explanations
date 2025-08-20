import os
from os.path import dirname, join, expanduser
import sys
import numpy as np
from imodelsx import submit_utils
from neuro.features.feat_select import get_alphas
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)
# python /home/chansingh/fmri/01_fit_encoding.py
params_shared_dict = {
    # things to average over
    'use_cache': [1],
    'nboots': [5],
    'use_test_setup': [1],
    'use_extract_only': [0],
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/feb13_2025_test_tabpfn'],
    'pc_components': [100],
    'feature_selection_alpha': [get_alphas('qa_embedder')[3]],
    'feature_selection_stability_seeds': [5],


    # next, we can use selected features to fit ridge #######################################
    'ndelays': [1, 2, 4, 8],
    # 'subject': [f'UTS0{k}' for k in range(1, 9)],
    'subject': [f'UTS0{k}' for k in range(1, 4)],
    'num_stories': [1, 5, 10, 15, 20],
    'encoding_model': ['mlp'],  # 'ridge', 'tabpfn'],
}

params_coupled_dict = {
    ('feature_space', 'qa_questions_version', 'qa_embedding_model'): [
        ('qa_embedder', 'v3_boostexamples_merged', 'ensemble2')
    ],
}
# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
script_name = join(repo_dir, 'experiments', '02_fit_encoding.py')
# amlt_kwargs = {
#     # 'amlt_file': join(repo_dir, 'scripts', 'launch_cpu.yaml'),
#     # 'sku': 'E4ads_v5',
#     # 'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
#     'amlt_file': join(repo_dir, 'launch.yaml'),  # change this to run a cpu job
#     'sku': '64G2-MI200-xGMI',
#     'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
# }
amlt_kwargs = {}
submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    # amlt_kwargs=amlt_kwargs,
    # n_cpus=6,
    # n_cpus=2,
    # gpu_ids=[0, 1, 2, 3],
    # actually_run=False,
    repeat_failed_jobs=True,
    shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; uv run python',
)
