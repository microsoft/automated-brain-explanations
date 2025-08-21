import os
import sys
from os.path import dirname, expanduser, join

from imodelsx import submit_utils

from neuro.features.questions.gpt4 import QS_35_STABLE

QS_ALL_COMPUTED = set()
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

params_shared_dict = {
    # things to average over
    'use_extract_only': [0],
    'pc_components': [100],
    # 'ndelays': [1],
    'ndelays': [4],  # [8, 16],
    'nboots': [50],

    # things to change
    'use_test_setup': [0],
    # 'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/aug16_gpt4'],
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/oct17_neurosynth_gemv'],
    # 'subject': ['UTS01', 'UTS02', 'UTS03'],
    # 'subject': ['UTS04', 'UTS05', 'UTS06', 'UTS07', 'UTS08'],
    # 'subject': ['UTS01', 'UTS02', 'UTS03', 'UTS04', 'UTS05', 'UTS06', 'UTS07', 'UTS08'],
    'subject': ['UTS03'],
    # 'use_added_wordrate_feature': [0, 1],
    # 'subject': ['UTS02'],
    'use_added_wordrate_feature': [0, 1],
}

params_coupled_dict = {
    ('qa_questions_version', 'qa_embedding_model',
     'use_random_subset_features', 'seed'):

    # single question (could also pass the other used questions here)
    [
        # (repr(QUESTIONS_GPT4_COMPUTED_FULL[i]), 'gpt4', None, None)
        # for i in range(len(QUESTIONS_GPT4_COMPUTED_FULL))
        (repr(QS_35_STABLE[i]), 'gpt4', None, None)
        for i in range(len(QS_35_STABLE))
    ]
    +
    # shapley features
    [
        # ('qs_35', 'gpt4', 1, seed)
        # for seed in range(50)
    ]
    +
    # full
    [
        # ('qs_35', 'gpt4', None, None)
    ]


}
QS_35_STABLE
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
    # unique_seeds='seed',
    # amlt_kwargs=amlt_kwargs,
    n_cpus=8,
    # actually_run=False,
    # repeat_failed_jobs=True,
    shuffle=False,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; .venv/bin/python',
)
