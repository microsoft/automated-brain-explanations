from neuro.features.feat_select import get_alphas
import os
from os.path import dirname, join, expanduser
import sys
from imodelsx import submit_utils
from neuro.features.questions.gpt4 import QS_HYPOTHESES, QS_HYPOTHESES_COMPUTED
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

params_shared_dict = {
    # things to average over
    'use_extract_only': [0],
    'pc_components': [100],
    'ndelays': [8],

    # things to change
    'use_test_setup': [0],
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/aug16_eng1000'],
    'subject': ['UTS01', 'UTS02', 'UTS03', 'UTS04', 'UTS05', 'UTS06', 'UTS07', 'UTS08'],
    'feature_space': ['eng1000'],
}

params_coupled_dict = {}
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
    n_cpus=4,
    # actually_run=False,
    # repeat_failed_jobs=True,
    shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; uv run python',
)
