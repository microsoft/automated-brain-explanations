import os
from os.path import dirname, join, expanduser
import sys
from imodelsx import submit_utils
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

params_shared_dict = {
    'use_test_setup': [0],
    'use_extract_only': [0],
    'pc_components': [100],

    'subject': [f'UTS0{k}' for k in range(1, 4)],

    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/aug4_agentic'],
    'ndelays': [8],

    'qa_embedding_model': ['meta-llama/Meta-Llama-3-8B-Instruct'], 
}

params_coupled_dict = {
    ('feature_space', 'qa_questions_version', 'num_questions_restrict'): 
    [
        ('qa_embedder', qa_questions_version, num_questions_restrict)
        for qa_questions_version in ['v3', 'hypothesae']
        for num_questions_restrict in [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200]
    ] + 
    [
        ('eng1000', None, num_questions_restrict)
        for num_questions_restrict in [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200]
    ]
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
    unique_seeds='seed_stories',
    # amlt_kwargs=amlt_kwargs_cpu,
    n_cpus=8,
    # actually_run=False,
    repeat_failed_jobs=True,
    # shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; uv run python',
)
