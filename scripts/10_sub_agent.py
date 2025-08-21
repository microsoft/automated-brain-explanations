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
    'ndelays': [8],
    'num_stories': [-1], 

    'feature_space': ['qa_agent'],
    'qa_embedding_model': ['meta-llama/Meta-Llama-3-8B-Instruct'],


    # 'subject': ['UTS02'],
    'seed': [1],
    'subject': [f'UTS0{k}' for k in range(1, 4)],
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/aug19_agentic'],
    # 'predict_subset': ['prefrontal', 'occipital', 'sensorimotor', 'cingulate', 'insula', 'parietal', 'temporal'],
    'predict_subset': ['all'], #, 'prefrontal'],

    # 8B model: 16 for 1x45 GB, 64 for 2x45 GB, 256 for 4x45 GB (but is slower)
    # 'qa_batch_size': [64], 
    'qa_batch_size': [128], 

    
    'agent_checkpoint': ['o4-mini', 'gpt-4.1', 'gpt-5'],
}

params_coupled_dict = {
    ('num_questions_brainstorm_target', 'num_questions_increment_target', 'num_agent_epochs'): [
        (25, 25, 25),
        (10, 10, 40),
        (50, 50, 20),
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
    # change this to run a cpu job
    'amlt_file': join(repo_dir, 'scripts', 'launch.yaml'),
    'sku': 'G1-A100',
    # 'sku': 'G2-A100',    
    # 'target___name': 'msrresrchvc',
    'target___name': 'msroctovc',

    # 'sku': '192G1-MI300X',
    # 'environment___image': 'amlt-sing/acpt-rocm6.2_ubuntu22.04_py3.10_pytorch2.5.1',
    # 'target___name': 'whitney16',

    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
}

submit_utils.run_args_list(
    args_list[::-1],
    script_name=script_name,
    unique_seeds='seed_stories',
    # amlt_kwargs=amlt_kwargs,
    # n_cpus=8,
    # actually_run=False,
    gpu_ids=[0, 1, 2, 3],
    repeat_failed_jobs=True,
    # shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; .venv/bin/python',
)
