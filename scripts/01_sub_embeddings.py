import os
import sys
from os.path import dirname, expanduser, join

from imodelsx import submit_utils

path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

# main models
MIST7B = 'mistralai/Mistral-7B-Instruct-v0.2'
LLAMA8B = 'meta-llama/Meta-Llama-3-8B-Instruct'
LLAMA8B_fewshot = 'meta-llama/Meta-Llama-3-8B-Instruct-fewshot'
LLAMA11B = 'meta-llama/Llama-3.2-11B-Vision-Instruct'

# other models (also -refined models)
LLAMA70B = 'meta-llama/Meta-Llama-3-70B-Instruct'
LLAMA70B_fewshot = 'meta-llama/Meta-Llama-3-70B-Instruct-fewshot'
MIXTMOE = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

params_shared_dict = {
    # things to average over
    'use_test_setup': [0],
    'pc_components': [100],

    # extract standard embeddings
    'use_extract_only': [1],
    # all subjects will be extracted regardless, this is just for prediction
    # 'subject': ['UTS03'],

    # extract braindrive embs
    # 'use_eval_brain_drive': [1],

    # 'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/may27'],
    # 'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/jun8'],
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/jul12_2025'],
    'seed_stories': range(8),
}

params_coupled_dict = {
    ('feature_space', 'qa_questions_version', 'qa_embedding_model', 'embedding_layer'):

    # baselines
    [
        # ('bert-base-uncased', None, None, None),
        # ('finetune_roberta-base-10', None, None, None),
        # ('finetune_roberta-base_binary-10', None, None, None),
    ]
    +

    # llama versions
    [
        # (llama, None, None, embedding_layer)
        # for llama in ['meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-70B']
        # for embedding_layer in [12, 24, 36, 48, 60]
    ]
    +

    # qa versions
    [
        ('qa_embedder', version, model, None)
        for version in ['hypothesae']
        # for version in ['v1neurosynth']
        # for version in ['v3_boostexamples']
        #     # ensemble1, v4, v5, v6, v4_boostexamples
        # # for version in ['v1', 'v2', 'v3_boostexamples', 'v3']
        for model in [LLAMA8B]
        # for model in [MIST7B, LLAMA8B, LLAMA8B_fewshot]
        # for model in [LLAMA8B_fewshot]
        # for model in [LLAMA70B]
        # for model in ['google/gemma-2-9b-it']
        # for model in ['Qwen/Qwen3-8B', 'google/gemma-2-9b-it', 'mistralai/Ministral-8B-Instruct-2410',]
        # for model in ['Qwen/Qwen3-8B', 'mistralai/Ministral-8B-Instruct-2410',]
    ]

    # let's just skip llama 7B/8B
    # [
        # (llama, None, None, embedding_layer)
        # for llama in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Meta-Llama-3-8B']
        # for embedding_layer in [6, 12, 18, 24, 30]
    # ]

}

args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)

# args_list = args_list[:1]

script_name = join(repo_dir, 'experiments', '02_fit_encoding.py')
amlt_kwargs = {
    # change this to run a cpu job
    'amlt_file': join(repo_dir, 'scripts', 'launch.yaml'),
    'sku': 'G1-A100',
    # 'sku': 'G2-A100',    
    'target___name': 'msrresrchvc',
    # 'target___name': 'msroctovc',

    # 'sku': '192G1-MI300X',
    # 'environment___image': 'amlt-sing/acpt-rocm6.2_ubuntu22.04_py3.10_pytorch2.5.1',
    # 'target___name': 'whitney16',

    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
}
submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    unique_seeds='seed_stories', # sets seed_stories to be randomly unique for each run
    amlt_kwargs=amlt_kwargs,
    # gpu_ids=[0, 1],
    # gpu_ids=[1, 2, 3],
    # gpu_ids=[[0, 1], [2, 3]],
    # gpu_ids=[[0, 1, 2, 3]],
    # actually_run=False,
    # shuffle=True,
    # cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; python',
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; .venv/bin/python',
)
