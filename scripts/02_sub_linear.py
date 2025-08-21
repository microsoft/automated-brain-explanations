import os
from os.path import dirname, join, expanduser
import sys
from imodelsx import submit_utils
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

# main models
MIST7B = 'mistralai/Mistral-7B-Instruct-v0.2'
LLAMA8B = 'meta-llama/Meta-Llama-3-8B-Instruct'
LLAMA8B_fewshot = 'meta-llama/Meta-Llama-3-8B-Instruct-fewshot'

# other models (also -refined models)
LLAMA70B = 'meta-llama/Meta-Llama-3-70B-Instruct'
LLAMA70B_fewshot = 'meta-llama/Meta-Llama-3-70B-Instruct-fewshot'
MIXTMOE = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

BEST_RUN = '/home/chansingh/mntv1/deep-fMRI/encoding/results_apr7/68936a10a548e2b4ce895d14047ac49e7a56c3217e50365134f78f990036c5f7'

# python 02_fit_encoding.py --encoding_model tabpfn --use_test_setup 1 --ndelays 1 --use_extract_only 0 --pc_components 100 --subject UTS03 --save_dir /home/chansingh/fmri/tabpfn_test/ --feature_space qa_embedder --qa_questions_version v3_boostexamples_merged --qa_embedding_model ensemble2 --feature_selection_alpha 0.28
# python 02_fit_encoding.py --encoding_model ridge --use_test_setup 1 --ndelays 1 --use_extract_only 0 --pc_components 100 --subject UTS03 --save_dir /home/chansingh/fmri/tabpfn_test/ --feature_space qa_embedder --qa_questions_version v3_boostexamples_merged --qa_embedding_model ensemble2 --feature_selection_alpha 0.28


params_shared_dict = {
    # things to average over
    # 'use_cache': [1],
    # 'nboots': [5],
    # 'encoding_model': ['ridge'],
    'use_test_setup': [0],
    'use_extract_only': [0],
    'pc_components': [100],

    'subject': [f'UTS0{k}' for k in range(1, 4)],
    # 'subject': [f'UTS0{k}' for k in range(4, 9)],
    # 'subject': ['UTS04'],

    # ['UTS01', 'UTS02', 'UTS03', 'UTS04', 'UTS05', 'UTS06', 'UTS07', 'UTS08']
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/jul21_2025_agentic_test'],
    # 'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/jun8'],
    # 'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/may7'],
    # 'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/may27'],
    'use_eval_brain_drive': [0],
    # 'ndelays': [4, 8],
    'ndelays': [8],

    # default is -1, SO4-SO8 have 24 or 25 stories
    'num_stories': [-1],  # , 5, 10, 20],
    # 'num_stories': [5, 10, 20],
}

params_coupled_dict = {
    ('feature_space', 'qa_questions_version', 'qa_embedding_model', 'embedding_layer'):

    [
        # baselines
        # ('eng1000', None, None, None),
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
        # ensemble1
        # questions: v4, v5, v6, v4_boostexamples, v1, v2, v3_boostexamples, v3, 'v3_boostexamples_merged'
        # ('qa_embedder', 'v3_boostexamples_merged', model, None)
        # ('qa_embedder', 'v2', model, None)
        # for model in [MIST7B, LLAMA8B, LLAMA8B_fewshot]
        # for model in ['ensemble2']  # , LLAMA8B, LLAMA70B]
    ]
    +
    # qa 70B
    [
        # ('qa_embedder', version, model, None)
        # for version in ['v1']  # , 'v2']
        # for version in ['v3_boostexamples']
        # for model in [LLAMA70B]
    ]
    +
    # qa agentic test 
    [
        # ensemble1
        # questions: v4, v5, v6, v4_boostexamples, v1, v2, v3_boostexamples, v3, 'v3_boostexamples_merged'
        ('qa_embedder', 'v1', model, None)
        # ('qa_embedder', 'v2', model, None)
        # for model in [MIST7B, LLAMA8B, LLAMA8B_fewshot]
        for model in [MIST7B, LLAMA8B, LLAMA8B_fewshot, LLAMA70B, 'Qwen/Qwen3-8B', 'mistralai/Ministral-8B-Instruct-2410',]  # , LLAMA8B, LLAMA70B]
    ]



    ####################
    # let's just skip llama 7B/8B
    # [
        # (llama, None, None, embedding_layer)
        # for llama in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Meta-Llama-3-8B']
        # for embedding_layer in [6, 12, 18, 24, 30]
    # ]

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
    shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; .venv/bin/python',
)
