import os
import sys
from os.path import join

import joblib
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import neuro.sasc
import neuro.sasc.generate_helper
from neuro import config
from neuro.repr_steering import generate_with_steering, get_avg_weight_llama70, select_weight_for_roi
from neuro import get_voxels
import imodelsx.llm

def get_rows_and_prompts_default(
    subject,
    setting,
    seed,
    n_examples_per_prompt_to_consider,
    n_examples_per_prompt,
    version,
    fname_suffix,
):
    # get voxels
    rows = get_voxels.get_rows_voxels(
        subject=subject, setting=setting, fname_suffix=fname_suffix)

    # shuffle order (this is the 1st place randomness is applied)
    rows = rows.sample(frac=1, random_state=seed, replace=False)

    # get prompt inputs
    expls = rows.expl.values
    examples_list = rows.top_ngrams_module_correct
    # n_examples from each list of examples (this is the 2nd and last place randomness is applied)
    # for pilot v0, just selected the first
    examples_list = neuro.sasc.generate_helper.select_top_examples_randomly(
        examples_list,
        n_examples_per_prompt_to_consider,
        n_examples_per_prompt,
        seed,
    )

    # get prompts
    PV = neuro.sasc.generate_helper.get_prompt_templates(version)
    prompts = neuro.sasc.generate_helper.get_prompts(
        expls, examples_list, version)
    if 'prompt_suffix' in rows.columns and not 'steered' in version:
        prompts = [p + row.prompt_suffix for p,
                   row in zip(prompts, rows.itertuples())]
    # for p in prompts:
        # print(p)

    return rows, prompts, PV


def get_rows_and_prompts_interactions(
    subject,
    setting,
    seed,
    n_examples_per_prompt_to_consider,
    n_examples_per_prompt,
    version,
):
    # get voxels
    rows1, rows2 = get_voxels.get_rows_voxels(subject=subject, setting=setting)
    print(rows1.expl.values)
    print(rows2.expl.values)

    # shuffle order (this is the 1st place randomness is applied)
    rows1 = rows1.sample(frac=1, random_state=seed, replace=False)
    rows2 = rows2.sample(frac=1, random_state=seed, replace=False)

    # get prompt inputs
    expls1 = rows1.expl.values
    expls2 = rows2.expl.values
    kwargs = dict(
        n_examples_per_prompt_to_consider=n_examples_per_prompt_to_consider,
        n_examples_per_prompt=n_examples_per_prompt,
        seed=seed,
    )

    examples_list1 = neuro.sasc.generate_helper.select_top_examples_randomly(
        rows1["top_ngrams_module_correct"].values.tolist(), **kwargs
    )
    examples_list2 = neuro.sasc.generate_helper.select_top_examples_randomly(
        rows2["top_ngrams_module_correct"].values.tolist(), **kwargs
    )
    prompts = neuro.sasc.generate_helper.get_prompts_interaction(
        expls1,
        expls2,
        examples_list1,
        examples_list2,
        version=version,
    )
    for p in prompts:
        print(p)
    PV = neuro.sasc.generate_helper.get_prompt_templates_interaction(version)

    return rows1, rows2, prompts, PV


if __name__ == "__main__":
    generate_paragraphs = False

    VERSIONS = {
        # "default": "v5_noun",
        "default": "v4_noun",
        "interactions": "v0",
        "polysemantic": "v5_noun",
        'qa': 'v6_noun',
        'roi': 'v6_noun',

        
        'default_steered': 'v1_steered',
        'roi_steered': 'v1_steered',
    }
    # iterate over seeds
    seeds = range(1, 4)
    # seeds = range(1, 2)
    # seeds = range(7, 12)
    # seeds = range(1, 10)
    # seeds = range(2, 3)
    # random.shuffle(seeds)

    # original stories
    # n_examples_per_prompt = 3
    # n_examples_per_prompt_to_consider = 6

    # increased for roi stories
    n_examples_per_prompt = 5
    n_examples_per_prompt_to_consider = 9
    # fname_suffix = '_v1'
    fname_suffix = ''
    pad_beginning_and_end = True
    for setting in [
        # "interactions",
        # "default",
        # "polysemantic",
        # 'qa',
        # 'roi',
        # 'default_steered',
        'roi_steered',
    ]:  # default, interactions, polysemantic
        for subject in [
            "UTS02",
            # "UTS03",
        ]:  # , "UTS03"]:  # ["UTS01", "UTS03"]:
            for seed in seeds:
                # for version in ["v5_noun"]:
                version = VERSIONS[setting]
                STORIES_DIR = config.STORIES_DIR_GCT

                # EXPT_NAME = f"{subject.lower()}___qa_may31___seed={seed}"
                # EXPT_NAME = f"{subject.lower()}___roi_may31___seed={seed}"
                # EXPT_NAME = f"{subject.lower()}___roi_nov30___seed={seed}{fname_suffix}"
                # EXPT_NAME = f"{subject.lower()}___qa_mar9_2025___seed={seed}{fname_suffix}"
                EXPT_NAME = f"{subject.lower()}____roi_steered_jan8_2026___seed={seed}{fname_suffix}"
                EXPT_DIR = join(STORIES_DIR, setting, EXPT_NAME)
                os.makedirs(EXPT_DIR, exist_ok=True)

                if setting in ["default", "polysemantic", 'qa', 'roi', 'default_steered', 'roi_steered']:
                    rows, prompts, PV = get_rows_and_prompts_default(
                        subject,
                        setting,
                        seed,
                        n_examples_per_prompt_to_consider,
                        n_examples_per_prompt,
                        version,
                        fname_suffix,
                    )
                    rows.to_csv(join(EXPT_DIR, "rows.csv"), index=False)
                    rows.to_pickle(join(EXPT_DIR, "rows.pkl"))

                elif setting == "interactions":
                    rows1, rows2, prompts, PV = get_rows_and_prompts_interactions(
                        subject,
                        setting,
                        seed,
                        n_examples_per_prompt_to_consider,
                        n_examples_per_prompt,
                        version,
                    )

                    # repeat
                    reps = [rows1.iloc[0]]
                    for i in range(0, len(rows1)):
                        reps.append(rows1.iloc[i])
                        reps.append(rows2.iloc[i])
                        reps.append(rows1.iloc[i])
                    rows1_rep = pd.concat(
                        reps,
                        ignore_index=True,
                        axis=1,
                    ).transpose()
                    rows1.to_csv(join(EXPT_DIR, "rows1.csv"), index=False)
                    rows1.to_pickle(join(EXPT_DIR, "rows1.pkl"))
                    rows2.to_csv(join(EXPT_DIR, "rows2.csv"), index=False)
                    rows2.to_pickle(join(EXPT_DIR, "rows2.pkl"))
                    rows1_rep.to_pickle(join(EXPT_DIR, "rows.pkl"))

                for p in prompts:
                    print('\n' + p)
                with open(join(EXPT_DIR, "prompts.txt"), "w") as f:
                    f.write("\n\n".join(prompts))

                # save
                # generate paragraphs
                if not 'steered' in version:
                    paragraphs = neuro.sasc.generate_helper.get_paragraphs(
                        prompts,
                        checkpoint="gpt-4o",
                        # checkpoint="gpt-4",
                        prefix_first=PV["prefix_first"] if "prefix_first" in PV else None,
                        prefix_next=PV["prefix_next"] if "prefix_next" in PV else None,
                        cache_dir="/home/chansingh/cache/llm_stories_may8",
                    )
                else:
                    print(rows)
                    lm = imodelsx.llm.get_llm("gpt-5")
                    model_name = "meta-llama/Meta-Llama-3-70B"
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                    ).eval()

                    subj = subject.replace('UTS', 'S')
                    wt, emb_layer = get_avg_weight_llama70(subj)
                    VEC_MULTIPLIERS_SELECTED = {
                        # 'S02': 14251.0,
                        'S02': 7017.0,
                        # 'S03': None, # to be filled later
                    }
                    max_new_tokens = 180
                    sampling_params = {
                        'do_sample': True,
                        # original tuning params
                        # 'top_p': 0.9,
                        # 'temperature': 0.8,

                        # more diverse text
                        'top_p': 0.6,
                        'temperature': 1.2,
                    }
                    
                    vec_multiplier = VEC_MULTIPLIERS_SELECTED[subj]
                    # print(rows.columns)
                    paragraphs = []
                    running_prompt = ''
                    MAX_N_CHARS = 300
                    for i in range(len(prompts)):
                        running_prompt += prompts[i]
                        row = rows.iloc[i]
                        roi_name = row.roi.replace('1', '').replace('2', '')
                        wt_vec = select_weight_for_roi(wt, subj, roi_name)
                        paragraph_steered = generate_with_steering(
                                model,
                                tokenizer,
                                emb_layer,
                                wt_vec,
                                prompt = running_prompt,
                                vec_multiplier = vec_multiplier,
                                max_new_tokens = max_new_tokens,
                                sampling_params = sampling_params,
                                seed = seed,
                        )
                        paragraph_concl = lm(
                            f'Return text that concludes the following paragraph coherently and non-repetitively in a sentence or less: {paragraph_steered}',
                            use_cache=False,
                            max_completion_tokens=30)
                        print('PARAGRAPH CONCLUSION SUGGESTION:', repr(paragraph_concl))
                        paragraph_steered += ' ' + paragraph_concl.strip()
                        
                        paragraphs.append(paragraph_steered)
                        running_prompt += paragraph_steered 
                        print('PROMPT START\n\n', running_prompt, '\n\nPROMPT END')
                        running_prompt = running_prompt[-MAX_N_CHARS:]  # rough estimate of chars per token

                if pad_beginning_and_end:
                    START_PARAGRAPH = 'You are about to read a story told in the first person. Please pay attention to the details of the story.'
                    END_PARAGRAPH = 'The narrative story has now concluded. Hope you enjoyed passively reading the story.'
                    paragraphs = [START_PARAGRAPH] + \
                        paragraphs + [END_PARAGRAPH]
                    prompts = ['START'] + prompts + ['END']
                    # DUMMY_START = pd.DataFrame(
                        # [{'expl': 'START', 'top_ngrams_module_correct': [],
                            # 'subject': subject, 'prompt_suffix': ''}])
                    # DUMMY_END = pd.DataFrame(
                        # [{'expl': 'END', 'top_ngrams_module_correct': [],
                            # 'subject': subject, 'prompt_suffix': ''}])
                    # create DUMMY_START and DUMMY_END rows based on the cols in rows
                    DUMMY_START = pd.DataFrame(
                        [{col: 'START' if col == 'expl' else [] if col == 'top_ngrams_module_correct' else subject if col == 'subject' else '' for col in rows.columns}])
                    DUMMY_END = pd.DataFrame(
                        [{col: 'END' if col == 'expl' else [] if col == 'top_ngrams_module_correct' else subject if col == 'subject' else '' for col in rows.columns}])
                    rows = pd.concat([DUMMY_START, rows, DUMMY_END])

                    # overwrite rows with pad
                    rows.to_csv(join(EXPT_DIR, "rows.csv"), index=False)
                    rows.to_pickle(join(EXPT_DIR, "rows.pkl"))

                with open(join(EXPT_DIR, "story.txt"), "w") as f:
                    f.write("\n\n".join(paragraphs))
                joblib.dump(
                    {"prompts": prompts, "paragraphs": paragraphs},
                    join(EXPT_DIR, "prompts_paragraphs.pkl"),
                )
