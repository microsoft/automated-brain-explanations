import os.path
from copy import deepcopy
from os.path import dirname
from typing import List

import imodelsx.llm
import imodelsx.util
import numpy as np
import numpy.random
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import neuro.sasc.data.data
import neuro.sasc.evaluate as evaluate
from neuro.sasc.modules.fmri_module import fMRIModule

repo_dir = dirname(dirname(os.path.abspath(__file__)))


def get_scandalous_paragraph(messages):

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = messages + [{"role": 'asssistant',
                            'content': 'Sure, here is the paragraph:\n'}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        # add_generation_prompt=True,
        return_tensors="pt"
    )[:, :-2].to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    paragraph_scandalous = tokenizer.decode(
        response, skip_special_tokens=True)[2:].strip()

    # cache based on input using dicthash
    # joblib.dump(paragraph_scandalous, 'scandalous_paragraph.pkl')
    return paragraph_scandalous


def get_paragraphs(
    prompts,
    checkpoint="gpt-4-0314",
    prefix_first="Write the beginning paragraph of a story about",
    prefix_next="Write the next paragraph of the story, but now make it about",
    cache_dir="~/.cache/llm_stories",
    apply_steering=False,
):
    """
    Example messages
    ----------------
    [
      {'role': 'system', 'content': 'You are a helpful assistant.'},
      {'role': 'user', 'content': 'Write the beginning paragraph of a story about "baseball". Make sure it contains several references to "baseball".'},
      {'role': 'assistant', 'content': 'The crack of the bat echoed through the stadium as the ball soared over the outfield fence. The crowd erupted into cheers, their excitement palpable. It was a beautiful day for baseball, with the sun shining down on the field and the smell of freshly cut grass filling the air. The players on the field were focused and determined, each one ready to give their all for their team. Baseball was more than just a game to them; it was a passion, a way of life. And as they took their positions on the field, they knew that anything was possible in this great game of baseball.'},
      {'role': 'user', 'content': 'Write the next paragraph of the story, but now make it about "animals". Make sure it contains several references to "animals".'},
    ]
    """

    llm = imodelsx.llm.get_llm(
        checkpoint, CACHE_DIR=cache_dir)  # , repeat_delay=10)
    response = None
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    all_content = []

    # for i in range(len(prompts)):
    #     messages.append({"role": "user", "content": prompts[i]})
    #     all_content.append(messages[-1])
    #     # for message in messages:
    #     # print(message)

    #     if i >= 10:
    #         response_text = llm(messages, use_cache=False)
    #     else:
    #         response_text = llm(messages, use_cache=True)
    #     messages.append({"role": "assistant", "content": response_text})
    #     all_content.append(messages[-1])

    #     for msg in messages:
    #         print(msg)
    #     print(i, 'token_length', len(
    #         ''.join([str(m) for m in messages]).split()))
    #     print('----------\n\n')
    #     print(i, len(prompts))
    #     time.sleep(0.5)

    # token_limit = {
    #     "gpt-3.5-turbo": 3200,
    #     "gpt-4-0314": 30000,
    #     "gpt-4": 30000,
    # }[checkpoint]
    token_limit = 30000

    for i in range(len(prompts)):
        messages.append({"role": "user", "content": prompts[i]})
        all_content.append(messages[-1])
        response = llm(messages, return_str=False, use_cache=True)
        # print('resp', response)
        # if response is not None:
        # response_text = response["choices"][0]["message"]["content"]

        # need to drop beginning of story whenever we approach the tok limit
        # gpt-3.5.turbo has a limit of 4096, and it cant generate beyond that
        # num_tokens = response["usage"]["total_tokens"]
        try:
            # tests whether call was successfull
            num_tokens = response.usage.total_tokens
            response_text = response.choices[0].message.content
        except Exception as e:
            print(e)
            print('Failed on prompt',
                  prompts[i], 'relying on backup scandalous paragraph')
            response_text = get_scandalous_paragraph(messages)
            print('scandalous paragraph', response_text)

        if "can't assist with that" in response_text:
            raise ValueError('Failed on prompt', prompts[i])

        messages.append({"role": "assistant", "content": response_text})
        all_content.append(messages[-1])

        # print('num_tokens', num_tokens)
        if num_tokens >= token_limit:
            # drop the first (assistant, user) pair in messages
            messages = [messages[0]] + messages[3:]

            # rewrite the original prompt to now say beginning paragraph rather than next paragraph
            if prefix_first and prefix_next:
                messages[1]["content"] = messages[1]["content"].replace(
                    prefix_next, prefix_first
                )

        for msg in messages:
            print(msg)
        print(i, 'token_length', len(
            ''.join([str(m) for m in messages]).split()))
        print('----------\n\n')
        print(i, len(prompts), num_tokens)
        # time.sleep(0.5)

    # extract out paragraphs
    paragraphs = [d["content"]
                  for d in all_content if d["role"] == "assistant"]
    assert len(paragraphs) == len(prompts)
    return paragraphs


def select_top_examples_randomly(
    examples_list: List[str],
    n_examples_per_prompt_to_consider: int,
    n_examples_per_prompt: int,
    seed: int,
) -> List[str]:
    rng = np.random.RandomState(seed)
    return [
        ", ".join(
            [
                f'"{x}"'
                for x in rng.choice(
                    examples[:n_examples_per_prompt_to_consider],
                    min(n_examples_per_prompt, len(
                        examples[:n_examples_per_prompt_to_consider])),
                    replace=False,
                ).tolist()
            ]
        )
        for examples in examples_list
    ]


def process_and_add_scores(r: pd.DataFrame, add_bert_scores=False):
    # metadata
    r["task_str"] = r.apply(
        lambda row: neuro.sasc.data.data.get_task_str(
            row["module_name"], row["module_num"]),
        axis=1,
    )
    r["task_keyword"] = r["task_str"].apply(
        lambda task_str: neuro.sasc.data.data.get_groundtruth_keyword(task_str)
    )
    r["task_name (groundtruth)"] = r["task_str"].apply(
        lambda s: s.split("_")[-1])
    r["ngrams_restricted"] = ~(r["module_num_restrict"] == -1)
    r["num_generated_explanations"] = r["explanation_init_strs"].apply(
        lambda x: len(x))

    # recompute recovery metrics
    r["score_contains_keywords"] = r.apply(
        lambda row: evaluate.compute_score_contains_keywords(
            row, row["explanation_init_strs"]
        ),
        axis=1,
    )
    if add_bert_scores:
        r["score_bert"] = r.progress_apply(
            lambda row: evaluate.compute_score_bert(
                row, row["explanation_init_strs"]),
            axis=1,
        )

    # metrics
    # for met_suff in ['contains_keywords']:
    for met_suff in ["contains_keywords", "bert"]:
        if "score_" + met_suff in r.columns:
            met_val = r["score_" + met_suff]
            r["top_" + met_suff] = met_val.apply(lambda x: x[0])
            r["any_" + met_suff] = met_val.apply(lambda x: np.max(x))
            r["mean_" + met_suff] = met_val.apply(lambda x: np.mean(x))
            r[f"mean_{met_suff}_weighted"] = (
                r[f"mean_{met_suff}"] * r["num_generated_explanations"]
            )
    r["row_count_helper"] = 1
    r = r.sort_values(by="top_score_synthetic", ascending=False).round(3)
    return r


def get_prompt_templates(version):
    PROMPTS = {
        # used without explanations
        'v1_steered': {
            "prefix_first": "Here is the first paragraph of an interesting story:",
            "prefix_next": "",
            "suffix": "",

        },
        # used for may 31 2024 onwards, helps clean up prompts
        "v6_noun": {
            "prefix_first": "Write the first paragraph of an interesting story told in first person. The story should have a plot and characters. The first paragraph of the story should contain many examples of",
            "prefix_next": "Write the next paragraph of the story, but now make it contain several examples of",
            "suffix": ' "{expl}". Make sure it contains many examples of "{expl}", such as {examples}.',
        },
        # make story "coherent" (this was used for UTS03)
        "v5_noun": {
            "prefix_first": "Write the beginning paragraph of a long, coherent story. The story should be about",
            "prefix_next": "Write the next paragraph of the story, staying consistent with the story so far, but now make it about",
            "suffix": ' "{expl}". Make sure it contains several words related to "{expl}", such as {examples}.',
        },
        # make story "interesting"
        "v4_noun": {
            "prefix_first": "Write the beginning paragraph of an interesting story told in first person. The story should have a plot and characters. The story should be about",
            "prefix_next": "Write the next paragraph of the story, but now make it about",
            "suffix": ' "{expl}". Make sure it contains several words related to "{expl}", such as {examples}.',
        },
        "v4": {
            "prefix_first": "Write the beginning paragraph of an interesting story told in first person. The story should have a plot and characters. The story should place a heavy focus on",
            "prefix_next": "Write the next paragraph of the story, but now make it emphasize",
            "suffix": " {expl} words. Make sure it contains several references to {expl} words, such as {examples}.",
        },
        # add example ngrams
        "v2": {
            "prefix_first": "Write the beginning paragraph of a story told in first person. The story should be about",
            "prefix_next": "Write the next paragraph of the story, but now make it about",
            "suffix": ' "{expl}". Make sure it contains several references to "{expl}", such as {examples}.',
        },
        # first-person
        "v1": {
            "prefix_first": "Write the beginning paragraph of a story told in first person. The story should be about",
            "prefix_next": "Write the next paragraph of the story, but now make it about",
            "suffix": ' "{expl}". Make sure it contains several references to "{expl}".',
        },
        "v0": {
            "prefix_first": "Write the beginning paragraph of a story about",
            "prefix_next": "Write the next paragraph of the story, but now make it about",
            "suffix": ' "{expl}". Make sure it contains several references to "{expl}".',
        },
    }
    return PROMPTS[version]


def get_prompts(expls: List[str], examples_list: List[str], version):
    # get templates
    PV = get_prompt_templates(version)
    prompt_init = PV["prefix_first"] + PV["suffix"]
    prompt_continue = PV["prefix_next"] + PV["suffix"]

    # create prompts
    if version in ["v0", "v1"]:
        prompts = [prompt_init.format(expl=expls[0])] + [
            prompt_continue.format(expl=expl) for expl in expls[1:]
        ]
    elif version in ["v2", "v4", "v4_noun", "v5_noun", 'v6_noun']:
        prompts = [prompt_init.format(expl=expls[0], examples=examples_list[0])] + [
            prompt_continue.format(expl=expl, examples=examples)
            for (expl, examples) in zip(expls[1:], examples_list[1:])
        ]
    elif 'steered' in version:
        prompts = [prompt_init] + ['Here is the next paragraph:'] * (len(expls) - 1)

    else:
        raise ValueError(version, "not supported in get_prompts")

    prompts = [_process_prompts(p) for p in prompts]

    return prompts


def _process_prompts(s):
    for k in ['repetition', 'repetition', 'numbers', 'specific times']:
        s = s.replace(f'about "{k}"', f'include {k}')
    return s


def get_prompt_templates_interaction(version):
    PROMPTS = {
        # make story "coherent"
        "v0": {
            "prefix_first": "Write the beginning paragraph of a long, coherent story. The story will cover diverse themes such as {expls}.",
            "prefix_one_not_two": 'Write the next paragraph of the story, staying consistent with the story so far, but now make it about "{expl_one}". Make sure it contains several words related to "{expl_one}", such as {examples_one}. Important: do not mention words related to "{expl_two}".',
            "prefix_one_and_two": 'Write the next paragraph of the story, staying consistent with the story so far, but now make it about "{expl_one}" and "{expl_two}". Make sure it contains several words related to "{expl_one}", such as {examples_one}, and several words related to "{expl_two}", such as {examples_two}.',
            "prefix_two_not_one": 'Write the next paragraph of the story, staying consistent with the story so far, but now make it about "{expl_two}". Make sure it contains several words related to "{expl_two}", such as {examples_two}. Important: do not mention words related to "{expl_one}".',
        },
    }
    return PROMPTS[version]


def get_prompts_interaction(
    expls_one: List[str],
    expls_two: List[str],
    examples_list_one: List[str],
    examples_list_two: List[str],
    version: str = "v0",
):
    """Get prompts for an entire story, alternative 3 paragraphs for each explanation"""
    # get templates
    assert len(expls_one) == len(expls_two)
    assert len(expls_one) == len(examples_list_one)
    assert len(expls_two) == len(examples_list_two)
    PV = get_prompt_templates_interaction(version)
    all_examples = [
        f'"{x}"'
        for x in np.unique(
            np.concatenate(
                (np.unique(expls_one[:3]), np.unique(expls_two[:3])))
        )
    ]
    prompts = [PV["prefix_first"].format(expls=", ".join(all_examples))]

    for i in range(len(expls_one)):
        # ex1 = ", ".join([f'"{x}"' for x in examples_list_one[i]])
        # ex2 = ", ".join([f'"{x}"' for x in examples_list_two[i]])
        ex1 = examples_list_one[i]
        ex2 = examples_list_two[i]
        prompts.append(
            PV["prefix_one_not_two"].format(
                expl_one=expls_one[i],
                examples_one=ex1,
                expl_two=expls_two[i],
            )
        )
        prompts.append(
            PV["prefix_one_and_two"].format(
                expl_one=expls_one[i],
                examples_one=ex1,
                expl_two=expls_two[i],
                examples_two=ex2,
            )
        )
        prompts.append(
            PV["prefix_two_not_one"].format(
                expl_one=expls_one[i],
                examples_two=ex2,
                expl_two=expls_two[i],
            )
        )

    prompts = [_process_prompts(p) for p in prompts]
    # print(expls_one)
    # for ex_list in examples_list_one[:3]:
    #     print(repr(ex_list))
    # print(all_examples)
    # print(prompts)
    # print(prompts)
    # for p in prompts:
    # print(p, end='\n\n')
    return prompts


def compute_expl_data_match_heatmap(val, expls, paragraphs):
    """
    Returns
    -------
    scores_mean: np.array (n, n) (voxels x paragraphs)
        Mean score for each story
    scores_full: List[np.array] n x (n x par_len) = voxels x (paragraphs x par_len)
    """
    n = len(expls)
    scores_mean = np.zeros((n, n))
    scores_full = []
    for i in tqdm(range(n)):
        expl = expls[i]
        scores_list = []
        for j in range(n):
            text = paragraphs[j].lower()
            words = text.split()

            ngrams = imodelsx.util.generate_ngrams_list(text, ngrams=3)
            ngrams = [words[0], words[0] + " " + words[1]] + ngrams

            # validator-based viz
            probs = np.array(val.validate_w_scores(expl, ngrams)) > 0.5
            scores_mean[i, j] = probs.mean()
            scores_list.append(deepcopy(probs))
        scores_full.append(scores_list)
    return scores_mean, scores_full


def compute_expl_module_match_heatmap(
    expls, paragraphs, voxel_nums: List[int], subjects: List[str], ngram_length=15, restrict_weights=True,
):
    """
    Note: this function also works if voxel_nums is List[List[int]]


    Returns
    -------
    scores: np.array (n, n) (voxels x paragraphs)
        Mean score for each story
    scores_max: np.array (n, n) (voxels x paragraphs)
        Max score for each story
    all_scores: List[np.array] (n, n)  (voxels x paragraphs)
        Scores for each voxel for each ngram
    """
    n = len(expls)
    if isinstance(subjects, str):
        subjects = [subjects] * n
    scores = np.zeros((n, n))
    scores_max = np.zeros((n, n))
    all_scores = []
    all_ngrams = []
    mod = fMRIModule(subject=subjects[0], restrict_weights=restrict_weights)
    for i in tqdm(range(n)):
        mod._init_fmri_voxel(subject=subjects[i], voxel_num_best=voxel_nums[i])
        ngrams_list = []
        ngrams_scores_list = []
        for j in range(n):
            text = paragraphs[j].lower()
            ngrams_paragraph = imodelsx.util.generate_ngrams_list(
                text, ngrams=ngram_length, pad_starting_ngrams=True
            )
            ngrams_list.append(ngrams_paragraph)

            # get mean score for each story
            ngrams_scores_paragraph = mod(ngrams_paragraph)
            ngrams_scores_list.append(ngrams_scores_paragraph)
            scores[i, j] = ngrams_scores_paragraph.mean()
            scores_max[i, j] = ngrams_scores_paragraph.max()

        all_scores.append(deepcopy(ngrams_scores_list))
        all_ngrams.append(deepcopy(ngrams_list))
    return scores, scores_max, all_scores


def compute_expl_module_match_heatmap_cached_single_subject(
    expls, paragraphs, voxel_nums: List[int], subject: List[str], ngram_length=15
):
    """Assume subject is the same for all stories - let's us easily run all voxels in parallel
    Returns exactly the same as compute_expl_module_match_heatmap

    Returns
    -------
    scores: np.array (n, n) (voxels x paragraphs)
        Mean score for each story
    scores_max: np.array (n, n) (voxels x paragraphs)
        Max score for each story
    all_scores: List[np.array] (n, n)  (voxels x paragraphs)
        Scores for each voxel for each ngram
    """
    n = len(expls)
    scores = np.zeros((n, n))
    scores_max = np.zeros((n, n))
    all_scores = []
    mod = fMRIModule(subject=subject, voxel_num_best=voxel_nums)

    # loop over paragraphs
    for idx_paragraph in tqdm(range(n)):
        text = paragraphs[idx_paragraph].lower()
        ngrams_paragraph = imodelsx.util.generate_ngrams_list(
            text, ngrams=ngram_length, pad_starting_ngrams=True
        )

        # get mean score for each paragraph
        ngrams_scores_paragraph = mod(ngrams_paragraph)
        all_scores.append(deepcopy(ngrams_scores_paragraph))
        scores[idx_paragraph] = ngrams_scores_paragraph.mean(axis=0)
        scores_max[idx_paragraph] = ngrams_scores_paragraph.max(axis=0)

    return scores.T, scores_max.T, all_scores


def compute_expl_module_match_heatmap_running(
    expls,
    paragraphs,
    voxel_nums,
    subjects,
    ngram_length=10,
    paragraph_start_offset_max=50,
):
    """Computes the heatmap of the match between explanations and the module scores.
    Allows for running the module on a sliding window of the story (with overlapping paragraphs)

    Params
    ------
    ngram_length: int
        The length of the ngrams to use for the module (longer will blend together paragraphs more)
    start_offset_max: int

    """
    paragraph_start_offset = min(ngram_length, paragraph_start_offset_max)
    n = len(expls)
    scores = np.zeros((n, n))
    scores_max = np.zeros((n, n))
    all_scores = []
    all_ngrams = []
    mod = fMRIModule()

    for i in tqdm(range(n)):
        mod._init_fmri(subject=subjects[i], voxel_num_best=voxel_nums[i])
        story = "\n".join(paragraphs).lower()
        ngrams_story = imodelsx.util.generate_ngrams_list(
            story, ngrams=ngram_length, pad_starting_ngrams=True
        )

        # each score corresponds to the position of the last word of the ngram
        ngrams_scores_story = mod(ngrams_story)

        story_start = 0
        for j in range(n):
            # get mean score for each story (after applying offset)
            story_end = story_start + len(paragraphs[j].split())
            ngrams_scores_paragraph = ngrams_scores_story[
                story_start + paragraph_start_offset: story_end
            ]
            story_start = story_end

            scores[i, j] = np.mean(ngrams_scores_paragraph)
            scores_max[i, j] = np.max(ngrams_scores_paragraph)

        all_scores.append(deepcopy(ngrams_scores_story))
        all_ngrams.append(deepcopy(ngrams_story))
    return scores, scores_max, all_scores, all_ngrams


def viz_paragraphs(
    paragraphs,
    scores_data_story,
    expls,
    prompts,
    normalize_to_range=True,
    moving_average=True,
    shift_to_range=True,
):
    s_data = ""
    for i in range(len(paragraphs)):
        scores_i = np.array(scores_data_story[i])

        # normalize to 0-1 range
        if normalize_to_range:
            scores_i = (scores_i - scores_i.min()) / \
                (scores_i.max() - scores_i.min())
        # scores_mod_i = scipy.special.softmax(scores_mod_i)
        if moving_average:
            scores_i = neuro.sasc.viz.moving_average(scores_i, n=3)
        if shift_to_range:
            scores_i = scores_i / 2 + 0.5  # shift to 0.5-1 range
        s_data += " " + neuro.sasc.viz.colorize(
            paragraphs[i].split(),
            scores_i,
            title=expls[i],
            subtitle=prompts[i],
            char_width_max=60,
        )
    return s_data


if __name__ == "__main__":
    expls = ["good", "bad"]
    paragraphs = ["this is a sample good story", "this is a sample bad story"]
    voxel_nums = [0, 1]
    subjects = ["UTS02", "UTS02"]

    (
        scores_mod,
        scores_max_mod,
        all_scores,
    ) = compute_expl_module_match_heatmap_cached_single_subject(
        expls, paragraphs, voxel_nums, subject="UTS02"
    )
    assert scores_mod.shape == (2, 2)

    scores_mod1, scores_max_mod1, all_scores1 = compute_expl_module_match_heatmap(
        expls, paragraphs, voxel_nums, subjects
    )
    assert scores_mod1.shape == (2, 2)

    assert np.allclose(scores_mod, scores_mod1)
    assert np.allclose(scores_max_mod, scores_max_mod1)
