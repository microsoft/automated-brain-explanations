import re
from os.path import dirname

import joblib
import numpy as np
import json
from os.path import join
import neuro.config
from neuro.features.questions.gpt4 import QS_35_STABLE, QS_HYPOTHESES
from neuro.features.questions.merge_v3_boostexamples import DICT_MERGE_V3_BOOSTEXAMPLES
from neuro.features.questions.qa_questions_base import *
from neuro.features.questions.qa_questions_data_boost import *
from neuro.features.questions.qa_questions_llama_boost import *

path_to_file = dirname(__file__)


def _split_bulleted_str(s, remove_parentheticals=False):
    qs = [q.strip('- ') for q in s.split('\n')]
    if remove_parentheticals:
        qs = [re.sub(r'\(.*?\)', '', q).strip() for q in qs]
    return qs


def _rewrite_to_focus_on_end(question, suffix='last'):
    '''Note, haven't screened questions beyond v4
    to make sure they're compatible with this
    '''
    if suffix == 'last':
        focus_text = 'In the last word of the input, '
    elif suffix == 'ending':
        focus_text = 'At the end of the input, '
    elif suffix == 'last10':
        focus_text = 'In the last ten words of the input, '
    else:
        raise ValueError(suffix)
    # replace nouns
    question = question.lower().replace('the sentence', 'the text').replace(
        'the story', 'the text').replace('the narrative', 'the text')
    question = question.replace(' in the input?', '?').replace(
        ' in the input text?', '?')
    return focus_text + question


def get_kwargs_list_for_version_str(version_str: str):
    '''This function is just for logic with dealing with merging the manually specified versions that depend on each other
    '''
    if version_str not in ['v1', 'v2', 'v3', 'v3_boostexamples',
                                 'v4_boostexamples', 'v4', 'v5', 'v3_boostexamples_merged']:
        # if '?' in version_str or 'neurosynth' in version_str or version_str == 'qs_35':
        return [{'qa_questions_version': version_str}]
    # version str contains version and suffix
    # v3 -> v1, v2, v3
    # v3-ending -> v1-ending, v2-ending, v3-ending
    # v3_boostbasic -> v1, v2, v3_boostbasic
    # v3_boostexamples -> v1, v2, v3_boostexamples
    # v3_boostexamples_merged -> v1, v2, v3_boostexamples

    # remove _merged
    version_str = version_str.replace('_merged', '')

    # check that there is no more than one hyphen or one underscore
    assert len(version_str.split('-')) <= 2
    assert len(version_str.split('_')) <= 2

    # deal with ending
    if '-' in version_str:
        suffix = '-' + version_str.split('-')[1]
        version_str = version_str.split('-')[0]
    else:
        suffix = ''

    if '_' not in version_str:
        version_num = int(version_str.replace('v', ''))
        kwargs_list = [{'qa_questions_version': f'v{i + 1}{suffix}'}
                       for i in range(version_num)]

    # v3_boost etc. require that we first get the base v1, v2 before adding v3_boost
    else:
        version_num = int(version_str.split('_')[0].replace('v', ''))
        boost_type = '_' + version_str.split('_')[1]
        assert version_num >= 3
        kwargs_list_base = [
            {'qa_questions_version': f'v1{suffix}'},
            {'qa_questions_version': f'v2{suffix}'}
        ]
        kwargs_list_boost = [{'qa_questions_version': f'v{i + 1}{boost_type}{suffix}'}
                             for i in range(2, version_num)]
        kwargs_list = kwargs_list_base + kwargs_list_boost

    return kwargs_list


def get_questions(version='v1', suffix=None, full=False):
    '''Different versions
    -last, -ending adds suffixes from last
    '''
    if len(version.split('-')) > 1:
        version, suffix = version.split('-')
    remove_parentheticals = False

    if version == 'v1':
        ans_list = [ANS_SEMANTIC, ANS_STORY, ANS_STORY_FOLLOWUP, ANS_WORDS]
        remove_list = []
    elif version == 'v2':
        ans_list = [ANS_NEURO, ANS_NEURO_FOLLOWUP]
        remove_list = ['v1']

    # boost versions
    elif version == 'v3_boostbasic':
        ans_list = [ANS_BOOST_LLAMA_v3_1, ANS_BOOST_LLAMA_v3_2]
        remove_list = ['v1', 'v2']

    elif version == 'v3_boostexamples':
        ans_list = [ANS_BOOST_LLAMA_v3_1_ex, ANS_BOOST_LLAMA_v3_2_ex]
        remove_parentheticals = True
        remove_list = ['v1', 'v2']

    elif version == 'v4_boostexamples':
        ans_list = [ANS_BOOST_LLAMA_v4_1_ex, ANS_BOOST_LLAMA_v4_2_ex]
        remove_parentheticals = True
        remove_list = ['v1', 'v2', 'v3_boostexamples']

    # non-boost versions
    elif version == 'v3':
        ans_list = [ANS_RANDOM_DATA_EXAMPLES, ANS_RANDOM_DATA_EXAMPLES_2]
        remove_parentheticals = True
        remove_list = ['v1', 'v2']

    elif version == 'v4':
        ans_list = [ANS_BOOST_1, ANS_BOOST_2]
        remove_list = ['v1', 'v2', 'v3']

    elif version == 'v5':
        ans_list = [ANS_BOOST_5, ANS_BOOST_5_2]
        remove_list = ['v1', 'v2', 'v3', 'v4']

    elif version == 'v6':
        ans_list = [ANS_BOOST_6, ANS_BOOST_6_2]
        remove_list = ['v1', 'v2', 'v3', 'v4', 'v5']

    # custom qs
    elif version == 'v1neurosynth':
        qs = QS_HYPOTHESES
        remove_list = []
    elif version == 'qs_35':
        qs = QS_35_STABLE
        remove_list = []
    elif version == 'hypothesae':
        qs = joblib.load(join(neuro.config.HYPOTHESAES_RESULTS_DIR, 'interpretations', "results_questions.pkl"))['questions_processed'].values.tolist()
        remove_list = []

    # special cases
    elif version == 'base':
        return get_questions(version='v2', suffix=suffix, full=True)

    elif version == 'all':
        return get_questions(version='v6', suffix=suffix, full=True)

    if 'neurosynth' not in version and not version == 'qs_35' and not version == 'hypothesae':
        qs = sum([_split_bulleted_str(ans, remove_parentheticals)
                  for ans in ans_list], [])

    if suffix is not None:
        qs = [_rewrite_to_focus_on_end(q, suffix) for q in qs]

    qs_remove = sum([get_questions(version=v, suffix=suffix)
                     for v in remove_list], [])
    qs_added = sorted(list(set(qs) - set(qs_remove)))


    if full:
        # be careful to always add things in the right order!!!!
        qs_final = qs_remove + qs_added
    else:
        qs_final = qs_added

    return qs_final


def _get_merged_keep_indices_v3_boostexamples():
    questions = get_questions(
        version='v3_boostexamples', full=True)
    questions_to_drop = [k for k in sum(DICT_MERGE_V3_BOOSTEXAMPLES.values(), [
    ]) if k not in DICT_MERGE_V3_BOOSTEXAMPLES]
    return np.array([i for i, q in enumerate(questions) if q not in questions_to_drop])


def get_merged_questions_v3_boostexamples():
    questions = get_questions(
        version='v3_boostexamples', full=True)
    for k, v in DICT_MERGE_V3_BOOSTEXAMPLES.items():
        # check that all questions are precise and have no spacing
        for q in v:
            assert q.strip() == q, q
            assert q.strip() in questions, q
        # check that questions are unique
        assert len(v) == len(set(v)), v

    idxs_to_keep = _get_merged_keep_indices_v3_boostexamples()
    return [q for i, q in enumerate(questions) if i in idxs_to_keep]


def get_question_num(question_version):
    if '-' in question_version:
        return int(question_version.split('-')[0][1:])
    else:
        return int(question_version[1])


if __name__ == "__main__":
    # for k in ['v1', 'v3', 'v6', 'v3_boostbasic', 'v3_boostexamples', 'v4-ending', 'v3_boostbasic-ending', 'v4_boostexamples']:
    # print(k, get_kwargs_list_for_version_str(k))

    # for v in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v1_neurosynth']:
    #     print(v, len(get_questions(v)))  # , get_questions(v)[:10])
    # print('total questions', len(get_questions('all')))

    # # boosting
    # for v in ['v3_boostbasic', 'v3_boostexamples', 'v4_boostexamples']:
    #     print(v, len(get_questions(v)))

    # # write all questions to a json file
    # with open(join(path_to_file, 'questions/all_questions.json'), 'w') as f:
        # json.dump(get_questions('all'), f, indent=4)
    # with open(join(path_to_file, 'questions/base_questions.json'), 'w') as f:
    #     json.dump(get_questions('base'), f, indent=4)
    # with open(join(path_to_file, 'questions/v3_boostexamples.json'), 'w') as f:
    #     json.dump(get_questions('v3_boostexamples', full=True), f, indent=4)
    with open(join(path_to_file, 'questions/qs_35_stable.json'), 'w') as f:
        json.dump(QS_35_STABLE, f, indent=4)
    

    # for q in get_questions('v4_boostexamples'):
    # print(q)

    # print(get_kwargs_list_for_version_str('v3_boostexamples_merged'))
    # idxs_to_keep = _get_merged_keep_indices_v3_boostexamples()
    # # print(idxs_to_keep.size)
