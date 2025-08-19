import logging
import os
from copy import deepcopy
from os.path import dirname, join
from typing import Dict, List

import datasets
import imodelsx.llm
import joblib
import numpy as np
from dict_hash import sha256
from tqdm import tqdm
from transformers import pipeline

import neuro.config as config
import neuro.features.qa_questions as qa_questions
from neuro.data.interp_data import expinterp2D, lanczosinterp2D
from neuro.data.semantic_model import SemanticModel
from neuro.data.utils_ds import apply_model_to_words
from neuro.features.qa_embedder import FinetunedQAEmbedder
from imodelsx.qaemb.qaemb import QAEmb
from neuro.features.questions.gpt4 import QS_35_STABLE, QS_HYPOTHESES_COMPUTED
from neuro.features.stim_utils import load_story_wordseqs_wrapper


def downsample_word_vectors(stories, word_vectors, wordseqs, strategy='lanczos'):
    """Get Lanczos downsampled word_vectors for specified stories.

    Args:
            stories: List of stories to obtain vectors for.
            word_vectors: Dictionary of {story: <float32>[num_story_words, vector_size]}

    Returns:
            Dictionary of {story: downsampled vectors}
    """
    downsampled_semanticseqs = dict()
    for story in stories:
        if strategy == 'lanczos':
            downsampled_semanticseqs[story] = lanczosinterp2D(
                word_vectors[story],
                oldtime=wordseqs[story].data_times,  # timing of the old data
                newtime=wordseqs[story].tr_times,  # timing of the new data
                window=3
            )
        elif strategy == 'exp':
            downsampled_semanticseqs[story] = expinterp2D(
                word_vectors[story],
                oldtime=wordseqs[story].data_times,  # timing of the old data
                newtime=wordseqs[story].tr_times,  # timing of the new data
                theta=1
            )
        else:
            raise ValueError(f"Strategy {strategy} not recognized")
    return downsampled_semanticseqs


def get_wordrate_vectors(wordseqs, story_names: List[str], downsample=True, **kwargs):
    """Get wordrate vectors for specified stories.
    Returns
    -------
    Dictionary of {story: downsampled vectors}
    """
    # wordseqs = load_story_wordseqs(story_names)
    vectors = {}
    for story in story_names:
        nwords = len(wordseqs[story].data)
        vectors[story] = np.ones([nwords, 1])
    if downsample:
        return downsample_word_vectors(story_names, vectors, wordseqs)
    else:
        return story_names, vectors, wordseqs


def get_eng1000_vectors(wordseqs, story_names: List[str], downsample='lanczos', **kwargs):
    """Get Eng1000 vectors (985-d) for specified stories.
    Returns
    -------
    Dictionary of {story: downsampled vectors}
    """
    eng1000 = SemanticModel.load(join(config.EM_DATA_DIR, "english1000sm.hf5"))
    # wordseqs = load_story_wordseqs(story_names)
    vectors = {}
    for story in story_names:
        sm = apply_model_to_words(wordseqs[story], eng1000, 985)
        vectors[story] = sm.data
    if downsample:
        return downsample_word_vectors(story_names, vectors, wordseqs, strategy=downsample)
    else:
        return story_names, vectors, wordseqs


def get_embs_from_text_list(text_list: List[str], embedding_function) -> List[np.ndarray]:
    """
    Params
    ------
    embedding_function (ngram -> fixed size vector)

    Returns
    -------
    embs: np.ndarray (len(text_list), embedding_size)
    """

    text = datasets.Dataset.from_dict({'text': text_list})

    # get embeddings
    def get_emb(x):
        return {'emb': embedding_function(x['text'])}
    embs_list = text.map(get_emb)['emb']  # embedding_function(text)

    # Convert to np array by averaging over len
    # Embeddings are already the same size
    # Can't just convert this since seq lens vary
    # Need to avg over seq_len dim
    logging.info('\tPostprocessing embs...')
    embs = np.zeros((len(embs_list), len(embs_list[0])))
    num_ngrams = len(embs_list)
    dim_size = len(embs_list[0][0][0])
    embs = np.zeros((num_ngrams, dim_size))
    for i in tqdm(range(num_ngrams)):
        # embs_list is (batch_size, 1, (seq_len + 2), 768) -- BERT adds initial / final tokens
        embs[i] = np.mean(embs_list[i], axis=1)  # avg over seq_len dim
    return embs


def get_gpt4_qa_embs_cached(
        story_name,
        questions: List[str] = None,
        qa_questions_version: str = None,
        return_ngrams=False,
):
    '''Returns (binary) embeddings for a story using GPT4 questions
    Params
    -----
    story_name: str
        Name of the story to get embeddings for
    questions: List[str]
        List of questions to get embeddings for. These end in '?'
    qa_questions_version: str, Optional
        If questions is not passed, get questions from this version


    Returns
    -------
    embs: np.ndarray (n_ngrams, n_questions)
        if a question is not found in the list, returns nan for that column in embs
    '''
    # set up question names
    CACHE_DIR_GPT = join(config.FMRI_DIR_BLOB, 'qa/cache_gpt')
    if questions is None or questions == []:
        if '?' in qa_questions_version:
            questions = [qa_questions_version]
        elif qa_questions_version == 'QS_HYPOTHESES_COMPUTED':
            questions = QS_HYPOTHESES_COMPUTED
        elif qa_questions_version == 'qs_35':
            questions = QS_35_STABLE
        else:
            questions = qa_questions.get_questions(
                version=qa_questions_version)
    ngrams_metadata = joblib.load(
        join(CACHE_DIR_GPT, 'ngrams_metadata.joblib'))
    wordseq_idxs = ngrams_metadata['wordseq_idxs'][story_name]
    story_len = wordseq_idxs[1] - wordseq_idxs[0]
    embs = np.zeros((story_len, len(questions)))
    for i, q in enumerate(questions):
        gpt4_cached_answers_file = join(CACHE_DIR_GPT, f'{q}.pkl')
        if os.path.exists(gpt4_cached_answers_file):
            embs[:, i] = joblib.load(gpt4_cached_answers_file)[
                wordseq_idxs[0]: wordseq_idxs[1]]
        else:
            embs[:, i] = np.nan
            print('warning, question not found in cache', q)
    if return_ngrams:
        return embs, ngrams_metadata['ngrams_list_total'][wordseq_idxs[0]: wordseq_idxs[1]]
    else:
        return embs

def _get_ngrams_list_from_words_list(words_list: List[str], ngram_size: int = 5) -> List[str]:
    """Concatenate running list of words into grams with spaces in between
    """
    ngrams_list = []
    for i in range(len(words_list)):
        l = max(0, i - ngram_size)
        ngram = ' '.join(words_list[l: i + 1])
        ngrams_list.append(ngram.strip())
    return ngrams_list

def _get_ngrams_list_from_chunks(chunks, num_trs=2):
    ngrams_list = []
    for i in range(len(chunks)):
        chunk_block = chunks[i - num_trs:i]
        if len(chunk_block) == 0:
            ngrams_list.append('')
        else:
            chunk_block = np.concatenate(chunk_block)
            ngrams_list.append(' '.join(chunk_block))
    return ngrams_list

def _get_ngrams_list_from_words_list_and_times(words_list: List[str], times_list: np.ndarray[float], sec_offset: float = 4) -> List[str]:
    words_arr = np.array(words_list)
    ngrams_list = []
    for i in range(len(times_list)):
        t = times_list[i]
        t_off = t - sec_offset
        idxs = np.where(np.logical_and(
            times_list >= t_off, times_list <= t))[0]
        ngrams_list.append(' '.join(words_arr[idxs]))
    return ngrams_list

def get_ngrams_list_main(ds, num_trs_context=None, num_secs_context_per_word=None, num_ngrams_context=None) -> List[str]:
    # get ngrams_list
    if num_trs_context is not None:
        # replace each TR with text from the current TR and the TRs immediately before it
        ngrams_list = _get_ngrams_list_from_chunks(
            ds.chunks(), num_trs=num_trs_context)
        assert len(ngrams_list) == len(ds.chunks())
    elif num_secs_context_per_word is not None:
        # replace each word with the ngrams in a time window leading up to that word
        ngrams_list = _get_ngrams_list_from_words_list_and_times(
            ds.data, ds.data_times, sec_offset=num_secs_context_per_word)
        assert len(ngrams_list) == len(ds.data)
    else:
        # replace each word with an ngram leading up to that word
        ngrams_list = _get_ngrams_list_from_words_list(
            ds.data, ngram_size=num_ngrams_context)
        assert len(ngrams_list) == len(ds.data)
    return ngrams_list


def get_llm_vectors(
    wordseqs,
    story_names,
    feature_space='bert-base-uncased',
    num_ngrams_context=10,
    num_trs_context=None,
    num_secs_context_per_word=None,
    layer_idx=None,
    qa_embedding_model='mistralai/Mistral-7B-v0.1',
    qa_questions_version='v1',
    downsample='lanczos',
    use_cache=True,
    batch_size=16,
    **kwargs
) -> Dict[str, np.ndarray]:
    """Get llm embedding vectors
    """

    def _get_embedding_model(feature_space, qa_questions_version, qa_embedding_model):
        print('loading embedding_model...')
        if feature_space in ['qa_embedder', 'qa_agent']:
            if isinstance(qa_questions_version, str):
                questions = qa_questions.get_questions(
                    version=qa_questions_version)
            elif isinstance(qa_questions_version, list):
                questions = qa_questions_version
            return QAEmb(
                # dont cache calls locally
                checkpoint=qa_embedding_model, questions=questions, use_cache=False, batch_size=batch_size)
        elif feature_space.startswith('finetune_'):
            return FinetunedQAEmbedder(
                feature_space.replace('finetune_', '').replace('_binary', ''), qa_questions_version=qa_questions_version)
        if not feature_space == 'qa_embedder':
            if 'bert' in feature_space.lower():
                return pipeline("feature-extraction", model=feature_space, device=0)
            elif layer_idx is not None:
                return imodelsx.llm.LLMEmbs(checkpoint=feature_space)

    assert not (
        num_trs_context and num_secs_context_per_word), 'num_trs_context and num_secs_context_per_word are mutually exclusive'
    vectors = {}
    ngrams_list_dict = {}
    embedding_model = None  # only initialize if needed
    if feature_space == 'qa_embedder' or feature_space == 'qa_agent':
        logging.info(
            f'extracting {feature_space} {qa_questions_version} {qa_embedding_model} embs...')
    else:
        logging.info(f'extracting {feature_space} {qa_questions_version} embs...')

    for story_num, story in enumerate(story_names):
        # qa agent does caching per-question per-story, so has its own handling
        if feature_space == 'qa_agent':
            assert qa_embedding_model != 'gpt4', \
                'qa_agent does not support gpt4, use qa_embedder instead'
            assert isinstance(qa_questions_version, list), \
                'qa_questions_version must be a list of questions for qa_agent'
            qa_vecs = {}
            for i, q in enumerate(qa_questions_version):
                loaded_from_cache = False
                args_cache = {'story': story, 'model': qa_embedding_model, 'question': q, 'qa_embedding_model': qa_embedding_model, 'story_gen': 'genstory' in story.lower()}
                cache_hash = sha256(args_cache)
                cache_file = join(
                    config.CACHE_EMBS_AGENT_DIR, qa_embedding_model.replace('/', '_'), f'{cache_hash}.jl')
                
                if os.path.exists(cache_file) and use_cache:
                    logging.info(
                        f'Loading cached {story_num}/{len(story_names)}: {story} Q{i}/{len(qa_questions_version)}: {q}')
                    try:
                        qa_vecs[q] = joblib.load(cache_file).astype(float)
                        loaded_from_cache = True
                    except:
                        print('Error loading', cache_file)
                if not loaded_from_cache:
                    logging.info(
                        f'Computing Story{story_num}/{len(story_names)}: {story} Question{i}/{len(qa_questions_version)}: {q}')
                    if embedding_model is None:
                        embedding_model = _get_embedding_model(
                            feature_space, qa_questions_version, qa_embedding_model)
                    if not story in ngrams_list_dict:
                        ngrams_list_dict[story] = get_ngrams_list_main(
                            wordseqs[story], num_trs_context, num_secs_context_per_word, num_ngrams_context)
                    embedding_model.questions = [q]
                    qa_vecs[q] = embedding_model(ngrams_list_dict[story], verbose=False)
                    os.makedirs(dirname(cache_file), exist_ok=True)
                    joblib.dump(qa_vecs[q].astype(bool), cache_file)
            
            # stack the vectors for all questions                    
            assert len(qa_vecs) == len(qa_questions_version), \
                f'Expected {len(qa_questions_version)} questions, got {len(qa_vecs)}'
            vectors[story] = np.hstack(
                [qa_vecs[q] for q in qa_questions_version])
        else:
            # non-agent: try loading from cache
            loaded_from_cache = False
            if qa_embedding_model != 'gpt4':
                args_cache = {'story': story, 'model': feature_space, 'ngram_size': num_ngrams_context,
                            'qa_embedding_model': qa_embedding_model, 'qa_questions_version': qa_questions_version,
                            'num_trs_context': num_trs_context, 'num_secs_context_per_word': num_secs_context_per_word}
                if layer_idx is not None:
                    args_cache['layer_idx'] = layer_idx
                if 'genstory' in story.lower():
                    args_cache['story_gen'] = True
                cache_hash = sha256(args_cache)
                cache_file = join(
                    config.CACHE_EMBS_DIR, qa_questions_version, feature_space.replace('/', '_'), f'{cache_hash}.jl')
                if os.path.exists(cache_file) and use_cache:
                    logging.info(
                        f'Loading cached {story_num}/{len(story_names)}: {story}')
                    try:
                        vectors[story] = joblib.load(cache_file)
                        loaded_from_cache = True
                        # print('Loaded', story, 'vectors', vectors[story].shape,
                        #   'unique', np.unique(vectors[story], return_counts=True))
                    except:
                        print('Error loading', cache_file)
            elif qa_embedding_model == 'gpt4':
                vectors[story] = get_gpt4_qa_embs_cached(
                            story, qa_questions_version=qa_questions_version)
                loaded_from_cache = True            

            # get ngrams_list if needed
            if not downsample or not loaded_from_cache:
                ngrams_list_dict[story] = get_ngrams_list_main(
                    wordseqs[story], num_trs_context, num_secs_context_per_word, num_ngrams_context)
            if not loaded_from_cache and not qa_embedding_model == 'gpt4':

                # embed the ngrams
                if embedding_model is None:
                    embedding_model = _get_embedding_model(
                        feature_space, qa_questions_version, qa_embedding_model)
                if feature_space == 'qa_embedder':
                    print(f'Extracting {story_num}/{len(story_names)}: {story}')
                    embs = embedding_model(ngrams_list_dict[story], verbose=False)
                elif feature_space.startswith('finetune_'):
                    embs = embedding_model.get_embs_from_text_list(ngrams_list_dict[story])
                    if '_binary' in feature_space:
                        embs = embs.argmax(axis=-1)  # get yes/no binarized`
                    else:
                        embs = embs[:, :, 1]  # get logit for yes
                elif 'bert' in feature_space:
                    embs = get_embs_from_text_list(
                        ngrams_list_dict[story], embedding_function=embedding_model)
                elif layer_idx is not None:
                    embs = embedding_model(
                        ngrams_list_dict[story], layer_idx=layer_idx, batch_size=8)
                else:
                    raise ValueError(feature_space)

                # if num_trs_context is None:
                    # embs = DataSequence(
                    # embs, ds.split_inds, ds.data_times, ds.tr_times).data
                vectors[story] = deepcopy(embs)
                os.makedirs(dirname(cache_file), exist_ok=True)
                joblib.dump(embs, cache_file)

    # return
    if num_trs_context is not None:
        return vectors
    elif not downsample:
        return story_names, vectors, wordseqs, ngrams_list_dict
    else:
        return downsample_word_vectors(
            story_names, vectors, wordseqs, strategy=downsample)


def _get_kwargs_extra(args):
    kwargs = {}
    if hasattr(args, 'input_chunking_type'):
        if args.input_chunking_type == 'ngram':
            kwargs['num_ngrams_context'] = args.input_chunking_size
        elif args.input_chunking_type == 'tr':
            kwargs['num_trs_context'] = args.input_chunking_size
        elif args.input_chunking_type == 'sec':
            kwargs['num_secs_context_per_word'] = args.input_chunking_size

    # also pass layer
    if hasattr(args, 'embedding_layer') and args.embedding_layer >= 0:
        kwargs['layer_idx'] = args.embedding_layer
    return kwargs


def get_features(args, feature_space, **kwargs):
    kwargs_extra = _get_kwargs_extra(args)
    logging.info('getting wordseqs..')
    wordseqs = load_story_wordseqs_wrapper(
        kwargs['story_names'], kwargs['use_brain_drive'])

    if feature_space == 'eng1000':
        return get_eng1000_vectors(wordseqs, **kwargs)
    elif feature_space == 'wordrate':
        return get_wordrate_vectors(wordseqs, **kwargs)
    else:
        return get_llm_vectors(wordseqs, feature_space=feature_space, batch_size=args.qa_batch_size, **kwargs, **kwargs_extra)

if __name__ == "__main__":
    kwargs = {
        'story_names': ['sloth', 'adollshouse'],
        'qa_embedding_model': 'mistralai/Mistral-7B-Instruct-v0.2',
        'use_huge': True, 'use_brain_drive': False,
        'num_ngrams_context': 10,
        'qa_batch_size': 16,
        
        # 'feature_space': 'qa_embedder',
        # 'qa_questions_version': 'v1',

        'feature_space': 'qa_agent',
        'qa_questions_version': ['Does the sentence contain a proper noun?'],
    }
    logging.basicConfig(level=logging.INFO)
    wordseqs = load_story_wordseqs_wrapper(
        kwargs['story_names'], kwargs['use_brain_drive'])
    llm_vecs = get_llm_vectors(
        wordseqs, **kwargs)