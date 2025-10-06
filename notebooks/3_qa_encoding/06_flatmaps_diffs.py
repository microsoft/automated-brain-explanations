import os
import sys
from os.path import dirname, join

import joblib
from tqdm import tqdm

from neuro import analyze_helper, config
import cortex
import imodelsx.process_results
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def _save_flatmap(vals, subject, fname_save, clab=None, with_rois=False, cmap='RdBu', with_borders=False, vabs=None):
    if vabs is None:
        vabs = max(np.abs(vals))

    # cmap = sns.diverging_palette(12, 210, as_cmap=True)
    # cmap = sns.diverging_palette(16, 240, as_cmap=True)

    vol = cortex.Volume(
        vals, 'UT' + subject, xfmname=f'UT{subject}_auto', vmin=-vabs, vmax=vabs, cmap=cmap)

    cortex.quickshow(vol,
                     with_rois=with_rois,
                     with_labels=False,
                     with_borders=with_borders,
                     with_colorbar=clab == None,  # if not None, save separate cbar
                     )
    os.makedirs(dirname(fname_save), exist_ok=True)
    plt.savefig(fname_save)
    plt.close()

    # save cbar
    norm = Normalize(vmin=-vabs, vmax=vabs)
    # need to invert this to match above
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig, ax = plt.subplots(figsize=(5, 0.35))
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    if clab:
        cbar.set_label(clab, fontsize='x-large')
        plt.savefig(fname_save.replace('flatmap.pdf',
                    'cbar.pdf'), bbox_inches='tight')
    plt.close()


def get_corrs_single_question(results_dir='/home/chansingh/mntv1/deep-fMRI/encoding/may15_single_question', subject='UTS03'):
    r = imodelsx.process_results.get_results_df(results_dir)
    return r[r.subject == subject]['corrs_test_selected'].values[0]


if __name__ == '__main__':
    # ['qa_embedder', 'bert', 'single_question', 'llama']
    # feature_spaces = ['single_question']
    feature_spaces = ['llama']
    vabs = 0.15



    results_dir = config.BEST_RESULTS_DIR_ENSEMBLE
    # out_dir = join(config.REPO_DIR, 'results', 'qa', 'encoding_flatmap_diffs')
    out_dir = 'flatmaps_diffs'
    os.makedirs(out_dir, exist_ok=True)
    # load the results in to a pandas dataframe
    # r, cols_varied, mets = analyze_helper.load_clean_results(results_dir)
    data = joblib.load(join(config.RESULTS_DIR_LOCAL, 'results_best_ensemble.pkl'))
    r, cols_varied, mets = data['r'], data['cols_varied'], data['mets']
    r = r[r.qa_questions_version.isin(['', 'v3_boostexamples'])]
    r = r[r.num_stories == -1]
    r = r[r.feature_selection_alpha == -1]
    r = r[~r.feature_space.isin(
        ['meta-llama/Llama-2-7b-hf', 'meta-llama/Meta-Llama-3-8B'])]
    r = r[~((r.feature_space == 'qa_embedder') &
            (r.qa_embedding_model != 'ensemble1'))]
    metric_sort = 'corrs_tune_pc_weighted_mean'

    for subject in tqdm(['S03', 'S02', 'S01']):
        args_qa = r[
            (r.subject == subject) *
            (r.feature_space.str.contains('qa_embedder'))
        ].sort_values(by=metric_sort, ascending=False).iloc[0]
        # , 'llama']:
        for feature_space in feature_spaces:
            corrs = []

            if feature_space == 'single_question':
                corrs_baseline = get_corrs_single_question(
                    subject='UT' + subject)

            else:
                corrs_baseline = r[
                    # (r.feature_space.str.contains('bert'))
                    (r.feature_space.str.contains(feature_space)) *
                    (r.subject == subject)
                    # (r.ndelays == 8)
                ].sort_values(by=metric_sort, ascending=False).iloc[0]['corrs_test']

            print('means', 'qa', args_qa['corrs_test'].mean(
            ), 'baseline', corrs_baseline)

            # fname_save = join(out_dir, f'diff_bert-qa.png')

            lab_name_dict = {
                'qa_embedder': 'QA-Emb',
                'bert': 'BERT',
                'llama': 'LLaMA',
                'single_question': 'Single question'
            }
            clab = f'Test correlation ({lab_name_dict.get(feature_space, feature_space)})'
            fname_save = join(
                out_dir, f'{subject}_{feature_space.replace("qa_embedder", "qa")}_flatmap.pdf')
            _save_flatmap(corrs_baseline, subject, fname_save, clab=clab, vabs=vabs)

            if not feature_space == 'qa_embedder':
                fname_save = join(
                    out_dir, f'{subject}_qa-{feature_space.replace("qa_embedder", "qa")}_flatmap.pdf')
                clab = f'Test correlation (QA - {lab_name_dict.get(feature_space, feature_space)})'
                _save_flatmap(
                    args_qa['corrs_test'] - corrs_baseline, subject, fname_save, clab=clab, cmap='RdBu_r', vabs=vabs)
