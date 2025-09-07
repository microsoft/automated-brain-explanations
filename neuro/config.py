import os.path
from os.path import abspath, dirname, expanduser, join

path_to_file = os.path.dirname(abspath(__file__))
REPO_DIR = dirname(path_to_file)
if 'chansingh' in expanduser('~'):
    MNT_DIR = '/home/chansingh/mntv1'
else:
    MNT_DIR = '/mntv1'

FMRI_DIR_BLOB = join(MNT_DIR, 'deep-fMRI')
RESULTS_DIR_LOCAL = join(REPO_DIR, 'results')
NEUROSYNTH_DATA_DIR = join(FMRI_DIR_BLOB, 'qa', 'neurosynth_data')

# save fitted model
BEST_RESULTS_DIR_ENSEMBLE = join(FMRI_DIR_BLOB, 'encoding', 'may7')
SPARSE_FEATS_DIR = join(FMRI_DIR_BLOB, 'qa', 'sparse_feats_shared')

STORIES_DIR_GCT = join(RESULTS_DIR_LOCAL, "stories")
SAVE_DIR_FMRI = join(FMRI_DIR_BLOB, 'sasc', 'rj_models')
CACHE_DIR = join(FMRI_DIR_BLOB, 'sasc', 'mprompt', 'cache')
PILOT_STORY_DATA_DIR = join(FMRI_DIR_BLOB, 'brain_tune/story_data')

CACHE_NGRAMS_DIR = join(FMRI_DIR_BLOB, 'sasc/mprompt/cache/cache_ngrams')
REGION_IDXS_DIR = join(FMRI_DIR_BLOB, 'sasc/brain_regions')


PROCESSED_DIR = join(FMRI_DIR_BLOB, 'qa', 'processed')
CACHE_EMBS_DIR = join(FMRI_DIR_BLOB, 'qa', 'cache_embs')
CACHE_EMBS_AGENT_DIR = join(FMRI_DIR_BLOB, 'qa', 'cache_embs_agent')
RESP_PROCESSING_DIR = join(FMRI_DIR_BLOB, 'qa', 'resp_processing_full')
GEMV_RESPS_DIR = join(FMRI_DIR_BLOB, 'brain_tune', 'story_data')
CACHE_OPENAI_AGENT_DIR = join(FMRI_DIR_BLOB, 'qa', 'cache_openai_agent')

EM_DATA_DIR = join(REPO_DIR, 'data', 'em_data')
NLP_UTILS_DIR = join(FMRI_DIR_BLOB, 'nlp_utils')

DECODING_DIR = join(REPO_DIR, 'data', 'decoding')

# agent stuff
HYPOTHESAES_RESULTS_DIR = join(FMRI_DIR_BLOB, 'qa', 'hypothesaes')


############## ECOG ###################
ECOG_DIR = join(MNT_DIR, 'ecog')


def setup_freesurfer():
    # set os environ SUBJECTS_DIR
    FREESURFER_VARS = {
        'FREESURFER_HOME': os.path.expanduser('~/freesurfer'),
        'FSL_DIR': os.path.expanduser('~/fsl'),
        'FSFAST_HOME': os.path.expanduser('~/freesurfer/fsfast'),
        'MNI_DIR': os.path.expanduser('~/freesurfer/mni'),
        # 'SUBJECTS_DIR': join(repo_dir, 'notebooks_gt_flatmaps'),
        'SUBJECTS_DIR': os.path.expanduser('~/freesurfer/subjects'),
        # add freesurfer bin to path
        'PATH': os.path.expanduser('~/freesurfer/bin') + ':' + os.environ['PATH'],
    }
    for k in FREESURFER_VARS.keys():
        os.environ[k] = FREESURFER_VARS[k]
