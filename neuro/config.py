import os.path
from os.path import abspath, dirname, expanduser, join

path_to_file = os.path.dirname(abspath(__file__))

# local stuff that comes with the repo
REPO_DIR = dirname(path_to_file)
EM_DATA_DIR = join(REPO_DIR, 'data', 'em_data')
DECODING_DIR = join(REPO_DIR, 'data', 'decoding')
RESULTS_DIR_LOCAL = join(REPO_DIR, 'results')
STORIES_DIR_GCT = join(RESULTS_DIR_LOCAL, "gct_stories")

if 'chansingh' in expanduser('~'):
    MNT_DIR = '/home/chansingh/mntv1'
else:
    MNT_DIR = '/mntv1'

FMRI_DIR_BLOB = join(MNT_DIR, 'deep-fMRI')

# save fitted model
BEST_RESULTS_DIR_ENSEMBLE = join(FMRI_DIR_BLOB, 'encoding', 'may7')
SPARSE_FEATS_DIR = join(FMRI_DIR_BLOB, 'qa', 'sparse_feats_shared')

# gct stuff
PROCESSED_DIR = join(FMRI_DIR_BLOB, 'qa', 'processed')
GCT_RESPS_DIR = join(FMRI_DIR_BLOB, 'brain_tune', 'story_data')

# general stuff
SAVE_DIR_FMRI = join(FMRI_DIR_BLOB, 'sasc', 'rj_models')
PILOT_STORY_DATA_DIR = join(FMRI_DIR_BLOB, 'brain_tune/story_data')
REGION_IDXS_DIR = join(FMRI_DIR_BLOB, 'sasc/brain_regions')

# qa stuff
CACHE_EMBS_DIR = join(FMRI_DIR_BLOB, 'qa', 'cache_embs')
CACHE_EMBS_AGENT_DIR = join(FMRI_DIR_BLOB, 'qa', 'cache_embs_agent')
RESP_PROCESSING_DIR = join(FMRI_DIR_BLOB, 'qa', 'resp_processing_full')
NEUROSYNTH_DATA_DIR = join(FMRI_DIR_BLOB, 'qa', 'neurosynth_data')
# copy over the best results from the repo_dir
# !rclone copy box:DeepTune/QA/cached_results/results_best_ensemble.pkl results/
# !rclone copy box:DeepTune/QA/cached_results/results_full_oct17.pkl results/


# agent stuff
HYPOTHESAES_RESULTS_DIR = join(FMRI_DIR_BLOB, 'qa', 'hypothesaes')
CACHE_OPENAI_AGENT_DIR = join(FMRI_DIR_BLOB, 'qa', 'cache_openai_agent')


############## ECOG ###################
ECOG_DIR = join('/home/chansingh/kalytv', 'ecog')


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
