import json
from os.path import join
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from neuro import config

VOXEL_DICT = {
    "UTS02": [
        # 8 voxels with 3 reps is ~15 mins
        337,
        122,
        168,
        171,
        79,
        299,
        368,
        398,  # automatically picked (clustered then ran synthetic stories)
        # 9 handpicked
        426,  # laughter
        155,  # death
        179,  # emotion
        248,  # negativity
        212,  # counting time
        339,  # physical injury or trauma
        154,  # hair and clothing
        442,  # birthdays
        342,  # rejection
    ],
    "UTS03": [
        337,  # love and joy (0.199992)
        160,  # age (0.243015)
        173,  # conflict resolution (0.245964)
        403,  # negative experiences (0.247915)
        99,  # body language (0.250279)
        280,  # communication (0.252942)
        395,  # vomiting, sickness (0.258195)
        9,  # numbers (0.263722)
        408,  # numbers or measurements (0.277489)
        158,  # action or movement (0.277547)
        458,  # agreement and questioning (0.278527)
        466,  # food and drinks (0.296890)
        109,  # age (0.299564)
        342,  # locations (0.300802)
        152,  # movement or action (0.300931)
        148,  # physical injury (0.321311)
        368,  # family and relationships (0.367439)
    ],
    "UTS01_original": [
        # 322,  # embarrassment (0.125209)
        # 311,  # vulgarities (0.132738)
        # 258,  # lonelineness (0.135058)
        # 469,  # physical contact (0.137269)
        # 106,  # communication (0.154657)
        # 162,  # communication (0.174878)
        484,  # repetition (0.179070)
        171,  # physical movement (0.182128)
        186,  # location (0.184146)
        378,  # college days (0.199483)
        458,  # numbers (0.213608)
        100,  # locations (0.214551)
        451,  # speaking (0.218006)
        187,  # relationships (0.221229)
        # 332,  # family and friends (0.233957)
        121,  # movement (0.235266)
        149,  # clothing (0.238112)
        # 192,  # family relationships (0.243154)
        39,  # physical movement (0.253595)
        232,  # violence and injury (0.266177)
        107,  # family and friends (0.285960)
        153,  # movement (0.286167)
        144,  # family relationships (0.298463)
        203,  # food and drink (0.313123)
        473,  # food (0.351168)
    ],
    "UTS01": [
        # 322,  # embarrassment (0.125209)
        # 311,  # vulgarities (0.132738)
        258,  # lonelineness (0.135058)
        469,  # physical contact (0.137269)
        # 106,  # communication (0.154657)
        162,  # communication (0.174878)
        484,  # repetition (0.179070)
        # 171,  # physical movement (0.182128)
        # 186,  # location (0.184146)
        378,  # college days (0.199483)
        458,  # numbers (0.213608)
        100,  # locations (0.214551)
        157,  # specific times (0.217809)
        451,  # speaking (0.218006)
        # 187,  # relationships (0.221229)
        # 332,  # family and friends (0.233957)
        # 121,  # movement (0.235266)
        149,  # clothing (0.238112)
        # 192,  # family relationships (0.243154)
        39,  # physical movement (0.253595)
        232,  # violence and injury (0.266177)
        107,  # family and friends (0.285960)
        153,  # movement (0.286167)
        144,  # family relationships (0.298463)
        203,  # food and drink (0.313123)
        473,  # food (0.351168)
    ],
}


INTERACTIONS_DICT = {
    "UTS01_original": [],  # overwrote these
    "UTS01": [
        # very related (0.74)
        (100, 149),  # location, clothing
        # high related (~0.5)
        (484, 473),  # repetition, food
        # medium (0.25)
        (153, 144),  # movement, family relationships
        # low (0.19)
        (144, 157),  # family relationships, specific times
        # very low (~0)
        (451, 258),  # speaking, loneliness
    ],
    "UTS02": [
        # related (>0.4)
        (212, 171),  # time, measurements
        # medium (~0.2)
        (339, 337),  # physical injury or trauma, moments
        (426, 442),  # laughter, birthdays
        # unrelated (< -0.3)
        (122, 299),  # locations, communication
        (398, 79),  # emotional expression, food preparation
    ],
    "UTS03": [
        # very related (0.63)
        (466, 158),  # food and drinks, action or movement
        # pretty related (0.38)
        (152, 403),  # movement or action, negative experiences
        # medium (0.24)
        (458, 9),  # agreement and questioning, numbers
        # low (0.1)
        (173, 109),  # conflict resolution, age
        # very low (~0)
        (342, 99),  # locations, body language
    ],
}


def _voxels_to_rows(
    voxels: List[Tuple], polysemantic_ngrams: Dict = None
) -> pd.DataFrame:
    """Add extra data (like ngrams) to each row"""
    r = pd.read_pickle(
        join(config.RESULTS_DIR, "sasc", "fmri_results_merged.pkl"))
    # put all voxel data into rows DataFrame
    rows = []
    expls = []
    for vox in voxels:
        expl, subj, vox_num = vox
        vox_num = int(vox_num)
        # try:
        row = r[(r.subject == subj) & (r.module_num == vox_num)].iloc[0]
        if polysemantic_ngrams is not None:
            row["top_ngrams_module_correct"] = polysemantic_ngrams[tuple(vox)]
        rows.append(row)
        expls.append(expl)

        # except:
        # print("skipping", vox)
    rows = pd.DataFrame(rows)
    rows["expl"] = expls
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', 200):
    # display(rows[['subject', 'module_num', 'expl', 'top_explanation_init_strs', 'top_ngrams_module_correct']])
    return rows


def get_rows_voxels(subject: str, setting="default", fname_suffix=""):
    """Select rows from fitted voxels

    Params
    ------
    subject : str
        UTS01, UTS02, UTS03
    setting : str
        default, interactions
    """

    if setting == "qa":
        # return joblib.load(
        # join(config.REPO_DIR, "notebooks_stories/0_voxel_select/rows_qa_may31.pkl")
        # )
        return joblib.load(
            join(config.REPO_DIR,
                 f"notebooks/1_gct_stories/0_voxel_select/rows_qa_mar9_2025{fname_suffix}.pkl")
        )

    elif setting == "roi" or setting == "roi_steered":
        if subject == 'UTS02':
            roi_rows_file = join(
                config.REPO_DIR, f"notebooks/1_gct_stories/0_voxel_select/rows_roi_{subject.lower()}_may31.pkl")
        else:
            raise ValueError('Double-check with notebooks')
            # roi_rows_file = join
                # config.REPO_DIR,
                # f"notebooks/1_gct_stories/0_voxel_select/rows_roi_{subject.lower()}_nov30{fname_suffix}.pkl",
            # )
        return joblib.load(roi_rows_file)

    # UTS02 - Pilot voxels
    if setting in ["default", "interactions", "default_steered"]:
        VOXEL_DICT_FNAMES = {
            "UTS02": "notebooks/1_gct_stories/0_voxel_select/default/uts02_concepts_pilot_mar22.json",
            "UTS01": "notebooks/1_gct_stories/0_voxel_select/default/uts01_concepts_pilot_jun14.json",
            "UTS03": "notebooks/1_gct_stories/0_voxel_select/default/uts03_concepts_pilot_jun14.json",
        }

    elif setting == "polysemantic":
        VOXEL_DICT_FNAMES = {
            k: f"notebooks/1_gct_stories/0_voxel_select/polysemantic/polysemantic_{k}.json"
            for k in ["UTS01", "UTS02", "UTS03"]
        }
    voxels_dict = json.load(
        open(join(config.REPO_DIR, VOXEL_DICT_FNAMES[subject]), "r"))
    vals = pd.DataFrame([tuple(x)
                        for x in sum(list(voxels_dict.values()), [])])
    vals.columns = ["expl", "subject", "module_num"]

    if setting in ["default", "polysemantic", "default_steered"]:
        # filter vals
        if setting == "default" or setting == "default_steered":
            voxel_nums = VOXEL_DICT[subject]
            print(
                "len(voxel_nums)",
                len(voxel_nums),
                "nunique",
                len(np.unique(voxel_nums)),
            )
            vals = vals[vals["module_num"].isin(voxel_nums)]
            for voxel_num in voxel_nums:
                if voxel_num not in vals["module_num"].values:
                    print("missing", voxel_num)
            assert vals["module_num"].nunique(
            ) == vals.shape[0], "no duplicates"
            assert len(vals) == len(voxel_nums), "all voxels found"
            assert len(voxel_nums) == 17
        else:
            print(
                "len(vals)", len(
                    vals), "nunique voxels", vals["module_num"].nunique()
            )

        # add extra data (like ngrams) to each row
        if setting == "polysemantic":
            polysemantic_ngrams = joblib.load(
                join(
                    config.REPO_DIR,
                    f"notebooks/1_gct_stories/0_voxel_select/polysemantic/polysemantic_ngrams_{subject}.pkl",
                )
            )
        else:
            polysemantic_ngrams = None
        rows = _voxels_to_rows(
            vals.values, polysemantic_ngrams=polysemantic_ngrams)
        return rows

    elif setting == "interactions":
        # interactions
        rows_list = []
        for i in range(2):
            voxel_nums = [x[i] for x in INTERACTIONS_DICT[subject]]
            print(
                i,
                "len(voxel_nums)",
                len(voxel_nums),
                "nunique",
                len(np.unique(voxel_nums)),
            )
            v = pd.concat(vals[vals["module_num"] == x] for x in voxel_nums)
            for voxel_num in voxel_nums:
                if voxel_num not in v["module_num"].values:
                    print("missing", voxel_num)
            assert v["module_num"].nunique() == v.shape[0], "no duplicates"
            assert len(v) == len(voxel_nums), "all voxels found"
            rows_list.append(_voxels_to_rows(v.values))
        return tuple(rows_list)
