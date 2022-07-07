import copy
from pathlib import Path
import yaml
from yaml.loader import BaseLoader
import json


def remove1(name, samples):
    keep, remove = [], []
    for s in samples:
        if name == s:
            remove.append(s)
        else:
            keep.append(s)

    return keep, remove


def remove2(name, samples):
    keep, remove = [], []
    for s in samples:
        if name in s[0]:
            remove.append(s)
        else:
            keep.append(s)

    return keep, remove




if __name__ == '__main__':
    with Path('data/MCD/build_datasets/train_test_dataset.json').open('r') as f_in:
        a = json.load(f_in)

    t = copy.deepcopy(a)

    train_to_test = ['black_and_decker_lithium_drill_driver',
                'domino_sugar_1lb',
                'frenchs_classic_yellow_mustard_14oz',
                'pringles_original',
                'rubbermaid_ice_guard_pitcher_blue',
              ]

    test_to_train = ['block_of_wood_6in'
        , 'cheerios_14oz'
        , 'melissa_doug_farm_fresh_fruit_banana'
        , 'play_go_rainbow_stakin_cups_2_orange'
                     ]

    remove_from_train = ['soft_scrub_2lb_4oz', 'avocado_poisson_000']



    for target in train_to_test:
        keep, remove = remove1(target, t['train_model_names'])
        t['train_model_names'] = keep
        t['holdout_model_names'] += remove

        keep, remove = remove2(target, t['train_models_train_views'])
        t['train_models_train_views'] = keep
        t['holdout_models_holdout_views'] += remove

        keep, remove = remove2(target, t['train_models_holdout_views'])
        t['train_models_holdout_views'] = keep
        t['holdout_models_holdout_views'] += remove

    for target in test_to_train:
        keep, remove = remove1(target, t['holdout_model_names'])
        t['holdout_model_names'] = keep
        t['train_model_names'] += remove

        keep, remove = remove2(target, t['holdout_models_holdout_views'])
        t['holdout_models_holdout_views'] = keep
        t['train_models_train_views'] += remove

    for target in remove_from_train:
        keep, _ = remove1(target, t['train_model_names'])
        t['train_model_names'] = keep

        keep, _ = remove2(target, t['train_models_train_views'])
        t['train_models_train_views'] = keep

        keep, _ = remove2(target, t['train_models_holdout_views'])
        t['train_models_holdout_views'] = keep

    with Path('data/MCD/build_datasets/grasping_dataset.json').open('w') as f_in:
        json.dump(t, f_in)