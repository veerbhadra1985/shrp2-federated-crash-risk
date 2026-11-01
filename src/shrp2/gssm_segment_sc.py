"""
S-C SEGMENTATION (Amendment 4, Section 5: S-C, the secondary GSSM model, retrained on the safe baseline trips of OUR
training split only). Runs the authors' segmentation (src_data_preparation/segment_datasets.py, main) for the
dataset 'SafeBaseline' only.

Why this wrapper exists: the authors' main() loops over highD, SafeBaseline and ArgoverseHV and stops when a dataset
is missing; S-C needs only SafeBaseline. The loop body below is copied from main() unchanged, and it
calls the authors' own read_dataset and ContextSegmenter; the copy is byte for byte (checked by a test). Their
random choices all use RandomState(131), as there.

Run it from a TRAINING-split GSSM workspace (gssm_workspace.py --split train), never from the validation or test
workspace: the authors' read_dataset draws random object sizes from the metadata of the events in the workspace,
so only training-split events may be there.
  cd data\\gssm_ws_train
  <GPY> ..\\..\\src\\shrp2\\gssm_segment_sc.py --gssm ..\\..\\baselines\\GSSM
Then train with the authors' unchanged script, from the same folder:
  <GPY> ..\\..\\baselines\\GSSM\\src_posterior_inference\\pi_train_eval.py --stage 1
(stage 1 = SafeBaseline/current, highD/current, ArgoverseHV/current; the latter two have no data here and are skipped
by the authors' own try/except). The trained model is PreparedData/PosteriorInference/SafeBaseline/current/*.pth.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np


def main(gssm_repo, path_prepared="./PreparedData/Segments/", path_processed="./ProcessedData/", manual_seed=131):
    man = Path("workspace_manifest.json")
    if not man.exists() or json.loads(man.read_text()).get("split") != "train":
        raise SystemExit("run this from a TRAINING-split workspace (gssm_workspace.py --split train)")
    sys.path.insert(0, os.path.abspath(os.path.join(gssm_repo, "src_data_preparation")))
    from represent_utils.utils_data_segmentation import ContextSegmenter, read_dataset          # authors' code
    np.random.seed(manual_seed)                                                                 # as their __main__
    os.makedirs(path_prepared, exist_ok=True)
    for dataset in ['SafeBaseline']:
        # ---- segment_datasets.main, loop body: copied byte for byte (test S1 checks this) ----
        path_save = f'{path_prepared}{dataset}/'
        os.makedirs(path_save, exist_ok=True)
        if os.path.exists(path_save + f'profiles_{dataset}_val.h5'):
            print(f'{dataset} set already segmented.')
            continue

        print('Loading data...')
        data_both = read_dataset(dataset, path_processed, manual_seed)
        # Separate all the events into train (80%) and val (20%) sets, the test set will be (near-)crashes in SHRP2
        '''
        highD: 420,077 train scenes (min. dist. 1.14 m) + 102,853 val scenes (min. dist. 2.84 m)
        SafeBaseline: 427,674 train scenes (min. dist. 1.78 m) + 105,671 val scenes (min. dist. 2.59 m)
        ArgoverseHV: 439,175 train scenes (min. dist. 0.04 m) + 109,462 val scenes (min. dist. 0.20 m)
        '''
        event_ids = data_both['event_id'].unique()
        len_event_ids = len(event_ids)
        train_event_ids = np.random.RandomState(manual_seed).choice(event_ids, int(0.8*len_event_ids), replace=False)
        val_event_ids = np.setdiff1d(event_ids, train_event_ids)
        data_train = data_both[data_both['event_id'].isin(train_event_ids)]
        data_val = data_both[data_both['event_id'].isin(val_event_ids)]

        # Segment and save scenes
        initial_scene_id = 0
        for data, suffix in zip([data_train, data_val], ['train', 'val']):
            print('Segmenting ' + dataset + ' ' + suffix + ' set...')
            segmenter = ContextSegmenter(data, initial_scene_id, dataset, manual_seed)
            print(f'{suffix} current examples:', segmenter.current_features_set.iloc[:5].to_string(), '\n', segmenter.current_features_set.describe().to_string())
            print(f'{suffix} profiles examples:', segmenter.profiles_set.iloc[:5].to_string(), '\n', segmenter.profiles_set.describe().to_string())
            segmenter.profiles_set.to_hdf(path_save + f'profiles_{dataset}_{suffix}.h5', key='profiles')
            segmenter.current_features_set.to_hdf(path_save + f'current_features_{dataset}_{suffix}.h5', key='features')
            if dataset=='SafeBaseline':
                print(f'{suffix} environment features examples:', segmenter.environment_features_set.iloc[:5].to_string(), '\n', segmenter.environment_features_set.describe().to_string())
                segmenter.environment_features_set.to_hdf(path_save + f'environment_features_{dataset}_{suffix}.h5', key='features')
            initial_scene_id = segmenter.current_features_set['scene_id'].max() + 1

            print('--------------------- ' + dataset + ' ' + suffix + ' set segmented ---------------------')
            print(f'Number of scenes: {initial_scene_id - segmenter.initial_scene_id}')
            print(f"Minimum dist.: {segmenter.current_features_set['s'].min():.2f}")
            print(f"Unique scene ids in current features set: {segmenter.current_features_set['scene_id'].nunique()}, should be the same as the profiles set: {segmenter.profiles_set['scene_id'].nunique()}")
            print('--------------------------------------------------------------------')
    seg = Path(path_prepared) / "SafeBaseline"
    return {k: int(len(__import__("pandas").read_hdf(seg / f"current_features_SafeBaseline_{k}.h5", "features"))) for k in ("train", "val")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gssm", required=True, help="path to the GSSM repository")
    print(main(ap.parse_args().gssm))
