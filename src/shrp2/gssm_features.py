"""
GSSM EVENT FEATURES from the released SafetyCriticalTestSet (Amendment 4, Section 5).

Why this wrapper exists: the authors' src_safety_evaluation/organise_events.py builds event_features.npz (needed by
evaluate_safety.py) only on its raw-data path. When event_data.h5 already exists - the released test set - the
variable `environment` is never assigned and the script stops with UnboundLocalError (found 11 Oct 2026 on
synthetic data in the released format). This wrapper runs the SAME feature lines of organise_events.main (its
"Save event features" block, copied below with only `environment` read from the released environment.csv, which is
exactly what the raw-data path writes) and the authors' own functions get_context_representations,
set_veh_dimensions and create_categorical_encoder. Output: ResultData/EventData/<category>/event_features.npz,
byte-for-byte the structure evaluate_safety.py reads. Run it from the GSSM workspace (gssm_workspace.py) with the
GSSM repository on the path:
  python <project>/src/shrp2/gssm_features.py --gssm <project>/baselines/GSSM     (cwd = the workspace)
"""
import argparse, os, sys
import numpy as np
import pandas as pd


def main(gssm_repo, path_result="ResultData/", event_table="RawData/SHRP2/FileToUse/InsightTables/Event_Table.csv"):
    sys.path.insert(0, os.path.abspath(gssm_repo))
    from src_safety_evaluation.organise_events import create_categorical_encoder          # authors' code
    from src_safety_evaluation.validation_utils.utils_evaluation import set_veh_dimensions
    from src_safety_evaluation.validation_utils.utils_features import get_context_representations
    np.random.seed(131)                                                                   # as organise_events
    events = pd.read_csv(event_table).set_index("eventID")
    environment_feature_names = ["lighting", "weather", "surfaceCondition", "trafficDensity"]
    one_hot_encoder = create_categorical_encoder(events, environment_feature_names)
    path_save = path_result + "EventData/"
    made = {}
    for event_cat in sorted(os.listdir(path_save)):
        event_meta = pd.read_csv(path_save + f"{event_cat}/event_meta.csv").set_index("event_id")
        data = pd.read_hdf(path_save + f"{event_cat}/event_data.h5", key="data")
        environment = pd.read_csv(path_save + f"{event_cat}/environment.csv").set_index("event_id")
        # ---- from organise_events.main, block "Save event features" (unchanged except the indent) ----
        avg_width = np.nanmean(event_meta['ego_width'].values)
        avg_length = np.nanmean(event_meta['ego_length'].values)
        veh_dimensions = set_veh_dimensions(event_meta, avg_width, avg_length)
        profiles_features = []; current_features = []; spacing_list = []; event_id_list = []
        event_meta = event_meta[event_meta['duration_enough'] & (event_meta['conflict'] != 'none')]
        target_ids = data[data['event_id'].isin(event_meta.index.values)].index.unique(level='target_id').values
        for target_id in target_ids:
            df = data.loc(axis=0)[target_id, :]
            if len(df) < 30:  # skip if the target was detected for less than 3 seconds
                continue
            segmented_features = get_context_representations(df, veh_dimensions.loc[df['event_id'].values[0]])
            profiles_features.append(segmented_features[0])
            current_features.append(segmented_features[1])
            spacing_list.append(segmented_features[2])
            event_id_list.append(segmented_features[3])
        if not current_features:
            made[event_cat] = 0
            continue
        profiles_features = np.concatenate(profiles_features, axis=0)
        current_features = np.concatenate(current_features, axis=0)
        spacing_list = np.concatenate(spacing_list, axis=0)
        event_id_list = np.concatenate(event_id_list, axis=0)
        environment_features = one_hot_encoder.transform(environment.loc[event_id_list[:, 0]].values)
        assert profiles_features.shape[0] == len(spacing_list) and profiles_features.shape[1] == 25
        np.savez(path_save + f'{event_cat}/event_features.npz',
                 profiles=profiles_features, current=current_features, environment=environment_features,
                 spacing=spacing_list, event_id=event_id_list)
        made[event_cat] = int(len(spacing_list))
    print("event features (moments per category):", made)
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gssm", required=True, help="path to the GSSM repository")
    main(ap.parse_args().gssm)
