"""
Single source of truth for the SHRP2 bird's-eye dataset layout (Jiao & Calvert, 2025; complete version
doi 10.15787/VTT1/EFYEJR; public version doi 10.15787/VTT1/T7UUC1).

Every fact below was taken from the dataset authors' own documentation, checked on 11 Oct 2026 BEFORE any
SHRP2 data were opened:
  [D1] OverallDataDictionary.pdf (complete version) and OverallDataDictionary_PublicVersion.pdf,
       github.com/Yiru-Jiao/BirdsEyeTrajectoryReconstructionSHRP2NDS/pdfs/
  [D2] README.md of the same repository (event counts, structure, 10 Hz, environment only for
       safety-critical events, no perception data)
  [D3] MethodExplanation.pdf (EKF reconstruction, 0.1 s grid, test-set filters -> 4,875 events)
  [D4] GSSM code, github.com/Yiru-Jiao/GSSM: organise_metadata.py (metadata_birdseye.csv columns),
       reconstruct_birdseye.py (h5 key 'data', SafeBaseline in chunks _0.._4, 'event' flag =
       eventStart..eventEnd), organise_events.py (test set = non-baseline events; environment.csv;
       duration_enough = an object present >= 50 frames), utils_evaluation.py (inclusion =
       duration_enough & conflict != 'none'), utils_ego_sur.py (object x, y = radar point placed
       relative to the subject's FRONT bumper; not shifted to the object's centre)
Anything the documentation does not settle is listed in OPEN_QUESTIONS and is checked by the audit.
"""

H5_KEY = "data"                                                     # [D4]
HZ = 10                                                             # [D2], [D3]
TS_PER_S = 1000                                                     # raw timestamp: 1,000 = 1 s [D1]

# ---- ReconstructedTrajectories (all trips) ----
RT_DIR = "ReconstructedTrajectories"
RT_FILES = ["metadata_birdseye.csv", "event_counts.csv", "ekf_parameters.csv"]
EGO_FILE, SUR_FILE = "Ego_birdseye", "Surrounding_birdseye"         # + ".h5" or "_<chunk>.h5" [D4]
BASELINE_CATEGORY = "SafeBaseline"                                  # [D4]

EGO_COLS_PUBLIC = ["event_id", "timestamp", "time", "x_ekf", "y_ekf", "psi_ekf", "v_ekf", "omega_ekf",
                   "acc_ekf", "event"]                              # [D1] public
EGO_COLS_RAW_ONLY = ["speed_comp", "yaw_rate", "acc_lat", "acc_lon", "brake", "wheel_steering",
                     "turn_signal"]                                 # [D1] complete version only
SUR_COLS_PUBLIC = ["event_id", "target_id", "timestamp", "time", "x_ekf", "y_ekf", "v_ekf", "psi_ekf"]
SUR_COLS_RAW_ONLY = ["local_dx", "local_dy", "delta_vx", "delta_vy", "x", "y", "speed_comp"]

# metadata_birdseye.csv, as written by GSSM organise_metadata.py + reconstruct_birdseye.py [D4]
META_COLS = ["event_id", "event_category", "first", "second", "ego_width", "ego_length",
             "target_width", "target_length", "other_width", "other_length",
             "ego_reconstructed", "surrounding_reconstructed"]
META_COLS_OPTIONAL = ["time_series_das", "time_series_honda", "file_dir", "file2use", "note"]

# ---- SafetyCriticalTestSet (crashes and near-crashes only) ----
ST_DIR = "SafetyCriticalTestSet"
EVENT_META_COLS = ["event_id", "event_category", "first", "second", "ego_width", "ego_length",
                   "target_width", "target_length", "other_width", "other_length", "start_timestamp",
                   "end_timestamp", "impact_timestamp", "reaction_timestamp", "severity_first",
                   "severity_second", "conflict", "duration_enough"]                    # [D1]
EVENT_META_COMPLETE_ONLY = ["narrative"]
ENV_COLS = ["event_id", "lighting", "weather", "surfaceCondition", "trafficDensity"]  # complete only [D1]
EVENT_DATA_COLS = ["event_id", "x_ego", "y_ego", "v_ego", "psi_ego", "acc_ego", "hx_ego", "hy_ego",
                   "x_sur", "y_sur", "v_sur", "psi_sur", "hx_sur", "hy_sur"]          # index target_id, time

# ---- registered definitions (pre-registration osf.io/y5zrp; Amendments 1-3) ----
HORIZONS_S = [1, 2, 3, 5]
PRIMARY_H = 3
HISTORY_FRAMES = 20                     # 2 s at 10 Hz, window t-19..t (as build_samples.py: first t = start + 19)
RETENTION_SHARE, RETENTION_EXTRA_S = 0.80, 2.0
BUFFER_S = (5.0, 7.0)                   # time to event in (5, 7] s excluded (registered "between 5 s and 7 s")
MIN_EVAL_UNIT = 100                     # Amendment 2: unit eligible with >= 100 frames evaluable at 3 s
EPS_LEVELS = {"0.05": 0.05, "0.10": 0.10, "op": 1.0 / 36000}
IMPLAUSIBLE = dict(speed=60.0, acc=15.0, acc_crash_window_s=1.0, jump=10.0, trip_share=0.10)
PUBLISHED = dict(test_set_events=4875, crashes=282, near_crashes=4593,            # Jiao et al. 2026
                 baselines_reconstructed=10919, events_reconstructed=8111)        # [D2]
SEVERITY = {0: "not applicable", 1: "crash-relevant", 2: "near-crash", 3: "crash"}  # [D1]

# Outcome type of an event trip = its SHRP2 event category, mapped as in the GSSM code
# (src_visualisation/visual_utils/utils_tabfig.py), which gives the published 282 crashes / 4,593 near-crashes.
# Frozen 20 Oct 2026 as clarification C25, after the audit and BEFORE any sample, split or model. The coded
# severities (severity_first/second) are not used for the outcome type. A category not listed here stops the audit.
OUTCOME_BY_CATEGORY = {
    BASELINE_CATEGORY: "baseline",
    "Crash": "crash", "Crash-Crash": "crash", "Crash-CrashRelevant": "crash", "Crash-NearCrash": "crash",
    "Crash-OtherConflict": "crash", "CrashRelevant-Crash": "crash", "NearCrash-Crash": "crash",
    "SecondaryCrash": "crash",
    "NearCrash": "near-crash", "NearCrash-NearCrash": "near-crash", "NearCrash-CrashRelevant": "near-crash",
    "NearCrash-OtherConflict": "near-crash", "CrashRelevant-NearCrash": "near-crash",
    "SecondaryNearCrash": "near-crash",
}

# Binary domain factors (registration Section 3). Category strings exactly as listed in [D1].
# 1 = the "shifted" level; None = unknown -> excluded from that factor only (registered rule).
FACTOR_MAPS = {
    "lighting": {"Daylight": 0, "Dawn": 1, "Dusk": 1, "Darkness, lighted": 1, "Darkness, not lighted": 1,
                 "Unknown": None},
    "weather": {"No Adverse Conditions": 0, "Fog": 1, "Sleeting": 1, "Mist/Light Rain": 1, "Rain and Fog": 1,
                "Raining": 1, "Snow/Sleet and Fog": 1, "Snowing": 1, "Unknown": None},
    "surfaceCondition": {"Dry": 0, "Wet": 1, "Gravel over Asphalt": 1, "Gravel/Dirt Road": 1, "Muddy": 1,
                         "Icy": 1, "Snowy": 1, "Unknown": None, "Other": None},
    # level of service: A-B = 0, C or worse = 1 (registration). [D1] lists "A1 to F"; exact strings are
    # checked by the audit, which prints every value it sees; any value not matched here is reported.
    "trafficDensity": "LOS",
}
LOS_LOW = ("A", "B")                    # first letter A or B -> 0; C, D, E, F -> 1

ID_PATTERNS = ["driver", "participant", "person", "subject_id", "vehicle_id", "vehicleid", "vin",
               "dsid", "participant_id", "driverid"]              # identifier search (unit rule)

OPEN_QUESTIONS = [
    "Q1 driver identifier: none documented -> registered rule gives unit = event (trip); audit searches all columns",
    "Q2 environment for safe baselines: documented only for safety-critical test-set events",
    "Q3 record lengths (s) per outcome type: not documented",
    "Q4 object reference point: radar point vs object centre (Surrounding x_ekf vs test-set x_sur)",
    "Q5 object size: not documented per object (only the primary/secondary target in event_meta)",
    "Q6 exact trafficDensity strings ('A1' ... 'F')",
    "Q7 time origin: 'time' = round(timestamp/1000, 1) in [D4]; impact_timestamp in the same raw units",
]
