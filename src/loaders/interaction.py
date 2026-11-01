"""
Load the INTERACTION dataset into the project's common trajectory format.
Reads every vehicle_tracks_*.csv under data/raw/interaction/
(skipping __MACOSX junk and duplicate copies of the same file)
and writes data/processed/interaction.parquet
"""
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/interaction")
OUT_FILE = Path("data/processed/interaction.parquet")


def load_track_file(csv_path: Path) -> pd.DataFrame:
    """Convert one vehicle_tracks_xxx.csv into the common format."""
    df = pd.read_csv(csv_path)

    location = csv_path.parent.name
    file_no = csv_path.stem.replace("vehicle_tracks_", "")
    if "case_id" in df.columns:
        scene = location + "/" + file_no + "/" + df["case_id"].astype(int).astype(str)
    else:
        scene = pd.Series(location + "/" + file_no, index=df.index)

    out = pd.DataFrame({
        "dataset": "interaction",
        "scene_id": scene,
        "time_s": df["timestamp_ms"] / 1000.0,
        "agent_id": df["track_id"].astype(str),
        "is_ego": False,
        "x": df["x"], "y": df["y"],
        "vx": df["vx"], "vy": df["vy"],
        "heading": df["psi_rad"],
        "length": df["length"], "width": df["width"],
        "agent_type": df["agent_type"],
    })

    # Acceleration is not provided: differentiate velocity per vehicle
    out = out.sort_values(["scene_id", "agent_id", "time_s"]).reset_index(drop=True)
    g = out.groupby(["scene_id", "agent_id"], sort=False)
    dt = g["time_s"].diff()
    out["ax"] = (g["vx"].diff() / dt).fillna(0.0)
    out["ay"] = (g["vy"].diff() / dt).fillna(0.0)
    return out


def find_track_files() -> list:
    """Find each track file once, ignoring __MACOSX and duplicate copies."""
    all_files = sorted(p for p in RAW_DIR.rglob("vehicle_tracks_*.csv")
                       if "__MACOSX" not in p.parts)
    files, seen = [], set()
    for p in all_files:
        key = (p.parent.name, p.name)   # same location + same file name = duplicate copy
        if key not in seen:
            seen.add(key)
            files.append(p)
    print(f"Found {len(all_files)} track files, "
          f"skipped {len(all_files) - len(files)} duplicate copies, "
          f"using {len(files)}")
    return files


def main():
    files = find_track_files()
    if not files:
        raise SystemExit(f"No vehicle_tracks_*.csv found under {RAW_DIR.resolve()}")

    parts = []
    for f in files:
        part = load_track_file(f)
        parts.append(part)
        print(f"  {f.parent.name}/{f.name}: {part['agent_id'].nunique():5d} vehicles, "
              f"{len(part):8d} rows")

    data = pd.concat(parts, ignore_index=True)
    cols = ["dataset", "scene_id", "time_s", "agent_id", "is_ego",
            "x", "y", "vx", "vy", "ax", "ay", "heading", "length", "width", "agent_type"]
    data = data[cols]

    # Final safety net: no vehicle can appear twice at the same moment in a scene
    before = len(data)
    data = data.drop_duplicates(["scene_id", "agent_id", "time_s"]).reset_index(drop=True)
    print(f"\nRemoved {before - len(data):,} duplicate rows")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUT_FILE, index=False)

    speed = np.hypot(data["vx"], data["vy"])
    print("\nSummary")
    print(f"  locations: {data['scene_id'].str.split('/').str[0].nunique()}")
    print(f"  scenes:    {data['scene_id'].nunique()}")
    print(f"  vehicles:  {data.groupby('scene_id')['agent_id'].nunique().sum():,}")
    print(f"  rows:      {len(data):,}")
    print(f"  speed:     median {speed.median():.1f} m/s, max {speed.max():.1f} m/s")
    print(f"  agent types: {data['agent_type'].value_counts().to_dict()}")
    print(f"\nSaved to {OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()