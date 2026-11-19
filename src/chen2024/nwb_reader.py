from pathlib import Path
from pynwb import NWBHDF5IO

def nwb_paths(root):
    root = Path(root)
    if root.is_file():
        if root.suffix.lower() != ".nwb":
            raise ValueError("Input file must be .nwb")
        return [root]
    return sorted(root.rglob("*.nwb"))

def describe(path):
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        units = nwb.units.to_dataframe() if nwb.units is not None else None
        trials = nwb.trials.to_dataframe() if nwb.trials is not None else None
        return {
            "file": str(path), "subject": getattr(nwb.subject, "subject_id", None),
            "session_id": nwb.session_id,
            "n_units": 0 if units is None else len(units),
            "n_trials": 0 if trials is None else len(trials),
            "unit_columns": [] if units is None else list(units.columns),
            "trial_columns": [] if trials is None else list(trials.columns),
            "areas": {} if units is None or "anno_name" not in units else
                {str(k): int(v) for k, v in units["anno_name"].value_counts().items()},
        }
