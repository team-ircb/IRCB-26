import yaml

def _normalize_subjects(cfg):
    subjects = cfg.get("subjects")
    if subjects is None:
        return {}
    if not isinstance(subjects, list):
        raise ValueError("subjects must be a list of {id, name}")
    normalized = {}
    for item in subjects:
        if not isinstance(item, dict) or "id" not in item:
            raise ValueError("Each subject entry must be a mapping with at least an 'id' key")
        subject_id = str(item["id"]).removeprefix("sub-").strip()
        if not subject_id:
            raise ValueError("Subject id cannot be empty")
        subject_name = str(item.get("name", f"Mouse {subject_id}")).strip()
        normalized[subject_id] = subject_name
    return normalized

def _normalize_trial_fields(cfg):
    metadata = cfg.get("metadata", {})
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a mapping")
    trial_fields = metadata.get("trial_fields")
    if trial_fields is None:
        return None
    if isinstance(trial_fields, list):
        return {str(name): str(name) for name in trial_fields}
    if isinstance(trial_fields, dict):
        result = {}
        for source, target in trial_fields.items():
            s = str(source).strip()
            t = str(target).strip()
            if not s or not t:
                raise ValueError("metadata.trial_fields keys/values cannot be empty")
            result[s] = t
        return result
    raise ValueError("metadata.trial_fields must be a list or mapping")

def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")
    binning = cfg.get("binning", {})
    mode = binning.get("mode", "fixed_bins")
    if mode != "fixed_bins":
        raise ValueError("binning.mode must be fixed_bins")
    bins = binning.get("n_bins", 500)
    if not isinstance(bins, int) or bins <= 0:
        raise ValueError("binning.n_bins must be a positive integer")
    flt = cfg.get("trial_filter", {})
    if "target_duration_s" in flt:
        raise ValueError("trial_filter.target_duration_s is not supported; trial duration is fixed at 5.0 s")
    if float(flt.get("duration_tolerance_s", 0.5)) < 0:
        raise ValueError("trial_filter.duration_tolerance_s must be >= 0")
    if flt.get("photostim", "exclude") not in ("exclude", "include", "only"):
        raise ValueError("trial_filter.photostim must be exclude/include/only")
    labels = cfg.get("region_labels", "abbreviation")
    if labels not in ("abbreviation", "full_name"):
        raise ValueError("region_labels must be abbreviation or full_name")
    cfg["region_labels"] = labels
    cfg["_subjects_by_id"] = _normalize_subjects(cfg)
    cfg["_trial_field_map"] = _normalize_trial_fields(cfg)
    return cfg
