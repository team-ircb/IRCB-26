import json
import logging
from pathlib import Path
import numpy as np
import h5py
from pynwb import NWBHDF5IO
from .trial_filtering import select_trials
from .region_mapping import region_for_annotation, unit_label
from .spike_binning import bin_unit
from .hdf5_writer import write_area, write_info_scalar, write_trial_column

LOG = logging.getLogger(__name__)

def _subject_id(nwb, path):
    sid = getattr(nwb.subject, "subject_id", None)
    if sid:
        return str(sid).removeprefix("sub-")
    # DANDI paths often contain sub-<id>
    import re
    match = re.search(r"sub-([A-Za-z0-9]+)", str(path))
    if match: return match.group(1)
    raise ValueError(f"Missing subject ID in {path}")

def _session_name(nwb, path):
    import re
    sid = nwb.session_id
    if not sid:
        # DANDI 000363 files leave session_id empty; their names carry it
        # (sub-440959_ses-20190225T142613_behavior+ecephys+image+ogen.nwb -> 20190225T142613).
        match = re.search(r"ses-([A-Za-z0-9]+)", Path(str(path)).name)
        sid = match.group(1) if match else Path(str(path)).stem
    # avoid HDF5 path separators; preserve stable source-derived session identity
    return "ses-" + re.sub(r"[^A-Za-z0-9_.-]", "_", str(sid)).removeprefix("ses-")

def _configured_subject_name(nwb, subject_id, cfg):
    configured = cfg.get("_subjects_by_id", {})
    if subject_id in configured:
        return configured[subject_id]
    fallback = getattr(nwb.subject, "description", None)
    if fallback:
        return str(fallback)
    return f"Mouse {subject_id}"

# Units-table columns the conversion never reads. Skipping them keeps streamed reads small
# (waveforms and electrode links are most of the table's size).
_UNUSED_UNIT_COLUMNS = {"electrodes", "electrode_group", "waveform_mean", "waveform_sd", "obs_intervals"}


def convert_file(path, output_dir, cfg):
    """Convert one local NWB file."""
    path = Path(path)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        return convert_nwb(io.read(), path, output_dir, cfg)


def convert_dandi_asset(asset, output_dir, cfg, version=None):
    """Convert one DANDI 000363 asset by streaming it (no local copy). ``asset`` is a dict
    from ``dandi_download.list_assets``; its path names the subject and session. ``version`` is
    the dandiset version it was listed from, recorded in the output (unit labels differ between
    versions)."""
    from .remote import asset_url, open_remote_nwb
    with open_remote_nwb(asset_url(asset)) as nwb:
        return convert_nwb(nwb, asset["path"], output_dir, cfg, dandiset_version=version)


def convert_nwb(nwb, path, output_dir, cfg, dandiset_version=None):
    """Convert an open NWBFile. ``path`` (local path or DANDI asset path) is used for the
    subject and session names when the file doesn't carry them, and is recorded as the source."""
    if nwb.units is None or nwb.trials is None:
        raise ValueError(f"Missing units or trials: {path}")
    exclude = _UNUSED_UNIT_COLUMNS & set(nwb.units.colnames)
    units, trials = nwb.units.to_dataframe(exclude=exclude), nwb.trials.to_dataframe()
    required = {"spike_times", "anno_name", "classification"}
    if not required.issubset(units.columns):
        raise ValueError(f"Missing unit columns: {sorted(required - set(units.columns))}")
    selected, _ = select_trials(trials, cfg.get("trial_filter", {}))
    if len(selected) == 0:
        LOG.warning("No accepted trials: %s", path)
        return None
    starts = trials["start_time"].to_numpy(dtype=float)[selected]
    stops = trials["stop_time"].to_numpy(dtype=float)[selected]
    quality = cfg.get("units", {}).get("classification", "good")
    valid_quality = np.ones(len(units), dtype=bool) if quality is None else (units["classification"].astype(str).to_numpy() == str(quality))
    regions = {}
    for row_idx, (unit_id, row) in enumerate(units.iterrows()):
        if not valid_quality[row_idx]: continue
        label = region_for_annotation(row["anno_name"], cfg.get("regions", {}),
                                      cfg.get("region_labels", "abbreviation"))
        if label is not None:
            regions.setdefault(label, []).append((row_idx, unit_id, row["anno_name"], row["spike_times"]))
    if not regions:
        LOG.warning("No matching units: %s", path)
        return None
    # A group's units follow the order its areas are listed in the config (all MRN units, then all
    # PPN units, ...), each area in NWB unit order, as the original loaders stacked them; held-out
    # neuron indices refer to positions in this order.
    region_cfg = cfg.get("regions", {}) or {}
    mode = cfg.get("region_labels", "abbreviation")
    for group, rows in regions.items():
        members = list(region_cfg.get(group, []))
        if members:
            def member_pos(r):
                own = unit_label(r[2], mode)
                full = str(r[2]).strip()
                return members.index(own) if own in members else members.index(full)
            rows.sort(key=lambda r: (member_pos(r), r[0]))
    subject = _subject_id(nwb, path)
    configured_subjects = cfg.get("_subjects_by_id", {})
    if configured_subjects and subject not in configured_subjects:
        LOG.info("Skipping %s because subject %s is not in config.subjects", path, subject)
        return None
    out = Path(output_dir) / f"sub-{subject}" / f"sub-{subject}.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    session = _session_name(nwb, path)
    subject_name = _configured_subject_name(nwb, subject, cfg)
    n_bins = cfg.get("binning", {}).get("n_bins", 500)
    trial_field_map = cfg.get("_trial_field_map")
    with h5py.File(out, "a") as h5:
        if session in h5:
            raise ValueError(f"Duplicate session {session} in {out}; refusing to overwrite")
        g = h5.create_group(session)
        try:
            g.attrs["source_nwb"] = str(path)
            g.attrs["source_dandiset"] = "000363"
            if dandiset_version:
                g.attrs["source_dandiset_version"] = str(dandiset_version)
            g.attrs["binning_mode"] = "fixed_bins"
            g.attrs["n_bins"] = n_bins
            config_snapshot = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
            g.attrs["config_json"] = json.dumps(config_snapshot, sort_keys=True)
            g.attrs["accepted_trial_count"] = len(selected)
            g.create_dataset("source_trial_row", data=selected)
            g["source_trial_row"].attrs["type"] = "info"
            write_info_scalar(g, "subject_id", subject)
            write_info_scalar(g, "subject_name", subject_name)
            write_info_scalar(g, "session_id", session.removeprefix("ses-"))
            selected_trials = trials.iloc[selected]
            if trial_field_map is None:
                trial_field_map = {str(col): str(col) for col in trials.columns}
            missing = [src for src in trial_field_map if src not in selected_trials.columns]
            if missing:
                raise ValueError(f"Configured trial metadata columns are missing: {missing}")
            for src_name, out_name in trial_field_map.items():
                if not write_trial_column(g, out_name, selected_trials[src_name].to_numpy()):
                    LOG.info("Skipped non-scalar trial metadata: %s", src_name)
            if "start_time" not in g or "stop_time" not in g:
                raise ValueError("Trial timing was not written")
            g.create_dataset("trial_duration_s", data=stops - starts)
            g["trial_duration_s"].attrs["type"] = "info"
            # Each trial has its own physical bin width, recorded explicitly.
            g.create_dataset("bin_width_s", data=(stops-starts)/n_bins)
            g["bin_width_s"].attrs["type"] = "info"
            for label, rows in sorted(regions.items()):
                counts = np.zeros((len(selected), len(rows), n_bins), dtype=np.int32)
                for j, (_, _, _, spikes) in enumerate(rows):
                    counts[:, j, :] = bin_unit(spikes, starts, stops, n_bins)
                write_area(g, label, counts, [r[1] for r in rows],
                           [r[2] for r in rows], [r[0] for r in rows])
            LOG.info("Wrote %s / %s: %d trials, %d regions", out, session, len(selected), len(regions))
        except Exception:
            del h5[session]
            raise
    return str(out)
