import numpy as np
import h5py

def write_info_scalar(group, name, value):
    if name in group:
        return False
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        ds = group.create_dataset(name, data=np.array(value, dtype=h5py.string_dtype("utf-8")))
    elif isinstance(value, (bool, np.bool_)):
        ds = group.create_dataset(name, data=np.array(value, dtype=np.bool_))
    elif isinstance(value, (int, np.integer)):
        ds = group.create_dataset(name, data=np.array(value, dtype=np.int64))
    elif isinstance(value, (float, np.floating)):
        ds = group.create_dataset(name, data=np.array(value, dtype=np.float64))
    else:
        return False
    ds.attrs["type"] = "info"
    return True

def write_trial_column(group, name, values):
    """Store scalar numeric/bool/text trial columns; skip ragged/object arrays."""
    a = np.asarray(values)
    if a.ndim != 1:
        return False
    if a.dtype.kind in "iufb":
        data = a
    elif a.dtype.kind in "USO":
        if not all(isinstance(x, (str, bytes, type(None))) for x in a):
            return False
        data = np.array([x.decode("utf8", "replace") if isinstance(x, bytes) else
                         "" if x is None else x for x in a], dtype=object)
    else:
        return False
    if name in group:
        return False
    ds = group.create_dataset(name, data=data, dtype=h5py.string_dtype("utf-8") if data.dtype.kind == "O" else None)
    ds.attrs["type"] = "info"
    return True

def write_area(group, label, counts, unit_ids, annotations, original_indices):
    """Legacy area dataset [trial, unit, time] plus aligned unit provenance."""
    from .region_mapping import safe_key
    key = "area-" + safe_key(label)
    if key in group:
        raise ValueError(f"HDF5 region name collision: {key}")
    ds = group.create_dataset(key, data=counts, compression="gzip", compression_opts=4,
                              chunks=(1, min(max(1, counts.shape[1]), 64), counts.shape[2]))
    ds.attrs["type"] = "data"
    ds.attrs["region_label"] = label
    ds.attrs["axis_order"] = "trials,units,time"
    meta = group.require_group("unit_metadata").create_group(safe_key(label))
    meta.create_dataset("unit_id", data=np.array([str(x) for x in unit_ids], dtype=object),
                        dtype=h5py.string_dtype("utf-8"))
    meta.create_dataset("anno_name", data=np.array([str(x) for x in annotations], dtype=object),
                        dtype=h5py.string_dtype("utf-8"))
    meta.create_dataset("source_unit_row", data=np.asarray(original_indices, dtype=np.int64))
    return ds
