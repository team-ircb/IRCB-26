import csv
import logging
import re
from functools import lru_cache
from importlib import resources

LOG = logging.getLogger(__name__)

LABEL_MODES = ("abbreviation", "full_name")


@lru_cache(maxsize=1)
def ccf_abbreviations():
    """Allen CCF v3 full structure name -> abbreviation (packaged CSV).

    DANDI 000363 stores full Allen names in ``anno_name`` (e.g. "Secondary motor area,
    layer 2/3"). the original extraction translated them to abbreviations ("MOs2/3")
    with the same table (allen_ccf_v3.xlsx); this keeps that behaviour.
    """
    text = resources.files(__package__).joinpath("data/allen_ccf_v3_names.csv").read_text(encoding="utf-8")
    rows = csv.DictReader(line for line in text.splitlines() if line and not line.startswith("#"))
    return {r["full_name"].strip(): r["abbreviation"].strip() for r in rows}


_warned = set()


def unit_label(annotation, mode="abbreviation"):
    """Label for one unit's ``anno_name``: its CCF abbreviation (default) or the name itself.

    Annotations already given as an abbreviation are kept. A name missing from the table
    is kept as is, with one warning per name.
    """
    if not annotation or str(annotation).strip().lower() in ("nan", "none"):
        return None
    name = str(annotation).strip()
    if mode == "full_name":
        return name
    if mode != "abbreviation":
        raise ValueError(f"region_labels must be one of {LABEL_MODES}")
    table = ccf_abbreviations()
    if name in table:
        return table[name]
    if name in set(table.values()):
        return name
    if name not in _warned:
        _warned.add(name)
        LOG.warning("anno_name %r is not in the Allen CCF table; keeping it as is", name)
    return name


def region_for_annotation(annotation, region_config, mode="abbreviation"):
    """Map a unit's ``anno_name`` to its configured output group, or to its own label.

    region_config: {"ALM": ["MOs2/3", "MOs5"], ...}. Entries may be abbreviations or full
    Allen names; matching is exact. If the mapping is empty, every unit keeps its own
    label (one ``area-*`` dataset per area, as in the original extraction).
    """
    label = unit_label(annotation, mode)
    if label is None:
        return None
    if not region_config:
        return label
    full = str(annotation).strip()
    for group, members in region_config.items():
        if label in members or full in members:
            return group
    return None


def safe_key(label):
    """HDF5-safe component; disallow path separators, keep readable names."""
    return re.sub(r"[/\\]+", "_", str(label)).strip()
