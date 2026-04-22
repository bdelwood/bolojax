"""This module contains functions to help manage configuration for the
offline analysis of LSST Electrical-Optical testing"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

CONFIG_DIR: str | None = None


def is_none(val: Any) -> bool:
    """Check for values equivalent to None (None, 'none', 'None')."""
    if not isinstance(val, type(None) | str):
        return False
    return val in [None, "none", "None"]


def is_not_none(val: Any) -> bool:
    """Check for values NOT equivalent to None."""
    if not isinstance(val, type(None) | str):
        return True
    return val not in [None, "none", "None"]


class CfgDir:
    """Tiny class to find configuration files"""

    def __init__(self) -> None:
        self.config_dir: str | None = None

    def set_dir(self, val: str | None) -> None:
        """Set the top-level configuration directory"""
        self.config_dir = val

    def get_dir(self) -> str | None:
        """Get the top-level configuration directory"""
        return self.config_dir

    def cfg_path(self, val: str) -> str:
        """Build a path using the top-level configuration directory"""
        return str(Path(self.config_dir) / val)


CFG_DIR = CfgDir()

set_config_dir = CFG_DIR.set_dir
get_config_dir = CFG_DIR.get_dir
cfg_path = CFG_DIR.cfg_path


def copy_dict(in_dict: dict[str, Any], def_dict: dict[str, Any]) -> dict[str, Any]:
    """Copy a set of key-value pairs to an new dict

    Parameters
    ----------
    in_dict : `dict`
        The dictionary with the input values
    def_dict : `dict`
        The dictionary with the default values

    Returns
    -------
    outdict : `dict`
        Dictionary with arguments selected from in_dict to override def_dict
    """
    return {key: in_dict.get(key, val) for key, val in def_dict.items()}


def pop_values(in_dict: dict[str, Any], keylist: list[str]) -> dict[str, Any]:
    """Pop a set of key-value pairs to an new dict

    Parameters
    ----------
    in_dict : `dict`
        The dictionary with the input values
    keylist : `list`
        The values to pop

    Returns
    -------
    outdict : `dict`
        Dictionary with only the arguments we have selected
    """
    outdict: dict[str, Any] = {}
    for key in keylist:
        if key in in_dict:
            outdict[key] = in_dict.pop(key)
    return outdict


def update_dict_from_string(
    o_dict: dict[str, Any],
    key: str,
    val: Any,
    subparser_dict: dict[str, Any] | None = None,
) -> None:
    """Update a dictionary with sub-dictionaries

    Parameters
    ----------
    o_dict : dict
        The output

    key : `str`
        The string we are parsing

    val : `str`
        The value

    subparser_dict : `dict` or `None`
        The subparsers used to parser the command line

    """
    idx = key.find(".")
    use_key = key[0:idx]
    remain = key[idx + 1 :]
    if subparser_dict is not None:
        try:
            subparser = subparser_dict[use_key[1:]]
        except KeyError:
            subparser = None
    else:
        subparser = None

    if use_key not in o_dict:
        o_dict[use_key] = {}

    def_val = None
    if subparser is not None:
        def_val = subparser.get_default(remain)
    if def_val == val:
        return

    if remain.find(".") < 0:
        o_dict[use_key][remain] = val
    else:
        update_dict_from_string(o_dict[use_key], remain, val)


def expand_dict_from_defaults_and_elements(
    default_dict: dict[str, Any], elem_dict: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Expand a dictionary by copying defaults to a set of elements

    Parameters
    ----------
    default_dict : `dict`
        The defaults

    elem_dict : `dict`
        The elements

    Returns
    -------
    o_dict : `dict`
        The output dict
    """
    o_dict: dict[str, dict[str, Any]] = {}
    for key, elem in elem_dict.items():
        o_dict[key] = default_dict.copy()
        if elem is None:
            continue
        o_dict[key].update(elem)
    return o_dict


def read_txt_to_np(fname: str) -> np.ndarray:
    """Read a txt file to a numpy array."""
    ext = Path(fname).suffix
    if ext.lower() == ".txt":
        delim = None
    elif ext.lower() == ".csv":
        delim = ","
    else:
        msg = f"File {fname} is not csv or txt"
        raise ValueError(msg)
    return np.loadtxt(str(Path(fname)), unpack=True, dtype=np.float64, delimiter=delim)


def reshape_array(
    val: float | np.ndarray, shape: tuple[int, ...]
) -> float | np.ndarray:
    """Reshape an array, but not a scalar

    This is useful for broadcasting many arrays to the same shape
    """
    if np.isscalar(val):
        return val
    return val.reshape(shape)
