# -*- coding: utf-8 -*-

"""Top-level package for Neat Panda."""

__author__ = """Henric Sundberg"""
__email__ = "henric.sundberg@gmail.com"
__version__ = "0.9.8.1"

from neat_panda._tidy import spread, gather, flatten_pivot

# from ._caretaker import clean_column_names, _clean_column_names
from neat_panda._caretaker import clean_column_names, CleanColumnNames, clean_strings

from neat_panda._set_operations import (
    difference,
    intersection,
    symmetric_difference,
    union,
    SetOperations,
)

from neat_panda._helpers import _get_version_from_toml

from neat_panda._clipboard_wsl import read_clipboard_wsl, to_clipboard_wsl
