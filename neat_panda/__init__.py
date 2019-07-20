# -*- coding: utf-8 -*-

"""Top-level package for Neat Panda."""

__author__ = """Henric Sundberg"""
__email__ = "henric.sundberg@gmail.com"
__version__ = "0.7.0"

from ._tidy import spread, gather
from ._caretaker import (
    clean_columnnames,
    _clean_columnnames,
    _clean_columnnames_dataframe,
    _clean_columnnames_list,
)
