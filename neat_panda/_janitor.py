# -*- coding: utf-8 -*-

import re
from collections import Counter
import pandas as pd
from typing import Union, Optional, List, Dict


def clean_columns(
    columns: List[str], convert_duplicates: bool = True, convert_camel_case: bool = True
) -> List[str]:
    """
    """
    # implement removal of leading and lagging non alphabetic character
    # implement control for duplicates
    # columns = [re.sub(r"\s+", " ", i).strip() for i in columns]
    # columns = [re.sub(r"\W+", "_", i).strip().lower() for i in columns]
    columns = _clean_columns(
        columns=columns,
        convert_camel_case=convert_camel_case,
        expressions=[
            r"i.lower()",
            r're.sub(r"\s+", " ", i).strip()',  # replace multiple spaces with one space
            r're.sub(r"\W+", "_", i).strip()',  # Replace all non-alphanumeric characters in a string except underscore to with underscore
            r'i.rstrip("_").lstrip("_")'
            # r're.sub(r"_+$", "", i).strip()',  # remove lagging underscores [1]
            # r're.sub(r"_+$", "", i).strip()',
        ],
        convert_duplicates=convert_duplicates,
    )
    return columns  # .rtrim("_", "").ltrim("_", "")


def clean_columns_dataframe(
    df: pd.DataFrame, convert_duplicates: bool = True, convert_camel_case: bool = True
) -> pd.DataFrame:
    """
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("write something")
    df.columns = clean_columns(
        columns=df.columns,
        convert_duplicates=convert_duplicates,
        convert_camel_case=convert_camel_case,
    )
    return df


def _clean_columns(
    columns: List[str],
    custom=None,
    expressions=None,
    convert_duplicates: bool = True,
    convert_camel_case: bool = True,
) -> List[str]:
    """
    """
    if type(columns) == pd.Index:
        columns = columns.to_list()
    if not isinstance(columns, list):
        raise TypeError("")
    columns = [str(i) for i in columns]
    if custom:
        for i, j in custom.items():
            columns = [k.replace(i, j) for k in columns]
    if convert_camel_case:
        columns = _camel_to_snake(columns=columns)
    if expressions:
        for reg in expressions:
            columns = [eval(reg, {}, {"i": i, "re": re}) for i in columns]
    if convert_duplicates:
        columns = _convert_duplicates(columns=columns)
    return columns


def _convert_duplicates(columns: List[str]) -> List:
    """
    """
    d = {a: list(range(1, b + 1)) if b > 1 else "" for a, b in Counter(columns).items()}
    columns = [i + str(d[i].pop(0)) if len(d[i]) else i for i in columns]  # [2]
    return columns


def _camel_to_snake(columns: List[str]) -> List:
    _cols = []
    for i in columns:
        i = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", i)
        i = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", i).lower().replace("__", "_")
        _cols.append(i)
    return _cols


# [1] https://stackoverflow.com/questions/40740646/python-trimming-underscores-from-end-of-string/40740861
# [2] https://stackoverflow.com/questions/30650474/python-rename-duplicates-in-list-with-progressive-numbers-without-sorting-list/30651843#30651843
# [3] psun https://stackoverflow.com/questions/12985456/replace-all-non-alphanumeric-characters-in-a-string/12985459
