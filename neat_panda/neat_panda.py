# -*- coding: utf-8 -*-

import pandas as pd
from typing import Union, Optional, List
from warnings import warn


def _control_types(
    _df,
    _key,
    _value,
    _fill="NaN",
    _convert=False,
    _sep=None,
    _columns=[],
    _drop_na=False,
    _invert_columns=False,
):
    # spread and gather
    if not isinstance(_df, pd.DataFrame):
        raise TypeError("write something")
    if not isinstance(_key, str):
        raise TypeError()
    if not isinstance(_value, str):
        raise TypeError()
    # spread
    if isinstance(_fill, bool):
        raise TypeError()
    if not isinstance(_fill, (str, float, int)):
        raise TypeError()
    if not isinstance(_convert, bool):
        raise TypeError()
    if not isinstance(_sep, (str, type(None))):
        raise TypeError()
    # gather
    if not isinstance(_columns, (list, range)):
        raise TypeError()
    if isinstance(_columns, range) and len(_df.columns) - 1 < _columns[-1]:
        raise IndexError()
    if not isinstance(_drop_na, bool):
        raise TypeError()
    if not isinstance(_invert_columns, bool):
        raise TypeError()


def _assure_consistent_value_dtypes(new_df, old_df, columns, value):
    """
    """
    _dtype = old_df[value].dtypes
    try:
        new_df[columns] = new_df[columns].astype(_dtype)
    except ValueError:
        warn(
            UserWarning(
                """Because the parameter drop is set to False and NA is generated
            when the dataframe is widened, the type of the new columns is
            set to Object."""
            )
        )
        new_df[columns] = new_df[columns].astype("O")
    return new_df


def _custom_columns(columns, new_columns, key, sep):
    _cols = [i for i in columns if i not in new_columns]
    _custom = [key + sep + i for i in new_columns]
    return _cols + _custom


def spread(
    df: pd.DataFrame,
    key: str,
    value: str,
    fill: Union[str, int, float] = "NaN",
    convert: bool = False,
    drop: bool = False,
    sep: Optional[str] = None,
) -> pd.DataFrame:
    """Behaves similar to the tidyr spread function.\n
    Does not work with multi index dataframes.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame
    key : str
        Column to use to make new frame’s columns
    value : str
        Column which contains values corresponding to the new frame’s columns
    fill : Union[str, int, float], optional
        Missing values will be replaced with this value.\n
        (the default is "NaN", which is numpy.nan)
    convert : bool, optional
        If True, the new columns is set to the same type as the original frame's value column.\n
        However, if fill is equal to "NaN" all columns is set to the object type since Numpy.nan is of that type\n
        (the default is False, which ...)
    drop : bool, optional
        If True, all rows that contains at least one "NaN" is dropped.
    sep : Optional[str], optional
        If set, the names of the new columns will be given by "<key_name><sep><key_value>".\n
        E.g. if set to '-' and the key column is called 'Year' and contains 2018 and 2019 the new columns will be\n
        'Year-2018' and 'Year-2019'. (the default is None, and using previous example, the new column names will be '2018' and '2019')

    Raises
    ------
    ValueError
        [description]
    TypError
        [description]

    Returns
    -------
    pd.DataFrame
        A widened dataframe

    Example
    -------
    from neat_panda import spread
    from gapminder import gapminder

    gapminder2 = gapminder[["country", "continent", "year", "pop"]]
    gapminder3 = spread(df=gapminder2, key="year", value="pop")
    # or
    gapminder3 = gapminder2.pipe(spread, key="year", value="pop")

    print(gapminder3)

           country continent      1952      1957      1962  ...
    0  Afghanistan      Asia   8425333   9240934  10267083  ...
    1      Albania    Europe   1282697   1476505   1728137  ...
    2      Algeria    Africa   9279525  10270856  11000948  ...
    3       Angola    Africa   4232095   4561361   4826015  ...
    4    Argentina  Americas  17876956  19610538  21283783  ...
    .          ...       ...       ...       ...       ...  ...
    """

    _control_types(
        _df=df, _key=key, _value=value, _fill=fill, _convert=convert, _sep=sep
    )
    _drop = [key, value]
    _columns = [i for i in df.columns.tolist() if i not in _drop]
    try:
        _df = df.set_index(_columns).pivot(columns=key)
    except ValueError:
        raise ValueError("something about that ")
    _df.columns = _df.columns.droplevel()
    new_df = pd.DataFrame(_df.to_records())
    _new_columns = [i for i in new_df.columns if i not in df.columns]
    if drop:
        new_df = new_df.dropna(how="any")
    if fill != "NaN":
        new_df[_new_columns] = new_df[_new_columns].fillna(fill)
    if convert:
        new_df = _assure_consistent_value_dtypes(new_df, df, _new_columns, value)
    if sep:
        custom_columns = _custom_columns(
            new_df.columns.to_list(), _new_columns, key, sep
        )
        new_df.columns = custom_columns
    return new_df


def gather(
    df: pd.DataFrame,
    key: str,
    value: str,
    columns: Union[List[str], range],
    drop_na: bool = False,
    convert: bool = False,
    invert_columns: bool = False,
) -> pd.DataFrame:
    """[summary]

    Parameters
    ----------
    df : pd.DataFrame
        [description]
    key : str
        [description]
    value : str
        [description]
    columns : List[str]
        [description]
    drop_na : bool, optional
        [description] (the default is False, which [default_description])
    convert : bool, optional
        [description] (the default is False, which [default_description])
    invert_columns : bool, optional
        [description] (the default is False, which [default_description])

    Returns
    -------
    pd.DataFrame
        [description]
    """

    _control_types(
        _df=df,
        _key=key,
        _value=value,
        _columns=columns,
        _drop_na=drop_na,
        _convert=convert,
        _invert_columns=invert_columns,
    )
    _all_columns = df.columns.to_list()
    if isinstance(columns, range):
        _temp_col = []
        _index = list(columns)
        for i, j in enumerate(_all_columns):
            if i in _index:
                _temp_col.append(j)
        columns = _temp_col
    if invert_columns:
        columns = [i for i in _all_columns if i not in columns]
    _id_vars = [i for i in _all_columns if i not in columns]
    new_df = pd.melt(
        frame=df, id_vars=_id_vars, value_vars=columns, value_name=value, var_name=key
    )
    if drop_na:
        new_df = new_df.dropna(how="all", subset=[value])
    if convert:
        _dtype = new_df[value].infer_objects().dtypes
        new_df[value] = new_df[value].astype(_dtype)
    return new_df


if __name__ == "__main__":
    pass
