#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `neat_panda` package."""

import pytest
import pandas as pd
import numpy as np

from neat_panda import (
    spread,
    gather,
    clean_column_names,
    _clean_column_names,
    difference,
    intersection,
    symmetric_difference,
    union,
    _get_version_from_toml,
    __version__,
)


df = pd.DataFrame(
    data={
        "country": ["Sweden", "Sweden", "Denmark"],
        "continent": ["Europe", "Europe", "Not known"],
        "year": [2018, 2019, 2018],
        "actual": [1, 2, 3],
    }
)


def test_version():
    assert "0.9.4.1-dev" == __version__ == _get_version_from_toml("pyproject.toml")


class TestsSpread:
    def test_input_types_1(self):
        with pytest.raises(TypeError):
            spread(df=[1, 2, 3], key="hello", value="Goodbye")

    def test_duplicate_rows(self, df=df):
        with pytest.raises(ValueError):
            df = df.append(df)
            spread(df=df, key="year", value="actual")

    def test_input_types_key(self):
        with pytest.raises(TypeError):
            spread(df=df, key=1, value="actual")

    def test_input_types_value(self):
        with pytest.raises(TypeError):
            spread(df=df, key="country", value=1)

    def test_input_types_fill_bool(self):
        with pytest.raises(TypeError):
            spread(df=df, key="year", value="actual", fill=True)

    def test_input_types_fill(self):
        with pytest.raises(TypeError):
            spread(df=df, key="year", value="actual", fill=pd.DataFrame)

    def test_input_types_convert(self):
        with pytest.raises(TypeError):
            spread(df=df, key="year", value="actual", fill=1, convert="True")

    def test_input_types_sep(self):
        with pytest.raises(TypeError):
            spread(df=df, key="year", value="actual", fill=1, convert=True, sep=1)

    def test_user_warning(self):
        with pytest.warns(UserWarning):
            _df = spread(df=df, key="year", value="actual", drop=False, convert=True)
            del _df

    def test_no_nan(self):
        _df = spread(df=df, key="year", value="actual", drop=True, convert=True)
        assert _df.isna().any().sum() == 0

    def test_fill_other_than_nan(self):
        df1 = spread(df=df, key="year", value="actual", fill="Hej", sep="_")
        df2 = spread(df=df, key="year", value="actual", sep="_")
        _idx1 = sorted(df1.query("year_2019=='Hej'").index.tolist())
        _idx2 = sorted(df2.query("year_2019.isna()").index.tolist())
        assert _idx1 == _idx2

    def test_spread(self):
        _df = spread(
            df=df,
            key="year",
            value="actual",
            fill="NaN",
            drop=False,
            convert=True,
            sep="_",
        )


class TestsGather:
    df_wide = spread(df=df, key="year", value="actual")

    def test_equal_df(self, df=df_wide):
        df1 = gather(
            df=df,
            key="year",
            value="actual",
            columns=["country", "continent"],
            invert_columns=True,
            # convert=True,
        )
        df2 = gather(df=df, key="year", value="actual", columns=["2018", "2019"])

        assert df1.equals(df2)

    def test_equal_df_method(self, df=df_wide):
        df1 = gather(
            df=df,
            key="year",
            value="actual",
            columns=["country", "continent"],
            invert_columns=True,
            # convert=True,
        )
        df2 = df.gather(key="year", value="actual", columns=["2018", "2019"])

        assert df1.equals(df2)

    def test_correct_length_range(self):
        with pytest.raises(IndexError):
            gather(df=df, key="year", value="actual", columns=range(2, 100))

    def test_correct_column_type(self):
        with pytest.raises(TypeError):
            gather(df=df, key="year", value="actual", columns="string")

    def test_correct_dropna_type(self):
        with pytest.raises(TypeError):
            gather(
                df=df,
                key="year",
                value="actual",
                columns=["2018", "2019"],
                drop_na="Yes",
            )

    def test_correct_invertcolumns_type(self):
        with pytest.raises(TypeError):
            gather(
                df=df,
                key="year",
                value="actual",
                columns=["2018", "2019"],
                invert_columns="Yes",
            )

    def test_gather(self, df=df_wide):
        __df = gather(
            df=df,
            key="year",
            value="actual",
            columns=range(2, 4),
            invert_columns=False,
            drop_na=True,
            # convert=True,
        )


class TestsCleanColumns:
    nasty = [
        "Name    ",
        "hej",
        "  name",
        "country",
        "Region5",
        "country",
        "country-name£---",
        "______country@name",
        "countryName",
        "countryName",
        "country_Name",
        1,
    ]

    clean = [
        "name1",
        "hej",
        "name2",
        "country1",
        "region5",
        "country2",
        "country_name1",
        "country_name2",
        "country_name3",
        "country_name4",
        "country_name5",
        "1",
    ]

    actual_camel_case_names = ["countryName", "subRegion", "iceHockey"]

    actual_pascal_case_names = ["CountryName", "SubRegion", "IceHockey"]

    faulty_camel_case_names = ["countryNaMe", "SUbRegion", "ICeHOckey"]

    snake_case_names = ["country_name", "sub_region", "ice_hockey"]

    def test_type_error(self, cols=clean):
        with pytest.raises(TypeError):
            clean_column_names(object_=tuple(cols))
            _clean_column_names(columns=tuple(cols))

    def test_assert_type(self, cols=clean, df=df):
        assert isinstance(clean_column_names(cols), list)
        assert isinstance(_clean_column_names(cols), list)
        assert isinstance(clean_column_names(df), pd.DataFrame)
        assert isinstance(clean_column_names(df.columns), list)
        assert isinstance(_clean_column_names(df.columns), list)

    def test_assert_correct_result_basic(self, old=nasty, new=clean):
        assert clean_column_names(old, case_type="snake") == new

    def test_assert_correct_result_camel_case1(
        self, old=actual_camel_case_names, new=snake_case_names
    ):
        assert clean_column_names(old, case_type="snake") == new

    def test_assert_correct_result_to_camelcase(
        self, old=snake_case_names, new=actual_camel_case_names
    ):
        assert clean_column_names(old, case_type="camel") == new

    def test_assert_correct_result_to_pascalcase(
        self, old=snake_case_names, new=actual_pascal_case_names
    ):
        assert clean_column_names(old, case_type="pascal") == new

    def test_assert_correct_result_to_camelcase_large_letters(
        self, old=snake_case_names, new=actual_camel_case_names
    ):
        old = clean_column_names(old, case_type="snake")
        old = [i.upper() for i in old]
        assert clean_column_names(old, case_type="camel") == new

    def test_assert_correct_result_to_pascalcase_large_letters(
        self, old=snake_case_names, new=actual_pascal_case_names
    ):
        old = clean_column_names(old, case_type="snake")
        old = [i.upper() for i in old]
        assert clean_column_names(old, case_type="pascal") == new

    def test_assert_errorenous_result_camel_case(
        self, old=faulty_camel_case_names, new=snake_case_names
    ):
        assert clean_column_names(old, case_type="snake") != new

    def test_assert_correct_result_custom(self, old=nasty, new=clean):
        cols3 = _clean_column_names(
            old,
            expressions=[
                r"column.lower()",
                r're.sub(r"\s+", " ", column).strip()',
                r're.sub(r"\W+", "_", column).strip()',
                r'column.rstrip("_").lstrip("_")',
            ],
            convert_duplicates=True,
            case_type="snake",
        )
        assert cols3 == new

    def test_assert_correct_result_custom2(self):
        a = ["-Hello-", "Goodbye?", "HelloGoodbye", "Hello_Goodbye"]
        b = ["hello", "goodbye!", "hello_goodbye1", "hello_goodbye2"]
        c = _clean_column_names(
            a,
            custom={"-": "", "?": "!"},
            case_type="snake",  # the expression 'column.lower()' is not needed since convert_camel_case invokes it
            convert_duplicates=True,
        )
        assert c == b

    def test_assert_correct_result_custom3(self):
        a = ["-Hello-", "Goodbye?", "HelloGoodbye", "Hello_Goodbye"]
        b = ["hello", "goodbye!", "hello_goodbye", "hello_goodbye"]
        c = _clean_column_names(
            a,
            custom={"-": "", "?": "!"},
            convert_duplicates=False,
            expressions=["column.lower()"],
        )
        assert c == b

    def test_assert_correct_result_custom4(self):
        a = ["-Hello-", "Goodbye?", "HelloGoodbye", "Hello_Goodbye"]
        b = ["hello", "goodbye!", "hello_goodbye1", "hello_goodbye2"]
        c = _clean_column_names(
            a,
            custom={"-": "", "?": "!"},
            convert_duplicates=True,
            expressions=["column.lower()"],
        )
        assert c == b

    def test_assert_correct_result_custom5(self):
        a = ["-Hello-", "Goodbye?", "HelloGoodbye", "Hello_Goodbye"]
        b = ["Hello", "Goodbye!", "Hello_Goodbye1", "Hello_Goodbye2"]
        c = _clean_column_names(
            a,
            custom={"-": "", "?": "!"},
            convert_duplicates=True,
            expressions=["column.title()"],
        )
        assert c == b

    def test_assert_correct_result_custom6(self):
        a = ["-Hello-", "Goodbye?", "HelloGoodbye", "Hello_Goodbye"]
        b = ["Hello", "Goodbye!", "Hello_goodbye1", "Hello_goodbye2"]
        c = _clean_column_names(
            a,
            custom={"-": "", "?": "!"},
            convert_duplicates=True,
            expressions=["column.capitalize()"],
        )
        assert c == b

    def test_assert_correct_result_custom7(self):
        a = ["SUbRegion", "helloHOwAReYou?"]  # hur kan denna göras till snake?
        b = ["sub_region", "hello_how_are_you"]
        c = clean_column_names(a, case_type="snake", convert_duplicates=False)
        assert c == b

    def test_assert_correct_result_dataframe(self, df=df):
        messy_cols = ["country    ", "continent£", "@@year   ", "actual"]
        clean_cols = df.columns.tolist()
        df.columns = messy_cols
        df = clean_column_names(
            df
        )  # convert camelcase can lead to unexpected behaviour when large and
        # small letters ar mixed and they are not camelcase. set camelcase
        # default as false. eg year becomes y_ear
        assert df.columns.tolist() == clean_cols

    def test_assert_correct_result_dataframe_method(self, df=df):
        messy_cols = ["country    ", "continent£", "@@year   ", "actual"]
        clean_cols = df.columns.tolist()
        df.columns = messy_cols
        df = (
            df.clean_column_names()
        )  # convert camelcase can lead to unexpected behaviour when large
        # and small letters ar mixed and they are not camelcase.
        assert df.columns.tolist() == clean_cols


class TestSetOperations:
    def test_no_difference(self, df=df):
        assert difference(df, df).empty

    def test_basic_difference1(self, df=df):
        df2 = pd.DataFrame(
            data={
                "country": ["Sweden", "Sweden", "Denmark", "Norway"],
                "continent": ["Europe", "Europe", "Not known", "Scandinavia"],
                "year": [2018, 2019, 2018, 2020],
                "actual": [1, 2, 3, 5],
            }
        )
        assert difference(df, df2).empty

    def test_basic_difference2(self, df=df):
        df2 = pd.DataFrame(
            data={
                "country": ["Sweden", "Sweden", "Denmark", "Norway"],
                "continent": ["Europe", "Europe", "Not known", "Scandinavia"],
                "year": [2018, 2019, 2018, 2020],
                "actual": [1, 2, 3, 5],
            }
        )
        df3 = pd.DataFrame(
            data={
                "country": ["Norway"],
                "continent": ["Scandinavia"],
                "year": [2020],
                "actual": [5],
            }
        )
        assert difference(df2, df).reset_index(drop=True).equals(df3)

    def test_basic_intersection(self, df=df):
        df2 = pd.DataFrame(
            data={
                "country": ["Sweden", "Denmark", "Iceland"],
                "continent": ["Europe", "Not known", "Europe"],
                "year": [2018, 2018, np.nan],
                "actual": [1, 3, 0],
            }
        )

        df3 = pd.DataFrame(
            data={
                "country": ["Sweden", "Denmark"],
                "continent": ["Europe", "Not known"],
                "year": [2018, 2018],
                "actual": [1, 3],
            }
        )

        assert intersection(df, df2).reset_index(drop=True).equals(df3)

    def test_basic_symmetric_difference_names(self, df=df):
        df1 = pd.DataFrame(
            data={
                "country": ["Sweden", "Sweden", "Finland"],
                "continent": ["Europe", "Europe", "Scandinavia"],
                "year": [2018, 2012, 2018],
                "actual": [1, 2, 3],
            }
        )
        df3 = pd.DataFrame(
            data={
                "country": ["Sweden", "Denmark", "Sweden", "Finland"],
                "continent": ["Europe", "Not known", "Europe", "Scandinavia"],
                "year": [2019, 2018, 2012, 2018],
                "actual": [2, 3, 2, 3],
                "original_dataframe": ["df", "df", "df1", "df1"],
            }
        )
        assert symmetric_difference(df, df1, dataframe_names=["df", "df1"]).equals(df3)

    def test_basic_symmetric_difference_no_names(self, df=df):
        df1 = pd.DataFrame(
            data={
                "country": ["Sweden", "Sweden", "Finland"],
                "continent": ["Europe", "Europe", "Scandinavia"],
                "year": [2018, 2012, 2018],
                "actual": [1, 2, 3],
            }
        )
        df3 = pd.DataFrame(
            data={
                "country": ["Sweden", "Denmark", "Sweden", "Finland"],
                "continent": ["Europe", "Not known", "Europe", "Scandinavia"],
                "year": [2019, 2018, 2012, 2018],
                "actual": [2, 3, 2, 3],
            }
        )
        assert symmetric_difference(df, df1).reset_index(drop=True).equals(df3)

    def test_basic_union(self):
        df1 = pd.DataFrame(
            data={
                "country": ["Sweden", "Sweden", "Finland"],
                "continent": ["Europe", "Europe", "Scandinavia"],
                "year": [2018, 2012, 2018],
                "actual": [1, 2, 3],
            }
        )
        df2 = pd.DataFrame(
            data={
                "country": ["Sweden", "Denmark", "Sweden", "Finland"],
                "continent": ["Europe", "Not known", "Europe", "Scandinavia"],
                "year": [2019, 2018, 2012, 2018],
                "actual": [2, 3, 2, 3],
            }
        )
        df3 = df1.append(df2).reset_index(drop=True)
        assert union(df1, df2).equals(df3)

    def test_warning_duplicates(self):
        df1 = pd.DataFrame(
            data={
                "country": ["Sweden", "Sweden", "Finland"],
                "continent": ["Europe", "Europe", "Scandinavia"],
                "year": [2018, 2018, 2018],
                "actual": [1, 1, 3],
            }
        )
        df2 = pd.DataFrame(
            data={
                "country": ["Sweden", "Denmark", "Sweden", "Finland"],
                "continent": ["Europe", "Not known", "Europe", "Scandinavia"],
                "year": [2019, 2018, 2019, 2018],
                "actual": [2, 3, 2, 3],
            }
        )
        with pytest.warns(UserWarning):
            union(df1, df2)

