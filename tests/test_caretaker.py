# noqa: E501
import pytest
import pandas as pd
import numpy as np

from neat_panda import clean_column_names, clean_strings


class TestsCleanColumns:
    def test_type_error(self, clean_columns):
        with pytest.raises(TypeError):
            clean_column_names(object_=tuple(clean_columns))

    def test_assert_type(self, dataframe_long, clean_columns):
        assert isinstance(clean_column_names(clean_columns), list)
        assert isinstance(clean_column_names(dataframe_long), pd.DataFrame)
        assert isinstance(clean_column_names(dataframe_long.columns), list)

    def test_assert_not_alter_original_dataframe(self, nasty_columns, clean_columns):
        df = pd.DataFrame(np.random.randint(0, 5, size=(5, 12)))
        df.columns = nasty_columns
        df2 = df.copy()
        df.clean_column_names()
        df2 = df2.clean_column_names()
        assert df.columns.to_list() == nasty_columns
        assert df2.columns.to_list() == clean_columns

    def test_assert_correct_result_basic(self, nasty_columns, clean_columns):
        assert clean_column_names(nasty_columns, case="snake") == clean_columns

    def test_assert_correct_result_camel_case1(
        self, actual_camel_case_names, snake_case_names
    ):
        assert (
            clean_column_names(actual_camel_case_names, case="snake")
            == snake_case_names
        )

    def test_assert_correct_result_to_camelcase(
        self, snake_case_names, actual_camel_case_names
    ):
        assert (
            clean_column_names(snake_case_names, case="camel")
            == actual_camel_case_names
        )

    def test_assert_correct_result_to_pascalcase(
        self, snake_case_names, actual_pascal_case_names
    ):
        assert (
            clean_column_names(snake_case_names, case="pascal")
            == actual_pascal_case_names
        )

    def test_assert_correct_result_to_camelcase_large_letters(
        self, snake_case_names, actual_camel_case_names
    ):
        old = clean_column_names(snake_case_names, case="snake")
        old = [i.upper() for i in old]
        assert clean_column_names(old, case="camel") == actual_camel_case_names

    def test_assert_correct_result_to_pascalcase_large_letters(
        self, snake_case_names, actual_pascal_case_names
    ):
        old = clean_column_names(snake_case_names, case="snake")
        old = [i.upper() for i in old]
        assert clean_column_names(old, case="pascal") == actual_pascal_case_names

    def test_assert_errorenous_result_camel_case(
        self, faulty_camel_case_names, snake_case_names
    ):
        assert (
            clean_column_names(faulty_camel_case_names, case="snake")
            != snake_case_names
        )

    def test_assert_correct_result_custom(self, nasty_columns, clean_columns):
        cols3 = clean_column_names(
            nasty_columns,
            custom_expressions=[
                r're.sub(r"\s+", " ", column).strip()',
                r're.sub(r"\W+", "_", column).strip()',
                r'column.rstrip("_").lstrip("_")',
            ],
            convert_duplicates=True,
            case="snake",
        )
        assert cols3 == clean_columns

    def test_assert_correct_result_custom2(self):
        a = ["-Hello-", "Goodbye?", "HelloGoodbye", "Hello_Goodbye"]
        b = ["#hello#", "goodbye!", "hello_goodbye1", "hello_goodbye2"]
        c = clean_column_names(
            a,
            basic_cleaning=False,
            custom_transformation={"-": "#", "?": "!"},
            case="snake",
            convert_duplicates=True,
        )
        assert c == b

    def test_assert_correct_result_custom3(self):
        a = ["-Hello-", "Goodbye?", "HelloGoodbye", "Hello_Goodbye"]
        b = ["#hello#", "goodbye!", "hellogoodbye", "hello_goodbye"]
        c = clean_column_names(
            a,
            basic_cleaning=False,
            custom_transformation={"-": "#", "?": "!"},
            convert_duplicates=False,
            custom_expressions=["column.lower()"],
        )
        assert c == b

    def test_assert_correct_result_custom4(self):
        a = ["-Hello-", "Goodbye?", "HelloGoodbye", "Hello_Goodbye"]
        b = ["hello", "goodbye!", "hello_goodbye1", "hello_goodbye2"]
        c = clean_column_names(
            a,
            basic_cleaning=False,
            custom_transformation={"-": "", "?": "!"},
            convert_duplicates=True,
        )
        assert c == b

    def test_assert_correct_result_custom5(self):
        a = ["-Hello-", "Goodbye?", "Hello_goodbye", "Hello_Goodbye"]
        b = ["Hello", "Goodbye!", "Hello_Goodbye1", "Hello_Goodbye2"]
        c = clean_column_names(
            a,
            basic_cleaning=False,
            custom_transformation={"-": "", "?": "!"},
            convert_duplicates=True,
            custom_expressions=["column.title()"],
            case=None,
        )
        assert c == b

    def test_assert_correct_result_custom6(self):
        a = ["-Hello-", "Goodbye?", "hello_Goodbye", "Hello_Goodbye"]
        b = ["Hello", "Goodbye!", "Hello_goodbye1", "Hello_goodbye2"]
        c = clean_column_names(
            a,
            basic_cleaning=False,
            custom_transformation={"-": "", "?": "!"},
            convert_duplicates=True,
            custom_expressions=["column.capitalize()"],
            case=None,
        )
        assert c == b

    def test_assert_correct_result_string(
        self, nasty_columns2, clean_snake, clean_pascal, clean_camel
    ):
        for n in nasty_columns2:
            assert clean_strings(n, case="snake") == clean_snake
            assert clean_strings(n, case="pascal") == clean_pascal
            assert clean_strings(n, case="camel") == clean_camel

    def test_assert_correct_result_dataframe(self, dataframe_long):
        messy_cols = ["country    ", "continent£", "@@year   ", "actual"]
        clean_cols = dataframe_long.columns.tolist()
        df = dataframe_long.copy()
        df.columns = messy_cols
        df = clean_column_names(df)
        assert df.columns.tolist() == clean_cols

    def test_assert_correct_result_dataframe_method(self, dataframe_long):
        df = dataframe_long.copy()
        messy_cols = ["country    ", "continent£", "@@year   ", "actual"]
        clean_cols = df.columns.tolist()
        df.columns = messy_cols
        df = (
            df.clean_column_names()
        )  # convert camelcase can lead to unexpected behaviour when large
        # and small letters ar mixed and they are not camelcase.
        assert df.columns.tolist() == clean_cols

    def test_assert_correct_result_dataframe_method2(self, dataframe_long):
        df = dataframe_long.copy()
        messy_cols = ["CountryName", "Continent", "yearNo", "ACTUAL"]
        clean_cols_snake = ["country_name", "continent", "year_no", "actual"]
        clean_cols_camel = ["countryName", "continent", "yearNo", "actual"]
        clean_cols_pascal = ["CountryName", "Continent", "YearNo", "Actual"]
        df.columns = messy_cols
        df_snake = df.clean_column_names(case="snake").copy()
        df_camel = df.clean_column_names(case="camel").copy()
        df_pascal = df.clean_column_names(case="pascal").copy()
        assert df_snake.columns.tolist() == clean_cols_snake
        assert df_camel.columns.tolist() == clean_cols_camel
        assert df_pascal.columns.tolist() == clean_cols_pascal

    def test_assert_correct_result_dataframe_basic_cleaning_false(self, dataframe_long):
        df = dataframe_long.copy()
        messy_cols = ["CountryName", "Continent", "yearNo", "ACTUAL"]
        clean_cols_snake = ["country_name", "continent", "year_no", "actual"]
        clean_cols_camel = ["countryName", "continent", "yearNo", "actual"]
        clean_cols_pascal = ["CountryName", "Continent", "YearNo", "Actual"]
        df.columns = messy_cols
        df_snake = df.clean_column_names(case="snake", basic_cleaning=False).copy()
        df_camel = df.clean_column_names(case="camel", basic_cleaning=False).copy()
        df_pascal = df.clean_column_names(case="pascal", basic_cleaning=False).copy()
        assert df_snake.columns.tolist() == clean_cols_snake
        assert df_camel.columns.tolist() == clean_cols_camel
        assert df_pascal.columns.tolist() == clean_cols_pascal

    def test_assert_correct_result_series(self, nasty_columns2, clean_columns2):
        df = pd.DataFrame(data=nasty_columns2, columns=["a"])
        df2 = pd.DataFrame(data=clean_columns2, columns=["a"])
        df.a = df.a.astype(str)
        df2.a = df2.a.astype(str)
        df3 = df.copy()
        df3.a = df3.a.astype("string")
        assert df.a.clean_column_names().equals(df2.a)
        assert df.a.clean_strings().equals(df2.a)
        assert df3.a.clean_strings().dtype.__str__() == "string"
