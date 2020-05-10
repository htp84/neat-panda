import pytest
import pandas as pd
from neat_panda import spread, gather

dataframe = pd.DataFrame(
    data={
        "country": ["Sweden", "Sweden", "Denmark"],
        "continent": ["Europe", "Europe", "Not known"],
        "year": [2018, 2019, 2018],
        "actual": [1, 2, 3],
    }
)


def test_cases_spread():
    return [
        (
            [1, 2, 3],
            "hello",
            "Goodbye",
            "NaN",
            False,
            False,
            None,
            TypeError,
            "input types",
        ),
        (
            dataframe.append(dataframe),
            "year",
            "actual",
            "NaN",
            False,
            False,
            None,
            ValueError,
            "duplicate rows",
        ),
        (dataframe, "year", 1, "NaN", False, False, None, TypeError, "input types key"),
        (
            dataframe,
            "country",
            1,
            "NaN",
            False,
            False,
            None,
            TypeError,
            "input types value",
        ),
        (
            dataframe,
            "year",
            "actual",
            pd.DataFrame,
            False,
            False,
            None,
            TypeError,
            "input types fill",
        ),
        (
            dataframe,
            "year",
            "actual",
            1,
            "True",
            False,
            None,
            TypeError,
            "input types convert",
        ),
        (
            dataframe,
            "year",
            "actual",
            pd.DataFrame,
            1,
            True,
            None,
            TypeError,
            "input types sep",
        ),
    ]


class TestsSpread:
    @pytest.mark.parametrize(
        " data, key, value, fill, convert, drop, sep, error, comment",
        test_cases_spread(),
    )
    def test_spread_errors(
        self, data, key, value, fill, convert, drop, sep, error, comment
    ):
        with pytest.raises(error):
            spread(
                df=data,
                key=key,
                value=value,
                fill=fill,
                convert=convert,
                drop=drop,
                sep=sep,
            )

    def test_user_warning(self, dataframe_long):
        with pytest.warns(UserWarning):
            spread(
                df=dataframe_long, key="year", value="actual", drop=False, convert=True
            )

    def test_no_nan(self, dataframe_long):
        _df = spread(
            df=dataframe_long, key="year", value="actual", drop=True, convert=True
        )
        assert _df.isna().any().sum() == 0

    def test_fill_other_than_nan(self, dataframe_long):
        df1 = spread(df=dataframe_long, key="year", value="actual", fill="Hej", sep="_")
        df2 = spread(df=dataframe_long, key="year", value="actual", sep="_")
        _idx1 = sorted(df1.query("year_2019=='Hej'").index.tolist())
        _idx2 = sorted(df2.query("year_2019.isna()").index.tolist())
        assert _idx1 == _idx2

    def test_spread(self, dataframe_long):
        _df = spread(
            df=dataframe_long,
            key="year",
            value="actual",
            fill="NaN",
            drop=False,
            convert=True,
            sep="_",
        )

    def test_user_warning(self, dataframe_long):
        with pytest.warns(UserWarning):
            spread(
                df=dataframe_long, key="year", value="actual", drop=False, convert=True
            )


class TestsGather:
    def test_equal_df(self, dataframe_wide):
        df1 = gather(
            df=dataframe_wide,
            key="year",
            value="actual",
            columns=["country", "continent"],
            invert_columns=True,
            # convert=True,
        )
        df2 = gather(
            df=dataframe_wide, key="year", value="actual", columns=["2018", "2019"]
        )

        assert df1.equals(df2)

    def test_equal_df_method(self, dataframe_wide):
        df1 = gather(
            df=dataframe_wide,
            key="year",
            value="actual",
            columns=["country", "continent"],
            invert_columns=True,
            # convert=True,
        )
        df2 = dataframe_wide.gather(
            key="year", value="actual", columns=["2018", "2019"]
        )

        assert df1.equals(df2)

    def test_correct_length_range(self, dataframe_wide):
        with pytest.raises(IndexError):
            gather(df=dataframe_wide, key="year", value="actual", columns=range(2, 100))

    def test_correct_column_type(self, dataframe_wide):
        with pytest.raises(TypeError):
            gather(df=dataframe_wide, key="year", value="actual", columns="string")

    def test_correct_dropna_type(self, dataframe_wide):
        with pytest.raises(TypeError):
            gather(
                df=dataframe_wide,
                key="year",
                value="actual",
                columns=["2018", "2019"],
                drop_na="Yes",
            )

    def test_correct_invertcolumns_type(self, dataframe_wide):
        with pytest.raises(TypeError):
            gather(
                df=dataframe_wide,
                key="year",
                value="actual",
                columns=["2018", "2019"],
                invert_columns="Yes",
            )

    def test_gather(self, dataframe_wide):
        __df = gather(
            df=dataframe_wide,
            key="year",
            value="actual",
            columns=range(2, 4),
            invert_columns=False,
            drop_na=True,
            # convert=True,
        )
