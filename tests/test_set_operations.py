import pytest
import pandas as pd
import numpy as np

from neat_panda import difference, intersection, symmetric_difference, union


class TestSetOperations:
    def test_no_difference(self, dataframe_long):
        assert difference(dataframe_long, dataframe_long).empty

    def test_basic_difference1(self, dataframe_long):
        df2 = pd.DataFrame(
            data={
                "country": ["Sweden", "Sweden", "Denmark", "Norway"],
                "continent": ["Europe", "Europe", "Not known", "Scandinavia"],
                "year": [2018, 2019, 2018, 2020],
                "actual": [1, 2, 3, 5],
            }
        )
        assert difference(dataframe_long, df2).empty

    def test_basic_difference2(self, dataframe_long):
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
        assert difference(df2, dataframe_long).reset_index(drop=True).equals(df3)

    def test_basic_intersection(self, dataframe_long):
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

        assert intersection(dataframe_long, df2).reset_index(drop=True).equals(df3)

    def test_basic_symmetric_difference_names(self, dataframe_long):
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
        assert symmetric_difference(
            dataframe_long, df1, dataframe_names=["df", "df1"]
        ).equals(df3)

    def test_basic_symmetric_difference_no_names(self):
        df = pd.DataFrame(
            data={
                "country": ["Sweden", "Sweden", "Denmark"],
                "continent": ["Europe", "Europe", "Not known"],
                "year": [2018, 2019, 2018],
                "actual": [1, 2, 3],
            }
        )
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
