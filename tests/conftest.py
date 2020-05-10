import pytest
import pandas as pd
from neat_panda import spread


@pytest.fixture()
def dataframe_long():
    return pd.DataFrame(
        data={
            "country": ["Sweden", "Sweden", "Denmark"],
            "continent": ["Europe", "Europe", "Not known"],
            "year": [2018, 2019, 2018],
            "actual": [1, 2, 3],
        }
    )


@pytest.fixture()
def dataframe_wide(dataframe_long):
    return spread(df=dataframe_long, key="year", value="actual")


@pytest.fixture()
def nasty_columns():
    return [
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


@pytest.fixture()
def clean_columns():
    return [
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


@pytest.fixture()
def nasty_columns2():
    return [
        "country____NAME",
        "CountryName",
        "Country_Name",
        "country_Name",
        "country_name",
        "country-name£---",
        "______country@name",
    ]


@pytest.fixture()
def clean_columns2():
    return ["country_name" for i in range(7)]


@pytest.fixture()
def clean_snake():
    return "country_name"


@pytest.fixture()
def clean_camel():
    return "countryName"


@pytest.fixture()
def clean_pascal():
    return "CountryName"


@pytest.fixture()
def actual_camel_case_names():
    return ["countryName", "subRegion", "iceHockey"]


@pytest.fixture()
def actual_pascal_case_names():
    return ["CountryName", "SubRegion", "IceHockey"]


@pytest.fixture()
def faulty_camel_case_names():
    return ["countryNaMe", "SUbRegion", "ICeHOckey"]


@pytest.fixture()
def snake_case_names():
    return ["country_name", "sub_region", "ice_hockey"]
