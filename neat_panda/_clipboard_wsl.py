import io
import pandas as pd
import pandas_flavor as pf

try:
    import pyperclip
except ImportError:
    print(
        f"""It is necessary to install 'pyperclip' for the function/methods read_clipboard_wsl, and to_clipboard to work.\n
            This can be done with via pip.\n
            """
    )
    raise


@pf.register_dataframe_method
def read_clipboard_wsl(decimal: str = ".") -> None:
    """read clipboard dataframe from wsl.

    Parameters
    ----------
    decimal : str, optional
         Character recognized as decimal separator. E.g. use ',' for European data. by default "."

    Returns
    -------
    [type]
        [description]
    """
    return pd.read_csv(io.StringIO(pyperclip.paste()), sep="\t", decimal=decimal).drop(
        columns="Unnamed: 0"
    )


@pf.register_dataframe_method
def to_clipboard_wsl(df: pd.DataFrame, decimal: str = ".") -> pd.DataFrame:
    """copy to clipboard in wsl

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to be copied to clipboard
    decimal : str, optional
         Character recognized as decimal separator. E.g. use ',' for European data. by default "."

    Returns
    -------
    pd.DataFrame
        [description]
    """
    return pyperclip.copy(df.to_csv(sep="\t", decimal=decimal))

