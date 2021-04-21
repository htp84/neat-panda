import pytest
import pandas as pd

from neat_panda import read_clipboard_wsl, to_clipboard_wsl


class TestsClipboardWsl:
    def test_clipboard_wsl(self, dataframe_long):
        to_clipboard_wsl(dataframe_long)
        _data = read_clipboard_wsl()
        assert dataframe_long.equals(_data)
