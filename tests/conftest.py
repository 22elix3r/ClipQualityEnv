"""Pytest configuration for the ClipQualityEnv test suite."""
from __future__ import annotations


def pytest_configure(config):
    """Suppress Gradio 6.0 deprecation warnings that originate from
    server/app.py's UI construction.  These are not test bugs — they are
    framework migration notices that will be resolved when the app migrates
    to the Gradio 6.0 API."""
    config.addinivalue_line(
        "filterwarnings",
        "ignore:The parameters have been moved from the Blocks constructor:UserWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:The `col_count` parameter is deprecated:UserWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:The `column_limits` parameter is not yet implemented:UserWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:The 'col_count' parameter will be removed:DeprecationWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Passing a tuple to 'col_count' will be removed:DeprecationWarning",
    )
