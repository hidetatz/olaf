import pytest


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests when no CUDA device is usable, so the rest of the suite runs anywhere."""
    if not any("gpu" in item.keywords for item in items):
        return
    from olaf.backend.cuda.driver import available

    if available():
        return
    skip = pytest.mark.skip(reason="no usable CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
