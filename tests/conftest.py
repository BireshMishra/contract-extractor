import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="run tests marked 'live', which call the real Anthropic API",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip_live = pytest.mark.skip(reason="live API test; opt in with --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
