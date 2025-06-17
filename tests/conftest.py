"""
Pytest configuration and shared fixtures for OpenBabel tool tests.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Load tool configurations
CONFIG_PATH = (
    Path(__file__).parent.parent
    / "src"
    / "tooluniverse"
    / "data"
    / "openbabel_tools.json"
)

if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        TOOL_CONFIGS = json.load(f)
else:
    TOOL_CONFIGS = {}

# Common test molecules
TEST_MOLECULES = {
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "water": "O",
    "ethanol": "CCO",
    "benzene": "c1ccccc1",
    "glucose": "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O",
}


@pytest.fixture(scope="session")
def tool_configs():
    """Provide tool configurations for tests"""
    return TOOL_CONFIGS


@pytest.fixture(scope="session")
def test_molecules():
    """Provide test molecules for tests"""
    return TEST_MOLECULES


@pytest.fixture
def mock_openbabel():
    """
    Fixture that provides mocked OpenBabel modules.
    Returns (pybel_mock, ob_mock) tuple.
    """
    pybel_mock = MagicMock()
    ob_mock = MagicMock()

    modules = {
        "openbabel": MagicMock(),
        "openbabel.pybel": pybel_mock,
        "openbabel.openbabel": ob_mock,
    }

    with patch.dict("sys.modules", modules):
        with patch("tooluniverse.openbabel_tool.pybel", pybel_mock):
            with patch("tooluniverse.openbabel_tool.ob", ob_mock):
                with patch("tooluniverse.openbabel_tool.OPENBABEL_AVAILABLE", True):
                    yield pybel_mock, ob_mock


@pytest.fixture
def mock_molecule():
    """
    Fixture that provides a mock molecule with common properties.
    """
    mol = MagicMock()
    mol.molwt = 180.16
    mol.exactmass = 180.063388
    mol.formula = "C6H12O6"
    mol.atoms = [MagicMock() for _ in range(24)]
    mol.calcdesc.return_value = {"logP": -2.5, "TPSA": 90.15, "MR": 40.25}
    mol.OBMol.Bonds.return_value = [MagicMock() for _ in range(25)]
    mol.write.return_value = "mock_output"
    return mol


@pytest.fixture
def basic_tool_config():
    """Provide a basic tool configuration for testing"""
    return {
        "tool_url": "/test_tool",
        "description": "Test tool",
        "parameter": {
            "type": "object",
            "properties": {
                "input_string": {"type": "string"},
                "input_format": {"type": "string", "default": "smiles"},
            },
            "required": ["input_string"],
        },
    }


# Pytest markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line("markers", "unit: marks tests as unit tests (with mocks)")
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (requires OpenBabel)"
    )
    config.addinivalue_line("markers", "slow: marks tests as slow tests")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location"""
    for item in items:
        # Mark tests in integration test file
        if "test_openbabel_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark tests in unit test file
        if "test_openbabel_tools" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
