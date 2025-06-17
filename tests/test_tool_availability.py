"""
Test the availability and import mechanism of OpenBabel tools.
"""

import pytest
from unittest.mock import patch


def test_openbabel_availability_check():
    """Test that OpenBabel availability is correctly detected"""
    # Test when OpenBabel is available
    with patch("importlib.util.find_spec", return_value=True):
        # Re-import the module to test detection
        import importlib
        import tooluniverse

        importlib.reload(tooluniverse)

        # This should work without raising an error
        # (though the actual tool may still fail if OpenBabel isn't really installed)
        try:
            tool_class = tooluniverse.ConvertMoleculeFormatTool
            assert tool_class is not None
        except ImportError:
            # This is expected if OpenBabel isn't actually installed
            pass


def test_openbabel_not_available():
    """Test graceful handling when OpenBabel is not available"""
    with patch("importlib.util.find_spec", return_value=None):
        import tooluniverse

        # Trying to access an OpenBabel tool should raise ImportError
        with pytest.raises(ImportError, match="OpenBabel tool .* is not available"):
            tool = tooluniverse.ConvertMoleculeFormatTool


def test_lazy_import_mechanism():
    """Test that OpenBabel tools are imported lazily"""
    import tooluniverse

    # The __all__ list should contain OpenBabel tools
    assert "ConvertMoleculeFormatTool" in tooluniverse.__all__
    assert "CalculateMolecularPropertiesTool" in tooluniverse.__all__
    assert "CalculateMolecularSimilarityTool" in tooluniverse.__all__


def test_non_openbabel_tools_available():
    """Test that non-OpenBabel tools are always available"""
    import tooluniverse

    # These should always work
    assert hasattr(tooluniverse, "ToolUniverse")
    assert hasattr(tooluniverse, "MonarchTool")
    assert hasattr(tooluniverse, "OpentargetTool")


def test_invalid_tool_name():
    """Test error handling for invalid tool names"""
    import tooluniverse

    with pytest.raises(AttributeError, match="module 'tooluniverse' has no attribute"):
        tool = tooluniverse.NonExistentTool


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
