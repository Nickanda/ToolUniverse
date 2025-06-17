import pytest
from unittest.mock import Mock, patch, MagicMock


class TestOpenBabelSimple:
    """Simple OpenBabel tests that don't require complex mocking"""

    def test_openbabel_availability_flag(self):
        """Test that the OPENBABEL_AVAILABLE flag exists"""
        from tooluniverse.openbabel_tool import OPENBABEL_AVAILABLE

        assert isinstance(OPENBABEL_AVAILABLE, bool)

    def test_base_tool_import_check(self):
        """Test that OpenBabel base tool can be imported when not available"""
        with patch("tooluniverse.openbabel_tool.OPENBABEL_AVAILABLE", False):
            from tooluniverse.openbabel_tool import OpenBabelBaseTool

            # Should raise error when trying to instantiate
            with pytest.raises(ImportError, match="OpenBabel is not available"):
                OpenBabelBaseTool({})

    def test_tool_classes_exist(self):
        """Test that all tool classes can be imported"""
        # This should work even if OpenBabel is not available
        # because we only check availability on instantiation
        from tooluniverse.openbabel_tool import (
            ConvertMoleculeFormatTool,
            Generate3DStructureTool,
            CalculateMolecularPropertiesTool,
            CalculateMolecularSimilarityTool,
            PerformSubstructureSearchTool,
            GenerateMoleculeImageTool,
            ChemicalFormulaToMassTool,
            GenerateConformersTool,
            MolecularDockingTool,
            AnalyzeBondsTool,
            MergeMultipleMoleculesTool,
            FragmentMoleculeTool,
        )

        # Just check they are classes
        assert isinstance(ConvertMoleculeFormatTool, type)
        assert isinstance(CalculateMolecularPropertiesTool, type)
        assert isinstance(CalculateMolecularSimilarityTool, type)

    def test_with_mocked_openbabel(self):
        """Test tool creation with mocked OpenBabel"""
        mock_modules = {
            "openbabel": MagicMock(),
            "openbabel.pybel": MagicMock(),
            "openbabel.openbabel": MagicMock(),
        }

        with patch.dict("sys.modules", mock_modules):
            with patch("tooluniverse.openbabel_tool.OPENBABEL_AVAILABLE", True):
                from tooluniverse.openbabel_tool import ConvertMoleculeFormatTool

                config = {
                    "parameter": {
                        "properties": {
                            "input_string": {"type": "string", "required": True}
                        }
                    }
                }

                # Should be able to create tool instance
                tool = ConvertMoleculeFormatTool(config)
                assert tool is not None
                assert tool.tool_config == config

    def test_tool_error_handling(self):
        """Test that tools handle errors gracefully"""
        mock_modules = {
            "openbabel": MagicMock(),
            "openbabel.pybel": MagicMock(),
            "openbabel.openbabel": MagicMock(),
        }

        with patch.dict("sys.modules", mock_modules):
            with patch("tooluniverse.openbabel_tool.OPENBABEL_AVAILABLE", True):
                from tooluniverse.openbabel_tool import ConvertMoleculeFormatTool

                config = {"parameter": {"properties": {}}}
                tool = ConvertMoleculeFormatTool(config)

                # Mock the _create_mol_from_input method to raise an exception
                with patch.object(
                    tool, "_create_mol_from_input", side_effect=Exception("Mock error")
                ):
                    result = tool.run({"input_string": "invalid"})

                    # Should return error dict instead of raising
                    assert isinstance(result, dict)
                    assert "error" in result
                    assert "Format conversion failed" in result["error"]

    def test_config_file_loading(self):
        """Test that configuration files can be loaded"""
        from pathlib import Path
        import json

        config_path = (
            Path(__file__).parent.parent
            / "src"
            / "tooluniverse"
            / "data"
            / "openbabel_tools.json"
        )

        if config_path.exists():
            with open(config_path) as f:
                configs = json.load(f)

            assert isinstance(configs, dict)
            assert len(configs) > 0

            # Check a few expected tools
            if "ConvertMoleculeFormat" in configs:
                config = configs["ConvertMoleculeFormat"]
                assert "description" in config
                assert "parameter" in config
                assert "properties" in config["parameter"]

    def test_basic_molecule_operations(self):
        """Test basic molecule operations with full mocking"""
        mock_pybel = MagicMock()
        mock_ob = MagicMock()

        # Mock molecule object
        mock_mol = Mock()
        mock_mol.molwt = 180.16
        mock_mol.formula = "C6H12O6"
        mock_mol.write.return_value = "mock_mol_data"

        mock_pybel.readstring.return_value = mock_mol

        mock_modules = {
            "openbabel": MagicMock(),
            "openbabel.pybel": mock_pybel,
            "openbabel.openbabel": mock_ob,
        }

        with patch.dict("sys.modules", mock_modules):
            with patch("tooluniverse.openbabel_tool.OPENBABEL_AVAILABLE", True):
                with patch("tooluniverse.openbabel_tool.pybel", mock_pybel):
                    from tooluniverse.openbabel_tool import ConvertMoleculeFormatTool

                    config = {"parameter": {"properties": {}}}
                    tool = ConvertMoleculeFormatTool(config)

                    # Test basic conversion
                    result = tool.run({"input_string": "CCO", "output_format": "mol"})

                    # Should return the mocked output
                    assert result == "mock_mol_data"
                    mock_mol.write.assert_called_with("mol")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
