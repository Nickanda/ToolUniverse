import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

# Test configuration for OpenBabel tools
TEST_CONFIG = {
    "ConvertMoleculeFormat": {
        "tool_url": "/convert_molecule_format",
        "description": "Convert a molecule from one chemical format to another",
        "parameter": {
            "type": "object",
            "properties": {
                "input_string": {"type": "string"},
                "input_format": {"type": "string", "default": "smiles"},
                "output_format": {"type": "string", "default": "smi"},
                "generate_3d": {"type": "boolean", "default": False},
            },
            "required": ["input_string"],
        },
    }
}

# Mock molecules for testing
MOCK_MOLECULES = {
    "aspirin_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine_smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "ibuprofen_smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "water_formula": "H2O",
    "glucose_formula": "C6H12O6",
}


class TestOpenBabelAvailability:
    """Test OpenBabel availability and graceful degradation"""

    def test_openbabel_not_available(self):
        """Test behavior when OpenBabel is not available"""
        with patch.dict("sys.modules", {"openbabel": None}):
            with patch(
                "importlib.import_module",
                side_effect=ImportError("No module named 'openbabel'"),
            ):
                # Test that trying to import OpenBabel tools raises appropriate error
                from tooluniverse import __getattr__ as get_tool

                with pytest.raises(
                    ImportError, match="OpenBabel tool .* is not available"
                ):
                    get_tool("ConvertMoleculeFormatTool")


class TestOpenBabelBaseTool:
    """Test the base OpenBabel tool functionality"""

    @pytest.fixture
    def mock_openbabel(self):
        """Mock OpenBabel modules"""
        with patch.dict(
            "sys.modules",
            {
                "openbabel": MagicMock(),
                "openbabel.pybel": MagicMock(),
                "openbabel.openbabel": MagicMock(),
            },
        ):
            yield

    @pytest.fixture
    def base_tool(self, mock_openbabel):
        """Create a base tool instance for testing"""
        with patch("tooluniverse.openbabel_tool.OPENBABEL_AVAILABLE", True):
            from tooluniverse.openbabel_tool import OpenBabelBaseTool

            return OpenBabelBaseTool(TEST_CONFIG["ConvertMoleculeFormat"])

    def test_base_tool_initialization(self, base_tool):
        """Test that base tool initializes correctly"""
        assert base_tool.tool_config == TEST_CONFIG["ConvertMoleculeFormat"]

    def test_base_tool_openbabel_not_available(self):
        """Test base tool fails when OpenBabel not available"""
        with patch("tooluniverse.openbabel_tool.OPENBABEL_AVAILABLE", False):
            from tooluniverse.openbabel_tool import OpenBabelBaseTool

            with pytest.raises(ImportError, match="OpenBabel is not available"):
                OpenBabelBaseTool(TEST_CONFIG["ConvertMoleculeFormat"])

    @patch("tooluniverse.openbabel_tool.pybel")
    def test_create_mol_from_input_smiles(self, mock_pybel, base_tool):
        """Test molecule creation from SMILES"""
        mock_mol = Mock()
        mock_pybel.readstring.return_value = mock_mol

        result = base_tool._create_mol_from_input("CCO", "smiles")

        mock_pybel.readstring.assert_called_once_with("smi", "CCO")
        assert result == mock_mol

    @patch("tooluniverse.openbabel_tool.pybel")
    def test_create_mol_from_input_other_format(self, mock_pybel, base_tool):
        """Test molecule creation from other formats"""
        mock_mol = Mock()
        mock_pybel.readstring.return_value = mock_mol

        result = base_tool._create_mol_from_input("mock_mol_data", "mol")

        mock_pybel.readstring.assert_called_once_with("mol", "mock_mol_data")
        assert result == mock_mol

    @patch("tooluniverse.openbabel_tool.pybel")
    def test_create_mol_from_input_error(self, mock_pybel, base_tool):
        """Test error handling in molecule creation"""
        mock_pybel.readstring.side_effect = Exception("Invalid molecule")

        with pytest.raises(ValueError, match="Error creating molecule from input"):
            base_tool._create_mol_from_input("invalid", "smiles")


class TestConvertMoleculeFormatTool:
    """Test molecule format conversion tool"""

    @pytest.fixture
    def mock_openbabel_modules(self):
        """Mock all OpenBabel related modules"""
        with patch.dict(
            "sys.modules",
            {
                "openbabel": MagicMock(),
                "openbabel.pybel": MagicMock(),
                "openbabel.openbabel": MagicMock(),
            },
        ):
            with patch("tooluniverse.openbabel_tool.OPENBABEL_AVAILABLE", True):
                yield

    @pytest.fixture
    def convert_tool(self, mock_openbabel_modules):
        """Create format conversion tool for testing"""
        from tooluniverse.openbabel_tool import ConvertMoleculeFormatTool

        return ConvertMoleculeFormatTool(TEST_CONFIG["ConvertMoleculeFormat"])

    @patch("tooluniverse.openbabel_tool.pybel")
    def test_convert_format_basic(self, mock_pybel, convert_tool):
        """Test basic format conversion"""
        # Mock molecule
        mock_mol = Mock()
        mock_mol.write.return_value = "mock_output"
        mock_pybel.readstring.return_value = mock_mol

        arguments = {
            "input_string": MOCK_MOLECULES["aspirin_smiles"],
            "input_format": "smiles",
            "output_format": "mol",
        }

        result = convert_tool.run(arguments)

        assert result == "mock_output"
        mock_mol.write.assert_called_once_with("mol")
        assert not mock_mol.make3D.called

    @patch("tooluniverse.openbabel_tool.pybel")
    def test_convert_format_with_3d(self, mock_pybel, convert_tool):
        """Test format conversion with 3D generation"""
        mock_mol = Mock()
        mock_mol.write.return_value = "mock_3d_output"
        mock_pybel.readstring.return_value = mock_mol

        arguments = {
            "input_string": MOCK_MOLECULES["aspirin_smiles"],
            "input_format": "smiles",
            "output_format": "pdb",
            "generate_3d": True,
        }

        result = convert_tool.run(arguments)

        assert result == "mock_3d_output"
        mock_mol.make3D.assert_called_once()
        mock_mol.write.assert_called_once_with("pdb")

    def test_convert_format_error_handling(self, convert_tool):
        """Test error handling in format conversion"""
        with patch.object(
            convert_tool, "_create_mol_from_input", side_effect=Exception("Mock error")
        ):
            arguments = {
                "input_string": "invalid",
                "input_format": "smiles",
                "output_format": "mol",
            }

            result = convert_tool.run(arguments)

            assert "error" in result
            assert "Format conversion failed" in result["error"]


class TestCalculateMolecularPropertiesTool:
    """Test molecular properties calculation tool"""

    @pytest.fixture
    def mock_openbabel_modules(self):
        with patch.dict(
            "sys.modules",
            {
                "openbabel": MagicMock(),
                "openbabel.pybel": MagicMock(),
                "openbabel.openbabel": MagicMock(),
            },
        ):
            with patch("tooluniverse.openbabel_tool.OPENBABEL_AVAILABLE", True):
                yield

    @pytest.fixture
    def properties_tool(self, mock_openbabel_modules):
        from tooluniverse.openbabel_tool import CalculateMolecularPropertiesTool

        config = {
            "parameter": {
                "properties": {
                    "input_string": {"required": True},
                    "input_format": {"required": False},
                    "properties": {"required": False},
                }
            }
        }
        return CalculateMolecularPropertiesTool(config)

    @patch("tooluniverse.openbabel_tool.pybel")
    @patch("tooluniverse.openbabel_tool.ob")
    def test_calculate_properties_basic(self, mock_ob, mock_pybel, properties_tool):
        """Test basic property calculation"""
        # Mock molecule with properties
        mock_mol = Mock()
        mock_mol.molwt = 180.16
        mock_mol.exactmass = 180.063388
        mock_mol.formula = "C6H12O6"
        mock_mol.atoms = [Mock() for _ in range(24)]  # 24 atoms in glucose
        mock_mol.calcdesc.return_value = {"logP": -2.5, "TPSA": 90.15, "MR": 40.25}
        mock_mol.OBMol.Bonds.return_value = [Mock() for _ in range(25)]  # Mock bonds

        mock_pybel.readstring.return_value = mock_mol

        arguments = {
            "input_string": MOCK_MOLECULES["glucose_formula"],
            "properties": ["molecular_weight", "logp", "formula", "atoms"],
        }

        result = properties_tool.run(arguments)

        assert result["molecular_weight"] == 180.16
        assert result["logp"] == -2.5
        assert result["formula"] == "C6H12O6"
        assert result["atoms"] == 24

    @patch("tooluniverse.openbabel_tool.pybel")
    def test_calculate_all_properties(self, mock_pybel, properties_tool):
        """Test calculation of all available properties"""
        mock_mol = Mock()
        mock_mol.molwt = 180.16
        mock_mol.exactmass = 180.063388
        mock_mol.formula = "C6H12O6"
        mock_mol.atoms = []
        mock_mol.calcdesc.return_value = {"logP": -2.5, "TPSA": 90.15, "MR": 40.25}
        mock_mol.OBMol.Bonds.return_value = []

        mock_pybel.readstring.return_value = mock_mol

        arguments = {
            "input_string": MOCK_MOLECULES["glucose_formula"],
            "properties": [],  # Empty list should calculate all
        }

        result = properties_tool.run(arguments)

        # Should contain all available properties
        expected_props = [
            "molecular_weight",
            "logp",
            "psa",
            "mr",
            "atoms",
            "bonds",
            "hba",
            "hbd",
            "rotatable_bonds",
            "exact_mass",
            "formula",
        ]
        for prop in expected_props:
            assert prop in result


class TestCalculateMolecularSimilarityTool:
    """Test molecular similarity calculation tool"""

    @pytest.fixture
    def mock_openbabel_modules(self):
        with patch.dict(
            "sys.modules",
            {
                "openbabel": MagicMock(),
                "openbabel.pybel": MagicMock(),
                "openbabel.openbabel": MagicMock(),
            },
        ):
            with patch("tooluniverse.openbabel_tool.OPENBABEL_AVAILABLE", True):
                yield

    @pytest.fixture
    def similarity_tool(self, mock_openbabel_modules):
        from tooluniverse.openbabel_tool import CalculateMolecularSimilarityTool

        config = {"parameter": {"properties": {}}}
        return CalculateMolecularSimilarityTool(config)

    @patch("tooluniverse.openbabel_tool.pybel")
    def test_similarity_tanimoto(self, mock_pybel, similarity_tool):
        """Test Tanimoto similarity calculation"""
        # Mock molecules and fingerprints
        mock_mol1 = Mock()
        mock_mol2 = Mock()
        mock_fp1 = Mock()
        mock_fp2 = Mock()

        mock_fp1.__or__ = Mock(return_value=0.75)  # Mock Tanimoto similarity
        mock_mol1.calcfp.return_value = mock_fp1
        mock_mol2.calcfp.return_value = mock_fp2

        mock_pybel.readstring.side_effect = [mock_mol1, mock_mol2]

        arguments = {
            "query_string": MOCK_MOLECULES["aspirin_smiles"],
            "target_string": MOCK_MOLECULES["ibuprofen_smiles"],
            "similarity_metric": "tanimoto",
        }

        result = similarity_tool.run(arguments)

        assert result["similarity"] == 0.75
        assert result["metric"] == "tanimoto"
        assert result["fp_type"] == "FP2"

    @patch("tooluniverse.openbabel_tool.pybel")
    def test_similarity_dice(self, mock_pybel, similarity_tool):
        """Test Dice similarity calculation"""
        mock_mol1 = Mock()
        mock_mol2 = Mock()
        mock_fp1 = Mock()
        mock_fp2 = Mock()

        mock_fp1.dice_similarity.return_value = 0.85
        mock_mol1.calcfp.return_value = mock_fp1
        mock_mol2.calcfp.return_value = mock_fp2

        mock_pybel.readstring.side_effect = [mock_mol1, mock_mol2]

        arguments = {
            "query_string": MOCK_MOLECULES["aspirin_smiles"],
            "target_string": MOCK_MOLECULES["ibuprofen_smiles"],
            "similarity_metric": "dice",
        }

        result = similarity_tool.run(arguments)

        assert result["similarity"] == 0.85
        assert result["metric"] == "dice"

    def test_similarity_invalid_metric(self, similarity_tool):
        """Test error handling for invalid similarity metric"""
        with patch.object(similarity_tool, "_create_mol_from_input"):
            arguments = {
                "query_string": MOCK_MOLECULES["aspirin_smiles"],
                "target_string": MOCK_MOLECULES["ibuprofen_smiles"],
                "similarity_metric": "invalid_metric",
            }

            result = similarity_tool.run(arguments)

            assert "error" in result
            assert "Unsupported similarity metric" in result["error"]


class TestPerformSubstructureSearchTool:
    """Test substructure search tool"""

    @pytest.fixture
    def mock_openbabel_modules(self):
        with patch.dict(
            "sys.modules",
            {
                "openbabel": MagicMock(),
                "openbabel.pybel": MagicMock(),
                "openbabel.openbabel": MagicMock(),
            },
        ):
            with patch("tooluniverse.openbabel_tool.OPENBABEL_AVAILABLE", True):
                yield

    @pytest.fixture
    def search_tool(self, mock_openbabel_modules):
        from tooluniverse.openbabel_tool import PerformSubstructureSearchTool

        config = {"parameter": {"properties": {}}}
        return PerformSubstructureSearchTool(config)

    @patch("tooluniverse.openbabel_tool.ob")
    @patch("tooluniverse.openbabel_tool.pybel")
    def test_substructure_search_found(self, mock_pybel, mock_ob, search_tool):
        """Test successful substructure search"""
        # Mock molecule
        mock_mol = Mock()
        mock_pybel.readstring.return_value = mock_mol

        # Mock SMARTS pattern
        mock_pattern = Mock()
        mock_pattern.Init.return_value = True
        mock_pattern.GetUMapList.return_value = [[1, 2, 3], [4, 5, 6]]
        mock_ob.OBSmartsPattern.return_value = mock_pattern

        arguments = {
            "molecule_string": MOCK_MOLECULES["aspirin_smiles"],
            "pattern_string": "c1ccccc1",  # Benzene ring
        }

        result = search_tool.run(arguments)

        assert result["count"] == 2
        assert result["has_match"] is True
        assert result["matches"] == [[1, 2, 3], [4, 5, 6]]

    @patch("tooluniverse.openbabel_tool.ob")
    @patch("tooluniverse.openbabel_tool.pybel")
    def test_substructure_search_not_found(self, mock_pybel, mock_ob, search_tool):
        """Test substructure search with no matches"""
        mock_mol = Mock()
        mock_pybel.readstring.return_value = mock_mol

        mock_pattern = Mock()
        mock_pattern.Init.return_value = True
        mock_pattern.GetUMapList.return_value = []
        mock_ob.OBSmartsPattern.return_value = mock_pattern

        arguments = {
            "molecule_string": MOCK_MOLECULES["aspirin_smiles"],
            "pattern_string": "[Pt]",  # Platinum (not in aspirin)
        }

        result = search_tool.run(arguments)

        assert result["count"] == 0
        assert result["has_match"] is False
        assert result["matches"] == []

    @patch("tooluniverse.openbabel_tool.ob")
    def test_substructure_search_invalid_pattern(self, mock_ob, search_tool):
        """Test error handling for invalid SMARTS pattern"""
        with patch.object(search_tool, "_create_mol_from_input"):
            mock_pattern = Mock()
            mock_pattern.Init.return_value = False
            mock_ob.OBSmartsPattern.return_value = mock_pattern

            arguments = {
                "molecule_string": MOCK_MOLECULES["aspirin_smiles"],
                "pattern_string": "invalid_smarts",
            }

            result = search_tool.run(arguments)

            assert "error" in result
            assert "Invalid SMARTS pattern" in result["error"]


class TestChemicalFormulaToMassTool:
    """Test chemical formula to mass conversion tool"""

    @pytest.fixture
    def mock_openbabel_modules(self):
        with patch.dict(
            "sys.modules",
            {
                "openbabel": MagicMock(),
                "openbabel.pybel": MagicMock(),
                "openbabel.openbabel": MagicMock(),
            },
        ):
            with patch("tooluniverse.openbabel_tool.OPENBABEL_AVAILABLE", True):
                yield

    @pytest.fixture
    def mass_tool(self, mock_openbabel_modules):
        from tooluniverse.openbabel_tool import ChemicalFormulaToMassTool

        config = {"parameter": {"properties": {}}}
        return ChemicalFormulaToMassTool(config)

    @patch("tooluniverse.openbabel_tool.ob")
    def test_formula_to_mass_water(self, mock_ob, mass_tool):
        """Test mass calculation for water"""
        mock_formula = Mock()
        mock_formula.ParseFormula.return_value = True
        mock_formula.GetMolecularWeight.return_value = 18.015
        mock_formula.GetExactMass.return_value = 18.010565
        mock_ob.OBMolecularFormula.return_value = mock_formula

        arguments = {"formula": MOCK_MOLECULES["water_formula"]}

        result = mass_tool.run(arguments)

        assert result["formula"] == "H2O"
        assert result["molecular_weight"] == 18.015
        assert result["exact_mass"] == 18.010565

    @patch("tooluniverse.openbabel_tool.ob")
    def test_formula_to_mass_invalid(self, mock_ob, mass_tool):
        """Test error handling for invalid formula"""
        mock_formula = Mock()
        mock_formula.ParseFormula.return_value = False
        mock_ob.OBMolecularFormula.return_value = mock_formula

        arguments = {"formula": "InvalidFormula123"}

        result = mass_tool.run(arguments)

        assert "error" in result
        assert "Invalid chemical formula" in result["error"]


class TestGenerateMoleculeImageTool:
    """Test molecule image generation tool"""

    @pytest.fixture
    def mock_openbabel_modules(self):
        with patch.dict(
            "sys.modules",
            {
                "openbabel": MagicMock(),
                "openbabel.pybel": MagicMock(),
                "openbabel.openbabel": MagicMock(),
            },
        ):
            with patch("tooluniverse.openbabel_tool.OPENBABEL_AVAILABLE", True):
                yield

    @pytest.fixture
    def image_tool(self, mock_openbabel_modules):
        from tooluniverse.openbabel_tool import GenerateMoleculeImageTool

        config = {"parameter": {"properties": {}}}
        return GenerateMoleculeImageTool(config)

    @patch("tooluniverse.openbabel_tool.pybel")
    @patch("builtins.open")
    @patch("tempfile.NamedTemporaryFile")
    @patch("os.unlink")
    @patch("tooluniverse.openbabel_tool.base64")
    def test_generate_image_png(
        self, mock_base64, mock_unlink, mock_temp, mock_open, mock_pybel, image_tool
    ):
        """Test PNG image generation"""
        # Mock molecule
        mock_mol = Mock()
        mock_pybel.readstring.return_value = mock_mol

        # Mock temporary file
        mock_file = Mock()
        mock_file.name = "/tmp/test.png"
        mock_temp.return_value.__enter__.return_value = mock_file

        # Mock file reading
        mock_open.return_value.__enter__.return_value.read.return_value = (
            b"fake_png_data"
        )

        # Mock base64 encoding
        mock_base64.b64encode.return_value = b"encoded_image_data"

        arguments = {
            "input_string": MOCK_MOLECULES["aspirin_smiles"],
            "image_format": "png",
            "width": 400,
            "height": 300,
        }

        result = image_tool.run(arguments)

        assert result["image_data"] == "encoded_image_data"
        assert result["format"] == "png"
        assert result["width"] == 400
        assert result["height"] == 300

        # Verify molecule methods were called
        mock_mol.make2D.assert_called_once()
        mock_mol.draw.assert_called_once()


class TestIntegrationWithMockedOpenBabel:
    """Integration tests with mocked OpenBabel"""

    @pytest.fixture
    def mock_complete_openbabel(self):
        """Complete OpenBabel mock for integration testing"""
        with patch.dict(
            "sys.modules",
            {
                "openbabel": MagicMock(),
                "openbabel.pybel": MagicMock(),
                "openbabel.openbabel": MagicMock(),
            },
        ):
            with patch("tooluniverse.openbabel_tool.OPENBABEL_AVAILABLE", True):
                yield

    def test_workflow_smiles_to_properties(self, mock_complete_openbabel):
        """Test complete workflow: SMILES -> properties"""
        from tooluniverse.openbabel_tool import (
            ConvertMoleculeFormatTool,
            CalculateMolecularPropertiesTool,
        )

        config = {"parameter": {"properties": {}}}

        with patch("tooluniverse.openbabel_tool.pybel") as mock_pybel:
            # Mock molecule with properties
            mock_mol = Mock()
            mock_mol.molwt = 180.16
            mock_mol.formula = "C6H12O6"
            mock_mol.write.return_value = "mock_mol_output"
            mock_pybel.readstring.return_value = mock_mol

            # Test conversion
            convert_tool = ConvertMoleculeFormatTool(config)
            mol_result = convert_tool.run(
                {
                    "input_string": MOCK_MOLECULES["glucose_formula"],
                    "output_format": "mol",
                }
            )

            # Test properties
            props_tool = CalculateMolecularPropertiesTool(config)
            props_result = props_tool.run(
                {
                    "input_string": MOCK_MOLECULES["glucose_formula"],
                    "properties": ["molecular_weight", "formula"],
                }
            )

            assert mol_result == "mock_mol_output"
            assert props_result["molecular_weight"] == 180.16
            assert props_result["formula"] == "C6H12O6"


class TestErrorHandling:
    """Test comprehensive error handling"""

    @pytest.fixture
    def mock_openbabel_modules(self):
        with patch.dict(
            "sys.modules",
            {
                "openbabel": MagicMock(),
                "openbabel.pybel": MagicMock(),
                "openbabel.openbabel": MagicMock(),
            },
        ):
            with patch("tooluniverse.openbabel_tool.OPENBABEL_AVAILABLE", True):
                yield

    def test_all_tools_handle_exceptions(self, mock_openbabel_modules):
        """Test that all tools handle exceptions gracefully"""
        from tooluniverse.openbabel_tool import (
            ConvertMoleculeFormatTool,
            CalculateMolecularPropertiesTool,
            CalculateMolecularSimilarityTool,
            PerformSubstructureSearchTool,
            ChemicalFormulaToMassTool,
        )

        config = {"parameter": {"properties": {}}}
        tools = [
            ConvertMoleculeFormatTool(config),
            CalculateMolecularPropertiesTool(config),
            CalculateMolecularSimilarityTool(config),
            PerformSubstructureSearchTool(config),
            ChemicalFormulaToMassTool(config),
        ]

        # Test with invalid arguments that should cause exceptions
        for tool in tools:
            with patch.object(
                tool, "_create_mol_from_input", side_effect=Exception("Mock error")
            ):
                result = tool.run({"input_string": "invalid"})
                assert "error" in result
                assert isinstance(result["error"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
