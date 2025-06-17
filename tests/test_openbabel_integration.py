"""
Integration tests for OpenBabel tools.
These tests run only if OpenBabel is actually installed.
"""

import pytest
import json
import os
from pathlib import Path

# Try to import OpenBabel to check if it's available
try:
    import openbabel

    OPENBABEL_AVAILABLE = True
except ImportError:
    OPENBABEL_AVAILABLE = False

# Load actual tool configurations
config_path = (
    Path(__file__).parent.parent
    / "src"
    / "tooluniverse"
    / "data"
    / "openbabel_tools.json"
)
if config_path.exists():
    with open(config_path) as f:
        TOOL_CONFIGS = json.load(f)
else:
    TOOL_CONFIGS = {}

# Test molecules
TEST_MOLECULES = {
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "water": "O",
    "ethanol": "CCO",
    "benzene": "c1ccccc1",
}


@pytest.mark.skipif(not OPENBABEL_AVAILABLE, reason="OpenBabel not installed")
class TestOpenBabelIntegration:
    """Integration tests with real OpenBabel library"""

    def test_convert_molecule_format_real(self):
        """Test real molecule format conversion"""
        if "ConvertMoleculeFormat" not in TOOL_CONFIGS:
            pytest.skip("ConvertMoleculeFormat config not found")

        from tooluniverse.openbabel_tool import ConvertMoleculeFormatTool

        tool = ConvertMoleculeFormatTool(TOOL_CONFIGS["ConvertMoleculeFormat"])

        # Convert aspirin SMILES to MOL format
        result = tool.run(
            {
                "input_string": TEST_MOLECULES["aspirin"],
                "input_format": "smiles",
                "output_format": "mol",
            }
        )

        # Should return a string containing MOL format data
        assert isinstance(result, str)
        assert "M  END" in result  # MOL format ending
        assert len(result) > 100  # Should be substantial content

    def test_calculate_molecular_properties_real(self):
        """Test real molecular property calculation"""
        if "CalculateMolecularProperties" not in TOOL_CONFIGS:
            pytest.skip("CalculateMolecularProperties config not found")

        from tooluniverse.openbabel_tool import CalculateMolecularPropertiesTool

        tool = CalculateMolecularPropertiesTool(
            TOOL_CONFIGS["CalculateMolecularProperties"]
        )

        # Calculate properties for water
        result = tool.run(
            {
                "input_string": TEST_MOLECULES["water"],
                "properties": ["molecular_weight", "formula", "atoms"],
            }
        )

        # Water should have specific properties
        assert isinstance(result, dict)
        assert "molecular_weight" in result
        assert "formula" in result
        assert "atoms" in result

        # Check that water has approximately correct molecular weight
        assert abs(result["molecular_weight"] - 18.015) < 0.1
        assert result["formula"] == "H2O"
        assert result["atoms"] == 3  # 2 H + 1 O

    def test_calculate_similarity_real(self):
        """Test real molecular similarity calculation"""
        if "CalculateMolecularSimilarity" not in TOOL_CONFIGS:
            pytest.skip("CalculateMolecularSimilarity config not found")

        from tooluniverse.openbabel_tool import CalculateMolecularSimilarityTool

        tool = CalculateMolecularSimilarityTool(
            TOOL_CONFIGS["CalculateMolecularSimilarity"]
        )

        # Compare water with itself (should be 1.0)
        result = tool.run(
            {
                "query_string": TEST_MOLECULES["water"],
                "target_string": TEST_MOLECULES["water"],
                "similarity_metric": "tanimoto",
            }
        )

        assert isinstance(result, dict)
        assert "similarity" in result
        assert result["similarity"] == 1.0  # Identical molecules
        assert result["metric"] == "tanimoto"

    def test_substructure_search_real(self):
        """Test real substructure search"""
        if "PerformSubstructureSearch" not in TOOL_CONFIGS:
            pytest.skip("PerformSubstructureSearch config not found")

        from tooluniverse.openbabel_tool import PerformSubstructureSearchTool

        tool = PerformSubstructureSearchTool(TOOL_CONFIGS["PerformSubstructureSearch"])

        # Search for benzene ring in aspirin
        result = tool.run(
            {
                "molecule_string": TEST_MOLECULES["aspirin"],
                "pattern_string": "c1ccccc1",  # Benzene ring SMARTS
            }
        )

        assert isinstance(result, dict)
        assert "has_match" in result
        assert "count" in result
        assert "matches" in result

        # Aspirin contains a benzene ring
        assert result["has_match"] is True
        assert result["count"] > 0

    def test_formula_to_mass_real(self):
        """Test real chemical formula to mass calculation"""
        if "ChemicalFormulaToMass" not in TOOL_CONFIGS:
            pytest.skip("ChemicalFormulaToMass config not found")

        from tooluniverse.openbabel_tool import ChemicalFormulaToMassTool

        tool = ChemicalFormulaToMassTool(TOOL_CONFIGS["ChemicalFormulaToMass"])

        # Calculate mass of water
        result = tool.run({"formula": "H2O"})

        assert isinstance(result, dict)
        assert "molecular_weight" in result
        assert "exact_mass" in result
        assert "formula" in result

        # Water molecular weight should be approximately 18.015
        assert abs(result["molecular_weight"] - 18.015) < 0.1
        assert result["formula"] == "H2O"


@pytest.mark.skipif(not OPENBABEL_AVAILABLE, reason="OpenBabel not installed")
class TestOpenBabelWorkflows:
    """Test complete workflows using OpenBabel tools"""

    def test_smiles_to_3d_workflow(self):
        """Test workflow: SMILES -> 3D structure"""
        if "Generate3DStructure" not in TOOL_CONFIGS:
            pytest.skip("Generate3DStructure config not found")

        from tooluniverse.openbabel_tool import Generate3DStructureTool

        tool = Generate3DStructureTool(TOOL_CONFIGS["Generate3DStructure"])

        # Generate 3D structure for ethanol
        result = tool.run(
            {"input_string": TEST_MOLECULES["ethanol"], "output_format": "pdb"}
        )

        assert isinstance(result, dict)
        assert "structure" in result
        assert "energy" in result

        # PDB format should contain coordinate information
        assert "ATOM" in result["structure"] or "HETATM" in result["structure"]
        assert isinstance(result["energy"], (int, float))

    def test_molecule_analysis_workflow(self):
        """Test workflow: molecule analysis with multiple tools"""
        configs_needed = [
            "CalculateMolecularProperties",
            "CalculateMolecularFingerprint",
        ]
        for config in configs_needed:
            if config not in TOOL_CONFIGS:
                pytest.skip(f"{config} config not found")

        from tooluniverse.openbabel_tool import (
            CalculateMolecularPropertiesTool,
            CalculateMolecularFingerprintTool,
        )

        # Analyze caffeine
        props_tool = CalculateMolecularPropertiesTool(
            TOOL_CONFIGS["CalculateMolecularProperties"]
        )
        fp_tool = CalculateMolecularFingerprintTool(
            TOOL_CONFIGS["CalculateMolecularFingerprint"]
        )

        # Get properties
        props_result = props_tool.run(
            {
                "input_string": TEST_MOLECULES["caffeine"],
                "properties": ["molecular_weight", "logp", "formula"],
            }
        )

        # Get fingerprint
        fp_result = fp_tool.run(
            {"input_string": TEST_MOLECULES["caffeine"], "fp_type": "FP2"}
        )

        # Both should succeed
        assert isinstance(props_result, dict)
        assert "molecular_weight" in props_result

        assert isinstance(fp_result, dict)
        assert "fingerprint" in fp_result
        assert isinstance(fp_result["fingerprint"], list)


class TestConfigurationValidation:
    """Test that tool configurations are valid"""

    def test_config_file_exists(self):
        """Test that OpenBabel config file exists"""
        assert config_path.exists(), "OpenBabel tools configuration file not found"

    def test_config_structure(self):
        """Test configuration file structure"""
        if not TOOL_CONFIGS:
            pytest.skip("No tool configurations loaded")

        required_tools = [
            "ConvertMoleculeFormat",
            "CalculateMolecularProperties",
            "CalculateMolecularSimilarity",
        ]

        for tool_name in required_tools:
            assert tool_name in TOOL_CONFIGS, f"Missing configuration for {tool_name}"

            tool_config = TOOL_CONFIGS[tool_name]
            assert "description" in tool_config
            assert "parameter" in tool_config
            assert "properties" in tool_config["parameter"]

    def test_required_parameters(self):
        """Test that required parameters are correctly defined"""
        if not TOOL_CONFIGS:
            pytest.skip("No tool configurations loaded")

        # Check ConvertMoleculeFormat requires input_string
        if "ConvertMoleculeFormat" in TOOL_CONFIGS:
            config = TOOL_CONFIGS["ConvertMoleculeFormat"]
            required = config["parameter"].get("required", [])
            assert "input_string" in required

        # Check CalculateMolecularSimilarity requires both molecules
        if "CalculateMolecularSimilarity" in TOOL_CONFIGS:
            config = TOOL_CONFIGS["CalculateMolecularSimilarity"]
            required = config["parameter"].get("required", [])
            assert "query_string" in required
            assert "target_string" in required


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
