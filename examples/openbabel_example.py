# Example usage of OpenBabel tools
import json
from tooluniverse.openbabel_tool import (
    ConvertMoleculeFormatTool,
    Generate3DStructureTool,
    CalculateMolecularPropertiesTool,
    CalculateMolecularSimilarityTool,
)

# Load tool configuration
with open("src/tooluniverse/data/openbabel_tools.json", "r") as f:
    tools_config = json.load(f)


def run_example():
    print("OpenBabel Tool Examples\n")

    # Example 1: Convert SMILES to PDB format
    print("Example 1: Convert SMILES to PDB format")
    convert_tool = ConvertMoleculeFormatTool(tools_config["ConvertMoleculeFormat"])

    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    result = convert_tool.run(
        {
            "input_string": aspirin_smiles,
            "input_format": "smiles",
            "output_format": "pdb",
            "generate_3d": True,
        }
    )

    print(f"Converted aspirin from SMILES to PDB with 3D coordinates:")
    # Print first few lines of the PDB file
    print("\n".join(result.strip().split("\n")[:10]))
    print("...\n")

    # Example 2: Calculate molecular properties
    print("Example 2: Calculate molecular properties")
    properties_tool = CalculateMolecularPropertiesTool(
        tools_config["CalculateMolecularProperties"]
    )

    result = properties_tool.run(
        {
            "input_string": aspirin_smiles,
            "properties": ["molecular_weight", "logp", "formula", "rotatable_bonds"],
        }
    )

    print("Aspirin properties:")
    for prop, value in result.items():
        print(f"  {prop}: {value}")
    print()

    # Example 3: Calculate molecular similarity
    print("Example 3: Calculate molecular similarity")
    similarity_tool = CalculateMolecularSimilarityTool(
        tools_config["CalculateMolecularSimilarity"]
    )

    # Compare aspirin and ibuprofen
    ibuprofen_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    result = similarity_tool.run(
        {
            "query_string": aspirin_smiles,
            "target_string": ibuprofen_smiles,
            "similarity_metric": "tanimoto",
        }
    )

    print(f"Similarity between aspirin and ibuprofen:")
    print(f"  Similarity score: {result['similarity']}")
    print(f"  Method: {result['metric']} using {result['fp_type']} fingerprints")
    print()

    # Example 4: Generate 3D structure
    print("Example 4: Generate 3D structure with energy minimization")
    structure_tool = Generate3DStructureTool(tools_config["Generate3DStructure"])

    caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    result = structure_tool.run(
        {
            "input_string": caffeine_smiles,
            "output_format": "mol2",
            "forcefield": "MMFF94",
            "steps": 250,
        }
    )

    print(f"Generated 3D structure for caffeine:")
    print(f"  Final energy: {result['energy']} kcal/mol")
    # Print first few lines of the MOL2 file
    print("\n".join(result["structure"].strip().split("\n")[:10]))
    print("...\n")


if __name__ == "__main__":
    try:
        run_example()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease make sure OpenBabel is installed:")
        print("pip install openbabel")
