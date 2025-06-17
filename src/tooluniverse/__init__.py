from .execute_function import ToolUniverse
from .restful_tool import MonarchTool, MonarchDiseasesForMultiplePhenoTool
from .graphql_tool import (
    OpentargetTool,
    OpentargetGeneticsTool,
    OpentargetToolDrugNameMatch,
)
from .openfda_tool import (
    FDADrugLabelTool,
    FDADrugLabelSearchTool,
    FDADrugLabelSearchIDTool,
    FDADrugLabelGetDrugGenericNameTool,
)
import importlib

# List of all tools
__all__ = [
    "ToolUniverse",
    "MonarchTool",
    "MonarchDiseasesForMultiplePhenoTool",
    "OpentargetTool",
    "OpentargetGeneticsTool",
    "OpentargetToolDrugNameMatch",
    "FDADrugLabelTool",
    "FDADrugLabelSearchTool",
    "FDADrugLabelSearchIDTool",
    "FDADrugLabelGetDrugGenericNameTool",
    # OpenBabel tools are included in __all__ but only loaded when accessed
    "ConvertMoleculeFormatTool",
    "Generate3DStructureTool",
    "CalculateMolecularPropertiesTool",
    "CalculateMolecularFingerprintTool",
    "CalculateMolecularSimilarityTool",
    "PerformSubstructureSearchTool",
    "GenerateMoleculeImageTool",
    "ChemicalFormulaToMassTool",
    "GenerateConformersTool",
    "MolecularDockingTool",
    "AnalyzeBondsTool",
    "MergeMultipleMoleculesTool",
    "FragmentMoleculeTool",
]

# OpenBabel tools that will be lazily imported
_OPENBABEL_TOOLS = {
    "ConvertMoleculeFormatTool",
    "Generate3DStructureTool",
    "CalculateMolecularPropertiesTool",
    "CalculateMolecularFingerprintTool",
    "CalculateMolecularSimilarityTool",
    "PerformSubstructureSearchTool",
    "GenerateMoleculeImageTool",
    "ChemicalFormulaToMassTool",
    "GenerateConformersTool",
    "MolecularDockingTool",
    "AnalyzeBondsTool",
    "MergeMultipleMoleculesTool",
    "FragmentMoleculeTool",
}


# Lazy import mechanism for OpenBabel tools
def __getattr__(name):
    if name in _OPENBABEL_TOOLS:
        try:
            # Import the OpenBabel tool module on demand
            module = importlib.import_module(".openbabel_tool", package="tooluniverse")
            return getattr(module, name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"The OpenBabel tool '{name}' is not available. "
                "Please install OpenBabel with 'pip install openbabel'"
            ) from e
    raise AttributeError(f"module 'tooluniverse' has no attribute '{name}'")
