# filepath: /Users/nicholas/Documents/GitHub/ToolUniverse/src/tooluniverse/openbabel_tool.py
# NOTE: The openbabel library needs to be installed separately with:
# pip install openbabel
try:
    from openbabel import pybel
    from openbabel import openbabel as ob

    OPENBABEL_AVAILABLE = True
except ImportError:
    # Allow for the module to be imported even if openbabel is not installed
    OPENBABEL_AVAILABLE = False

import tempfile
import os
import base64
from .base_tool import BaseTool


class OpenBabelBaseTool(BaseTool):
    """Base class for OpenBabel tools"""

    def __init__(self, tool_config):
        super().__init__(tool_config)
        if not OPENBABEL_AVAILABLE:
            raise ImportError(
                "OpenBabel is not available. Please install it with 'pip install openbabel'"
            )

    def _create_mol_from_input(self, input_string, input_format):
        """
        Helper method to create a molecule object from input string.

        Args:
            input_string (str): Molecular representation (SMILES, MOL, etc.)
            input_format (str): Format of the input_string (smiles, mol, etc.)

        Returns:
            pybel.Molecule: A molecule object
        """
        try:
            if input_format.lower() == "smiles":
                mol = pybel.readstring("smi", input_string)
            else:
                mol = pybel.readstring(input_format.lower(), input_string)
            return mol
        except Exception as e:
            raise ValueError(f"Error creating molecule from input: {str(e)}")


class ConvertMoleculeFormatTool(OpenBabelBaseTool):
    """Tool to convert molecular representation between different chemical file formats"""

    def __init__(self, tool_config):
        super().__init__(tool_config)

    def run(self, arguments):
        """
        Convert a molecule from one format to another.

        Args:
            arguments (dict): Contains:
                - input_string: Molecular representation in the input format
                - input_format: Format of the input string (smiles, mol, pdb, etc.)
                - output_format: Desired output format
                - generate_3d: Whether to generate 3D coordinates (default: False)

        Returns:
            str: Molecule in the requested output format
        """
        input_string = arguments.get("input_string")
        input_format = arguments.get("input_format", "smiles")
        output_format = arguments.get("output_format", "smi")
        generate_3d = arguments.get("generate_3d", False)

        try:
            mol = self._create_mol_from_input(input_string, input_format)

            if generate_3d:
                # Generate 3D coordinates if requested
                mol.make3D()

            # Convert to the output format
            return mol.write(output_format)

        except Exception as e:
            return {"error": f"Format conversion failed: {str(e)}"}


class Generate3DStructureTool(OpenBabelBaseTool):
    """Tool to generate 3D structures from SMILES or other molecular inputs"""

    def __init__(self, tool_config):
        super().__init__(tool_config)

    def run(self, arguments):
        """
        Generate a 3D structure from a molecular representation.

        Args:
            arguments (dict): Contains:
                - input_string: Molecular representation (typically SMILES)
                - input_format: Format of the input string (default: smiles)
                - output_format: Desired output format (default: pdb)
                - forcefield: Force field to use for optimization (default: MMFF94)
                - steps: Number of optimization steps (default: 500)

        Returns:
            dict: Contains:
                - structure: 3D structure in the requested format
                - energy: Final energy after optimization
        """
        input_string = arguments.get("input_string")
        input_format = arguments.get("input_format", "smiles")
        output_format = arguments.get("output_format", "pdb")
        forcefield_name = arguments.get("forcefield", "MMFF94")
        steps = arguments.get("steps", 500)

        try:
            mol = self._create_mol_from_input(input_string, input_format)

            # Generate 3D coordinates
            mol.make3D()

            # Optimize the structure
            ff = pybel._forcefields[forcefield_name]
            success = ff.Setup(mol.OBMol)
            if not success:
                return {"error": f"Could not set up force field {forcefield_name}"}

            ff.SteepestDescent(steps)
            ff.GetCoordinates(mol.OBMol)

            energy = ff.Energy()

            # Get the structure in the requested format
            structure = mol.write(output_format)

            return {"structure": structure, "energy": energy}

        except Exception as e:
            return {"error": f"3D structure generation failed: {str(e)}"}


class CalculateMolecularPropertiesTool(OpenBabelBaseTool):
    """Tool to calculate various molecular properties"""

    def __init__(self, tool_config):
        super().__init__(tool_config)

    def run(self, arguments):
        """
        Calculate various molecular properties of a compound.

        Args:
            arguments (dict): Contains:
                - input_string: Molecular representation (SMILES, MOL, etc.)
                - input_format: Format of the input string (default: smiles)
                - properties: List of properties to calculate (default: all available)

        Returns:
            dict: Calculated properties
        """
        input_string = arguments.get("input_string")
        input_format = arguments.get("input_format", "smiles")
        requested_props = arguments.get("properties", [])

        # Define available properties
        available_properties = [
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

        # If no specific properties requested, calculate all
        if not requested_props:
            requested_props = available_properties

        try:
            mol = self._create_mol_from_input(input_string, input_format)
            result = {}

            # Calculate requested properties
            for prop in requested_props:
                if prop == "molecular_weight":
                    result["molecular_weight"] = mol.molwt
                elif prop == "logp":
                    result["logp"] = mol.calcdesc()["logP"]
                elif prop == "psa":
                    result["psa"] = mol.calcdesc()["TPSA"]
                elif prop == "mr":
                    result["mr"] = mol.calcdesc()["MR"]
                elif prop == "atoms":
                    result["atoms"] = len(mol.atoms)
                elif prop == "bonds":
                    result["bonds"] = len(mol.OBMol.Bonds())
                elif prop == "hba":
                    result["hba"] = len(
                        [atom for atom in mol.atoms if atom.OBAtom.IsHbondAcceptor()]
                    )
                elif prop == "hbd":
                    result["hbd"] = len(
                        [atom for atom in mol.atoms if atom.OBAtom.IsHbondDonor()]
                    )
                elif prop == "rotatable_bonds":
                    result["rotatable_bonds"] = len(
                        [bond for bond in ob.OBMolBondIter(mol.OBMol) if bond.IsRotor()]
                    )
                elif prop == "exact_mass":
                    result["exact_mass"] = mol.exactmass
                elif prop == "formula":
                    result["formula"] = mol.formula

            return result

        except Exception as e:
            return {"error": f"Property calculation failed: {str(e)}"}


class CalculateMolecularFingerprintTool(OpenBabelBaseTool):
    """Tool to calculate molecular fingerprints"""

    def __init__(self, tool_config):
        super().__init__(tool_config)

    def run(self, arguments):
        """
        Calculate a molecular fingerprint.

        Args:
            arguments (dict): Contains:
                - input_string: Molecular representation (SMILES, MOL, etc.)
                - input_format: Format of the input string (default: smiles)
                - fp_type: Fingerprint type (default: FP2)
                  Options: FP2, FP3, FP4, MACCS
                - bits: Number of bits for fingerprint (used for some types)

        Returns:
            dict: Contains the fingerprint as a bit vector
        """
        input_string = arguments.get("input_string")
        input_format = arguments.get("input_format", "smiles")
        fp_type = arguments.get("fp_type", "FP2")
        # bits parameter is available for custom fingerprint types but not used for standard ones

        try:
            mol = self._create_mol_from_input(input_string, input_format)

            # Calculate fingerprint
            fp = mol.calcfp(fp_type)

            # Convert fingerprint to bit vector
            bit_vector = [1 if i in fp.bits else 0 for i in range(fp.bits.size)]

            return {
                "fingerprint": bit_vector,
                "fp_type": fp_type,
                "num_bits": len(bit_vector),
            }

        except Exception as e:
            return {"error": f"Fingerprint calculation failed: {str(e)}"}


class CalculateMolecularSimilarityTool(OpenBabelBaseTool):
    """Tool to calculate similarity between two molecules"""

    def __init__(self, tool_config):
        super().__init__(tool_config)

    def run(self, arguments):
        """
        Calculate similarity between two molecules.

        Args:
            arguments (dict): Contains:
                - query_string: First molecular representation (SMILES, MOL, etc.)
                - target_string: Second molecular representation
                - input_format: Format of the input strings (default: smiles)
                - fp_type: Fingerprint type (default: FP2)
                - similarity_metric: Similarity metric (default: tanimoto)
                  Options: tanimoto, dice

        Returns:
            dict: Contains similarity score
        """
        query_string = arguments.get("query_string")
        target_string = arguments.get("target_string")
        input_format = arguments.get("input_format", "smiles")
        fp_type = arguments.get("fp_type", "FP2")
        similarity_metric = arguments.get("similarity_metric", "tanimoto").lower()

        try:
            # Create molecules
            mol1 = self._create_mol_from_input(query_string, input_format)
            mol2 = self._create_mol_from_input(target_string, input_format)

            # Calculate fingerprints
            fp1 = mol1.calcfp(fp_type)
            fp2 = mol2.calcfp(fp_type)

            # Calculate similarity
            if similarity_metric == "tanimoto":
                similarity = fp1 | fp2
            elif similarity_metric == "dice":
                similarity = fp1.dice_similarity(fp2)
            else:
                return {"error": f"Unsupported similarity metric: {similarity_metric}"}

            return {
                "similarity": similarity,
                "fp_type": fp_type,
                "metric": similarity_metric,
            }

        except Exception as e:
            return {"error": f"Similarity calculation failed: {str(e)}"}


class PerformSubstructureSearchTool(OpenBabelBaseTool):
    """Tool to perform substructure search"""

    def __init__(self, tool_config):
        super().__init__(tool_config)

    def run(self, arguments):
        """
        Search for a substructure pattern within a molecule.

        Args:
            arguments (dict): Contains:
                - molecule_string: Molecule to search within (SMILES, MOL, etc.)
                - pattern_string: SMARTS pattern to search for
                - input_format: Format of molecule_string (default: smiles)

        Returns:
            dict: Contains match information
        """
        molecule_string = arguments.get("molecule_string")
        pattern_string = arguments.get("pattern_string")
        input_format = arguments.get("input_format", "smiles")

        try:
            # Create molecule
            mol = self._create_mol_from_input(molecule_string, input_format)

            # Create SMARTS pattern
            pattern = ob.OBSmartsPattern()
            if not pattern.Init(pattern_string):
                return {"error": "Invalid SMARTS pattern"}

            # Perform search
            pattern.Match(mol.OBMol)
            matches = pattern.GetUMapList()

            # Convert matches to atom indices (1-based in OpenBabel)
            matches_list = []
            for match in matches:
                matches_list.append(list(match))

            return {
                "matches": matches_list,
                "count": len(matches_list),
                "has_match": len(matches_list) > 0,
            }

        except Exception as e:
            return {"error": f"Substructure search failed: {str(e)}"}


class GenerateMoleculeImageTool(OpenBabelBaseTool):
    """Tool to generate 2D molecule images"""

    def __init__(self, tool_config):
        super().__init__(tool_config)

    def run(self, arguments):
        """
        Generate a 2D depiction of a molecule.

        Args:
            arguments (dict): Contains:
                - input_string: Molecular representation (SMILES, MOL, etc.)
                - input_format: Format of the input string (default: smiles)
                - image_format: Format of the output image (default: png)
                - width: Image width in pixels (default: 300)
                - height: Image height in pixels (default: 200)

        Returns:
            dict: Contains image data as base64 encoded string
        """
        input_string = arguments.get("input_string")
        input_format = arguments.get("input_format", "smiles")
        image_format = arguments.get("image_format", "png")
        width = arguments.get("width", 300)
        height = arguments.get("height", 200)

        try:
            mol = self._create_mol_from_input(input_string, input_format)

            # Generate 2D coordinates if needed
            mol.make2D()

            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                suffix=f".{image_format}", delete=False
            ) as tmp:
                tmp_path = tmp.name

            # Generate the image
            mol.draw(
                show=False,
                filename=tmp_path,
                width=width,
                height=height,
                usecoords=False,
            )

            # Read back the image and encode as base64
            with open(tmp_path, "rb") as f:
                image_data = f.read()

            # Clean up
            os.unlink(tmp_path)

            # Encode to base64
            encoded_image = base64.b64encode(image_data).decode("utf-8")

            return {
                "image_data": encoded_image,
                "format": image_format,
                "width": width,
                "height": height,
            }

        except Exception as e:
            return {"error": f"Image generation failed: {str(e)}"}


class ChemicalFormulaToMassTool(OpenBabelBaseTool):
    """Tool to calculate mass from a chemical formula"""

    def __init__(self, tool_config):
        super().__init__(tool_config)

    def run(self, arguments):
        """
        Calculate the mass of a chemical formula.

        Args:
            arguments (dict): Contains:
                - formula: Chemical formula (e.g., "H2O", "C6H12O6")

        Returns:
            dict: Contains the calculated mass
        """
        formula = arguments.get("formula")

        try:
            # Create an OB molecular formula object
            mol_formula = ob.OBMolecularFormula()

            # Parse the formula
            parsed_formula = mol_formula.ParseFormula(formula)

            if not parsed_formula:
                return {"error": "Invalid chemical formula"}

            # Calculate weight
            weight = mol_formula.GetMolecularWeight()
            exact_mass = mol_formula.GetExactMass()

            return {
                "formula": formula,
                "molecular_weight": weight,
                "exact_mass": exact_mass,
            }

        except Exception as e:
            return {"error": f"Formula parsing failed: {str(e)}"}


class GenerateConformersTool(OpenBabelBaseTool):
    """Tool to generate multiple conformers of a molecule"""

    def __init__(self, tool_config):
        super().__init__(tool_config)

    def run(self, arguments):
        """
        Generate multiple conformers of a molecule.

        Args:
            arguments (dict): Contains:
                - input_string: Molecular representation (SMILES, MOL, etc.)
                - input_format: Format of the input string (default: smiles)
                - output_format: Desired output format (default: sdf)
                - num_conformers: Number of conformers to generate (default: 10)
                - rmsd_cutoff: RMSD cutoff for conformer diversity (default: 0.5)
                - energy_window: Maximum energy difference from lowest energy conformer (default: 10.0)

        Returns:
            dict: Contains conformers in the requested format
        """
        input_string = arguments.get("input_string")
        input_format = arguments.get("input_format", "smiles")
        output_format = arguments.get("output_format", "sdf")
        num_conformers = arguments.get("num_conformers", 10)
        # These parameters are documented but not used in current implementation
        # as OpenBabel's conformer search has its own defaults

        try:
            # Create initial molecule
            mol = self._create_mol_from_input(input_string, input_format)

            # Generate initial 3D coordinates
            mol.make3D()

            # Use Open Babel's genetic algorithm conformer search
            obConversion = ob.OBConversion()
            obConversion.SetOutFormat(output_format)

            # Configure conformer search
            conformer_search = ob.OBConformerSearch()
            conformer_search.Setup(mol.OBMol, num_conformers)

            # Run conformer search
            conformer_search.Search()
            conformer_search.GetConformers(mol.OBMol)

            # Get number of conformers
            num_generated = mol.OBMol.NumConformers()

            # Extract conformers
            conformers = []
            energies = []

            # Use MMFF94 to score conformers
            ff = ob.OBForceField.FindForceField("MMFF94")

            # Score each conformer
            for i in range(num_generated):
                mol.OBMol.SetConformer(i)
                ff.Setup(mol.OBMol)
                energy = ff.Energy()
                conformers.append(obConversion.WriteString(mol.OBMol))
                energies.append(energy)

            return {
                "num_conformers": len(conformers),
                "conformers": conformers,
                "energies": energies,
                "format": output_format,
            }

        except Exception as e:
            return {"error": f"Conformer generation failed: {str(e)}"}


class MolecularDockingTool(OpenBabelBaseTool):
    """Tool to perform simple molecular docking"""

    def __init__(self, tool_config):
        super().__init__(tool_config)

    def run(self, arguments):
        """
        Perform a simple molecular docking using OpenBabel's built-in scoring function.

        Args:
            arguments (dict): Contains:
                - receptor_string: Receptor molecule (protein) representation
                - ligand_string: Ligand molecule representation
                - receptor_format: Format of receptor (default: pdb)
                - ligand_format: Format of ligand (default: sdf)
                - output_format: Format for docked results (default: sdf)
                - num_poses: Number of poses to generate (default: 10)

        Returns:
            dict: Contains docked poses and scores
        """
        receptor_string = arguments.get("receptor_string")
        ligand_string = arguments.get("ligand_string")
        receptor_format = arguments.get("receptor_format", "pdb")
        ligand_format = arguments.get("ligand_format", "sdf")
        output_format = arguments.get("output_format", "sdf")
        num_poses = arguments.get("num_poses", 10)

        try:
            # Create receptor and ligand molecules
            receptor = self._create_mol_from_input(receptor_string, receptor_format)
            ligand = self._create_mol_from_input(ligand_string, ligand_format)

            # Set up docking with AutoDock Vina scoring through pybel
            dock = ob.OBDock()
            dock.SetDimension(20.0)  # Set box size

            # Get approximate binding site - center of the receptor
            center_x = center_y = center_z = 0
            count = 0
            for atom in receptor.atoms:
                center_x += atom.coords[0]
                center_y += atom.coords[1]
                center_z += atom.coords[2]
                count += 1

            if count > 0:
                center_x /= count
                center_y /= count
                center_z /= count

            # Set box center
            dock.SetCenter(center_x, center_y, center_z)

            # Prepare for docking
            dock.Setup(receptor.OBMol, ligand.OBMol)

            # Perform docking
            success = dock.Dock(num_poses)
            if not success:
                return {"error": "Docking failed"}

            # Get docked poses
            poses = []
            scores = []

            obConversion = ob.OBConversion()
            obConversion.SetOutFormat(output_format)

            # Extract poses
            for i in range(dock.NumPoses()):
                # Get the pose
                posed_mol = ob.OBMol(ligand.OBMol)
                dock.GetPose(posed_mol, i)

                # Get the score
                score = dock.GetScore(i)

                # Convert to string
                pose_str = obConversion.WriteString(posed_mol)

                poses.append(pose_str)
                scores.append(score)

            return {
                "num_poses": len(poses),
                "poses": poses,
                "scores": scores,
                "format": output_format,
            }

        except Exception as e:
            return {"error": f"Docking failed: {str(e)}"}


class AnalyzeBondsTool(OpenBabelBaseTool):
    """Tool to analyze bonds in a molecule"""

    def __init__(self, tool_config):
        super().__init__(tool_config)

    def run(self, arguments):
        """
        Analyze the bonds in a molecule.

        Args:
            arguments (dict): Contains:
                - input_string: Molecular representation (SMILES, MOL, etc.)
                - input_format: Format of the input string (default: smiles)

        Returns:
            dict: Bond analysis results
        """
        input_string = arguments.get("input_string")
        input_format = arguments.get("input_format", "smiles")

        try:
            mol = self._create_mol_from_input(input_string, input_format)

            # Analyze bonds
            bonds = []

            for bond in ob.OBMolBondIter(mol.OBMol):
                # Get atoms involved in the bond
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()

                # Get element symbols
                begin_symbol = ob.etab.GetSymbol(begin_atom.GetAtomicNum())
                end_symbol = ob.etab.GetSymbol(end_atom.GetAtomicNum())

                # Get bond order
                bond_order = bond.GetBondOrder()

                # Determine bond type
                bond_type = "unknown"
                if bond.IsAromatic():
                    bond_type = "aromatic"
                elif bond.IsAmide():
                    bond_type = "amide"
                elif bond.IsPrimaryAmide():
                    bond_type = "primary amide"
                elif bond.IsSecondaryAmide():
                    bond_type = "secondary amide"
                elif bond.IsTertiaryAmide():
                    bond_type = "tertiary amide"
                elif bond.IsEster():
                    bond_type = "ester"
                elif bond.IsCarbonyl():
                    bond_type = "carbonyl"
                elif bond.IsSingle():
                    bond_type = "single"
                elif bond.IsDouble():
                    bond_type = "double"
                elif bond.IsTriple():
                    bond_type = "triple"

                # Get bond length if 3D coordinates are available
                bond_length = None
                if mol.OBMol.Has3D():
                    begin_coords = begin_atom.GetVector()
                    end_coords = end_atom.GetVector()
                    x_diff = begin_coords.GetX() - end_coords.GetX()
                    y_diff = begin_coords.GetY() - end_coords.GetY()
                    z_diff = begin_coords.GetZ() - end_coords.GetZ()
                    bond_length = (x_diff**2 + y_diff**2 + z_diff**2) ** 0.5

                # Store bond information
                bond_info = {
                    "begin_atom": begin_atom.GetIdx(),
                    "end_atom": end_atom.GetIdx(),
                    "begin_element": begin_symbol,
                    "end_element": end_symbol,
                    "bond_order": bond_order,
                    "bond_type": bond_type,
                    "is_rotatable": bond.IsRotor(),
                    "in_ring": bond.IsInRing(),
                }

                if bond_length is not None:
                    bond_info["bond_length"] = bond_length

                bonds.append(bond_info)

            return {"molecule": mol.formula, "num_bonds": len(bonds), "bonds": bonds}

        except Exception as e:
            return {"error": f"Bond analysis failed: {str(e)}"}


class MergeMultipleMoleculesTool(OpenBabelBaseTool):
    """Tool to merge multiple molecules into a single file"""

    def __init__(self, tool_config):
        super().__init__(tool_config)

    def run(self, arguments):
        """
        Merge multiple molecules into a single output file.

        Args:
            arguments (dict): Contains:
                - input_strings: List of molecular representations
                - input_format: Format of the input strings (default: smiles)
                - output_format: Format for output file (default: sdf)
                - generate_3d: Whether to generate 3D coordinates (default: False)

        Returns:
            str: Combined molecules in the requested format
        """
        input_strings = arguments.get("input_strings", [])
        input_format = arguments.get("input_format", "smiles")
        output_format = arguments.get("output_format", "sdf")
        generate_3d = arguments.get("generate_3d", False)

        if not input_strings:
            return {"error": "No input molecules provided"}

        try:
            # Initialize a combined string to hold all molecules
            combined_output = ""

            # Convert each molecule
            for input_str in input_strings:
                mol = self._create_mol_from_input(input_str, input_format)

                if generate_3d:
                    mol.make3D()

                # Add to the combined output
                combined_output += mol.write(output_format)

            return {
                "num_molecules": len(input_strings),
                "combined_data": combined_output,
                "format": output_format,
            }

        except Exception as e:
            return {"error": f"Molecule merging failed: {str(e)}"}


class FragmentMoleculeTool(OpenBabelBaseTool):
    """Tool to fragment a molecule into smaller parts"""

    def __init__(self, tool_config):
        super().__init__(tool_config)

    def run(self, arguments):
        """
        Fragment a molecule into smaller parts based on common fragmentation rules.

        Args:
            arguments (dict): Contains:
                - input_string: Molecular representation (SMILES, MOL, etc.)
                - input_format: Format of the input string (default: smiles)
                - output_format: Format for output fragments (default: smiles)
                - method: Fragmentation method (default: 'rotatable_bonds')
                  Options: 'rotatable_bonds', 'rings', 'functional_groups'

        Returns:
            dict: Contains the fragments
        """
        input_string = arguments.get("input_string")
        input_format = arguments.get("input_format", "smiles")
        output_format = arguments.get("output_format", "smiles")
        method = arguments.get("method", "rotatable_bonds")

        try:
            mol = self._create_mol_from_input(input_string, input_format)

            # Create a list to store fragments
            fragments = []

            if method == "rotatable_bonds":
                # Fragment at rotatable bonds
                # Create a list of rotatable bonds
                rot_bonds = []
                for bond in ob.OBMolBondIter(mol.OBMol):
                    if bond.IsRotor() and not bond.IsInRing():
                        rot_bonds.append(bond.GetIdx())

                if not rot_bonds:
                    # No rotatable bonds, return the original molecule
                    fragments.append(mol.write(output_format))
                else:
                    # Fragment at each rotatable bond
                    for bond_idx in rot_bonds:
                        # Create a copy of the molecule
                        work_mol = ob.OBMol(mol.OBMol)

                        # Delete the bond
                        work_mol.DeleteBond(work_mol.GetBond(bond_idx))

                        # Find all disconnected fragments
                        fragments_found = work_mol.Separate()

                        # Convert fragments to desired output format
                        for frag in fragments_found:
                            pybel_frag = pybel.Molecule(frag)
                            fragments.append(pybel_frag.write(output_format))

            elif method == "rings":
                # Fragment the molecule and keep the rings intact
                for bond in ob.OBMolBondIter(mol.OBMol):
                    if not bond.IsInRing() and bond.GetBondOrder() == 1:
                        # Create a copy of the molecule
                        work_mol = ob.OBMol(mol.OBMol)

                        # Delete the bond
                        work_mol.DeleteBond(work_mol.GetBond(bond.GetIdx()))

                        # Find all disconnected fragments
                        fragments_found = work_mol.Separate()

                        # Convert fragments to desired output format
                        for frag in fragments_found:
                            pybel_frag = pybel.Molecule(frag)
                            fragments.append(pybel_frag.write(output_format))

                if not fragments:
                    # No suitable bonds for fragmentation, return the original molecule
                    fragments.append(mol.write(output_format))

            elif method == "functional_groups":
                # This is a simplified approach to identify common functional groups
                # A more comprehensive implementation would use SMARTS patterns

                # Define SMARTS patterns for common functional groups
                functional_groups = {
                    "carboxylic_acid": "[CX3](=O)[OX2H]",
                    "alcohol": "[OX2H]",
                    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
                    "amide": "[NX3][CX3](=[OX1])",
                    "ester": "[CX3](=O)[OX2][CX4]",
                    "aldehyde": "[CX3H1](=O)",
                    "ketone": "[CX3](=O)[CX4]",
                    "ether": "[OX2]([CX4])[CX4]",
                }

                # Identify functional groups
                group_matches = {}
                for group_name, smarts in functional_groups.items():
                    pattern = ob.OBSmartsPattern()
                    pattern.Init(smarts)
                    pattern.Match(mol.OBMol)
                    matches = pattern.GetUMapList()

                    if matches:
                        group_matches[group_name] = matches

                # Create fragments around functional groups
                original_smiles = mol.write("smiles").strip()
                for group_name, matches in group_matches.items():
                    for match in matches:
                        # For simplicity, just highlight the matched atoms
                        # A real implementation would do proper fragmentation
                        smiles_with_highlight = original_smiles
                        fragments.append(
                            {
                                "group": group_name,
                                "smiles": smiles_with_highlight,
                                "atoms": list(match),
                            }
                        )

            else:
                return {"error": f"Unsupported fragmentation method: {method}"}

            # Remove duplicates
            unique_fragments = []
            for frag in fragments:
                if isinstance(frag, str) and frag not in unique_fragments:
                    unique_fragments.append(frag)
                elif isinstance(frag, dict) and frag not in unique_fragments:
                    unique_fragments.append(frag)

            return {
                "input_molecule": mol.write("smiles").strip(),
                "num_fragments": len(unique_fragments),
                "fragments": unique_fragments,
                "method": method,
            }

        except Exception as e:
            return {"error": f"Fragmentation failed: {str(e)}"}
