"""
Protein structure prediction and analysis module.

This module provides interfaces to various structure prediction tools including
AlphaFold2/3, RoseTTAFold, and custom diffusion models for 3D structure prediction.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import requests
import tempfile
import os
from pathlib import Path
from loguru import logger

try:
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb
    import biotite.database.rcsb as rcsb
    BIOTITE_AVAILABLE = True
except ImportError:
    BIOTITE_AVAILABLE = False
    logger.warning("Biotite not available. Some structure analysis features will be limited.")

try:
    from Bio import PDB
    from Bio.PDB import DSSP
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    logger.warning("BioPython not available. Some PDB parsing features will be limited.")


class StructurePredictor:
    """
    Unified interface for protein structure prediction and analysis.
    
    Supports multiple prediction methods:
    - AlphaFold database lookup
    - Custom structure prediction models
    - Structure analysis and comparison
    """
    
    def __init__(
        self,
        alphafold_db_url: str = "https://alphafold.ebi.ac.uk/api/prediction/",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize structure predictor.
        
        Args:
            alphafold_db_url: URL for AlphaFold database API
            cache_dir: Directory to cache downloaded structures
        """
        self.alphafold_db_url = alphafold_db_url
        self.cache_dir = cache_dir or tempfile.gettempdir()
        
        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Structure predictor initialized")
        
    def predict_structure(
        self, 
        sequence: str, 
        method: str = "alphafold",
        uniprot_id: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Predict protein structure for given sequence.
        
        Args:
            sequence: Protein sequence
            method: Prediction method ('alphafold', 'custom')
            uniprot_id: UniProt ID for AlphaFold lookup
            
        Returns:
            Dictionary containing structure prediction results
        """
        if method == "alphafold":
            return self._predict_alphafold(sequence, uniprot_id)
        elif method == "custom":
            return self._predict_custom(sequence)
        else:
            raise ValueError(f"Unsupported prediction method: {method}")
            
    def _predict_alphafold(
        self, 
        sequence: str, 
        uniprot_id: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Get structure prediction from AlphaFold database.
        
        Args:
            sequence: Protein sequence
            uniprot_id: UniProt ID for lookup
            
        Returns:
            Structure prediction results
        """
        if not uniprot_id:
            logger.warning("UniProt ID required for AlphaFold lookup")
            return {"error": "UniProt ID required for AlphaFold database lookup"}
            
        try:
            # Query AlphaFold database
            url = f"{self.alphafold_db_url}{uniprot_id}"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Download PDB file if available
                pdb_url = data[0].get("pdbUrl") if data else None
                if pdb_url:
                    pdb_content = self._download_pdb(pdb_url, uniprot_id)
                    structure_analysis = self._analyze_structure(pdb_content)
                    
                    return {
                        "uniprot_id": uniprot_id,
                        "sequence": sequence,
                        "alphafold_data": data[0],
                        "pdb_content": pdb_content,
                        "structure_analysis": structure_analysis,
                        "confidence_scores": self._extract_confidence_scores(data[0]),
                        "method": "alphafold"
                    }
                else:
                    return {
                        "error": f"No structure available for {uniprot_id}",
                        "alphafold_data": data
                    }
            else:
                return {
                    "error": f"AlphaFold lookup failed: {response.status_code}",
                    "uniprot_id": uniprot_id
                }
                
        except Exception as e:
            logger.error(f"AlphaFold prediction failed: {e}")
            return {"error": str(e)}
            
    def _predict_custom(self, sequence: str) -> Dict[str, any]:
        """
        Custom structure prediction (placeholder for future implementation).
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Structure prediction results
        """
        logger.info("Custom structure prediction not yet implemented")
        return {
            "sequence": sequence,
            "method": "custom",
            "status": "not_implemented",
            "message": "Custom structure prediction will be implemented in future versions"
        }
        
    def _download_pdb(self, pdb_url: str, identifier: str) -> Optional[str]:
        """
        Download PDB file from URL.
        
        Args:
            pdb_url: URL to PDB file
            identifier: Identifier for caching
            
        Returns:
            PDB file content as string
        """
        try:
            cache_path = Path(self.cache_dir) / f"{identifier}.pdb"
            
            # Check cache first
            if cache_path.exists():
                logger.info(f"Loading cached PDB file: {cache_path}")
                return cache_path.read_text()
                
            # Download PDB file
            logger.info(f"Downloading PDB file: {pdb_url}")
            response = requests.get(pdb_url, timeout=60)
            
            if response.status_code == 200:
                pdb_content = response.text
                
                # Cache the file
                cache_path.write_text(pdb_content)
                logger.success(f"PDB file cached: {cache_path}")
                
                return pdb_content
            else:
                logger.error(f"Failed to download PDB: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"PDB download failed: {e}")
            return None
            
    def _analyze_structure(self, pdb_content: Optional[str]) -> Dict[str, any]:
        """
        Analyze protein structure from PDB content.
        
        Args:
            pdb_content: PDB file content
            
        Returns:
            Structure analysis results
        """
        if not pdb_content:
            return {"error": "No PDB content available"}
            
        analysis = {}
        
        try:
            if BIOTITE_AVAILABLE:
                analysis.update(self._biotite_analysis(pdb_content))
            
            if BIOPYTHON_AVAILABLE:
                analysis.update(self._biopython_analysis(pdb_content))
                
            if not BIOTITE_AVAILABLE and not BIOPYTHON_AVAILABLE:
                analysis = self._basic_pdb_analysis(pdb_content)
                
        except Exception as e:
            logger.error(f"Structure analysis failed: {e}")
            analysis["error"] = str(e)
            
        return analysis
        
    def _biotite_analysis(self, pdb_content: str) -> Dict[str, any]:
        """Analyze structure using Biotite library."""
        analysis = {}
        
        try:
            # Parse PDB content
            pdb_file = pdb.PDBFile.read(pdb_content.split('\n'))
            structure = pdb_file.get_structure()
            
            # Basic structure information
            analysis["num_atoms"] = len(structure)
            analysis["num_residues"] = len(struc.get_residues(structure))
            analysis["num_chains"] = len(struc.get_chains(structure))
            
            # Secondary structure assignment (if possible)
            try:
                ca_atoms = structure[struc.filter_amino_acids(structure)]
                ca_atoms = ca_atoms[ca_atoms.atom_name == "CA"]
                
                if len(ca_atoms) > 0:
                    # Compute basic geometric properties
                    coords = ca_atoms.coord
                    center_of_mass = np.mean(coords, axis=0)
                    radius_of_gyration = np.sqrt(np.mean(np.sum((coords - center_of_mass)**2, axis=1)))
                    
                    analysis["center_of_mass"] = center_of_mass.tolist()
                    analysis["radius_of_gyration"] = float(radius_of_gyration)
                    analysis["ca_atoms"] = len(ca_atoms)
                    
            except Exception as e:
                logger.warning(f"Biotite geometric analysis failed: {e}")
                
        except Exception as e:
            logger.warning(f"Biotite structure analysis failed: {e}")
            
        return analysis
        
    def _biopython_analysis(self, pdb_content: str) -> Dict[str, any]:
        """Analyze structure using BioPython."""
        analysis = {}
        
        try:
            # Parse PDB content
            parser = PDB.PDBParser(QUIET=True)
            
            # Write to temporary file for BioPython
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_file:
                tmp_file.write(pdb_content)
                tmp_file_path = tmp_file.name
                
            try:
                structure = parser.get_structure('protein', tmp_file_path)
                
                # Count chains, residues, atoms
                chains = list(structure.get_chains())
                residues = list(structure.get_residues())
                atoms = list(structure.get_atoms())
                
                analysis.update({
                    "biopython_num_chains": len(chains),
                    "biopython_num_residues": len(residues),
                    "biopython_num_atoms": len(atoms)
                })
                
                # Secondary structure analysis with DSSP (if available)
                try:
                    dssp = DSSP(structure[0], tmp_file_path)
                    ss_counts = {"H": 0, "B": 0, "E": 0, "G": 0, "I": 0, "T": 0, "S": 0, "-": 0}
                    
                    for residue in dssp:
                        ss = residue[2]
                        if ss in ss_counts:
                            ss_counts[ss] += 1
                        else:
                            ss_counts["-"] += 1
                            
                    analysis["secondary_structure"] = ss_counts
                    
                except Exception as e:
                    logger.warning(f"DSSP analysis failed: {e}")
                    
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
        except Exception as e:
            logger.warning(f"BioPython structure analysis failed: {e}")
            
        return analysis
        
    def _basic_pdb_analysis(self, pdb_content: str) -> Dict[str, any]:
        """Basic PDB analysis without external libraries."""
        lines = pdb_content.split('\n')
        
        atom_count = 0
        residue_set = set()
        chain_set = set()
        
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_count += 1
                
                if len(line) >= 26:
                    chain_id = line[21:22].strip()
                    res_num = line[22:26].strip()
                    
                    if chain_id:
                        chain_set.add(chain_id)
                    if res_num:
                        residue_set.add((chain_id, res_num))
                        
        return {
            "basic_num_atoms": atom_count,
            "basic_num_residues": len(residue_set),
            "basic_num_chains": len(chain_set),
            "chains": list(chain_set)
        }
        
    def _extract_confidence_scores(self, alphafold_data: Dict) -> Dict[str, any]:
        """
        Extract confidence scores from AlphaFold data.
        
        Args:
            alphafold_data: AlphaFold database response
            
        Returns:
            Confidence score information
        """
        confidence_info = {}
        
        try:
            if "confidenceScore" in alphafold_data:
                confidence_info["overall_confidence"] = alphafold_data["confidenceScore"]
                
            if "confidenceType" in alphafold_data:
                confidence_info["confidence_type"] = alphafold_data["confidenceType"]
                
            # Extract per-residue confidence if available
            if "confidenceScores" in alphafold_data:
                confidence_info["per_residue_confidence"] = alphafold_data["confidenceScores"]
                
        except Exception as e:
            logger.warning(f"Failed to extract confidence scores: {e}")
            
        return confidence_info
        
    def compare_structures(
        self, 
        structure1: Dict[str, any], 
        structure2: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Compare two protein structures.
        
        Args:
            structure1, structure2: Structure prediction results
            
        Returns:
            Structure comparison results
        """
        comparison = {
            "structure1_id": structure1.get("uniprot_id", "unknown"),
            "structure2_id": structure2.get("uniprot_id", "unknown"),
        }
        
        # Compare basic properties
        analysis1 = structure1.get("structure_analysis", {})
        analysis2 = structure2.get("structure_analysis", {})
        
        if analysis1 and analysis2:
            # Compare atom/residue counts
            for key in ["num_atoms", "num_residues", "num_chains"]:
                if key in analysis1 and key in analysis2:
                    comparison[f"{key}_diff"] = abs(analysis1[key] - analysis2[key])
                    
            # Compare radius of gyration if available
            if "radius_of_gyration" in analysis1 and "radius_of_gyration" in analysis2:
                rog_diff = abs(analysis1["radius_of_gyration"] - analysis2["radius_of_gyration"])
                comparison["radius_of_gyration_diff"] = rog_diff
                
        return comparison
        
    def analyze_binding_site(
        self, 
        structure_data: Dict[str, any],
        ligand_coords: Optional[List[float]] = None
    ) -> Dict[str, any]:
        """
        Analyze potential binding sites in protein structure.
        
        Args:
            structure_data: Structure prediction results
            ligand_coords: Known ligand coordinates (optional)
            
        Returns:
            Binding site analysis
        """
        # Placeholder for binding site analysis
        # This would typically involve cavity detection algorithms
        
        analysis = {
            "method": "placeholder",
            "message": "Binding site analysis will be implemented with cavity detection algorithms",
            "structure_id": structure_data.get("uniprot_id", "unknown")
        }
        
        if ligand_coords:
            analysis["ligand_coords"] = ligand_coords
            analysis["has_reference_ligand"] = True
        else:
            analysis["has_reference_ligand"] = False
            
        return analysis
        
    def __repr__(self) -> str:
        return f"StructurePredictor(cache_dir={self.cache_dir})"