"""
Protein function prediction module.

This module provides tools for predicting protein function using graph neural networks,
sequence analysis, and interaction network analysis.
"""

from typing import Dict, List, Optional, Tuple, Union, Set
import numpy as np
from loguru import logger
import requests
import json

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available. Graph analysis features will be limited.")


class FunctionPredictor:
    """
    Protein function prediction using multiple approaches:
    - Sequence-based function prediction
    - Protein-protein interaction networks
    - Domain and motif analysis
    - GO term prediction
    """
    
    def __init__(
        self,
        string_db_url: str = "https://string-db.org/api",
        uniprot_api_url: str = "https://rest.uniprot.org/uniprotkb",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize function predictor.
        
        Args:
            string_db_url: STRING database API URL
            uniprot_api_url: UniProt API URL  
            cache_dir: Directory to cache results
        """
        self.string_db_url = string_db_url
        self.uniprot_api_url = uniprot_api_url
        self.cache_dir = cache_dir
        
        # Function prediction models (placeholders for future ML models)
        self.go_predictor = None
        self.domain_predictor = None
        
        logger.info("Function predictor initialized")
        
    def predict_function(
        self,
        sequence: str,
        uniprot_id: Optional[str] = None,
        methods: List[str] = ["sequence", "network", "domains"]
    ) -> Dict[str, any]:
        """
        Predict protein function using multiple methods.
        
        Args:
            sequence: Protein sequence
            uniprot_id: UniProt identifier
            methods: List of prediction methods to use
            
        Returns:
            Function prediction results
        """
        results = {
            "sequence": sequence,
            "uniprot_id": uniprot_id,
            "methods_used": methods,
            "predictions": {}
        }
        
        # Sequence-based prediction
        if "sequence" in methods:
            results["predictions"]["sequence_based"] = self._predict_from_sequence(
                sequence, uniprot_id
            )
            
        # Network-based prediction
        if "network" in methods and uniprot_id:
            results["predictions"]["network_based"] = self._predict_from_network(
                uniprot_id
            )
            
        # Domain-based prediction
        if "domains" in methods:
            results["predictions"]["domain_based"] = self._predict_from_domains(
                sequence, uniprot_id
            )
            
        # Integrate predictions
        results["integrated_prediction"] = self._integrate_predictions(
            results["predictions"]
        )
        
        return results
        
    def _predict_from_sequence(
        self, 
        sequence: str, 
        uniprot_id: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Predict function from protein sequence.
        
        Args:
            sequence: Protein sequence
            uniprot_id: UniProt identifier
            
        Returns:
            Sequence-based function predictions
        """
        prediction = {
            "method": "sequence_analysis",
            "sequence_length": len(sequence),
            "composition": self._analyze_composition(sequence),
        }
        
        # Get UniProt annotations if ID is available
        if uniprot_id:
            uniprot_data = self._get_uniprot_annotations(uniprot_id)
            prediction["uniprot_annotations"] = uniprot_data
            
        # Placeholder for ML-based GO term prediction
        prediction["go_terms"] = self._predict_go_terms(sequence)
        
        # Analyze sequence features
        prediction["sequence_features"] = self._analyze_sequence_features(sequence)
        
        return prediction
        
    def _predict_from_network(self, uniprot_id: str) -> Dict[str, any]:
        """
        Predict function from protein interaction networks.
        
        Args:
            uniprot_id: UniProt identifier
            
        Returns:
            Network-based function predictions
        """
        prediction = {
            "method": "network_analysis",
            "uniprot_id": uniprot_id,
        }
        
        try:
            # Get protein interactions from STRING database
            interactions = self._get_string_interactions(uniprot_id)
            prediction["interactions"] = interactions
            
            if interactions and NETWORKX_AVAILABLE:
                # Build interaction network
                network = self._build_interaction_network(interactions)
                prediction["network_analysis"] = self._analyze_network(network, uniprot_id)
                
                # Predict function based on neighbors
                prediction["neighbor_functions"] = self._predict_from_neighbors(
                    network, uniprot_id
                )
                
        except Exception as e:
            logger.error(f"Network-based prediction failed: {e}")
            prediction["error"] = str(e)
            
        return prediction
        
    def _predict_from_domains(
        self, 
        sequence: str, 
        uniprot_id: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Predict function from protein domains and motifs.
        
        Args:
            sequence: Protein sequence
            uniprot_id: UniProt identifier
            
        Returns:
            Domain-based function predictions
        """
        prediction = {
            "method": "domain_analysis",
            "sequence_length": len(sequence),
        }
        
        # Identify potential domains (placeholder)
        prediction["predicted_domains"] = self._identify_domains(sequence)
        
        # Get domain annotations from UniProt if available
        if uniprot_id:
            uniprot_data = self._get_uniprot_annotations(uniprot_id)
            if uniprot_data and "features" in uniprot_data:
                prediction["known_domains"] = [
                    feature for feature in uniprot_data["features"]
                    if feature.get("type") in ["domain", "repeat", "motif"]
                ]
                
        # Predict function based on domains
        prediction["domain_functions"] = self._predict_domain_functions(
            prediction.get("predicted_domains", [])
        )
        
        return prediction
        
    def _analyze_composition(self, sequence: str) -> Dict[str, any]:
        """Analyze amino acid composition of sequence."""
        composition = {}
        
        # Count amino acids
        aa_counts = {}
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
            
        # Calculate frequencies
        total_length = len(sequence)
        aa_frequencies = {aa: count/total_length for aa, count in aa_counts.items()}
        
        composition["amino_acid_counts"] = aa_counts
        composition["amino_acid_frequencies"] = aa_frequencies
        
        # Calculate physicochemical properties
        hydrophobic_aas = set("AILMFPWYV")
        charged_aas = set("DEKRHK")
        polar_aas = set("STYNQC")
        
        composition["hydrophobic_fraction"] = sum(
            aa_frequencies.get(aa, 0) for aa in hydrophobic_aas
        )
        composition["charged_fraction"] = sum(
            aa_frequencies.get(aa, 0) for aa in charged_aas
        )
        composition["polar_fraction"] = sum(
            aa_frequencies.get(aa, 0) for aa in polar_aas
        )
        
        return composition
        
    def _analyze_sequence_features(self, sequence: str) -> Dict[str, any]:
        """Analyze various sequence features."""
        features = {}
        
        # Signal peptides (simple prediction)
        features["has_signal_peptide"] = self._predict_signal_peptide(sequence)
        
        # Transmembrane regions (simple prediction)
        features["transmembrane_regions"] = self._predict_transmembrane(sequence)
        
        # Low complexity regions
        features["low_complexity_regions"] = self._find_low_complexity(sequence)
        
        # Repeat regions
        features["repeat_regions"] = self._find_repeats(sequence)
        
        return features
        
    def _predict_signal_peptide(self, sequence: str) -> bool:
        """Simple signal peptide prediction based on N-terminal properties."""
        if len(sequence) < 20:
            return False
            
        n_terminal = sequence[:20]
        
        # Simple heuristic: high hydrophobic content in N-terminal
        hydrophobic_aas = set("AILMFPWYV")
        hydrophobic_count = sum(1 for aa in n_terminal if aa in hydrophobic_aas)
        
        return hydrophobic_count / len(n_terminal) > 0.6
        
    def _predict_transmembrane(self, sequence: str) -> List[Tuple[int, int]]:
        """Simple transmembrane region prediction."""
        transmembrane_regions = []
        
        # Simple sliding window approach
        window_size = 20
        hydrophobic_aas = set("AILMFPWYV")
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            hydrophobic_count = sum(1 for aa in window if aa in hydrophobic_aas)
            
            if hydrophobic_count / window_size > 0.7:
                transmembrane_regions.append((i, i + window_size))
                
        return transmembrane_regions
        
    def _find_low_complexity(self, sequence: str) -> List[Tuple[int, int]]:
        """Find low complexity regions in sequence."""
        low_complexity = []
        
        # Simple approach: regions with limited amino acid diversity
        window_size = 20
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            unique_aas = len(set(window))
            
            if unique_aas <= 5:  # Low diversity threshold
                low_complexity.append((i, i + window_size))
                
        return low_complexity
        
    def _find_repeats(self, sequence: str) -> List[Dict[str, any]]:
        """Find repeat regions in sequence."""
        repeats = []
        
        # Simple tandem repeat detection
        for repeat_len in range(2, 11):  # Check repeat lengths 2-10
            for i in range(len(sequence) - repeat_len * 2 + 1):
                motif = sequence[i:i + repeat_len]
                
                # Count consecutive repeats
                repeat_count = 1
                pos = i + repeat_len
                
                while pos + repeat_len <= len(sequence):
                    if sequence[pos:pos + repeat_len] == motif:
                        repeat_count += 1
                        pos += repeat_len
                    else:
                        break
                        
                if repeat_count >= 3:  # At least 3 repeats
                    repeats.append({
                        "start": i,
                        "end": pos,
                        "motif": motif,
                        "count": repeat_count
                    })
                    
        return repeats
        
    def _predict_go_terms(self, sequence: str) -> Dict[str, any]:
        """Predict GO terms (placeholder for ML model)."""
        # Placeholder for actual GO term prediction model
        return {
            "method": "placeholder",
            "predicted_terms": [],
            "confidence_scores": [],
            "note": "GO term prediction model not yet implemented"
        }
        
    def _identify_domains(self, sequence: str) -> List[Dict[str, any]]:
        """Identify protein domains (placeholder)."""
        # Placeholder for domain identification
        return [
            {
                "method": "placeholder",
                "note": "Domain identification will be implemented with HMM models"
            }
        ]
        
    def _predict_domain_functions(self, domains: List[Dict]) -> Dict[str, any]:
        """Predict function based on identified domains."""
        return {
            "method": "domain_function_mapping",
            "functions": [],
            "note": "Domain-function mapping not yet implemented"
        }
        
    def _get_uniprot_annotations(self, uniprot_id: str) -> Optional[Dict[str, any]]:
        """Get annotations from UniProt database."""
        try:
            url = f"{self.uniprot_api_url}/{uniprot_id}"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"UniProt lookup failed for {uniprot_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"UniProt annotation retrieval failed: {e}")
            return None
            
    def _get_string_interactions(self, uniprot_id: str) -> Optional[List[Dict]]:
        """Get protein interactions from STRING database."""
        try:
            # Convert UniProt ID to STRING identifier
            url = f"{self.string_db_url}/tsv/get_string_ids"
            params = {
                "identifiers": uniprot_id,
                "species": 9606,  # Human
                "caller_identity": "protein_science_ai"
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                if len(lines) > 1:  # Header + data
                    string_id = lines[1].split('\t')[2]  # STRING ID column
                    
                    # Get interactions
                    interaction_url = f"{self.string_db_url}/tsv/interaction_partners"
                    interaction_params = {
                        "identifiers": string_id,
                        "species": 9606,
                        "caller_identity": "protein_science_ai"
                    }
                    
                    interaction_response = requests.get(
                        interaction_url, 
                        params=interaction_params, 
                        timeout=30
                    )
                    
                    if interaction_response.status_code == 200:
                        interactions = []
                        lines = interaction_response.text.strip().split('\n')
                        
                        if len(lines) > 1:
                            headers = lines[0].split('\t')
                            for line in lines[1:]:
                                values = line.split('\t')
                                interaction = dict(zip(headers, values))
                                interactions.append(interaction)
                                
                        return interactions
                        
            return None
            
        except Exception as e:
            logger.error(f"STRING interaction retrieval failed: {e}")
            return None
            
    def _build_interaction_network(self, interactions: List[Dict]) -> Optional[object]:
        """Build interaction network graph."""
        if not NETWORKX_AVAILABLE:
            return None
            
        try:
            G = nx.Graph()
            
            for interaction in interactions:
                node1 = interaction.get("stringId_A")
                node2 = interaction.get("stringId_B") 
                score = float(interaction.get("score", 0))
                
                if node1 and node2:
                    G.add_edge(node1, node2, weight=score)
                    
            return G
            
        except Exception as e:
            logger.error(f"Network building failed: {e}")
            return None
            
    def _analyze_network(self, network: object, target_node: str) -> Dict[str, any]:
        """Analyze interaction network properties."""
        if not NETWORKX_AVAILABLE or not network:
            return {}
            
        try:
            analysis = {
                "num_nodes": network.number_of_nodes(),
                "num_edges": network.number_of_edges(),
                "density": nx.density(network),
            }
            
            if target_node in network:
                analysis["target_degree"] = network.degree(target_node)
                analysis["target_clustering"] = nx.clustering(network, target_node)
                
                # Centrality measures
                betweenness = nx.betweenness_centrality(network)
                closeness = nx.closeness_centrality(network)
                
                analysis["target_betweenness"] = betweenness.get(target_node, 0)
                analysis["target_closeness"] = closeness.get(target_node, 0)
                
            return analysis
            
        except Exception as e:
            logger.error(f"Network analysis failed: {e}")
            return {}
            
    def _predict_from_neighbors(self, network: object, target_node: str) -> Dict[str, any]:
        """Predict function based on network neighbors."""
        if not NETWORKX_AVAILABLE or not network or target_node not in network:
            return {}
            
        try:
            neighbors = list(network.neighbors(target_node))
            
            prediction = {
                "num_neighbors": len(neighbors),
                "neighbors": neighbors[:10],  # Limit for display
                "note": "Neighbor function analysis requires annotation data"
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Neighbor-based prediction failed: {e}")
            return {}
            
    def _integrate_predictions(self, predictions: Dict[str, any]) -> Dict[str, any]:
        """Integrate predictions from multiple methods."""
        integrated = {
            "methods_used": list(predictions.keys()),
            "integration_method": "consensus",
        }
        
        # Simple consensus approach (placeholder)
        confidence_scores = []
        predicted_functions = []
        
        for method, prediction in predictions.items():
            if isinstance(prediction, dict) and "error" not in prediction:
                confidence_scores.append(0.5)  # Placeholder confidence
                predicted_functions.append(f"function_from_{method}")
                
        integrated["average_confidence"] = np.mean(confidence_scores) if confidence_scores else 0.0
        integrated["predicted_functions"] = predicted_functions
        integrated["note"] = "Integration method is placeholder - will be improved with ML models"
        
        return integrated
        
    def __repr__(self) -> str:
        return f"FunctionPredictor(cache_dir={self.cache_dir})"