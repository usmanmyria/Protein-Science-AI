"""
Main protein science AI agent with autonomous reasoning capabilities.

This agent integrates protein language models, structure prediction, function analysis,
and simulation tools to provide intelligent assistance for protein science research.
"""

from typing import Dict, List, Optional, Union, Any
import asyncio
from loguru import logger
from datetime import datetime

# Import foundation components
from ..foundation.protein_models import ProteinLanguageModel
from ..foundation.structure_predictor import StructurePredictor
from ..foundation.function_predictor import FunctionPredictor

try:
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.agents import AgentExecutor
    from langchain.tools import Tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. Agent orchestration features will be limited.")


class ProteinAgent:
    """
    Autonomous protein science AI agent.
    
    This agent can:
    - Analyze protein sequences and structures
    - Predict protein function and interactions
    - Plan and execute multi-step analysis workflows
    - Reason about experimental results
    - Suggest follow-up experiments
    """
    
    def __init__(
        self,
        plm_model: str = "facebook/esm2_t33_650M_UR50D",
        enable_structure_prediction: bool = True,
        enable_function_prediction: bool = True,
        working_directory: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the protein science agent.
        
        Args:
            plm_model: Protein language model to use
            enable_structure_prediction: Whether to enable structure prediction
            enable_function_prediction: Whether to enable function prediction
            working_directory: Directory for caching and temporary files
            verbose: Whether to enable verbose logging
        """
        self.verbose = verbose
        self.working_directory = working_directory
        
        # Initialize core components
        logger.info("Initializing Protein Science AI Agent")
        
        # Protein language model
        self.plm = ProteinLanguageModel(
            model_name=plm_model,
            cache_dir=working_directory
        )
        
        # Structure predictor
        if enable_structure_prediction:
            self.structure_predictor = StructurePredictor(
                cache_dir=working_directory
            )
        else:
            self.structure_predictor = None
            
        # Function predictor
        if enable_function_prediction:
            self.function_predictor = FunctionPredictor(
                cache_dir=working_directory
            )
        else:
            self.function_predictor = None
            
        # Agent state and memory
        self.conversation_history = []
        self.analysis_cache = {}
        self.current_hypotheses = []
        
        # Available tools
        self.tools = self._initialize_tools()
        
        logger.success("Protein Science AI Agent initialized successfully")
        
    def _initialize_tools(self) -> List[Dict[str, Any]]:
        """Initialize available tools for the agent."""
        tools = []
        
        # Sequence analysis tool
        tools.append({
            "name": "analyze_sequence",
            "description": "Analyze protein sequence using language models",
            "function": self._tool_analyze_sequence
        })
        
        # Structure prediction tool
        if self.structure_predictor:
            tools.append({
                "name": "predict_structure",
                "description": "Predict protein 3D structure",
                "function": self._tool_predict_structure
            })
            
        # Function prediction tool
        if self.function_predictor:
            tools.append({
                "name": "predict_function",
                "description": "Predict protein function and interactions",
                "function": self._tool_predict_function
            })
            
        # Literature search tool (placeholder)
        tools.append({
            "name": "search_literature",
            "description": "Search scientific literature for protein information",
            "function": self._tool_search_literature
        })
        
        return tools
        
    async def analyze_protein(
        self,
        protein_input: Union[str, Dict[str, Any]],
        analysis_type: str = "comprehensive",
        include_structure: bool = True,
        include_function: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive protein analysis.
        
        Args:
            protein_input: Protein sequence or dict with sequence/ID
            analysis_type: Type of analysis ('quick', 'comprehensive', 'custom')
            include_structure: Whether to include structure prediction
            include_function: Whether to include function prediction
            
        Returns:
            Complete analysis results
        """
        start_time = datetime.now()
        
        # Parse input
        if isinstance(protein_input, str):
            sequence = protein_input
            uniprot_id = None
        elif isinstance(protein_input, dict):
            sequence = protein_input.get("sequence")
            uniprot_id = protein_input.get("uniprot_id")
        else:
            raise ValueError("Invalid protein input format")
            
        if not sequence:
            raise ValueError("Protein sequence is required")
            
        logger.info(f"Starting {analysis_type} analysis for protein sequence")
        
        # Initialize results
        results = {
            "input": {
                "sequence": sequence,
                "uniprot_id": uniprot_id,
                "analysis_type": analysis_type
            },
            "timestamp": start_time.isoformat(),
            "agent_version": "0.1.0"
        }
        
        try:
            # 1. Sequence analysis using PLM
            logger.info("Performing sequence analysis...")
            sequence_analysis = await self._analyze_sequence_comprehensive(sequence)
            results["sequence_analysis"] = sequence_analysis
            
            # 2. Structure prediction
            if include_structure and self.structure_predictor:
                logger.info("Performing structure prediction...")
                structure_analysis = await self._predict_structure_comprehensive(
                    sequence, uniprot_id
                )
                results["structure_analysis"] = structure_analysis
                
            # 3. Function prediction
            if include_function and self.function_predictor:
                logger.info("Performing function prediction...")
                function_analysis = await self._predict_function_comprehensive(
                    sequence, uniprot_id
                )
                results["function_analysis"] = function_analysis
                
            # 4. Generate insights and hypotheses
            logger.info("Generating insights...")
            insights = self._generate_insights(results)
            results["insights"] = insights
            
            # 5. Suggest follow-up experiments
            suggestions = self._suggest_experiments(results)
            results["experiment_suggestions"] = suggestions
            
            # Calculate analysis time
            end_time = datetime.now()
            results["analysis_time_seconds"] = (end_time - start_time).total_seconds()
            
            # Cache results
            cache_key = self._generate_cache_key(sequence, analysis_type)
            self.analysis_cache[cache_key] = results
            
            logger.success(f"Analysis completed in {results['analysis_time_seconds']:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            results["error"] = str(e)
            results["status"] = "failed"
            return results
            
    async def _analyze_sequence_comprehensive(self, sequence: str) -> Dict[str, Any]:
        """Comprehensive sequence analysis using PLM."""
        # Use the PLM to analyze the sequence
        plm_analysis = self.plm.analyze_sequence(sequence)
        
        # Add additional sequence-level insights
        analysis = {
            "plm_analysis": plm_analysis,
            "basic_properties": {
                "length": len(sequence),
                "molecular_weight_approx": len(sequence) * 110,  # Rough estimate
                "unique_residues": len(set(sequence)),
                "most_common_residue": max(set(sequence), key=sequence.count),
            }
        }
        
        # Analyze sequence patterns
        analysis["patterns"] = self._analyze_sequence_patterns(sequence)
        
        return analysis
        
    async def _predict_structure_comprehensive(
        self, 
        sequence: str, 
        uniprot_id: Optional[str]
    ) -> Dict[str, Any]:
        """Comprehensive structure prediction and analysis."""
        if not self.structure_predictor:
            return {"error": "Structure predictor not available"}
            
        # Predict structure
        structure_result = self.structure_predictor.predict_structure(
            sequence=sequence,
            method="alphafold",
            uniprot_id=uniprot_id
        )
        
        # Add structural insights
        if "error" not in structure_result:
            structure_result["structural_insights"] = self._analyze_structural_features(
                structure_result
            )
            
        return structure_result
        
    async def _predict_function_comprehensive(
        self, 
        sequence: str, 
        uniprot_id: Optional[str]
    ) -> Dict[str, Any]:
        """Comprehensive function prediction and analysis."""
        if not self.function_predictor:
            return {"error": "Function predictor not available"}
            
        # Predict function
        function_result = self.function_predictor.predict_function(
            sequence=sequence,
            uniprot_id=uniprot_id,
            methods=["sequence", "network", "domains"]
        )
        
        return function_result
        
    def _analyze_sequence_patterns(self, sequence: str) -> Dict[str, Any]:
        """Analyze patterns in protein sequence."""
        patterns = {}
        
        # Find simple motifs
        common_motifs = {
            "RGDS": "cell adhesion motif",
            "KDEL": "ER retention signal",
            "NLS": "nuclear localization signal patterns"
        }
        
        found_motifs = []
        for motif, description in common_motifs.items():
            if motif in sequence:
                positions = [i for i in range(len(sequence)) if sequence.startswith(motif, i)]
                found_motifs.append({
                    "motif": motif,
                    "description": description,
                    "positions": positions
                })
                
        patterns["known_motifs"] = found_motifs
        
        # Analyze charge distribution
        charge_pattern = []
        for aa in sequence:
            if aa in "KRH":
                charge_pattern.append(1)  # Positive
            elif aa in "DE":
                charge_pattern.append(-1)  # Negative
            else:
                charge_pattern.append(0)  # Neutral
                
        patterns["charge_distribution"] = {
            "pattern": charge_pattern,
            "net_charge": sum(charge_pattern),
            "charge_density": sum(charge_pattern) / len(sequence)
        }
        
        return patterns
        
    def _analyze_structural_features(self, structure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze structural features from prediction results."""
        insights = {}
        
        # Extract basic structural properties
        if "structure_analysis" in structure_data:
            analysis = structure_data["structure_analysis"]
            
            insights["structural_summary"] = {
                "has_structural_data": True,
                "analysis_methods": list(analysis.keys()),
            }
            
            # Analyze confidence if available
            if "confidence_scores" in structure_data:
                confidence = structure_data["confidence_scores"]
                insights["confidence_analysis"] = {
                    "has_confidence_data": True,
                    "overall_confidence": confidence.get("overall_confidence"),
                    "confidence_type": confidence.get("confidence_type")
                }
        else:
            insights["structural_summary"] = {
                "has_structural_data": False,
                "reason": "No structural data available"
            }
            
        return insights
        
    def _generate_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level insights from analysis results."""
        insights = {
            "summary": "Automated analysis insights",
            "key_findings": [],
            "confidence_level": "medium",
            "areas_of_interest": []
        }
        
        # Analyze sequence properties
        if "sequence_analysis" in analysis_results:
            seq_analysis = analysis_results["sequence_analysis"]
            
            # Length-based insights
            length = seq_analysis.get("basic_properties", {}).get("length", 0)
            if length > 1000:
                insights["key_findings"].append("Large protein (>1000 residues) - possibly multi-domain")
            elif length < 100:
                insights["key_findings"].append("Small protein (<100 residues) - possibly peptide hormone or domain")
                
            # Pattern-based insights
            patterns = seq_analysis.get("patterns", {})
            motifs = patterns.get("known_motifs", [])
            if motifs:
                insights["key_findings"].append(f"Contains {len(motifs)} known functional motifs")
                
        # Analyze structure confidence
        if "structure_analysis" in analysis_results:
            struct_analysis = analysis_results["structure_analysis"]
            if "confidence_scores" in struct_analysis:
                insights["areas_of_interest"].append("High-confidence structural regions available")
                
        # Analyze function predictions
        if "function_analysis" in analysis_results:
            func_analysis = analysis_results["function_analysis"]
            methods_used = func_analysis.get("methods_used", [])
            insights["key_findings"].append(f"Function analysis used {len(methods_used)} different methods")
            
        return insights
        
    def _suggest_experiments(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest follow-up experiments based on analysis."""
        suggestions = []
        
        # Structure-based experiments
        if "structure_analysis" in analysis_results:
            suggestions.append({
                "type": "structural_validation",
                "experiment": "X-ray crystallography or NMR",
                "rationale": "Validate predicted structure with experimental methods",
                "priority": "high"
            })
            
        # Function-based experiments
        if "function_analysis" in analysis_results:
            suggestions.append({
                "type": "functional_assay",
                "experiment": "Enzyme activity assay",
                "rationale": "Confirm predicted enzymatic function",
                "priority": "medium"
            })
            
        # Interaction experiments
        suggestions.append({
            "type": "interaction_study",
            "experiment": "Yeast two-hybrid or co-immunoprecipitation",
            "rationale": "Identify protein-protein interactions",
            "priority": "medium"
        })
        
        # Mutation experiments
        suggestions.append({
            "type": "mutagenesis",
            "experiment": "Site-directed mutagenesis of key residues",
            "rationale": "Determine structure-function relationships",
            "priority": "high"
        })
        
        return suggestions
        
    def _generate_cache_key(self, sequence: str, analysis_type: str) -> str:
        """Generate cache key for analysis results."""
        import hashlib
        sequence_hash = hashlib.md5(sequence.encode()).hexdigest()[:8]
        return f"{sequence_hash}_{analysis_type}"
        
    # Tool functions for agent orchestration
    def _tool_analyze_sequence(self, sequence: str) -> str:
        """Tool wrapper for sequence analysis."""
        try:
            result = asyncio.run(self._analyze_sequence_comprehensive(sequence))
            return f"Sequence analysis completed. Length: {result.get('basic_properties', {}).get('length', 'unknown')} residues."
        except Exception as e:
            return f"Sequence analysis failed: {str(e)}"
            
    def _tool_predict_structure(self, sequence: str, uniprot_id: str = None) -> str:
        """Tool wrapper for structure prediction."""
        try:
            result = asyncio.run(self._predict_structure_comprehensive(sequence, uniprot_id))
            if "error" in result:
                return f"Structure prediction failed: {result['error']}"
            return "Structure prediction completed successfully."
        except Exception as e:
            return f"Structure prediction failed: {str(e)}"
            
    def _tool_predict_function(self, sequence: str, uniprot_id: str = None) -> str:
        """Tool wrapper for function prediction."""
        try:
            result = asyncio.run(self._predict_function_comprehensive(sequence, uniprot_id))
            methods = result.get("methods_used", [])
            return f"Function prediction completed using {len(methods)} methods."
        except Exception as e:
            return f"Function prediction failed: {str(e)}"
            
    def _tool_search_literature(self, query: str) -> str:
        """Tool wrapper for literature search (placeholder)."""
        return f"Literature search for '{query}' - feature not yet implemented"
        
    def reason_about_results(self, analysis_results: Dict[str, Any], question: str) -> str:
        """
        Reason about analysis results to answer specific questions.
        
        Args:
            analysis_results: Results from protein analysis
            question: Question to answer about the results
            
        Returns:
            Reasoning-based answer
        """
        # Simple rule-based reasoning (placeholder for more sophisticated methods)
        
        if "function" in question.lower():
            if "function_analysis" in analysis_results:
                func_data = analysis_results["function_analysis"]
                return f"Based on function prediction analysis using {len(func_data.get('methods_used', []))} methods, the protein shows evidence of specific functional domains and interaction patterns."
            else:
                return "Function analysis was not performed or is not available."
                
        elif "structure" in question.lower():
            if "structure_analysis" in analysis_results:
                struct_data = analysis_results["structure_analysis"]
                if "error" not in struct_data:
                    return "Structural analysis indicates the protein has a well-defined 3D structure with specific domains and confidence scores available."
                else:
                    return f"Structure analysis encountered issues: {struct_data.get('error', 'unknown error')}"
            else:
                return "Structure analysis was not performed or is not available."
                
        elif "experiment" in question.lower():
            suggestions = analysis_results.get("experiment_suggestions", [])
            if suggestions:
                high_priority = [s for s in suggestions if s.get("priority") == "high"]
                return f"Based on the analysis, I recommend {len(high_priority)} high-priority experiments, including {high_priority[0]['experiment'] if high_priority else 'functional assays'}."
            else:
                return "No specific experimental suggestions are available based on current analysis."
                
        else:
            # General reasoning
            insights = analysis_results.get("insights", {})
            key_findings = insights.get("key_findings", [])
            
            if key_findings:
                return f"Key insights from the analysis include: {'; '.join(key_findings[:3])}."
            else:
                return "The analysis completed successfully but specific insights require more detailed examination of the results."
                
    def __repr__(self) -> str:
        return f"ProteinAgent(plm={self.plm.model_name}, tools={len(self.tools)})"