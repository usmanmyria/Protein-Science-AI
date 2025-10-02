"""
Agentic AI System for Protein Science

A comprehensive autonomous AI system that integrates protein language models,
molecular simulation tools, and multi-agent reasoning to accelerate drug discovery,
protein engineering, and synthetic biology research.
"""

__version__ = "0.1.0"
__author__ = "Protein Science AI Team"

# Import foundation components
from .foundation.protein_models import ProteinLanguageModel
from .foundation.structure_predictor import StructurePredictor
from .foundation.function_predictor import FunctionPredictor

# Import agent components
from .agents.protein_agent import ProteinAgent

# Import collaboration components
from .collaboration.coordinator import AgentCoordinator

__all__ = [
    "ProteinLanguageModel",
    "StructurePredictor", 
    "ProteinAgent",
    "AgentCoordinator",
]