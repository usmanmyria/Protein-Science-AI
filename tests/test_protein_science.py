"""Test suite for the protein science AI system."""

import pytest
import asyncio
from unittest.mock import Mock, patch

# Test configuration
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture
def sample_protein_sequence():
    """Sample protein sequence for testing."""
    return "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"


@pytest.fixture 
def sample_uniprot_id():
    """Sample UniProt ID for testing."""
    return "P53_HUMAN"


@pytest.fixture
def sample_mutations():
    """Sample mutations for testing."""
    return ["A10G", "T20A", "R30K"]


class TestProteinAgent:
    """Test cases for the main protein agent."""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test protein agent initialization."""
        from protein_science import ProteinAgent
        
        agent = ProteinAgent()
        assert agent is not None
        assert hasattr(agent, 'plm')
        assert hasattr(agent, 'tools')
    
    @pytest.mark.asyncio
    async def test_sequence_analysis(self, sample_protein_sequence):
        """Test basic sequence analysis."""
        from protein_science import ProteinAgent
        
        agent = ProteinAgent()
        
        # Mock the analysis to avoid long-running operations in tests
        with patch.object(agent, '_analyze_sequence_comprehensive') as mock_analysis:
            mock_analysis.return_value = {
                "basic_properties": {
                    "length": len(sample_protein_sequence),
                    "molecular_weight_approx": len(sample_protein_sequence) * 110
                }
            }
            
            result = await agent.analyze_protein(
                sample_protein_sequence,
                analysis_type="quick",
                include_structure=False,
                include_function=False
            )
            
            assert result is not None
            assert "input" in result
            assert result["input"]["sequence"] == sample_protein_sequence
    
    def test_sequence_similarity(self, sample_protein_sequence):
        """Test sequence similarity computation."""
        from protein_science import ProteinAgent
        
        agent = ProteinAgent()
        
        # Test with identical sequences
        similarity = agent.plm.compute_similarity(
            sample_protein_sequence, 
            sample_protein_sequence
        )
        
        # Should be very similar (close to 1.0)
        assert similarity > 0.99
    
    def test_reasoning_capability(self, sample_protein_sequence):
        """Test the reasoning capability of the agent."""
        from protein_science import ProteinAgent
        
        agent = ProteinAgent()
        
        # Mock analysis results
        mock_results = {
            "sequence_analysis": {
                "basic_properties": {
                    "length": len(sample_protein_sequence)
                }
            }
        }
        
        answer = agent.reason_about_results(
            mock_results, 
            "What is the length of this protein?"
        )
        
        assert answer is not None
        assert isinstance(answer, str)
        assert len(answer) > 0


class TestAgentCoordinator:
    """Test cases for the agent coordinator."""
    
    @pytest.mark.asyncio
    async def test_coordinator_initialization(self):
        """Test coordinator initialization."""
        from protein_science import ProteinAgent
        from protein_science.collaboration import AgentCoordinator
        
        agent = ProteinAgent()
        coordinator = AgentCoordinator(main_agent=agent)
        
        assert coordinator is not None
        assert coordinator.main_agent == agent
        assert hasattr(coordinator, 'specialized_agents')
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, sample_protein_sequence):
        """Test workflow execution."""
        from protein_science import ProteinAgent
        from protein_science.collaboration import AgentCoordinator
        
        agent = ProteinAgent()
        coordinator = AgentCoordinator(main_agent=agent)
        
        # Mock workflow execution
        with patch.object(coordinator, '_execute_comprehensive_analysis') as mock_exec:
            mock_exec.return_value = {
                "main_analysis": {"status": "completed"},
                "integrated_analysis": {"consensus_insights": []}
            }
            
            result = await coordinator.execute_workflow(
                workflow_type="comprehensive_analysis",
                protein_input=sample_protein_sequence
            )
            
            assert result is not None
            assert "workflow_id" in result
            assert "status" in result


class TestAPI:
    """Test cases for the REST API."""
    
    def test_api_initialization(self):
        """Test API initialization."""
        try:
            from protein_science.interface import ProteinScienceAPI
            
            api = ProteinScienceAPI()
            assert api is not None
            assert hasattr(api, 'app')
            assert hasattr(api, 'protein_agent')
            assert hasattr(api, 'coordinator')
            
        except ImportError:
            pytest.skip("FastAPI not available")


class TestFoundationComponents:
    """Test cases for foundation layer components."""
    
    def test_protein_language_model(self, sample_protein_sequence):
        """Test protein language model functionality."""
        from protein_science.foundation import ProteinLanguageModel
        
        # Use a smaller model for testing
        plm = ProteinLanguageModel(model_name="facebook/esm2_t6_8M_UR50D")
        
        # Test sequence encoding
        embeddings = plm.get_sequence_embeddings([sample_protein_sequence])
        
        assert embeddings is not None
        assert len(embeddings) == 1
        assert embeddings[0].ndim == 1  # Should be 1D vector
    
    def test_structure_predictor(self, sample_uniprot_id):
        """Test structure predictor functionality."""
        from protein_science.foundation import StructurePredictor
        
        predictor = StructurePredictor()
        
        # Test with mock data to avoid external API calls
        with patch.object(predictor, '_predict_alphafold') as mock_predict:
            mock_predict.return_value = {
                "uniprot_id": sample_uniprot_id,
                "method": "alphafold",
                "structure_analysis": {"num_atoms": 1000}
            }
            
            result = predictor.predict_structure(
                sequence="MKTVRQ",
                method="alphafold", 
                uniprot_id=sample_uniprot_id
            )
            
            assert result is not None
            assert "method" in result
    
    def test_function_predictor(self, sample_protein_sequence, sample_uniprot_id):
        """Test function predictor functionality."""
        from protein_science.foundation import FunctionPredictor
        
        predictor = FunctionPredictor()
        
        # Test function prediction
        result = predictor.predict_function(
            sequence=sample_protein_sequence,
            uniprot_id=sample_uniprot_id,
            methods=["sequence"]  # Only use sequence method to avoid external calls
        )
        
        assert result is not None
        assert "sequence" in result
        assert "methods_used" in result
        assert "predictions" in result


class TestCLI:
    """Test cases for command-line interface."""
    
    def test_cli_parser(self):
        """Test CLI argument parsing."""
        from protein_science.cli import create_parser
        
        parser = create_parser()
        
        # Test analyze command
        args = parser.parse_args(["analyze", "--sequence", "MKTVRQ"])
        assert args.command == "analyze"
        assert args.sequence == "MKTVRQ"
        
        # Test serve command
        args = parser.parse_args(["serve", "--port", "8080"])
        assert args.command == "serve"
        assert args.port == 8080


# Integration tests
class TestIntegration:
    """Integration test cases."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_analysis(self, sample_protein_sequence):
        """Test end-to-end protein analysis."""
        from protein_science import ProteinAgent
        
        agent = ProteinAgent()
        
        # Run a quick analysis
        result = await agent.analyze_protein(
            sample_protein_sequence,
            analysis_type="quick",
            include_structure=False,  # Skip to avoid external calls
            include_function=False    # Skip to avoid external calls
        )
        
        # Verify basic structure
        assert "input" in result
        assert "sequence_analysis" in result
        assert "timestamp" in result
        assert "agent_version" in result
        
        # Verify sequence analysis
        seq_analysis = result["sequence_analysis"]
        assert "basic_properties" in seq_analysis
        
        basic_props = seq_analysis["basic_properties"]
        assert basic_props["length"] == len(sample_protein_sequence)
        assert basic_props["molecular_weight_approx"] > 0


# Utility functions for tests
def mock_external_api_call(url, **kwargs):
    """Mock external API calls for testing."""
    return Mock(status_code=200, json=lambda: {"mock": "data"})


# Performance tests
class TestPerformance:
    """Performance test cases."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_analysis_performance(self, sample_protein_sequence):
        """Test analysis performance with timing."""
        from protein_science import ProteinAgent
        import time
        
        agent = ProteinAgent()
        
        start_time = time.time()
        
        result = await agent.analyze_protein(
            sample_protein_sequence,
            analysis_type="quick",
            include_structure=False,
            include_function=False
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Analysis should complete within reasonable time
        assert duration < 30.0  # 30 seconds max for quick analysis
        assert "analysis_time_seconds" in result