"""
Command-line interface for the Protein Science AI system.

This module provides a CLI for protein analysis, workflow execution,
and system management.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional
from loguru import logger

# Configure logger for CLI
logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{level}</level> | {message}")

try:
    from .agents.protein_agent import ProteinAgent
    from .collaboration.coordinator import AgentCoordinator
    from .interface.api import ProteinScienceAPI
    from .interface.streamlit_app import create_streamlit_app
except ImportError as e:
    print(f"Error importing protein_science modules: {e}")
    print("Please ensure the package is properly installed.")
    sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Protein Science AI - Agentic system for protein analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  protein-ai analyze --sequence MKTVRQERLK --output results.json
  protein-ai workflow --type drug_discovery --input protein.fasta
  protein-ai serve --port 8000
  protein-ai app --port 8501
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a protein sequence")
    analyze_parser.add_argument(
        "--sequence", "-s",
        type=str,
        help="Protein sequence to analyze"
    )
    analyze_parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Input file containing protein sequence"
    )
    analyze_parser.add_argument(
        "--uniprot-id", "-u",
        type=str,
        help="UniProt identifier"
    )
    analyze_parser.add_argument(
        "--analysis-type", "-t",
        choices=["quick", "comprehensive", "custom"],
        default="comprehensive",
        help="Type of analysis to perform"
    )
    analyze_parser.add_argument(
        "--no-structure",
        action="store_true",
        help="Skip structure prediction"
    )
    analyze_parser.add_argument(
        "--no-function",
        action="store_true",
        help="Skip function prediction"
    )
    analyze_parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for results (JSON format)"
    )
    
    # Workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Execute multi-agent workflow")
    workflow_parser.add_argument(
        "--type",
        choices=["comprehensive_analysis", "drug_discovery", "protein_engineering", "mutation_analysis"],
        required=True,
        help="Type of workflow to execute"
    )
    workflow_parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input file containing protein data"
    )
    workflow_parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Configuration file for workflow (JSON format)"
    )
    workflow_parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for results (JSON format)"
    )
    
    # Mutation command
    mutation_parser = subparsers.add_parser("mutations", help="Analyze protein mutations")
    mutation_parser.add_argument(
        "--sequence", "-s",
        type=str,
        help="Wild-type protein sequence"
    )
    mutation_parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Input file containing wild-type sequence"
    )
    mutation_parser.add_argument(
        "--mutations", "-m",
        type=str,
        nargs="+",
        required=True,
        help="Mutations to analyze (e.g., A123G T456A)"
    )
    mutation_parser.add_argument(
        "--uniprot-id", "-u",
        type=str,
        help="UniProt identifier"
    )
    mutation_parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for results (JSON format)"
    )
    
    # Serve command (REST API)
    serve_parser = subparsers.add_parser("serve", help="Start REST API server")
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    # App command (Streamlit)
    app_parser = subparsers.add_parser("app", help="Start Streamlit web application")
    app_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port for Streamlit app (default: 8501)"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    
    return parser


def read_sequence_from_file(file_path: Path) -> str:
    """Read protein sequence from file."""
    try:
        content = file_path.read_text().strip()
        
        # Handle FASTA format
        if content.startswith(">"):
            lines = content.split("\n")
            sequence = "".join(line for line in lines[1:] if not line.startswith(">"))
        else:
            sequence = content.replace("\n", "").replace(" ", "")
            
        return sequence.upper()
        
    except Exception as e:
        logger.error(f"Failed to read sequence from {file_path}: {e}")
        sys.exit(1)


def save_results(results: dict, output_path: Optional[Path]):
    """Save results to file or print to stdout."""
    try:
        if output_path:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.success(f"Results saved to {output_path}")
        else:
            print(json.dumps(results, indent=2, default=str))
            
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        sys.exit(1)


async def cmd_analyze(args):
    """Handle analyze command."""
    logger.info("Starting protein analysis...")
    
    # Get sequence
    if args.sequence:
        sequence = args.sequence.replace(" ", "").upper()
    elif args.input:
        sequence = read_sequence_from_file(args.input)
    else:
        logger.error("Either --sequence or --input must be provided")
        sys.exit(1)
    
    # Validate sequence
    if not sequence.replace(" ", "").isalpha():
        logger.error("Invalid protein sequence. Must contain only amino acid letters.")
        sys.exit(1)
    
    # Initialize agent
    agent = ProteinAgent()
    
    # Prepare input
    protein_input = {
        "sequence": sequence,
        "uniprot_id": args.uniprot_id
    }
    
    # Run analysis
    try:
        result = await agent.analyze_protein(
            protein_input=protein_input,
            analysis_type=args.analysis_type,
            include_structure=not args.no_structure,
            include_function=not args.no_function
        )
        
        logger.success(f"Analysis completed in {result.get('analysis_time_seconds', 0):.2f} seconds")
        save_results(result, args.output)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


async def cmd_workflow(args):
    """Handle workflow command."""
    logger.info(f"Starting {args.type} workflow...")
    
    # Read input data
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Determine input format
    if args.input.suffix.lower() == ".json":
        protein_input = json.loads(args.input.read_text())
    else:
        # Assume sequence file
        sequence = read_sequence_from_file(args.input)
        protein_input = {"sequence": sequence}
    
    # Read config if provided
    config = None
    if args.config:
        if args.config.exists():
            config = json.loads(args.config.read_text())
        else:
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)
    
    # Initialize coordinator
    agent = ProteinAgent()
    coordinator = AgentCoordinator(main_agent=agent)
    
    # Execute workflow
    try:
        result = await coordinator.execute_workflow(
            workflow_type=args.type,
            protein_input=protein_input,
            workflow_config=config
        )
        
        duration = result.get("duration_seconds", 0)
        logger.success(f"Workflow completed in {duration:.2f} seconds")
        save_results(result, args.output)
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        sys.exit(1)


async def cmd_mutations(args):
    """Handle mutations command."""
    logger.info(f"Starting mutation analysis for {len(args.mutations)} mutations...")
    
    # Get sequence
    if args.sequence:
        sequence = args.sequence.replace(" ", "").upper()
    elif args.input:
        sequence = read_sequence_from_file(args.input)
    else:
        logger.error("Either --sequence or --input must be provided")
        sys.exit(1)
    
    # Prepare input
    protein_input = {
        "sequence": sequence,
        "uniprot_id": args.uniprot_id
    }
    
    config = {
        "mutations": args.mutations
    }
    
    # Initialize coordinator
    agent = ProteinAgent()
    coordinator = AgentCoordinator(main_agent=agent)
    
    # Execute mutation analysis
    try:
        result = await coordinator.execute_workflow(
            workflow_type="mutation_analysis",
            protein_input=protein_input,
            workflow_config=config
        )
        
        duration = result.get("duration_seconds", 0)
        logger.success(f"Mutation analysis completed in {duration:.2f} seconds")
        save_results(result, args.output)
        
    except Exception as e:
        logger.error(f"Mutation analysis failed: {e}")
        sys.exit(1)


def cmd_serve(args):
    """Handle serve command."""
    logger.info(f"Starting REST API server on {args.host}:{args.port}")
    
    try:
        api = ProteinScienceAPI()
        api.run(
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        sys.exit(1)


def cmd_app(args):
    """Handle app command."""
    logger.info(f"Starting Streamlit app on port {args.port}")
    
    try:
        import subprocess
        import os
        
        # Get the path to the Streamlit app
        app_path = Path(__file__).parent.parent / "interface" / "streamlit_app.py"
        
        # Start Streamlit
        env = os.environ.copy()
        env["STREAMLIT_SERVER_PORT"] = str(args.port)
        
        subprocess.run([
            "streamlit", "run", str(app_path),
            "--server.port", str(args.port),
            "--server.address", "0.0.0.0"
        ], env=env)
        
    except Exception as e:
        logger.error(f"Failed to start Streamlit app: {e}")
        sys.exit(1)


def cmd_info(args):
    """Handle info command."""
    print("Protein Science AI System Information")
    print("=" * 40)
    
    try:
        # System info
        print(f"Version: 0.1.0")
        print(f"Python: {sys.version}")
        
        # Check dependencies
        print("\nDependencies:")
        
        deps_to_check = [
            "torch",
            "transformers", 
            "biopython",
            "fair-esm",
            "langchain",
            "fastapi",
            "streamlit",
            "loguru"
        ]
        
        for dep in deps_to_check:
            try:
                __import__(dep)
                print(f"  ✅ {dep}")
            except ImportError:
                print(f"  ❌ {dep} (not installed)")
        
        # Agent info
        print("\nAgent Status:")
        try:
            agent = ProteinAgent()
            print(f"  ✅ Protein Agent: {agent.plm.model_name}")
            
            coordinator = AgentCoordinator(main_agent=agent)
            print(f"  ✅ Coordinator: {len(coordinator.specialized_agents)} specialized agents")
            
        except Exception as e:
            print(f"  ❌ Agent initialization failed: {e}")
            
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")


async def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == "analyze":
        await cmd_analyze(args)
    elif args.command == "workflow":
        await cmd_workflow(args)
    elif args.command == "mutations":
        await cmd_mutations(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "app":
        cmd_app(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


def cli_main():
    """Entry point for console script."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()