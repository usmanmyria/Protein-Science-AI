"""
Streamlit-based interactive web application for protein science AI.

This module provides a user-friendly web interface for protein analysis,
visualization, and interactive exploration of results.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import asyncio
import json
from datetime import datetime

# Import core components
try:
    from ..agents.protein_agent import ProteinAgent
    from ..collaboration.coordinator import AgentCoordinator
except ImportError as e:
    st.error(f"Failed to import core components: {e}")
    st.stop()


def create_streamlit_app():
    """Create and configure the Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Protein Science AI",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'protein_agent' not in st.session_state:
        with st.spinner("Initializing Protein Science AI..."):
            st.session_state.protein_agent = ProteinAgent()
            st.session_state.coordinator = AgentCoordinator(
                main_agent=st.session_state.protein_agent
            )
    
    # Main header
    st.markdown('<h1 class="main-header">🧬 Protein Science AI</h1>', unsafe_allow_html=True)
    st.markdown("### Agentic AI system for autonomous protein analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["Protein Analysis", "Multi-Agent Workflows", "Mutation Analysis", "Results Explorer", "About"]
        )
        
        st.markdown("---")
        st.header("Quick Analysis")
        
        # Quick sequence input
        quick_sequence = st.text_area(
            "Paste protein sequence:",
            placeholder="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            height=100
        )
        
        if st.button("Quick Analyze") and quick_sequence:
            with st.spinner("Running quick analysis..."):
                try:
                    # Run analysis
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(
                        st.session_state.protein_agent.analyze_protein(
                            quick_sequence.strip(),
                            analysis_type="quick"
                        )
                    )
                    st.session_state.analysis_results = result
                    st.success("Analysis complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
    
    # Main content based on selected page
    if page == "Protein Analysis":
        render_protein_analysis_page()
    elif page == "Multi-Agent Workflows":
        render_workflow_page()
    elif page == "Mutation Analysis":
        render_mutation_analysis_page()
    elif page == "Results Explorer":
        render_results_explorer_page()
    elif page == "About":
        render_about_page()


def render_protein_analysis_page():
    """Render the main protein analysis page."""
    st.markdown('<h2 class="section-header">Protein Sequence Analysis</h2>', unsafe_allow_html=True)
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Protein Data")
        
        # Sequence input
        sequence = st.text_area(
            "Protein Sequence:",
            height=150,
            placeholder="Enter protein sequence in single-letter amino acid code...",
            help="Enter a protein sequence using standard single-letter amino acid codes (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)"
        )
        
        # UniProt ID input
        uniprot_id = st.text_input(
            "UniProt ID (optional):",
            placeholder="e.g., P53_HUMAN",
            help="UniProt identifier for additional annotations and structure data"
        )
        
    with col2:
        st.subheader("Analysis Options")
        
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["comprehensive", "quick", "custom"],
            help="Type of analysis to perform"
        )
        
        include_structure = st.checkbox("Include Structure Prediction", value=True)
        include_function = st.checkbox("Include Function Prediction", value=True)
        
        # Analysis button
        analyze_button = st.button(
            "🔬 Analyze Protein",
            type="primary",
            use_container_width=True
        )
    
    # Run analysis
    if analyze_button and sequence:
        if not sequence.replace(" ", "").replace("\n", "").isalpha():
            st.error("Please enter a valid protein sequence containing only amino acid letters.")
        else:
            with st.spinner("Analyzing protein... This may take a few minutes."):
                try:
                    # Prepare input
                    protein_input = {
                        "sequence": sequence.replace(" ", "").replace("\n", "").upper(),
                        "uniprot_id": uniprot_id if uniprot_id else None
                    }
                    
                    # Run analysis
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(
                        st.session_state.protein_agent.analyze_protein(
                            protein_input=protein_input,
                            analysis_type=analysis_type,
                            include_structure=include_structure,
                            include_function=include_function
                        )
                    )
                    
                    st.session_state.analysis_results = result
                    st.success("Analysis completed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    
    # Display results
    if st.session_state.analysis_results:
        display_analysis_results(st.session_state.analysis_results)


def render_workflow_page():
    """Render the multi-agent workflow page."""
    st.markdown('<h2 class="section-header">Multi-Agent Workflows</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Execute complex protein science workflows using multiple specialized AI agents
    working together to provide comprehensive analysis.
    """)
    
    # Workflow selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        workflow_type = st.selectbox(
            "Select Workflow:",
            [
                "comprehensive_analysis",
                "drug_discovery", 
                "protein_engineering",
                "mutation_analysis"
            ],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        # Workflow descriptions
        workflow_descriptions = {
            "comprehensive_analysis": "Complete analysis using all available agents",
            "drug_discovery": "Focus on drug target analysis and binding sites",
            "protein_engineering": "Optimization strategies and design recommendations", 
            "mutation_analysis": "Compare wild-type with mutant variants"
        }
        
        st.info(workflow_descriptions[workflow_type])
    
    with col2:
        # Protein input
        workflow_sequence = st.text_area(
            "Protein Sequence:",
            height=100,
            key="workflow_sequence"
        )
        
        workflow_uniprot = st.text_input(
            "UniProt ID (optional):",
            key="workflow_uniprot"
        )
    
    # Additional configuration
    st.subheader("Workflow Configuration")
    
    if workflow_type == "mutation_analysis":
        mutations = st.text_input(
            "Mutations (comma-separated):",
            placeholder="e.g., A123G,T456A,R789K",
            help="Enter mutations in format: OriginalAA + Position + NewAA"
        )
        config = {"mutations": [m.strip() for m in mutations.split(",") if m.strip()]} if mutations else None
    else:
        config = None
    
    # Execute workflow
    if st.button("🚀 Execute Workflow", type="primary"):
        if not workflow_sequence:
            st.error("Please enter a protein sequence")
        else:
            with st.spinner(f"Executing {workflow_type.replace('_', ' ')} workflow..."):
                try:
                    protein_input = {
                        "sequence": workflow_sequence.replace(" ", "").replace("\n", "").upper(),
                        "uniprot_id": workflow_uniprot if workflow_uniprot else None
                    }
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(
                        st.session_state.coordinator.execute_workflow(
                            workflow_type=workflow_type,
                            protein_input=protein_input,
                            workflow_config=config
                        )
                    )
                    
                    st.session_state.workflow_results = result
                    st.success("Workflow completed!")
                    
                    # Display workflow results
                    display_workflow_results(result)
                    
                except Exception as e:
                    st.error(f"Workflow failed: {str(e)}")


def render_mutation_analysis_page():
    """Render the mutation analysis page."""
    st.markdown('<h2 class="section-header">Mutation Effect Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Analyze the effects of specific mutations on protein structure and function.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Wild-type Protein")
        
        wt_sequence = st.text_area(
            "Wild-type Sequence:",
            height=150,
            key="wt_sequence"
        )
        
        wt_uniprot = st.text_input(
            "UniProt ID (optional):",
            key="wt_uniprot"
        )
    
    with col2:
        st.subheader("Mutations to Analyze")
        
        # Mutation input methods
        input_method = st.radio(
            "Input Method:",
            ["Manual Entry", "Upload File"]
        )
        
        if input_method == "Manual Entry":
            mutations_text = st.text_area(
                "Mutations (one per line):",
                placeholder="A123G\nT456A\nR789K",
                height=150,
                help="Format: OriginalAA + Position + NewAA"
            )
            mutations = [m.strip() for m in mutations_text.split("\n") if m.strip()]
        else:
            uploaded_file = st.file_uploader(
                "Upload mutation file:",
                type=['txt', 'csv'],
                help="Upload a file with mutations, one per line"
            )
            mutations = []
            if uploaded_file:
                content = uploaded_file.read().decode()
                mutations = [m.strip() for m in content.split("\n") if m.strip()]
    
    # Display mutations
    if mutations:
        st.subheader("Mutations to Analyze")
        df_mutations = pd.DataFrame({
            "Mutation": mutations,
            "Original AA": [m[0] for m in mutations if len(m) >= 3],
            "Position": [m[1:-1] for m in mutations if len(m) >= 3],
            "New AA": [m[-1] for m in mutations if len(m) >= 3]
        })
        st.dataframe(df_mutations, use_container_width=True)
    
    # Run mutation analysis
    if st.button("🧬 Analyze Mutations", type="primary"):
        if not wt_sequence:
            st.error("Please enter the wild-type sequence")
        elif not mutations:
            st.error("Please specify mutations to analyze")
        else:
            with st.spinner("Analyzing mutation effects..."):
                try:
                    protein_input = {
                        "sequence": wt_sequence.replace(" ", "").replace("\n", "").upper(),
                        "uniprot_id": wt_uniprot if wt_uniprot else None
                    }
                    
                    config = {"mutations": mutations}
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(
                        st.session_state.coordinator.execute_workflow(
                            workflow_type="mutation_analysis",
                            protein_input=protein_input,
                            workflow_config=config
                        )
                    )
                    
                    st.session_state.mutation_results = result
                    st.success("Mutation analysis completed!")
                    
                    # Display results
                    display_mutation_results(result)
                    
                except Exception as e:
                    st.error(f"Mutation analysis failed: {str(e)}")


def render_results_explorer_page():
    """Render the results explorer page."""
    st.markdown('<h2 class="section-header">Results Explorer</h2>', unsafe_allow_html=True)
    
    if st.session_state.analysis_results:
        st.subheader("Analysis Results")
        display_analysis_results(st.session_state.analysis_results)
    
    if hasattr(st.session_state, 'workflow_results'):
        st.subheader("Workflow Results")
        display_workflow_results(st.session_state.workflow_results)
    
    if hasattr(st.session_state, 'mutation_results'):
        st.subheader("Mutation Analysis Results")
        display_mutation_results(st.session_state.mutation_results)
    
    if not any([
        st.session_state.analysis_results,
        hasattr(st.session_state, 'workflow_results'),
        hasattr(st.session_state, 'mutation_results')
    ]):
        st.info("No results to display. Run an analysis first.")


def render_about_page():
    """Render the about page."""
    st.markdown('<h2 class="section-header">About Protein Science AI</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Overview
    
    Protein Science AI is a comprehensive agentic AI system that integrates protein language models,
    molecular simulation tools, and autonomous reasoning layers to accelerate protein science research.
    
    ## 🏗️ System Architecture
    
    ### Foundation Layer
    - **Protein Language Models**: ESM-2, ProtBERT, AlphaFold embeddings
    - **Structure Prediction**: AlphaFold2/3, RoseTTAFold integration
    - **Function Prediction**: Graph neural networks for protein interactions
    
    ### Agentic Layer
    - **Planner Module**: Chain-of-thought reasoning for experimental design
    - **Tool Integration**: Molecular dynamics, docking, database access
    - **Feedback Loop**: Iterative hypothesis refinement
    
    ### Collaboration Layer
    - **Specialized Agents**: Docking, Mutation, Pathway analysis
    - **Coordinator Agent**: Workflow management and consistency
    
    ## 🚀 Core Capabilities
    
    - **Structure Prediction & Analysis**: Folding, domains, disorder regions
    - **Hypothesis Generation**: Mutation effects, stability enhancement
    - **Molecular Simulation**: MD simulations, binding optimization
    - **Autonomous Discovery**: Literature mining, enzyme design
    
    ## 🔬 Applications
    
    - **Drug Discovery**: Protein-ligand docking, off-target prediction
    - **Protein Engineering**: Stability enhancement, enzyme optimization
    - **Synthetic Biology**: Novel protein circuit design
    - **Disease Research**: Mutation impact analysis
    
    ## 🛠️ Technology Stack
    
    - **AI Models**: PyTorch, Transformers (Hugging Face)
    - **Simulations**: GROMACS, OpenMM, Rosetta
    - **Databases**: PDB, UniProt, ChEMBL
    - **Interface**: Streamlit, FastAPI
    """)
    
    # System status
    st.subheader("System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Protein Agent", "Active", delta="✅")
    with col2:
        st.metric("Coordinator", "Active", delta="✅")
    with col3:
        agents_count = len(st.session_state.coordinator.specialized_agents)
        st.metric("Specialized Agents", agents_count, delta="🤖")


def display_analysis_results(results: Dict[str, Any]):
    """Display protein analysis results."""
    st.markdown('<h3 class="section-header">Analysis Results</h3>', unsafe_allow_html=True)
    
    # Basic info
    input_data = results.get("input", {})
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sequence Length", len(input_data.get("sequence", "")))
    with col2:
        st.metric("Analysis Type", input_data.get("analysis_type", "unknown").title())
    with col3:
        analysis_time = results.get("analysis_time_seconds", 0)
        st.metric("Analysis Time", f"{analysis_time:.1f}s")
    with col4:
        st.metric("UniProt ID", input_data.get("uniprot_id", "N/A"))
    
    # Sequence analysis
    if "sequence_analysis" in results:
        st.subheader("🧬 Sequence Analysis")
        seq_analysis = results["sequence_analysis"]
        
        # Basic properties
        if "basic_properties" in seq_analysis:
            props = seq_analysis["basic_properties"]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Molecular Weight", f"{props.get('molecular_weight_approx', 0):,} Da")
            with col2:
                st.metric("Unique Residues", props.get('unique_residues', 0))
            with col3:
                st.metric("Most Common AA", props.get('most_common_residue', 'N/A'))
        
        # Patterns
        if "patterns" in seq_analysis:
            patterns = seq_analysis["patterns"]
            
            # Known motifs
            if "known_motifs" in patterns and patterns["known_motifs"]:
                st.subheader("Known Motifs")
                motifs_data = []
                for motif_info in patterns["known_motifs"]:
                    motifs_data.append({
                        "Motif": motif_info["motif"],
                        "Description": motif_info["description"],
                        "Positions": ", ".join(map(str, motif_info["positions"]))
                    })
                
                if motifs_data:
                    df_motifs = pd.DataFrame(motifs_data)
                    st.dataframe(df_motifs, use_container_width=True)
            
            # Charge distribution
            if "charge_distribution" in patterns:
                charge_info = patterns["charge_distribution"]
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Net Charge", charge_info.get("net_charge", 0))
                with col2:
                    st.metric("Charge Density", f"{charge_info.get('charge_density', 0):.3f}")
    
    # Structure analysis
    if "structure_analysis" in results:
        st.subheader("🏗️ Structure Analysis")
        struct_analysis = results["structure_analysis"]
        
        if "error" not in struct_analysis:
            # Display structure information
            if "alphafold_data" in struct_analysis:
                st.success("AlphaFold structure data available")
                
                # Confidence scores
                if "confidence_scores" in struct_analysis:
                    conf_info = struct_analysis["confidence_scores"]
                    if "overall_confidence" in conf_info:
                        st.metric("Overall Confidence", conf_info["overall_confidence"])
            
            # Structure analysis details
            if "structure_analysis" in struct_analysis:
                struct_details = struct_analysis["structure_analysis"]
                
                # Create metrics from available data
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    if "num_atoms" in struct_details:
                        st.metric("Atoms", struct_details["num_atoms"])
                    elif "basic_num_atoms" in struct_details:
                        st.metric("Atoms", struct_details["basic_num_atoms"])
                        
                with metrics_col2:
                    if "num_residues" in struct_details:
                        st.metric("Residues", struct_details["num_residues"])
                    elif "basic_num_residues" in struct_details:
                        st.metric("Residues", struct_details["basic_num_residues"])
                        
                with metrics_col3:
                    if "num_chains" in struct_details:
                        st.metric("Chains", struct_details["num_chains"])
                    elif "basic_num_chains" in struct_details:
                        st.metric("Chains", struct_details["basic_num_chains"])
        else:
            st.warning(f"Structure analysis failed: {struct_analysis['error']}")
    
    # Function analysis
    if "function_analysis" in results:
        st.subheader("⚙️ Function Analysis")
        func_analysis = results["function_analysis"]
        
        # Methods used
        methods = func_analysis.get("methods_used", [])
        st.info(f"Analysis methods: {', '.join(methods)}")
        
        # Show predictions from each method
        predictions = func_analysis.get("predictions", {})
        
        for method, prediction in predictions.items():
            if isinstance(prediction, dict) and "error" not in prediction:
                with st.expander(f"{method.replace('_', ' ').title()} Results"):
                    st.json(prediction)
    
    # Insights
    if "insights" in results:
        st.subheader("💡 Key Insights")
        insights = results["insights"]
        
        # Key findings
        key_findings = insights.get("key_findings", [])
        if key_findings:
            for finding in key_findings:
                st.info(f"• {finding}")
        
        # Areas of interest
        areas = insights.get("areas_of_interest", [])
        if areas:
            st.subheader("Areas of Interest")
            for area in areas:
                st.success(f"• {area}")
    
    # Experiment suggestions
    if "experiment_suggestions" in results:
        st.subheader("🧪 Experimental Suggestions")
        suggestions = results["experiment_suggestions"]
        
        for suggestion in suggestions:
            priority_color = {
                "high": "🔴",
                "medium": "🟡", 
                "low": "🟢"
            }
            
            priority = suggestion.get("priority", "medium")
            color = priority_color.get(priority, "⚪")
            
            with st.expander(f"{color} {suggestion.get('experiment', 'Unknown experiment')} ({priority} priority)"):
                st.write(f"**Type:** {suggestion.get('type', 'N/A')}")
                st.write(f"**Rationale:** {suggestion.get('rationale', 'N/A')}")


def display_workflow_results(results: Dict[str, Any]):
    """Display workflow results."""
    st.markdown('<h3 class="section-header">Workflow Results</h3>', unsafe_allow_html=True)
    
    # Workflow info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Workflow ID", results.get("workflow_id", "unknown"))
    with col2:
        st.metric("Status", results.get("status", "unknown").title())
    with col3:
        duration = results.get("duration_seconds", 0)
        st.metric("Duration", f"{duration:.1f}s")
    
    # Agent outputs
    if "agent_outputs" in results:
        st.subheader("Agent Outputs")
        
        agent_outputs = results["agent_outputs"]
        
        for agent_name, output in agent_outputs.items():
            with st.expander(f"{agent_name.replace('_', ' ').title()} Results"):
                if isinstance(output, dict):
                    # Display summary metrics if available
                    if "insights" in output:
                        insights = output["insights"]
                        if "key_findings" in insights:
                            st.write("**Key Findings:**")
                            for finding in insights["key_findings"]:
                                st.write(f"• {finding}")
                
                # Show raw data
                st.json(output)
    
    # Integrated results
    if "results" in results and "integrated_analysis" in results["results"]:
        st.subheader("Integrated Analysis")
        integrated = results["results"]["integrated_analysis"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Participating Agents", len(integrated.get("participating_agents", [])))
        with col2:
            st.metric("Consensus Insights", integrated.get("consensus_count", 0))
        with col3:
            conf = integrated.get("average_confidence", 0)
            st.metric("Average Confidence", f"{conf:.2f}")
        
        # Consensus insights
        consensus = integrated.get("consensus_insights", [])
        if consensus:
            st.subheader("Consensus Insights")
            for insight in consensus:
                st.success(f"• {insight}")


def display_mutation_results(results: Dict[str, Any]):
    """Display mutation analysis results."""
    st.markdown('<h3 class="section-header">Mutation Analysis Results</h3>', unsafe_allow_html=True)
    
    if "results" in results and "comparative_analysis" in results["results"]:
        comparative = results["results"]["comparative_analysis"]
        
        st.metric("Mutations Analyzed", comparative.get("mutant_count", 0))
        
        # Comparison table
        if "comparisons" in comparative:
            comparisons = comparative["comparisons"]
            
            comparison_data = []
            for mutation, comparison in comparisons.items():
                comparison_data.append({
                    "Mutation": mutation,
                    "Structural Changes": comparison.get("structural_changes", "Pending"),
                    "Functional Changes": comparison.get("functional_changes", "Pending"),
                    "Stability Changes": comparison.get("stability_changes", "Pending")
                })
            
            if comparison_data:
                df_comparisons = pd.DataFrame(comparison_data)
                st.dataframe(df_comparisons, use_container_width=True)
    
    # Show raw workflow results
    with st.expander("Raw Results"):
        st.json(results)


# Main entry point
if __name__ == "__main__":
    create_streamlit_app()