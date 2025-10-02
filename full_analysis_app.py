#!/usr/bin/env python3
"""
Full Analysis Web App for Protein Science AI - Clean Version
"""

import streamlit as st
import sys
import os
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import json

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Protein Science AI - Full Analysis",
    page_icon="🧬",
    layout="wide"
)

def run_async(coro):
    """Helper to run async functions in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def main():
    st.title("🧬 Agentic AI System for Protein Science")
    st.markdown("### Full Analysis Interface with ESM-2 AI Models")
    st.markdown("---")

    # Sidebar for navigation
    with st.sidebar:
        st.header("🔧 Analysis Options")
        analysis_type = st.selectbox(
            "Choose Analysis Type:",
            ["Quick Analysis", "Full AI Analysis", "Comparative Analysis"]
        )
        
        st.markdown("### 📊 System Status")
        
        # Test imports and show status
        try:
            from protein_science.agents.protein_agent import ProteinAgent
            st.success("✅ ProteinAgent Ready")
            
            from protein_science.foundation.protein_models import ProteinLanguageModel
            st.success("✅ ESM-2 Model Ready")
            
            system_ready = True
            
        except Exception as e:
            st.error(f"❌ System Error: {e}")
            system_ready = False

    if not system_ready:
        st.error("System not ready. Please check installation.")
        return

    # Main content area
    if analysis_type == "Quick Analysis":
        show_quick_analysis()
    elif analysis_type == "Full AI Analysis":
        show_full_ai_analysis()
    elif analysis_type == "Comparative Analysis":
        show_comparative_analysis()

def show_quick_analysis():
    st.header("⚡ Quick Protein Analysis")
    st.markdown("Fast analysis without AI model loading")
    
    sequence = st.text_area(
        "Enter protein sequence:",
        value="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        height=100,
        help="Paste your protein sequence here (single letter amino acid codes)"
    )
    
    if st.button("🔍 Analyze Now", type="primary"):
        if sequence:
            with st.spinner("Analyzing protein..."):
                results = analyze_quick(sequence)
                display_quick_results(results)

def show_full_ai_analysis():
    st.header("🤖 Full AI Analysis with ESM-2")
    st.markdown("Deep analysis using protein language models and autonomous AI agent")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        sequence = st.text_area(
            "Enter protein sequence:",
            value="MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
            height=150,
            help="Enter your protein sequence for comprehensive AI analysis"
        )
        
        uniprot_id = st.text_input(
            "UniProt ID (optional):",
            value="",
            help="Optional UniProt ID for enhanced analysis"
        )
    
    with col2:
        st.markdown("### 🎯 Analysis Features")
        st.markdown("""
        🧠 **ESM-2 Embeddings** (Deep Learning)  
        🏗️ **Structure Prediction**  
        ⚡ **Function Prediction**  
        🤔 **AI Reasoning**  
        🔬 **Experiment Suggestions**  
        📊 **Interactive Visualizations**  
        🎯 **Embedding Heatmaps**
        """)
    
    if st.button("🚀 Run Full AI Analysis", type="primary"):
        if sequence:
            with st.spinner("Running comprehensive AI analysis... This may take 30-60 seconds"):
                results = run_full_analysis(sequence, uniprot_id)
                display_full_results(results)

def show_comparative_analysis():
    st.header("📊 Comparative Protein Analysis")
    st.markdown("Compare multiple proteins side by side")
    
    num_proteins = st.number_input("Number of proteins to compare:", min_value=2, max_value=5, value=2)
    
    sequences = {}
    for i in range(num_proteins):
        name = st.text_input(f"Protein {i+1} name:", value=f"Protein_{i+1}")
        sequence = st.text_area(
            f"Protein {i+1} sequence:",
            height=100,
            help=f"Enter sequence for protein {i+1}"
        )
        if name and sequence:
            sequences[name] = sequence
    
    if st.button("📊 Compare Proteins", type="primary"):
        if len(sequences) >= 2:
            with st.spinner("Comparing proteins..."):
                results = compare_proteins(sequences)
                display_comparison_results(results)

def analyze_quick(sequence):
    """Quick protein analysis without AI models"""
    try:
        import protein_science.foundation.protein_models as pm
        
        # Basic sequence analysis
        length = len(sequence)
        
        # Calculate molecular weight (approximate)
        aa_weights = {
            'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
            'Q': 146.1, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
            'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
            'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
        }
        
        molecular_weight = sum(aa_weights.get(aa, 110) for aa in sequence.upper()) - (length - 1) * 18.015
        
        # Basic composition analysis
        composition = {}
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            composition[aa] = sequence.upper().count(aa)
        
        # Calculate properties
        charged_positive = sequence.count('K') + sequence.count('R') + sequence.count('H')
        charged_negative = sequence.count('D') + sequence.count('E')
        net_charge = charged_positive - charged_negative
        
        hydrophobic_aas = 'AILMFWYV'
        hydrophobic_count = sum(sequence.upper().count(aa) for aa in hydrophobic_aas)
        hydrophobicity = hydrophobic_count / length if length > 0 else 0
        
        cysteine_count = sequence.upper().count('C')
        
        return {
            'sequence': sequence,
            'length': length,
            'molecular_weight': round(molecular_weight, 2),
            'composition': composition,
            'net_charge': net_charge,
            'hydrophobicity': round(hydrophobicity, 3),
            'cysteine_count': cysteine_count,
            'charged_positive': charged_positive,
            'charged_negative': charged_negative
        }
        
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return None

def run_full_analysis(sequence, uniprot_id=None):
    """Run full AI analysis with ESM-2 models"""
    try:
        from protein_science.agents.protein_agent import ProteinAgent
        
        # Create agent
        agent = ProteinAgent()
        
        # Run AI analysis
        async def analyze():
            return await agent.analyze_protein({
                'sequence': sequence,
                'uniprot_id': uniprot_id
            })
        
        st.info("🤖 Running AI analysis with ESM-2 models...")
        ai_results = run_async(analyze())
        
        # Add quick analysis data
        quick_results = analyze_quick(sequence)
        
        # Merge results properly
        final_results = {}
        if quick_results:
            final_results.update(quick_results)
        
        if ai_results:
            # Extract embeddings from nested structure
            sequence_analysis = ai_results.get('sequence_analysis', {})
            plm_analysis = sequence_analysis.get('plm_analysis', {})
            
            # Check for embeddings in different possible locations
            embeddings = None
            if 'sequence_embedding' in plm_analysis:
                embeddings = plm_analysis['sequence_embedding']
            elif 'residue_embeddings' in plm_analysis:
                embeddings = plm_analysis['residue_embeddings']
            elif 'embeddings' in plm_analysis:
                embeddings = plm_analysis['embeddings']
            
            # If we found embeddings, add them to results
            if embeddings is not None:
                final_results['embeddings'] = embeddings
                st.success("✅ ESM-2 embeddings successfully extracted!")
            else:
                st.warning("⚠️ Embeddings not found in expected locations")
                # Debug: show what's available
                st.write("**Debug - PLM Analysis keys:**", list(plm_analysis.keys()) if plm_analysis else "No PLM analysis")
            
            # Store the full AI analysis results
            final_results['ai_analysis'] = ai_results
            
            # Extract other analysis components
            if 'structure_analysis' in ai_results:
                final_results['structure_prediction'] = ai_results['structure_analysis']
            if 'function_analysis' in ai_results:
                final_results['function_prediction'] = ai_results['function_analysis']
            if 'insights' in ai_results:
                final_results['insights'] = ai_results['insights']
            if 'experiment_suggestions' in ai_results:
                final_results['experiment_suggestions'] = ai_results['experiment_suggestions']
        
        st.success("✅ AI analysis completed!")
        return final_results
        
    except Exception as e:
        st.error(f"Full analysis error: {e}")
        st.warning("Falling back to quick analysis...")
        return analyze_quick(sequence)  # Fallback to quick analysis

def display_quick_results(results):
    """Display quick analysis results"""
    if not results:
        return
    
    st.markdown("## 📊 Quick Analysis Results")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Length", f"{results['length']} aa")
    with col2:
        st.metric("Molecular Weight", f"{results['molecular_weight']:,.0f} Da")
    with col3:
        st.metric("Net Charge", results['net_charge'])
    with col4:
        st.metric("Hydrophobicity", f"{results['hydrophobicity']:.3f}")
    
    # Composition analysis
    st.markdown("### 🧬 Amino Acid Composition")
    
    composition_data = []
    for aa, count in results['composition'].items():
        if count > 0:
            composition_data.append({
                'Amino Acid': aa,
                'Count': count,
                'Percentage': round(100 * count / results['length'], 1)
            })
    
    df_comp = pd.DataFrame(composition_data)
    df_comp = df_comp.sort_values('Count', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(df_comp)
    
    with col2:
        fig = px.bar(df_comp.head(10), x='Amino Acid', y='Count', 
                     title='Top 10 Amino Acids by Count')
        st.plotly_chart(fig, use_container_width=True)
    
    # Properties summary
    st.markdown("### 📈 Sequence Properties")
    properties_df = pd.DataFrame([
        {'Property': 'Sequence Length', 'Value': f"{results['length']} amino acids"},
        {'Property': 'Molecular Weight', 'Value': f"{results['molecular_weight']:,.1f} Da"},
        {'Property': 'Net Charge', 'Value': f"{results['net_charge']:+d}"},
        {'Property': 'Positive Charges', 'Value': f"{results['charged_positive']}"},
        {'Property': 'Negative Charges', 'Value': f"{results['charged_negative']}"},
        {'Property': 'Hydrophobicity', 'Value': f"{results['hydrophobicity']:.3f}"},
        {'Property': 'Cysteine Count', 'Value': f"{results['cysteine_count']}"},
    ])
    
    st.dataframe(properties_df, use_container_width=True)

def display_full_results(results):
    """Display full AI analysis results"""
    if not results:
        st.error("No results to display")
        return
    
    st.markdown("## 🤖 Full AI Analysis Results")
    
    # Debug: Show what keys are in results (only in debug mode)
    with st.expander("🔍 Debug Information"):
        st.write("**Available result keys:**", list(results.keys()))
        
        # Show AI analysis structure if available
        if 'ai_analysis' in results:
            ai_keys = list(results['ai_analysis'].keys())
            st.write("**AI Analysis keys:**", ai_keys)
            
            # Show sequence analysis structure
            if 'sequence_analysis' in results['ai_analysis']:
                seq_analysis = results['ai_analysis']['sequence_analysis']
                st.write("**Sequence Analysis keys:**", list(seq_analysis.keys()))
                
                if 'plm_analysis' in seq_analysis:
                    plm_keys = list(seq_analysis['plm_analysis'].keys())
                    st.write("**PLM Analysis keys:**", plm_keys)
    
    # First show quick results
    display_quick_results(results)
    
    # AI Analysis section
    if 'ai_analysis' in results or any(key in results for key in ['embeddings', 'structure_prediction', 'function_prediction']):
        st.markdown("## 🧠 AI Analysis & Insights")
        
        # Check for embeddings
        embeddings_found = False
        embeddings = None
        
        if 'embeddings' in results:
            embeddings = results['embeddings']
            embeddings_found = True
        elif 'ai_analysis' in results:
            # Try to extract from nested structure
            ai_results = results['ai_analysis']
            seq_analysis = ai_results.get('sequence_analysis', {})
            plm_analysis = seq_analysis.get('plm_analysis', {})
            
            # Check different possible embedding keys
            for key in ['sequence_embedding', 'residue_embeddings', 'embeddings']:
                if key in plm_analysis:
                    embeddings = plm_analysis[key]
                    embeddings_found = True
                    st.info(f"✅ Found embeddings under key: {key}")
                    break
        
        if embeddings_found and embeddings is not None:
            st.markdown("### 🎯 ESM-2 Protein Embeddings")
            st.success("✅ ESM-2 embeddings computed successfully! "
                       "This represents the AI model's deep understanding of your protein.")
            
            # Convert to numpy array for analysis
            try:
                embeddings_array = np.array(embeddings)
                
                # Enhanced visualization of embedding statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if len(embeddings_array.shape) > 1:
                        st.metric("Embedding Dimension", embeddings_array.shape[-1])
                        st.metric("Sequence Length", embeddings_array.shape[0])
                    else:
                        st.metric("Embedding Dimension", len(embeddings_array))
                        st.metric("Type", "Sequence-level")
                with col2:
                    st.metric("Mean Activation", f"{embeddings_array.mean():.4f}")
                with col3:
                    st.metric("Std Activation", f"{embeddings_array.std():.4f}")
                with col4:
                    st.metric("Total Parameters", f"{embeddings_array.size:,}")
                
                # Show embedding heatmap for visualization
                if len(embeddings_array.shape) > 1 and embeddings_array.shape[0] > 1:
                    st.markdown("#### Embedding Heatmap")
                    # Take a subset for visualization
                    max_pos = min(50, embeddings_array.shape[0])
                    max_dim = min(100, embeddings_array.shape[1])
                    viz_embeddings = embeddings_array[:max_pos, :max_dim]
                    
                    fig = px.imshow(viz_embeddings, 
                                   title=f"ESM-2 Embeddings Heatmap (First {max_pos} positions, First {max_dim} dimensions)",
                                   labels=dict(x="Embedding Dimension", y="Sequence Position", color="Activation"),
                                   aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # For 1D embeddings, show as a line plot
                    st.markdown("#### Sequence-Level Embedding")
                    fig = px.line(y=embeddings_array, title="ESM-2 Sequence Embedding Values")
                    fig.update_layout(xaxis_title="Dimension", yaxis_title="Activation")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show embedding statistics
                st.markdown("#### Embedding Analysis")
                embedding_stats = pd.DataFrame({
                    'Statistic': ['Min Value', 'Max Value', 'Mean', 'Std Dev', 'Zero Count', 'Shape'],
                    'Value': [
                        f"{embeddings_array.min():.6f}",
                        f"{embeddings_array.max():.6f}", 
                        f"{embeddings_array.mean():.6f}",
                        f"{embeddings_array.std():.6f}",
                        f"{(embeddings_array == 0).sum():,}",
                        str(embeddings_array.shape)
                    ]
                })
                st.dataframe(embedding_stats, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error processing embeddings: {e}")
                st.write("Raw embeddings data:", type(embeddings))
        else:
            st.warning("⚠️ No embeddings found in results")
            
            # Show what we tried to extract
            if 'ai_analysis' in results:
                st.write("**Attempted to extract embeddings from:**")
                ai_results = results['ai_analysis']
                seq_analysis = ai_results.get('sequence_analysis', {})
                plm_analysis = seq_analysis.get('plm_analysis', {})
                st.write(f"- PLM Analysis keys: {list(plm_analysis.keys())}")
        
        # Get AI analysis data
        ai_results = results.get('ai_analysis', {})
        
        # Structure prediction
        structure_pred = results.get('structure_prediction') or ai_results.get('structure_analysis')
        if structure_pred:
            st.markdown("### 🏗️ Structure Prediction")
            
            # Show structure prediction results
            if 'error' not in structure_pred:
                st.success("✅ Structure prediction completed")
                
                # Show any available structural insights
                if 'structural_insights' in structure_pred:
                    insights = structure_pred['structural_insights']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Has Structural Data", 
                                 "Yes" if insights.get('structural_summary', {}).get('has_structural_data') else "No")
                    with col2:
                        st.metric("Analysis Methods", 
                                 len(insights.get('structural_summary', {}).get('analysis_methods', [])))
                    with col3:
                        confidence_data = insights.get('confidence_analysis', {})
                        if confidence_data.get('has_confidence_data'):
                            st.metric("Overall Confidence", 
                                     f"{confidence_data.get('overall_confidence', 'N/A')}")
            else:
                st.warning(f"⚠️ Structure prediction failed: {structure_pred.get('error')}")
        
        # Function prediction
        func_pred = results.get('function_prediction') or ai_results.get('function_analysis')
        if func_pred:
            st.markdown("### ⚡ Function Prediction")
            
            # Show function prediction results
            if 'error' not in func_pred:
                st.success("✅ Function prediction completed")
                
                # Show methods used
                methods_used = func_pred.get('methods_used', [])
                if methods_used:
                    st.write(f"**Analysis methods used:** {', '.join(methods_used)}")
                
                # Show predicted functions if available
                if 'predicted_functions' in func_pred:
                    st.write("**Predicted functions:**")
                    for func in func_pred['predicted_functions']:
                        st.write(f"• **{func.get('category', 'Unknown')}**: {func.get('description', 'N/A')} "
                                f"(Confidence: {func.get('confidence', 'N/A')})")
            else:
                st.warning(f"⚠️ Function prediction failed: {func_pred.get('error')}")
        
        # AI insights
        insights = ai_results.get('insights', {})
        if insights:
            st.markdown("### 🤔 AI Insights")
            
            # Key findings
            key_findings = insights.get('key_findings', [])
            if key_findings:
                st.write("**Key Findings:**")
                for finding in key_findings:
                    st.write(f"• {finding}")
            
            # Areas of interest
            areas = insights.get('areas_of_interest', [])
            if areas:
                st.write("**Areas of Interest:**")
                for area in areas:
                    st.write(f"• {area}")
            
            # Confidence level
            confidence = insights.get('confidence_level')
            if confidence:
                st.write(f"**Analysis Confidence Level:** {confidence}")
        
        # Experiment suggestions
        suggestions = ai_results.get('experiment_suggestions', [])
        if suggestions:
            st.markdown("### 🔬 Suggested Experiments")
            for suggestion in suggestions:
                priority_emoji = "🔴" if suggestion.get('priority') == 'high' else "🟡" if suggestion.get('priority') == 'medium' else "🟢"
                st.write(f"{priority_emoji} **{suggestion.get('type', 'Unknown')}**: {suggestion.get('experiment', 'N/A')}")
                st.write(f"   *Rationale*: {suggestion.get('rationale', 'N/A')}")
    else:
        st.warning("⚠️ AI analysis data not found in results. Only basic analysis available.")

def compare_proteins(sequences):
    """Compare multiple proteins"""
    results = {}
    
    for name, sequence in sequences.items():
        results[name] = analyze_quick(sequence)
    
    return results

def display_comparison_results(results):
    """Display protein comparison results"""
    st.markdown("## 📊 Protein Comparison Results")
    
    # Create comparison dataframe
    comparison_data = []
    for name, data in results.items():
        comparison_data.append({
            'Protein': name,
            'Length': data['length'],
            'MW (Da)': data['molecular_weight'],
            'Net Charge': data['net_charge'],
            'Hydrophobicity': data['hydrophobicity'],
            'Cysteines': data['cysteine_count']
        })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(df, x='Protein', y='Length', title='Protein Length Comparison')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(df, x='Net Charge', y='Hydrophobicity', 
                        text='Protein', title='Charge vs Hydrophobicity')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()