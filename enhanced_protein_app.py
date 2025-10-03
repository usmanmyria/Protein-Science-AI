#!/usr/bin/env python3
"""
Enhanced Protein Analysis App with Tripeptide, Tetrapeptide and BioPython Properties
"""

import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from collections import Counter
import itertools
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt

# BioPython imports for physicochemical properties
try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    from Bio.SeqUtils import molecular_weight
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

# ESM-2 model imports
try:
    import torch
    from transformers import EsmModel, EsmTokenizer
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global model cache
@st.cache_resource
def load_esm2_model():
    """Load ESM-2 model with caching"""
    if not ESM_AVAILABLE:
        raise ImportError("PyTorch and transformers not available")
    
    try:
        model_name = "facebook/esm2_t33_650M_UR50D"
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name)
        model.eval()
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "model_name": model_name
        }
    except Exception as e:
        # Fallback to smaller model
        try:
            model_name = "facebook/esm2_t12_35M_UR50D"
            tokenizer = EsmTokenizer.from_pretrained(model_name)
            model = EsmModel.from_pretrained(model_name)
            model.eval()
            
            return {
                "model": model,
                "tokenizer": tokenizer,
                "model_name": model_name
            }
        except Exception as e2:
            raise Exception(f"Failed to load ESM-2 models: {str(e)}, {str(e2)}")

def generate_esm2_embeddings(sequence):
    """Generate ESM-2 embeddings directly"""
    try:
        esm_data = load_esm2_model()
        model = esm_data["model"]
        tokenizer = esm_data["tokenizer"]
        
        # Tokenize sequence
        tokens = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**tokens)
            last_hidden_states = outputs.last_hidden_state
        
        # Extract sequence and residue embeddings
        sequence_embedding = last_hidden_states.mean(dim=1).squeeze().numpy()
        residue_embeddings = last_hidden_states.squeeze()[1:-1].numpy()  # Remove start/end tokens
        
        return {
            "embeddings": sequence_embedding,
            "residue_embeddings": residue_embeddings,
            "sequence": sequence,
            "model_name": esm_data["model_name"]
        }
        
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None

st.set_page_config(
    page_title="Protein Science AI App",
    page_icon="🧬",
    layout="wide"
)

def main():
    st.title("🧬 Protein Science AI")
    st.markdown("### ESM-2 Analysis with Tripeptide, Tetrapeptide & BioPython Properties")
    st.markdown("---")

    if not BIOPYTHON_AVAILABLE:
        st.warning("⚠️ BioPython not available. Install with: pip install biopython")

    # Sidebar for navigation
    with st.sidebar:
        st.header("🔧 Analysis Options")
        analysis_type = st.selectbox(
            "Choose Analysis Type:",
            ["Protein Analysis", "ESM-2 Embeddings", "Protein Comparison"]
        )
        
        st.markdown("### 📊 System Status")
        
        # Test ESM-2 availability
        system_ready = True
        esm_status = check_esm2_availability()
        
        if esm_status["available"]:
            st.success("✅ ESM-2 Model Ready")
        else:
            st.error(f"❌ ESM-2 Error: {esm_status['error']}")
            system_ready = False
        
        if BIOPYTHON_AVAILABLE:
            st.success("✅ BioPython Ready")
        else:
            st.warning("⚠️ BioPython Not Available")

    if not system_ready:
        st.error("System not ready. Please check installation.")
        st.markdown("### 🔧 Troubleshooting:")
        st.markdown("1. Ensure all dependencies are installed:")
        st.code("pip install torch transformers fair-esm biopython", language="bash")
        st.markdown("2. Check internet connection for model download")
        st.markdown("3. Verify sufficient memory (>4GB recommended)")
        return

def check_esm2_availability():
    """Check if ESM-2 model can be loaded"""
    if not ESM_AVAILABLE:
        return {
            "available": False,
            "error": "PyTorch or transformers not installed",
            "model_name": None
        }
    
    try:
        # Try to load the model (this will use cache if already loaded)
        esm_data = load_esm2_model()
        
        return {
            "available": True,
            "error": None,
            "model_name": esm_data["model_name"]
        }
        
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
            "model_name": None
        }

    if not system_ready:
        st.error("System not ready. Please check installation.")
        return

    # Main content area
    if analysis_type == "Protein Analysis":
        show_enhanced_analysis()
    elif analysis_type == "ESM-2 Embeddings":
        show_esm2_analysis()
    elif analysis_type == "Protein Comparison":
        show_enhanced_comparison()

def get_peptide_compositions(sequence, k=3):
    """Get k-peptide composition (tripeptide, tetrapeptide, etc.)"""
    peptides = []
    for i in range(len(sequence) - k + 1):
        peptides.append(sequence[i:i+k])
    
    # Count occurrences
    peptide_counts = Counter(peptides)
    total_peptides = len(peptides)
    
    # Convert to percentages
    peptide_freq = {}
    for peptide, count in peptide_counts.items():
        peptide_freq[peptide] = (count / total_peptides) * 100 if total_peptides > 0 else 0
    
    return peptide_freq, peptide_counts

def get_biopython_properties(sequence):
    """Get physicochemical properties using BioPython"""
    if not BIOPYTHON_AVAILABLE:
        return {}
    
    try:
        # Remove any invalid characters
        clean_sequence = ''.join([aa for aa in sequence.upper() if aa in 'ACDEFGHIKLMNPQRSTVWY'])
        
        if not clean_sequence:
            return {}
        
        analysis = ProteinAnalysis(clean_sequence)
        
        properties = {
            'molecular_weight_bp': analysis.molecular_weight(),
            'aromaticity': analysis.aromaticity(),
            'instability_index': analysis.instability_index(),
            'isoelectric_point': analysis.isoelectric_point(),
            'gravy': analysis.gravy(),  # Grand average of hydropathy
            'flexibility': analysis.flexibility(),
            'secondary_structure_fraction': analysis.secondary_structure_fraction(),
            'molar_extinction_coefficient': analysis.molar_extinction_coefficient(),
        }
        
        # Add amino acid percentages
        aa_percent = analysis.get_amino_acids_percent()
        properties['aa_percentages'] = aa_percent
        
        return properties
        
    except Exception as e:
        st.warning(f"BioPython analysis error: {e}")
        return {}

def analyze_enhanced(sequence):
    """Enhanced protein analysis with peptide compositions and BioPython properties"""
    try:
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
        
        # Get peptide compositions
        tripeptide_freq, tripeptide_counts = get_peptide_compositions(sequence, 3)
        tetrapeptide_freq, tetrapeptide_counts = get_peptide_compositions(sequence, 4)
        
        # Get BioPython properties
        bp_properties = get_biopython_properties(sequence)
        
        results = {
            'sequence': sequence,
            'length': length,
            'molecular_weight': round(molecular_weight, 2),
            'composition': composition,
            'net_charge': net_charge,
            'hydrophobicity': round(hydrophobicity, 3),
            'cysteine_count': cysteine_count,
            'charged_positive': charged_positive,
            'charged_negative': charged_negative,
            'tripeptide_freq': tripeptide_freq,
            'tripeptide_counts': tripeptide_counts,
            'tetrapeptide_freq': tetrapeptide_freq,
            'tetrapeptide_counts': tetrapeptide_counts,
            'biopython_properties': bp_properties
        }
        
        return results
        
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return None

def show_enhanced_analysis():
    st.header("🔬 Enhanced Protein Analysis")
    st.markdown("Comprehensive analysis with tripeptide, tetrapeptide compositions and BioPython properties")
    
    sequence = st.text_area(
        "Enter protein sequence:",
        value="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        height=100,
        help="Paste your protein sequence here (single letter amino acid codes)"
    )
    
    if st.button("🚀 Run Enhanced Analysis", type="primary"):
        if sequence:
            with st.spinner("Running enhanced analysis..."):
                results = analyze_enhanced(sequence)
                display_enhanced_results(results)

def show_esm2_analysis():
    st.header("🤖 ESM-2 Deep Analysis")
    st.markdown("Deep analysis using ESM-2 protein language model with enhanced features")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        sequence = st.text_area(
            "Enter protein sequence:",
            value="MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
            height=150,
            help="Enter your protein sequence for ESM-2 analysis"
        )
    
    with col2:
        st.markdown("### 🎯 Enhanced Features")
        st.markdown("""
        🧠 **ESM-2 Embeddings** (1280-dim)  
        🔬 **Tripeptide Analysis**  
        🧪 **Tetrapeptide Analysis**  
        ⚗️ **BioPython Properties**  
        📊 **Physicochemical Analysis**  
        🎯 **Sequence Representations**
        """)
    
    if st.button("🚀 Run ESM-2 Enhanced Analysis", type="primary"):
        if sequence:
            with st.spinner("Running ESM-2 enhanced analysis... This may take 30-60 seconds"):
                results = run_esm2_enhanced_analysis(sequence)
                display_esm2_enhanced_results(results)

def show_enhanced_comparison():
    st.header("📊 Enhanced Protein Comparison")
    st.markdown("Compare multiple proteins with tripeptide, tetrapeptide compositions and BioPython properties")
    
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
    
    if st.button("📊 Enhanced Comparison", type="primary"):
        if len(sequences) >= 2:
            with st.spinner("Running enhanced comparison..."):
                results = compare_proteins_enhanced(sequences)
                display_enhanced_comparison_results(results)

def run_esm2_enhanced_analysis(sequence):
    """Run enhanced ESM-2 analysis with all features"""
    try:
        # Generate ESM-2 embeddings directly
        st.info("🤖 Loading ESM-2 model...")
        
        st.info("🧠 Generating embeddings...")
        embedding_results = generate_esm2_embeddings(sequence)
        
        if embedding_results is None:
            st.error("Failed to generate ESM-2 embeddings")
            return None
        
        st.info("🔬 Computing enhanced features...")
        # Get enhanced analysis
        enhanced_results = analyze_enhanced(sequence)
        
        # Combine results
        results = {
            'sequence': sequence,
            'length': len(sequence),
            'embeddings': embedding_results['embeddings'],  # Main embeddings for display
            'residue_embeddings': embedding_results['residue_embeddings'],  # 2D array
            'model_name': embedding_results['model_name'],
            'timestamp': datetime.now().isoformat()
        }
        
        if enhanced_results:
            results.update(enhanced_results)
        
        st.success("✅ Enhanced ESM-2 analysis completed!")
        return results
        
    except Exception as e:
        st.error(f"ESM-2 analysis error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return analyze_enhanced(sequence)  # Fallback

def compare_proteins_enhanced(sequences):
    """Compare proteins with enhanced features"""
    try:
        st.info("🤖 Loading ESM-2 model for enhanced comparison...")
        
        results = {}
        
        # Get embeddings for all sequences
        all_sequences = list(sequences.values())
        all_names = list(sequences.keys())
        
        st.info("🧠 Computing embeddings for all proteins...")
        
        # Generate embeddings for each sequence
        all_embeddings = []
        for i, (name, sequence) in enumerate(sequences.items()):
            st.info(f"Processing {name} ({i+1}/{len(sequences)})...")
            
            embedding_results = generate_esm2_embeddings(sequence)
            if embedding_results is None:
                st.error(f"Failed to generate embeddings for {name}")
                continue
                
            enhanced = analyze_enhanced(sequence)
            results[name] = {
                **enhanced,
                'embeddings': embedding_results['embeddings'],
                'residue_embeddings': embedding_results['residue_embeddings'],
                'embedding_dim': len(embedding_results['embeddings'])
            }
            all_embeddings.append(embedding_results['embeddings'])
        
        # Compute similarities using cosine similarity
        similarities = {}
        embeddings_array = np.array(all_embeddings)
        
        for i, name1 in enumerate(all_names):
            for j, name2 in enumerate(all_names):
                if i < j and i < len(embeddings_array) and j < len(embeddings_array):  # Only compute upper triangle
                    # Use cosine similarity
                    similarity = cosine_similarity([embeddings_array[i]], [embeddings_array[j]])[0][0]
                    similarities[f"{name1} vs {name2}"] = similarity
        
        results['similarities'] = similarities
        
        st.success("✅ Enhanced protein comparison completed!")
        return results
        
    except Exception as e:
        st.error(f"Comparison error: {e}")
        # Fallback to basic comparison
        results = {}
        for name, sequence in sequences.items():
            results[name] = analyze_enhanced(sequence)
        return results

def display_enhanced_results(results):
    """Display enhanced analysis results"""
    if not results:
        return
    
    st.markdown("## 📊 Enhanced Analysis Results")
    
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
    
    # BioPython properties if available
    bp_props = results.get('biopython_properties', {})
    if bp_props:
        st.markdown("### 🧪 BioPython Physicochemical Properties")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if 'isoelectric_point' in bp_props:
                st.metric("Isoelectric Point", f"{bp_props['isoelectric_point']:.2f}")
        with col2:
            if 'aromaticity' in bp_props:
                st.metric("Aromaticity", f"{bp_props['aromaticity']:.3f}")
        with col3:
            if 'instability_index' in bp_props:
                st.metric("Instability Index", f"{bp_props['instability_index']:.2f}")
        with col4:
            if 'gravy' in bp_props:
                st.metric("GRAVY Score", f"{bp_props['gravy']:.3f}")
        
        # Secondary structure fractions
        if 'secondary_structure_fraction' in bp_props:
            ss_frac = bp_props['secondary_structure_fraction']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Helix Fraction", f"{ss_frac[0]:.3f}")
            with col2:
                st.metric("Turn Fraction", f"{ss_frac[1]:.3f}")
            with col3:
                st.metric("Sheet Fraction", f"{ss_frac[2]:.3f}")
    
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
    
    # Peptide composition analysis
    st.markdown("### 🔬 Peptide Composition Analysis")
    
    # Tripeptide analysis
    if 'tripeptide_freq' in results and results['tripeptide_freq']:
        st.markdown("#### Tripeptide Composition")
        
        # Show top tripeptides
        top_tripeptides = sorted(results['tripeptide_freq'].items(), key=lambda x: x[1], reverse=True)[:20]
        tripeptide_df = pd.DataFrame(top_tripeptides, columns=['Tripeptide', 'Frequency (%)'])
        tripeptide_df['Frequency (%)'] = tripeptide_df['Frequency (%)'].round(3)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(tripeptide_df)
        with col2:
            fig = px.bar(tripeptide_df.head(10), x='Tripeptide', y='Frequency (%)', 
                        title='Top 10 Tripeptides by Frequency')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tetrapeptide analysis
    if 'tetrapeptide_freq' in results and results['tetrapeptide_freq']:
        st.markdown("#### Tetrapeptide Composition")
        
        # Show top tetrapeptides
        top_tetrapeptides = sorted(results['tetrapeptide_freq'].items(), key=lambda x: x[1], reverse=True)[:15]
        tetrapeptide_df = pd.DataFrame(top_tetrapeptides, columns=['Tetrapeptide', 'Frequency (%)'])
        tetrapeptide_df['Frequency (%)'] = tetrapeptide_df['Frequency (%)'].round(3)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(tetrapeptide_df)
        with col2:
            fig = px.bar(tetrapeptide_df.head(8), x='Tetrapeptide', y='Frequency (%)', 
                        title='Top 8 Tetrapeptides by Frequency')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

def analyze_embedding_details(embeddings, residue_embeddings, sequence):
    """Perform detailed analysis of ESM-2 embeddings"""
    analysis = {}
    
    # Convert to numpy arrays
    seq_emb = np.array(embeddings)
    res_emb = np.array(residue_embeddings)
    
    # Basic statistics
    analysis['basic_stats'] = {
        'seq_embedding_shape': seq_emb.shape,
        'residue_embedding_shape': res_emb.shape,
        'seq_mean': float(seq_emb.mean()),
        'seq_std': float(seq_emb.std()),
        'seq_min': float(seq_emb.min()),
        'seq_max': float(seq_emb.max()),
        'seq_sparsity': float((seq_emb == 0).sum() / seq_emb.size),
        'res_mean': float(res_emb.mean()),
        'res_std': float(res_emb.std()),
        'res_sparsity': float((res_emb == 0).sum() / res_emb.size)
    }
    
    # Distribution analysis
    analysis['distribution'] = {
        'seq_skewness': float(stats.skew(seq_emb)),
        'seq_kurtosis': float(stats.kurtosis(seq_emb)),
        'seq_normality_p': float(stats.normaltest(seq_emb)[1]),
        'res_skewness': float(stats.skew(res_emb.flatten())),
        'res_kurtosis': float(stats.kurtosis(res_emb.flatten())),
    }
    
    # Activation patterns
    analysis['activation_patterns'] = {
        'high_activation_dims': np.where(seq_emb > seq_emb.mean() + 2*seq_emb.std())[0].tolist(),
        'low_activation_dims': np.where(seq_emb < seq_emb.mean() - 2*seq_emb.std())[0].tolist(),
        'dominant_dims': np.argsort(np.abs(seq_emb))[-20:].tolist(),
    }
    
    # Residue-level analysis
    if len(res_emb.shape) == 2 and res_emb.shape[0] > 1:
        # Per-residue statistics
        residue_norms = np.linalg.norm(res_emb, axis=1)
        analysis['residue_analysis'] = {
            'residue_norms': residue_norms.tolist(),
            'most_informative_residues': np.argsort(residue_norms)[-10:].tolist(),
            'least_informative_residues': np.argsort(residue_norms)[:10].tolist(),
            'avg_residue_norm': float(residue_norms.mean()),
            'residue_norm_std': float(residue_norms.std())
        }
        
        # Similarity between adjacent residues
        adjacent_similarities = []
        for i in range(len(res_emb) - 1):
            sim = cosine_similarity([res_emb[i]], [res_emb[i+1]])[0][0]
            adjacent_similarities.append(sim)
        
        analysis['residue_analysis']['adjacent_similarities'] = adjacent_similarities
        analysis['residue_analysis']['avg_adjacent_similarity'] = float(np.mean(adjacent_similarities))
    
    # Dimensionality analysis using PCA
    try:
        if len(res_emb.shape) == 2 and res_emb.shape[0] > 5:
            pca = PCA()
            pca.fit(res_emb)
            
            # Find number of dimensions needed for 95% variance
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            dims_95 = np.argmax(cumsum_variance >= 0.95) + 1
            dims_99 = np.argmax(cumsum_variance >= 0.99) + 1
            
            analysis['dimensionality'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_[:50].tolist(),  # Top 50
                'dims_for_95_variance': int(dims_95),
                'dims_for_99_variance': int(dims_99),
                'effective_rank': float(np.sum(pca.explained_variance_ratio_)**2 / np.sum(pca.explained_variance_ratio_**2))
            }
    except Exception as e:
        analysis['dimensionality'] = {'error': str(e)}
    
    # Clustering analysis
    try:
        if len(res_emb.shape) == 2 and res_emb.shape[0] > 10:
            # K-means clustering
            n_clusters = min(8, res_emb.shape[0] // 3)
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(res_emb)
                
                analysis['clustering'] = {
                    'n_clusters': n_clusters,
                    'cluster_labels': cluster_labels.tolist(),
                    'cluster_centers': kmeans.cluster_centers_.tolist(),
                    'inertia': float(kmeans.inertia_)
                }
                
                # Analyze clusters
                cluster_sizes = Counter(cluster_labels)
                analysis['clustering']['cluster_sizes'] = dict(cluster_sizes)
    except Exception as e:
        analysis['clustering'] = {'error': str(e)}
    
    return analysis

def display_detailed_embedding_analysis(embeddings, residue_embeddings, sequence):
    """Display comprehensive embedding analysis"""
    st.markdown("### 🔬 Detailed Embedding Analysis")
    
    # Get detailed analysis
    with st.spinner("Performing detailed embedding analysis..."):
        analysis = analyze_embedding_details(embeddings, residue_embeddings, sequence)
    
    # Basic Statistics
    st.markdown("#### 📊 Basic Statistics")
    basic_stats = analysis['basic_stats']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Activation", f"{basic_stats['seq_mean']:.4f}")
    with col2:
        st.metric("Std Deviation", f"{basic_stats['seq_std']:.4f}")
    with col3:
        st.metric("Value Range", f"{basic_stats['seq_max']-basic_stats['seq_min']:.3f}")
    with col4:
        st.metric("Sparsity %", f"{basic_stats['seq_sparsity']*100:.2f}%")
    
    # Distribution Analysis
    st.markdown("#### 📈 Distribution Analysis")
    dist_stats = analysis['distribution']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Skewness", f"{dist_stats['seq_skewness']:.3f}")
        if abs(dist_stats['seq_skewness']) < 0.5:
            st.success("Nearly symmetric distribution")
        else:
            st.warning("Skewed distribution")
    
    with col2:
        st.metric("Kurtosis", f"{dist_stats['seq_kurtosis']:.3f}")
        if abs(dist_stats['seq_kurtosis']) < 0.5:
            st.success("Normal-like tails")
        else:
            st.info("Heavy/light tails")
    
    with col3:
        st.metric("Normality p-value", f"{dist_stats['seq_normality_p']:.4f}")
        if dist_stats['seq_normality_p'] > 0.05:
            st.success("Normally distributed")
        else:
            st.info("Non-normal distribution")
    
    # Embedding Distribution Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Sequence Embedding Distribution")
        fig = px.histogram(x=embeddings, nbins=50, title="Embedding Values Distribution")
        fig.update_layout(xaxis_title="Activation Value", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Embedding Dimension Analysis")
        dim_importance = np.abs(embeddings)
        top_dims = np.argsort(dim_importance)[-20:]
        
        fig = px.bar(x=top_dims, y=dim_importance[top_dims], 
                    title="Top 20 Most Important Dimensions")
        fig.update_layout(xaxis_title="Dimension Index", yaxis_title="|Activation|")
        st.plotly_chart(fig, use_container_width=True)
    
    # Activation Patterns
    st.markdown("#### 🎯 Activation Patterns")
    patterns = analysis['activation_patterns']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("High Activation Dims", len(patterns['high_activation_dims']))
        if patterns['high_activation_dims']:
            st.write(f"Dimensions: {patterns['high_activation_dims'][:10]}")
    
    with col2:
        st.metric("Low Activation Dims", len(patterns['low_activation_dims']))
        if patterns['low_activation_dims']:
            st.write(f"Dimensions: {patterns['low_activation_dims'][:10]}")
    
    with col3:
        st.metric("Dominant Dims", len(patterns['dominant_dims']))
        st.write(f"Top dims: {patterns['dominant_dims'][-5:]}")
    
    # Residue-level Analysis
    if 'residue_analysis' in analysis:
        st.markdown("#### 🧬 Residue-Level Analysis")
        res_analysis = analysis['residue_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Residue Importance")
            residue_norms = res_analysis['residue_norms']
            
            # Create residue importance plot
            residue_data = pd.DataFrame({
                'Position': range(1, len(residue_norms) + 1),
                'Amino Acid': list(sequence),
                'Importance': residue_norms
            })
            
            fig = px.line(residue_data, x='Position', y='Importance', 
                         title="Residue Importance (L2 Norm)")
            fig.update_traces(mode='lines+markers')
            fig.update_layout(xaxis_title="Residue Position", yaxis_title="Embedding Norm")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### Adjacent Residue Similarity")
            adj_sims = res_analysis['adjacent_similarities']
            
            fig = px.line(x=range(1, len(adj_sims) + 1), y=adj_sims,
                         title="Adjacent Residue Similarity")
            fig.update_layout(xaxis_title="Position", yaxis_title="Cosine Similarity")
            st.plotly_chart(fig, use_container_width=True)
        
        # Most/Least informative residues
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Most Informative Residues")
            most_info = res_analysis['most_informative_residues']
            most_info_data = []
            for pos in most_info:
                if pos < len(sequence):
                    most_info_data.append({
                        'Position': pos + 1,
                        'Amino Acid': sequence[pos],
                        'Importance': residue_norms[pos]
                    })
            
            if most_info_data:
                df_most = pd.DataFrame(most_info_data)
                st.dataframe(df_most)
        
        with col2:
            st.markdown("##### Least Informative Residues")
            least_info = res_analysis['least_informative_residues']
            least_info_data = []
            for pos in least_info:
                if pos < len(sequence):
                    least_info_data.append({
                        'Position': pos + 1,
                        'Amino Acid': sequence[pos],
                        'Importance': residue_norms[pos]
                    })
            
            if least_info_data:
                df_least = pd.DataFrame(least_info_data)
                st.dataframe(df_least)
    
    # Dimensionality Analysis
    if 'dimensionality' in analysis and 'error' not in analysis['dimensionality']:
        st.markdown("#### 📏 Dimensionality Analysis (PCA)")
        dim_analysis = analysis['dimensionality']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dims for 95% Variance", dim_analysis['dims_for_95_variance'])
        with col2:
            st.metric("Dims for 99% Variance", dim_analysis['dims_for_99_variance'])
        with col3:
            st.metric("Effective Rank", f"{dim_analysis['effective_rank']:.2f}")
        
        # Plot explained variance
        if 'explained_variance_ratio' in dim_analysis:
            variance_ratio = dim_analysis['explained_variance_ratio']
            cumsum_variance = np.cumsum(variance_ratio)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, len(variance_ratio) + 1)), 
                                   y=variance_ratio,
                                   mode='lines+markers',
                                   name='Individual Variance',
                                   yaxis='y'))
            fig.add_trace(go.Scatter(x=list(range(1, len(cumsum_variance) + 1)), 
                                   y=cumsum_variance,
                                   mode='lines+markers',
                                   name='Cumulative Variance',
                                   yaxis='y2'))
            
            fig.update_layout(
                title='PCA Explained Variance Analysis',
                xaxis_title='Principal Component',
                yaxis=dict(title='Individual Variance Ratio', side='left'),
                yaxis2=dict(title='Cumulative Variance Ratio', side='right', overlaying='y'),
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Clustering Analysis
    if 'clustering' in analysis and 'error' not in analysis['clustering']:
        st.markdown("#### 🎭 Clustering Analysis")
        clustering = analysis['clustering']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Number of Clusters", clustering['n_clusters'])
            st.metric("Cluster Inertia", f"{clustering['inertia']:.2f}")
        
        with col2:
            # Show cluster sizes
            cluster_sizes = clustering['cluster_sizes']
            cluster_df = pd.DataFrame([
                {'Cluster': k, 'Size': v} for k, v in cluster_sizes.items()
            ])
            
            fig = px.bar(cluster_df, x='Cluster', y='Size', 
                        title='Cluster Sizes')
            st.plotly_chart(fig, use_container_width=True)
        
        # Visualize clusters on sequence
        if 'cluster_labels' in clustering:
            cluster_labels = clustering['cluster_labels']
            
            # Create sequence with cluster colors
            cluster_sequence_data = pd.DataFrame({
                'Position': range(1, len(sequence) + 1),
                'Amino Acid': list(sequence),
                'Cluster': cluster_labels
            })
            
            fig = px.scatter(cluster_sequence_data, x='Position', y='Cluster',
                           color='Cluster', hover_data=['Amino Acid'],
                           title='Sequence Clustering')
            fig.update_layout(xaxis_title="Residue Position", yaxis_title="Cluster ID")
            st.plotly_chart(fig, use_container_width=True)
    
    # Embedding Quality Assessment
    st.markdown("#### ✅ Embedding Quality Assessment")
    
    # Calculate quality metrics
    quality_score = 0
    quality_comments = []
    
    # Check distribution normality
    if dist_stats['seq_normality_p'] > 0.01:
        quality_score += 20
        quality_comments.append("✅ Well-distributed activations")
    else:
        quality_comments.append("⚠️ Non-normal activation distribution")
    
    # Check sparsity
    if basic_stats['seq_sparsity'] < 0.1:
        quality_score += 20
        quality_comments.append("✅ Good activation density")
    else:
        quality_comments.append("⚠️ High sparsity in embeddings")
    
    # Check variance coverage
    if 'dimensionality' in analysis and 'dims_for_95_variance' in analysis['dimensionality']:
        if analysis['dimensionality']['dims_for_95_variance'] < embeddings.shape[0] * 0.8:
            quality_score += 20
            quality_comments.append("✅ Good dimensional efficiency")
        else:
            quality_comments.append("⚠️ High dimensional requirements")
    
    # Check residue differentiation
    if 'residue_analysis' in analysis:
        norm_cv = analysis['residue_analysis']['residue_norm_std'] / analysis['residue_analysis']['avg_residue_norm']
        if norm_cv > 0.1:
            quality_score += 20
            quality_comments.append("✅ Good residue differentiation")
        else:
            quality_comments.append("⚠️ Low residue differentiation")
    
    # Display quality assessment
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Quality Score", f"{quality_score}/80")
        if quality_score >= 60:
            st.success("High quality embeddings")
        elif quality_score >= 40:
            st.warning("Moderate quality embeddings")
        else:
            st.error("Low quality embeddings")
    
    with col2:
        st.markdown("**Quality Assessment:**")
        for comment in quality_comments:
            st.write(comment)

def display_enhanced_comparison_results(results):
    """Display enhanced protein comparison results"""
    st.markdown("## 📊 Enhanced Protein Comparison Results")
    
    # Enhanced comparison with all features
    comparison_data = []
    embeddings_data = {}
    peptide_data = {}
    
    for name, data in results.items():
        if name == 'similarities':
            continue
        
        # Basic properties
        row = {
            'Protein': name,
            'Length': data['length'],
            'MW (Da)': data['molecular_weight'],
            'Net Charge': data['net_charge'],
            'Hydrophobicity': data['hydrophobicity'],
            'Embedding Dim': data.get('embedding_dim', 'N/A')
        }
        
        # Add BioPython properties if available
        bp_props = data.get('biopython_properties', {})
        if bp_props:
            row.update({
                'Isoelectric Point': round(bp_props.get('isoelectric_point', 0), 2) if bp_props.get('isoelectric_point') else 'N/A',
                'Aromaticity': round(bp_props.get('aromaticity', 0), 3) if bp_props.get('aromaticity') else 'N/A',
                'Instability Index': round(bp_props.get('instability_index', 0), 2) if bp_props.get('instability_index') else 'N/A',
                'GRAVY Score': round(bp_props.get('gravy', 0), 3) if bp_props.get('gravy') else 'N/A'
            })
        
        comparison_data.append(row)
        
        if 'embedding' in data:
            embeddings_data[name] = data['embedding']
        
        # Store peptide data for comparison
        if 'tripeptide_freq' in data:
            peptide_data[name] = {
                'tripeptides': data['tripeptide_freq'],
                'tetrapeptides': data.get('tetrapeptide_freq', {})
            }
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df)
    
    # BioPython properties comparison
    if any('Isoelectric Point' in row for row in comparison_data):
        st.markdown("### 🧪 BioPython Properties Comparison")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Filter out N/A values for plotting
            plot_df = df[df['Isoelectric Point'] != 'N/A'].copy()
            if not plot_df.empty:
                plot_df['Isoelectric Point'] = pd.to_numeric(plot_df['Isoelectric Point'])
                fig = px.bar(plot_df, x='Protein', y='Isoelectric Point', 
                           title='Isoelectric Point Comparison')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            plot_df = df[df['GRAVY Score'] != 'N/A'].copy()
            if not plot_df.empty:
                plot_df['GRAVY Score'] = pd.to_numeric(plot_df['GRAVY Score'])
                fig = px.bar(plot_df, x='Protein', y='GRAVY Score', 
                           title='GRAVY Score Comparison')
                st.plotly_chart(fig, use_container_width=True)
    
    # Peptide composition comparison
    if peptide_data:
        st.markdown("### 🔬 Peptide Composition Comparison")
        
        # Compare tripeptide diversity
        tripeptide_diversity = {}
        tetrapeptide_diversity = {}
        
        for name, data in peptide_data.items():
            tripeptide_diversity[name] = len(data['tripeptides'])
            tetrapeptide_diversity[name] = len(data['tetrapeptides'])
        
        diversity_df = pd.DataFrame({
            'Protein': list(tripeptide_diversity.keys()),
            'Tripeptide Diversity': list(tripeptide_diversity.values()),
            'Tetrapeptide Diversity': list(tetrapeptide_diversity.values())
        })
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(diversity_df, x='Protein', y='Tripeptide Diversity', 
                        title='Tripeptide Diversity Comparison')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(diversity_df, x='Protein', y='Tetrapeptide Diversity', 
                        title='Tetrapeptide Diversity Comparison')
            st.plotly_chart(fig, use_container_width=True)
        
        # Show common tripeptides
        if len(peptide_data) >= 2:
            st.markdown("#### Common Tripeptides Analysis")
            
            all_tripeptides = set()
            for data in peptide_data.values():
                all_tripeptides.update(data['tripeptides'].keys())
            
            common_tripeptides = []
            for tripeptide in all_tripeptides:
                frequencies = []
                proteins = []
                for name, data in peptide_data.items():
                    if tripeptide in data['tripeptides']:
                        frequencies.append(data['tripeptides'][tripeptide])
                        proteins.append(name)
                
                if len(frequencies) >= 2:  # Present in at least 2 proteins
                    common_tripeptides.append({
                        'Tripeptide': tripeptide,
                        'Proteins': ', '.join(proteins),
                        'Avg Frequency': np.mean(frequencies),
                        'Std Frequency': np.std(frequencies)
                    })
            
            if common_tripeptides:
                common_df = pd.DataFrame(common_tripeptides)
                common_df = common_df.sort_values('Avg Frequency', ascending=False).head(10)
                st.dataframe(common_df)
    
    # Similarity matrix if available
    if 'similarities' in results:
        st.markdown("### 🎯 ESM-2 Similarity Matrix")
        similarities = results['similarities']
        
        sim_data = []
        for pair, similarity in similarities.items():
            prot1, prot2 = pair.split(' vs ')
            sim_data.append({
                'Protein 1': prot1,
                'Protein 2': prot2, 
                'ESM-2 Similarity': f"{similarity:.4f}"
            })
        
        sim_df = pd.DataFrame(sim_data)
        st.dataframe(sim_df)

def display_esm2_enhanced_results(results):
    """Display ESM-2 enhanced analysis results with detailed embedding analysis"""
    if not results:
        st.error("No results to display")
        return
    
    st.markdown("## 🤖 ESM-2 Enhanced Analysis Results")
    
    # First show enhanced results
    display_enhanced_results(results)
    
    # ESM-2 specific results
    if 'embeddings' in results and results['embeddings'] is not None:
        st.markdown("## 🧠 ESM-2 Embeddings Analysis")
        
        embeddings = np.array(results['embeddings'])
        residue_embeddings = np.array(results['residue_embeddings'])
        sequence = results['sequence']
        
        st.success("✅ ESM-2 embeddings generated successfully!")
        
        # Basic Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Embedding Dimension", embeddings.shape[0])
        with col2:
            st.metric("Sequence Length", residue_embeddings.shape[0])
        with col3:
            st.metric("Mean Activation", f"{embeddings.mean():.4f}")
        with col4:
            st.metric("Std Activation", f"{embeddings.std():.4f}")
        
        # Basic Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Sequence-Level Embedding")
            fig = px.line(y=embeddings, title="ESM-2 Sequence Embedding (1280-dim)")
            fig.update_layout(xaxis_title="Dimension", yaxis_title="Activation")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Residue-Level Embeddings Heatmap")
            # Show heatmap of residue embeddings (subset for visualization)
            max_pos = min(50, residue_embeddings.shape[0])
            max_dim = min(100, residue_embeddings.shape[1])
            viz_embeddings = residue_embeddings[:max_pos, :max_dim]
            
            fig = px.imshow(viz_embeddings, 
                           title=f"Residue Embeddings Heatmap ({max_pos}x{max_dim})",
                           labels=dict(x="Embedding Dimension", y="Residue Position", color="Activation"),
                           aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        
        # Advanced Embedding Visualizations
        st.markdown("### 🎨 Advanced Embedding Visualizations")
        
        # Only show if we have enough residues
        if len(residue_embeddings) > 10:
            tab1, tab2, tab3 = st.tabs(["PCA Projection", "t-SNE Projection", "Similarity Matrix"])
            
            with tab1:
                st.markdown("#### PCA Projection of Residue Embeddings")
                try:
                    # Perform PCA
                    pca = PCA(n_components=min(3, residue_embeddings.shape[0]))
                    pca_result = pca.fit_transform(residue_embeddings)
                    
                    # Create PCA visualization
                    pca_df = pd.DataFrame({
                        'PC1': pca_result[:, 0],
                        'PC2': pca_result[:, 1] if pca_result.shape[1] > 1 else np.zeros(len(pca_result)),
                        'Position': range(1, len(sequence) + 1),
                        'Amino_Acid': list(sequence),
                        'Residue_Label': [f"{aa}{i+1}" for i, aa in enumerate(sequence)]
                    })
                    
                    if pca_result.shape[1] >= 3:
                        pca_df['PC3'] = pca_result[:, 2]
                        fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3',
                                           color='Position', hover_data=['Residue_Label'],
                                           title=f"3D PCA of Residue Embeddings")
                    else:
                        fig = px.scatter(pca_df, x='PC1', y='PC2',
                                       color='Position', hover_data=['Residue_Label'],
                                       title="2D PCA of Residue Embeddings")
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show explained variance
                    explained_var = pca.explained_variance_ratio_
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("PC1 Variance", f"{explained_var[0]*100:.1f}%")
                    with col2:
                        if len(explained_var) > 1:
                            st.metric("PC2 Variance", f"{explained_var[1]*100:.1f}%")
                    with col3:
                        if len(explained_var) > 2:
                            st.metric("PC3 Variance", f"{explained_var[2]*100:.1f}%")
                
                except Exception as e:
                    st.error(f"PCA visualization error: {e}")
            
            with tab2:
                st.markdown("#### t-SNE Projection of Residue Embeddings")
                try:
                    if len(residue_embeddings) >= 20:  # t-SNE needs more points
                        # Perform t-SNE
                        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(residue_embeddings)//3))
                        tsne_result = tsne.fit_transform(residue_embeddings)
                        
                        # Create t-SNE visualization
                        tsne_df = pd.DataFrame({
                            'TSNE1': tsne_result[:, 0],
                            'TSNE2': tsne_result[:, 1],
                            'Position': range(1, len(sequence) + 1),
                            'Amino_Acid': list(sequence),
                            'Residue_Label': [f"{aa}{i+1}" for i, aa in enumerate(sequence)]
                        })
                        
                        fig = px.scatter(tsne_df, x='TSNE1', y='TSNE2',
                                       color='Position', hover_data=['Residue_Label'],
                                       title="t-SNE of Residue Embeddings")
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("t-SNE requires at least 20 residues for meaningful visualization")
                
                except Exception as e:
                    st.error(f"t-SNE visualization error: {e}")
            
            with tab3:
                st.markdown("#### Residue Similarity Matrix")
                try:
                    # Compute cosine similarity matrix
                    similarity_matrix = cosine_similarity(residue_embeddings)
                    
                    # Create similarity heatmap
                    fig = px.imshow(similarity_matrix,
                                   title="Residue-Residue Similarity Matrix",
                                   labels=dict(x="Residue Position", y="Residue Position", color="Cosine Similarity"),
                                   aspect="auto")
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show most similar residue pairs
                    st.markdown("##### Most Similar Residue Pairs")
                    
                    # Get upper triangle indices
                    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
                    similar_pairs = []
                    
                    for i in range(len(similarity_matrix)):
                        for j in range(i+1, len(similarity_matrix)):
                            if mask[i, j]:
                                similar_pairs.append({
                                    'Residue 1': f"{sequence[i]}{i+1}",
                                    'Residue 2': f"{sequence[j]}{j+1}",
                                    'Similarity': similarity_matrix[i, j],
                                    'Distance': abs(i - j)
                                })
                    
                    # Sort by similarity and show top 10
                    similar_pairs.sort(key=lambda x: x['Similarity'], reverse=True)
                    top_pairs_df = pd.DataFrame(similar_pairs[:10])
                    st.dataframe(top_pairs_df)
                
                except Exception as e:
                    st.error(f"Similarity matrix error: {e}")
        
        else:
            st.info("Advanced visualizations require at least 10 residues")

        # Detailed Embedding Analysis
        display_detailed_embedding_analysis(embeddings, residue_embeddings, sequence)

if __name__ == "__main__":
    main()