# 🧬 Agentic AI System for Protein Science

A comprehensive agentic AI system for protein science research, integrating protein language models (ESM-2), molecular analysis tools, and enhanced sequence analysis capabilities.

## ✨ Key Features

- **🤖 ESM-2 Integration**: State-of-the-art protein language model embeddings
- **🔬 Enhanced Analysis**: Tripeptide & tetrapeptide composition analysis
- **🧪 BioPython Properties**: Comprehensive physicochemical property analysis
- **📊 Interactive Interface**: Streamlit-based web applications
- **🎯 Protein Comparison**: Multi-protein comparative analysis
- **🧠 Autonomous Reasoning**: AI-powered insights and experimental suggestions

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone [repository-url]
cd protein_science

# Install dependencies
pip install -r requirements.txt

# Install BioPython for enhanced features
pip install biopython
```

### Usage

#### 1. **Enhanced Protein Analysis App** (Recommended)
```bash
python -m streamlit run enhanced_protein_app.py --server.port 8502
```
Open: http://localhost:8502

Features:
- Enhanced single protein analysis with tripeptide/tetrapeptide compositions
- ESM-2 embeddings with enhanced features
- Multi-protein comparison with all advanced features

#### 2. **Full Analysis App** (ESM-2 Focus)
```bash
python -m streamlit run full_analysis_app.py
```
Open: http://localhost:8501

Features:
- Focus on ESM-2 embeddings and AI analysis
- Deep learning representations
- AI-powered insights

#### 3. **Command Line Interface**
```bash
# Test installation
python test_installation.py

# Test enhanced features
python test_enhanced_features.py

# CLI analysis
python -m protein_science.cli analyze "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
```

## 📁 Project Structure

```
protein_science/
├── protein_science/           # Core package
│   ├── __init__.py
│   ├── cli.py                # Command line interface
│   ├── agents/               # AI agents
│   │   ├── __init__.py
│   │   └── protein_agent.py  # Main protein AI agent
│   ├── collaboration/        # Multi-agent coordination
│   │   ├── __init__.py
│   │   └── coordinator.py
│   ├── foundation/           # Core models
│   │   ├── __init__.py
│   │   ├── protein_models.py # ESM-2 and PLM integration
│   │   ├── structure_predictor.py
│   │   └── function_predictor.py
│   └── interface/            # User interfaces
│       ├── __init__.py
│       ├── api.py           # REST API
│       └── streamlit_app.py # Web interface
├── data/                     # Data storage
├── examples/                 # Example notebooks and tutorials
├── tests/                    # Unit and integration tests
├── enhanced_protein_app.py   # Enhanced Streamlit app ⭐
├── full_analysis_app.py      # ESM-2 focused app
├── test_installation.py     # Installation verification
├── test_enhanced_features.py # Feature demonstration
├── requirements.txt          # Dependencies
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## 🔬 Analysis Capabilities

### Enhanced Sequence Analysis
- **Amino Acid Composition**: Detailed breakdown and statistics
- **Tripeptide Analysis**: 3-residue pattern analysis and frequencies
- **Tetrapeptide Analysis**: 4-residue pattern analysis and frequencies
- **Peptide Diversity**: Comparison of sequence complexity

### BioPython Physicochemical Properties
- **Isoelectric Point (pI)**: pH at which protein has no net charge
- **Aromaticity**: Proportion of aromatic residues
- **Instability Index**: Protein stability prediction
- **GRAVY Score**: Grand Average of Hydropathy
- **Secondary Structure Fractions**: Helix, turn, and sheet propensities
- **Molecular Weight**: Precise mass calculation

### ESM-2 Deep Learning Features
- **Sequence Embeddings**: 1280-dimensional representations
- **Residue Embeddings**: Per-residue deep features
- **Attention Patterns**: Model attention analysis
- **Similarity Scoring**: Sequence similarity using embeddings

### Multi-Protein Comparison
- **Comparative Analysis**: Side-by-side protein comparison
- **Peptide Diversity Comparison**: Tripeptide/tetrapeptide richness
- **Common Motifs**: Shared sequence patterns
- **Property Clustering**: Group proteins by physicochemical properties

## 🛠️ Development

### Core Dependencies
- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: Transformer models
- **ESM (fair-esm)**: Protein language models
- **BioPython**: Bioinformatics tools
- **Streamlit**: Web interface
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing

### Optional Dependencies
- **LangChain**: Agent orchestration (advanced features)
- **FastAPI**: REST API server
- **Loguru**: Enhanced logging

## 📊 Example Use Cases

1. **Single Protein Analysis**: Analyze any protein sequence for composition, properties, and patterns
2. **Protein Family Comparison**: Compare related proteins to identify conserved and variable regions
3. **Drug Target Analysis**: Analyze potential drug targets for physicochemical properties
4. **Evolutionary Studies**: Compare orthologs across species
5. **Protein Engineering**: Design variants based on property analysis

## 🧪 Testing

```bash
# Verify installation
python test_installation.py

# Test enhanced features
python test_enhanced_features.py

# Run unit tests
python -m pytest tests/
```

## 🤝 Contributing

This project integrates state-of-the-art protein language models with comprehensive sequence analysis tools. Contributions are welcome for:

- Additional physicochemical property calculations
- New visualization methods
- Enhanced comparison algorithms
- Integration with other protein analysis tools

## 📜 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Meta AI Research**: ESM-2 protein language models
- **Hugging Face**: Transformers library
- **BioPython**: Comprehensive bioinformatics toolkit
- **Streamlit**: Interactive web applications

---

**Ready to explore protein sequences with AI? Start with the enhanced app! 🚀**