# Protein Science AI - Getting Started Guide

This notebook demonstrates the basic usage of the Protein Science AI system for autonomous protein analysis.

## Prerequisites

First, ensure you have installed the required dependencies:

```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Initialize the Protein Agent

```python
from protein_science import ProteinAgent

# Initialize the main protein agent
agent = ProteinAgent()
print(f"Agent initialized: {agent}")
```

### 2. Analyze a Protein Sequence

```python
import asyncio

# Example protein sequence (p53 tumor suppressor, partial)
p53_sequence = """
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD
""".replace("\n", "").replace(" ", "")

# Run comprehensive analysis
result = await agent.analyze_protein(
    protein_input=p53_sequence,
    analysis_type="comprehensive",
    include_structure=True,
    include_function=True
)

print(f"Analysis completed in {result['analysis_time_seconds']:.2f} seconds")
```

### 3. Explore Analysis Results

```python
# Basic sequence properties
seq_analysis = result['sequence_analysis']
basic_props = seq_analysis['basic_properties']

print("Sequence Properties:")
print(f"- Length: {basic_props['length']} residues")
print(f"- Molecular Weight: ~{basic_props['molecular_weight_approx']:,} Da")
print(f"- Unique residues: {basic_props['unique_residues']}")
print(f"- Most common AA: {basic_props['most_common_residue']}")
```

### 4. Structure Analysis

```python
# Check if structure prediction was successful
if 'structure_analysis' in result and 'error' not in result['structure_analysis']:
    struct_data = result['structure_analysis']
    print("Structure Analysis:")
    
    if 'alphafold_data' in struct_data:
        print("- AlphaFold structure available")
        
        # Confidence scores
        if 'confidence_scores' in struct_data:
            conf = struct_data['confidence_scores']
            print(f"- Overall confidence: {conf.get('overall_confidence', 'N/A')}")
    
    # Basic structural properties
    if 'structure_analysis' in struct_data:
        struct_analysis = struct_data['structure_analysis']
        print(f"- Number of atoms: {struct_analysis.get('num_atoms', 'N/A')}")
        print(f"- Number of residues: {struct_analysis.get('num_residues', 'N/A')}")
else:
    print("Structure analysis not available or failed")
```

### 5. Function Prediction

```python
# Function analysis
if 'function_analysis' in result:
    func_data = result['function_analysis']
    methods_used = func_data.get('methods_used', [])
    print(f"Function prediction used {len(methods_used)} methods: {', '.join(methods_used)}")
    
    # Show integrated prediction
    if 'integrated_prediction' in func_data:
        integrated = func_data['integrated_prediction']
        print(f"Average confidence: {integrated.get('average_confidence', 'N/A')}")
```

### 6. Key Insights and Suggestions

```python
# AI-generated insights
if 'insights' in result:
    insights = result['insights']
    
    print("Key Findings:")
    for finding in insights.get('key_findings', []):
        print(f"- {finding}")
    
    print("\nAreas of Interest:")
    for area in insights.get('areas_of_interest', []):
        print(f"- {area}")

# Experimental suggestions
if 'experiment_suggestions' in result:
    suggestions = result['experiment_suggestions']
    
    print("\nRecommended Experiments:")
    for suggestion in suggestions:
        priority = suggestion.get('priority', 'medium')
        experiment = suggestion.get('experiment', 'Unknown')
        rationale = suggestion.get('rationale', 'No rationale provided')
        
        print(f"- [{priority.upper()}] {experiment}")
        print(f"  Rationale: {rationale}")
```

## Multi-Agent Workflows

### Drug Discovery Workflow

```python
from protein_science.collaboration import AgentCoordinator

# Initialize coordinator
coordinator = AgentCoordinator(main_agent=agent)

# Execute drug discovery workflow
drug_workflow = await coordinator.execute_workflow(
    workflow_type="drug_discovery",
    protein_input={
        "sequence": p53_sequence,
        "uniprot_id": "P04637"  # p53 UniProt ID
    }
)

print(f"Drug discovery workflow status: {drug_workflow['status']}")
print(f"Duration: {drug_workflow.get('duration_seconds', 0):.2f} seconds")
```

### Mutation Analysis Workflow

```python
# Analyze specific mutations
mutations_to_test = ["R175H", "G245S", "R273H"]  # Common p53 mutations

mutation_workflow = await coordinator.execute_workflow(
    workflow_type="mutation_analysis",
    protein_input={
        "sequence": p53_sequence,
        "uniprot_id": "P04637"
    },
    workflow_config={
        "mutations": mutations_to_test
    }
)

print(f"Mutation analysis status: {mutation_workflow['status']}")

# Examine comparative results
if 'results' in mutation_workflow and 'comparative_analysis' in mutation_workflow['results']:
    comparative = mutation_workflow['results']['comparative_analysis']
    print(f"Analyzed {comparative['mutant_count']} mutations")
```

## Reasoning and Question Answering

```python
# Ask specific questions about the analysis
questions = [
    "What can you tell me about the protein's function?",
    "Is this protein likely to be stable?",
    "What experiments would you recommend?",
    "How confident are you in the structure prediction?"
]

for question in questions:
    answer = agent.reason_about_results(result, question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

## Visualization and Export

```python
import json
import pandas as pd

# Export results to JSON
with open('p53_analysis_results.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)

# Create summary dataframe
summary_data = {
    'Property': [
        'Sequence Length',
        'Molecular Weight (Da)',
        'Analysis Time (s)',
        'Structure Available',
        'Function Methods',
        'Key Findings Count'
    ],
    'Value': [
        basic_props['length'],
        basic_props['molecular_weight_approx'],
        result['analysis_time_seconds'],
        'structure_analysis' in result and 'error' not in result['structure_analysis'],
        len(result.get('function_analysis', {}).get('methods_used', [])),
        len(result.get('insights', {}).get('key_findings', []))
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\nAnalysis Summary:")
print(summary_df.to_string(index=False))
```

## Next Steps

1. **Try Different Proteins**: Test with your own protein sequences
2. **Explore Workflows**: Use specialized workflows for different research questions  
3. **Customize Analysis**: Adjust analysis parameters for your specific needs
4. **Integration**: Use the REST API for programmatic access
5. **Visualization**: Use the Streamlit interface for interactive exploration

For more examples, see the `examples/` directory in the repository.