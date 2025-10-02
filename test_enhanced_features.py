#!/usr/bin/env python3
"""
Test script to demonstrate tripeptide, tetrapeptide, and BioPython functionality
"""

import sys
import os
import pandas as pd
from collections import Counter

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BioPython imports
try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    BIOPYTHON_AVAILABLE = True
    print("✅ BioPython is available")
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("❌ BioPython not available - install with: pip install biopython")

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
        
        return properties
        
    except Exception as e:
        print(f"BioPython analysis error: {e}")
        return {}

def test_peptide_compositions():
    """Test tripeptide and tetrapeptide analysis"""
    print("\n🔬 Testing Peptide Composition Analysis")
    print("=" * 50)
    
    # Test protein sequences
    sequences = {
        "Insulin B chain": "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
        "Lysozyme fragment": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "Hemoglobin α": "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
    }
    
    for name, sequence in sequences.items():
        print(f"\n🧬 Analyzing {name}")
        print(f"Sequence: {sequence[:50]}...")
        print(f"Length: {len(sequence)} residues")
        
        # Tripeptide analysis
        tripeptide_freq, tripeptide_counts = get_peptide_compositions(sequence, 3)
        print(f"\n📊 Tripeptide Analysis:")
        print(f"Total unique tripeptides: {len(tripeptide_freq)}")
        print(f"Total tripeptide positions: {sum(tripeptide_counts.values())}")
        
        # Top 10 tripeptides
        top_tripeptides = sorted(tripeptide_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"Top 10 tripeptides:")
        for tripeptide, freq in top_tripeptides:
            print(f"  {tripeptide}: {freq:.2f}% ({tripeptide_counts[tripeptide]} occurrences)")
        
        # Tetrapeptide analysis
        tetrapeptide_freq, tetrapeptide_counts = get_peptide_compositions(sequence, 4)
        print(f"\n📊 Tetrapeptide Analysis:")
        print(f"Total unique tetrapeptides: {len(tetrapeptide_freq)}")
        print(f"Total tetrapeptide positions: {sum(tetrapeptide_counts.values())}")
        
        # Top 5 tetrapeptides
        top_tetrapeptides = sorted(tetrapeptide_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Top 5 tetrapeptides:")
        for tetrapeptide, freq in top_tetrapeptides:
            print(f"  {tetrapeptide}: {freq:.2f}% ({tetrapeptide_counts[tetrapeptide]} occurrences)")

def test_biopython_properties():
    """Test BioPython physicochemical properties"""
    print("\n🧪 Testing BioPython Physicochemical Properties")
    print("=" * 50)
    
    if not BIOPYTHON_AVAILABLE:
        print("❌ BioPython not available. Please install with: pip install biopython")
        return
    
    # Test sequences
    sequences = {
        "Insulin B chain": "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
        "Lysozyme fragment": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "Basic peptide": "KKKRRRHHHKKK",  # Very basic
        "Acidic peptide": "DDDEEEDDDEEE",  # Very acidic
        "Hydrophobic peptide": "AAAFILVWYAAA"  # Very hydrophobic
    }
    
    results = []
    
    for name, sequence in sequences.items():
        print(f"\n🧬 Analyzing {name}")
        print(f"Sequence: {sequence}")
        
        properties = get_biopython_properties(sequence)
        
        if properties:
            print(f"✅ Properties calculated:")
            print(f"  Molecular Weight: {properties['molecular_weight_bp']:.2f} Da")
            print(f"  Isoelectric Point: {properties['isoelectric_point']:.2f}")
            print(f"  Aromaticity: {properties['aromaticity']:.3f}")
            print(f"  Instability Index: {properties['instability_index']:.2f}")
            print(f"  GRAVY Score: {properties['gravy']:.3f}")
            
            if 'secondary_structure_fraction' in properties:
                ss_frac = properties['secondary_structure_fraction']
                print(f"  Secondary Structure - Helix: {ss_frac[0]:.3f}, Turn: {ss_frac[1]:.3f}, Sheet: {ss_frac[2]:.3f}")
            
            results.append({
                'Protein': name,
                'Length': len(sequence),
                'MW (Da)': properties['molecular_weight_bp'],
                'pI': properties['isoelectric_point'],
                'Aromaticity': properties['aromaticity'],
                'Instability': properties['instability_index'],
                'GRAVY': properties['gravy']
            })
        else:
            print("❌ Properties could not be calculated")
    
    # Create comparison table
    if results:
        print(f"\n📊 Comparison Table:")
        df = pd.DataFrame(results)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df.round(3))

def test_comparison_features():
    """Test comparison features between proteins"""
    print("\n📊 Testing Protein Comparison Features")
    print("=" * 50)
    
    # Two similar proteins for comparison
    sequences = {
        "Human Insulin A": "GIVEQCCTSICSLYQLENYCN",
        "Pig Insulin A": "GIVEQCCTSICSLYQLENYCN",  # Same as human
        "Bovine Insulin A": "GIVEQCCASVCSLYQLENYCN",  # Slightly different
        "Human Insulin B": "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"
    }
    
    print("Analyzing peptide composition similarities...")
    
    # Get tripeptide compositions for all
    all_tripeptides = {}
    all_tetrapeptides = {}
    
    for name, sequence in sequences.items():
        trip_freq, _ = get_peptide_compositions(sequence, 3)
        tetra_freq, _ = get_peptide_compositions(sequence, 4)
        
        all_tripeptides[name] = trip_freq
        all_tetrapeptides[name] = tetra_freq
        
        print(f"\n{name}:")
        print(f"  Unique tripeptides: {len(trip_freq)}")
        print(f"  Unique tetrapeptides: {len(tetra_freq)}")
    
    # Find common tripeptides
    print(f"\n🔍 Common Tripeptide Analysis:")
    all_unique_tripeptides = set()
    for trip_dict in all_tripeptides.values():
        all_unique_tripeptides.update(trip_dict.keys())
    
    common_tripeptides = []
    for tripeptide in all_unique_tripeptides:
        present_in = []
        frequencies = []
        
        for name, trip_dict in all_tripeptides.items():
            if tripeptide in trip_dict:
                present_in.append(name)
                frequencies.append(trip_dict[tripeptide])
        
        if len(present_in) >= 2:  # Present in at least 2 proteins
            common_tripeptides.append({
                'Tripeptide': tripeptide,
                'Count': len(present_in),
                'Proteins': ', '.join(present_in),
                'Avg_Freq': sum(frequencies) / len(frequencies),
                'Max_Freq': max(frequencies)
            })
    
    # Sort by frequency and show top 10
    common_tripeptides.sort(key=lambda x: x['Avg_Freq'], reverse=True)
    
    print(f"Top 10 common tripeptides:")
    for i, trip in enumerate(common_tripeptides[:10]):
        print(f"  {i+1}. {trip['Tripeptide']}: Avg {trip['Avg_Freq']:.2f}% in {trip['Count']} proteins")

def main():
    """Run all tests"""
    print("🧬 Enhanced Protein Analysis - Feature Testing")
    print("=" * 60)
    
    test_peptide_compositions()
    test_biopython_properties() 
    test_comparison_features()
    
    print(f"\n✅ All tests completed!")
    print(f"\n📱 To see these features in action:")
    print(f"   1. Run: python -m streamlit run enhanced_protein_app.py --server.port 8502")
    print(f"   2. Open: http://localhost:8502")
    print(f"   3. Try the 'Enhanced Comparison' mode with multiple proteins!")

if __name__ == "__main__":
    main()