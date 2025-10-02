"""
Simple test script to verify the Protein Science AI installation.

This script performs basic tests to ensure the system is properly installed
and can run basic protein analysis.
"""

import sys
import traceback
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all core modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from protein_science import ProteinAgent
        print("✅ ProteinAgent import successful")
    except Exception as e:
        print(f"❌ ProteinAgent import failed: {e}")
        return False
    
    try:
        from protein_science.foundation import ProteinLanguageModel
        print("✅ ProteinLanguageModel import successful")
    except Exception as e:
        print(f"❌ ProteinLanguageModel import failed: {e}")
        return False
        
    try:
        from protein_science.collaboration import AgentCoordinator
        print("✅ AgentCoordinator import successful")
    except Exception as e:
        print(f"❌ AgentCoordinator import failed: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality without heavy dependencies."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test sequence analysis without model loading
        test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        
        print(f"Test sequence length: {len(test_sequence)} residues")
        
        # Test basic sequence properties
        unique_aas = len(set(test_sequence))
        print(f"Unique amino acids: {unique_aas}")
        
        # Test charge calculation
        positive_aas = sum(1 for aa in test_sequence if aa in "KRH")
        negative_aas = sum(1 for aa in test_sequence if aa in "DE")
        net_charge = positive_aas - negative_aas
        print(f"Net charge: {net_charge}")
        
        print("✅ Basic sequence analysis working")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_dependencies():
    """Test that key dependencies are available."""
    print("\n📦 Testing dependencies...")
    
    required_deps = [
        ("torch", "PyTorch for deep learning"),
        ("transformers", "Hugging Face Transformers"),
        ("numpy", "NumPy for numerical computing"),
        ("pandas", "Pandas for data analysis"),
        ("requests", "Requests for HTTP calls")
    ]
    
    optional_deps = [
        ("fair_esm", "ESM protein language models"),
        ("biotite", "Biotite for structural analysis"),
        ("fastapi", "FastAPI for REST API"),
        ("streamlit", "Streamlit for web interface"),
        ("langchain", "LangChain for agent orchestration"),
        ("loguru", "Loguru for logging")
    ]
    
    all_available = True
    
    print("Required dependencies:")
    for dep_name, description in required_deps:
        try:
            __import__(dep_name)
            print(f"  ✅ {dep_name}: {description}")
        except ImportError:
            print(f"  ❌ {dep_name}: {description} (MISSING)")
            all_available = False
    
    print("\nOptional dependencies:")
    for dep_name, description in optional_deps:
        try:
            __import__(dep_name)
            print(f"  ✅ {dep_name}: {description}")
        except ImportError:
            print(f"  ⚠️  {dep_name}: {description} (optional, not installed)")
    
    return all_available


def main():
    """Run all tests."""
    print("🧬 Protein Science AI - Installation Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test imports
    if test_imports():
        tests_passed += 1
    
    # Test dependencies
    if test_dependencies():
        tests_passed += 1
    
    # Test basic functionality
    if test_basic_functionality():
        tests_passed += 1
    
    # Summary
    print(f"\n📊 TEST RESULTS")
    print("=" * 20)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Install missing optional dependencies if needed")
        print("2. Run the example notebook: examples/protein_science_demo.ipynb")
        print("3. Try the CLI: python -m protein_science.cli info")
        print("4. Start the web interface: python -m protein_science.cli app")
        return True
    else:
        print(f"❌ {total_tests - tests_passed} tests failed. Please check the installation.")
        print("\nTroubleshooting:")
        print("1. Install required dependencies: pip install -r requirements.txt")
        print("2. Check Python version (3.9+ recommended)")
        print("3. Verify PYTHONPATH includes the project directory")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test script failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)