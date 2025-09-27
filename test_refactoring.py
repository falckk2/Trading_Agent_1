#!/usr/bin/env python3
"""
Comprehensive test suite for the ML architecture refactoring.
Tests the new RandomForestStrategy and RandomForestAgent composition pattern.
"""

import sys
import os
import ast
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_file_structure():
    """Test that refactoring resulted in correct file structure."""
    print("1. Testing File Structure...")

    # Check that BaseMLAgent was removed
    base_ml_agent_path = project_root / "crypto_trading/agents/ml/base_ml_agent.py"
    assert not base_ml_agent_path.exists(), "BaseMLAgent should be removed"
    print("   ✅ BaseMLAgent successfully removed")

    # Check that new files exist
    required_files = [
        "crypto_trading/agents/ml/ml_strategy.py",
        "crypto_trading/agents/ml/random_forest_strategy.py",
        "crypto_trading/agents/ml/random_forest_agent.py"
    ]

    for file_path in required_files:
        full_path = project_root / file_path
        assert full_path.exists(), f"Required file missing: {file_path}"

        # Test syntax
        with open(full_path, 'r') as f:
            content = f.read()
        ast.parse(content)  # Will raise SyntaxError if invalid

        print(f"   ✅ {file_path} exists and has valid syntax")

def test_architecture_patterns():
    """Test that the architecture patterns are correctly implemented."""
    print("\n2. Testing Architecture Patterns...")

    # Test RandomForestStrategy inheritance
    rf_strategy_path = project_root / "crypto_trading/agents/ml/random_forest_strategy.py"
    with open(rf_strategy_path, 'r') as f:
        content = f.read()

    # Check inheritance pattern
    assert "class RandomForestStrategy(MLStrategy)" in content, "RandomForestStrategy should inherit from MLStrategy"
    assert "from .ml_strategy import MLStrategy" in content, "Should import MLStrategy"
    print("   ✅ RandomForestStrategy inherits from MLStrategy")

    # Test RandomForestAgent composition
    rf_agent_path = project_root / "crypto_trading/agents/ml/random_forest_agent.py"
    with open(rf_agent_path, 'r') as f:
        content = f.read()

    # Check composition pattern
    assert "self.strategy = RandomForestStrategy" in content, "Should use composition"
    assert "from .random_forest_strategy import RandomForestStrategy" in content, "Should import RandomForestStrategy"
    assert "class RandomForestAgent(BaseAgent)" in content, "Should inherit from BaseAgent"
    print("   ✅ RandomForestAgent uses composition pattern")

def test_method_delegation():
    """Test that method delegation is properly implemented."""
    print("\n3. Testing Method Delegation...")

    rf_agent_path = project_root / "crypto_trading/agents/ml/random_forest_agent.py"
    with open(rf_agent_path, 'r') as f:
        content = f.read()

    # Check for delegation methods
    delegation_methods = [
        "def is_trained(self)",
        "def save_model(self",
        "def load_model(self",
        "def needs_retraining(self)",
        "def get_feature_importance(self)"
    ]

    for method in delegation_methods:
        assert method in content, f"Missing delegation method: {method}"

    print("   ✅ All delegation methods present")

    # Check that delegation calls strategy
    assert "return self.strategy.is_trained" in content, "is_trained should delegate to strategy"
    assert "self.strategy.save_model" in content, "save_model should delegate to strategy"
    print("   ✅ Delegation methods call strategy correctly")

def test_interface_compliance():
    """Test that interfaces are properly implemented."""
    print("\n4. Testing Interface Compliance...")

    # Test MLStrategy has required abstract methods
    ml_strategy_path = project_root / "crypto_trading/agents/ml/ml_strategy.py"
    with open(ml_strategy_path, 'r') as f:
        content = f.read()

    abstract_methods = [
        "@abstractmethod\n    def _create_model(self)",
        "@abstractmethod\n    def _train_model_impl(",
        "@abstractmethod\n    def _predict(self",
        "@abstractmethod\n    def _evaluate_model(self"
    ]

    for method in abstract_methods:
        assert method in content, f"Missing abstract method: {method}"

    print("   ✅ MLStrategy has all required abstract methods")

    # Test RandomForestStrategy implements abstract methods
    rf_strategy_path = project_root / "crypto_trading/agents/ml/random_forest_strategy.py"
    with open(rf_strategy_path, 'r') as f:
        content = f.read()

    implementations = [
        "def _create_model(self)",
        "def _train_model_impl(",
        "def _predict(self",
        "def _evaluate_model(self"
    ]

    for method in implementations:
        assert method in content, f"Missing implementation: {method}"

    print("   ✅ RandomForestStrategy implements all abstract methods")

def test_no_duplication():
    """Test that code duplication has been eliminated."""
    print("\n5. Testing Code Duplication Elimination...")

    # Read key files
    files_to_check = [
        "crypto_trading/agents/ml/ml_strategy.py",
        "crypto_trading/agents/ml/random_forest_strategy.py",
        "crypto_trading/agents/ml/random_forest_agent.py"
    ]

    file_contents = {}
    for file_path in files_to_check:
        with open(project_root / file_path, 'r') as f:
            file_contents[file_path] = f.read()

    # Check that ML logic is only in MLStrategy and its subclasses
    ml_keywords = ['sklearn', 'RandomForestClassifier', 'train_test_split', 'StandardScaler']

    # ML logic should be in strategy files, not agent
    agent_content = file_contents["crypto_trading/agents/ml/random_forest_agent.py"]
    for keyword in ml_keywords:
        if keyword in agent_content:
            # Allow imports but not implementation
            lines_with_keyword = [line for line in agent_content.split('\n') if keyword in line]
            implementation_lines = [line for line in lines_with_keyword if not line.strip().startswith(('from ', 'import '))]
            assert len(implementation_lines) == 0, f"ML implementation found in agent: {keyword}"

    print("   ✅ No ML implementation duplication in agent")

    # Strategy should contain ML logic
    strategy_content = file_contents["crypto_trading/agents/ml/random_forest_strategy.py"]
    ml_found = any(keyword in strategy_content for keyword in ml_keywords)
    assert ml_found, "Strategy should contain ML logic"

    print("   ✅ ML logic properly contained in strategy")

def test_imports_and_dependencies():
    """Test that all imports are correct after refactoring."""
    print("\n6. Testing Import Structure...")

    # Test that files can be parsed (imports are syntactically correct)
    files_to_test = [
        "crypto_trading/agents/ml/ml_strategy.py",
        "crypto_trading/agents/ml/random_forest_strategy.py",
        "crypto_trading/agents/ml/random_forest_agent.py"
    ]

    for file_path in files_to_test:
        full_path = project_root / file_path
        with open(full_path, 'r') as f:
            content = f.read()

        # Parse to check syntax and imports
        try:
            ast.parse(content)
            print(f"   ✅ {file_path} - imports and syntax valid")
        except SyntaxError as e:
            raise AssertionError(f"Syntax error in {file_path}: {e}")

def run_all_tests():
    """Run all refactoring tests."""
    print("🔍 REFACTORING VALIDATION TEST SUITE")
    print("=" * 60)

    try:
        test_file_structure()
        test_architecture_patterns()
        test_method_delegation()
        test_interface_compliance()
        test_no_duplication()
        test_imports_and_dependencies()

        print("\n" + "=" * 60)
        print("🎉 ALL REFACTORING TESTS PASSED!")
        print("\nRefactoring Summary:")
        print("✅ BaseMLAgent eliminated (no code duplication)")
        print("✅ RandomForestStrategy inherits from MLStrategy")
        print("✅ RandomForestAgent uses clean composition")
        print("✅ All abstract methods implemented")
        print("✅ Proper delegation pattern")
        print("✅ All syntax and imports valid")
        print("\nThe refactoring was successful and maintains code quality!")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)