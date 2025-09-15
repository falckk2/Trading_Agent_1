#!/usr/bin/env python3
"""
Test runner for the cryptocurrency trading system.
Runs all tests and generates coverage reports.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(coverage=True, verbose=False, specific_test=None):
    """Run the test suite with optional coverage reporting."""

    # Ensure we're in the project root
    project_root = Path(__file__).parent

    # Build pytest command
    cmd = ["python", "-m", "pytest"]

    if coverage:
        cmd.extend([
            "--cov=crypto_trading",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])

    if verbose:
        cmd.append("-v")

    # Add specific test if provided
    if specific_test:
        cmd.append(f"crypto_trading/tests/{specific_test}")
    else:
        cmd.append("crypto_trading/tests/")

    # Add additional pytest options
    cmd.extend([
        "--tb=short",
        "--strict-markers",
        "--disable-warnings"
    ])

    print(f"Running command: {' '.join(cmd)}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run cryptocurrency trading system tests")

    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Run tests without coverage reporting"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Run tests in verbose mode"
    )

    parser.add_argument(
        "--test",
        type=str,
        help="Run specific test file (e.g., test_agent_manager.py)"
    )

    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies before running tests"
    )

    args = parser.parse_args()

    if args.install_deps:
        print("Installing test dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "pytest", "pytest-cov", "pytest-asyncio", "pytest-mock"
        ])

    # Run tests
    coverage = not args.no_coverage
    return_code = run_tests(
        coverage=coverage,
        verbose=args.verbose,
        specific_test=args.test
    )

    if return_code == 0:
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        if coverage:
            print("📊 Coverage report generated in 'htmlcov' directory")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ Some tests failed!")
        print("=" * 50)

    return return_code


if __name__ == "__main__":
    sys.exit(main())