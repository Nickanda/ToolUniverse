#!/usr/bin/env python3
"""
Test runner for OpenBabel tools.

This script runs different types of tests:
- Unit tests (with mocked OpenBabel)
- Integration tests (with real OpenBabel if available)
- Configuration validation tests

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --config           # Run only configuration tests
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_pytest(args):
    """Run pytest with given arguments"""
    cmd = [sys.executable, "-m", "pytest"] + args
    return subprocess.run(cmd, cwd=Path(__file__).parent.parent)


def main():
    parser = argparse.ArgumentParser(description="Run OpenBabel tool tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--config", action="store_true", help="Run configuration tests only"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")

    args = parser.parse_args()

    # Build pytest arguments
    pytest_args = []

    if args.verbose:
        pytest_args.append("-v")

    if args.coverage:
        pytest_args.extend(
            ["--cov=src/tooluniverse", "--cov-report=html", "--cov-report=term"]
        )

    # Select test files based on arguments
    if args.unit:
        pytest_args.append("tests/test_openbabel_tools.py")
        print("Running unit tests...")
    elif args.integration:
        pytest_args.append("tests/test_openbabel_integration.py")
        print("Running integration tests...")
    elif args.config:
        pytest_args.extend(["-k", "TestConfigurationValidation"])
        print("Running configuration tests...")
    else:
        pytest_args.append("tests/")
        print("Running all tests...")

    # Run tests
    result = run_pytest(pytest_args)

    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with return code {result.returncode}")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
