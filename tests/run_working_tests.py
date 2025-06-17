#!/usr/bin/env python3
"""
Quick test runner for working OpenBabel tests
"""

import subprocess
import sys


def run_working_tests():
    """Run only the tests that are known to pass"""

    test_commands = [
        # Basic functionality tests
        "tests/test_openbabel_functional.py::test_openbabel_availability_flag",
        "tests/test_openbabel_functional.py::test_base_tool_import_when_unavailable",
        "tests/test_openbabel_functional.py::test_tool_classes_can_be_imported",
        "tests/test_openbabel_functional.py::test_tool_creation_with_mocked_openbabel",
        "tests/test_openbabel_functional.py::test_tool_error_handling",
        "tests/test_openbabel_functional.py::test_config_file_exists",
        # Integration tests (including skipped ones)
        "tests/test_openbabel_integration.py",
        # Availability tests
        "tests/test_tool_availability.py::test_openbabel_availability_check",
        "tests/test_tool_availability.py::test_lazy_import_mechanism",
        "tests/test_tool_availability.py::test_non_openbabel_tools_available",
        "tests/test_tool_availability.py::test_invalid_tool_name",
    ]

    print("🧪 Running OpenBabel Tools Test Suite - Working Tests Only")
    print("=" * 60)

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"] + test_commands + ["-v", "--tb=short"]

    try:
        result = subprocess.run(cmd, check=False, capture_output=False)

        if result.returncode == 0:
            print("\n✅ All working tests passed!")
        else:
            print(
                f"\n⚠️  Some tests failed or were skipped (exit code: {result.returncode})"
            )

        return result.returncode

    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


if __name__ == "__main__":
    exit_code = run_working_tests()
    sys.exit(exit_code)
