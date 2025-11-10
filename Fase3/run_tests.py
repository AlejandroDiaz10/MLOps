"""
Test runner script with common testing scenarios.

Usage:
    python run_tests.py                # Run all tests
    python run_tests.py --unit         # Run only unit tests
    python run_tests.py --integration  # Run only integration tests
    python run_tests.py --api          # Run only API tests
    python run_tests.py --quick        # Run fast tests only
    python run_tests.py --coverage     # Run with coverage report
"""

import sys
import subprocess
from pathlib import Path
import typer
from loguru import logger

app = typer.Typer(help="Test runner for German Credit Risk project")


def run_command(cmd: list, description: str = "Running tests"):
    """Execute a command and handle output."""
    logger.info(f"{description}...")
    logger.info(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        logger.success("✅ Tests passed!")
        return True
    else:
        logger.error("❌ Tests failed!")
        return False


@app.command()
def all(
    verbose: bool = typer.Option(True, "--verbose", "-v"),
    coverage: bool = typer.Option(False, "--coverage", "-c"),
    html: bool = typer.Option(False, "--html", help="Generate HTML report"),
):
    """
    Run ALL tests (unit + integration + API).

    Examples:
        python run_tests.py all
        python run_tests.py all --coverage
        python run_tests.py all --coverage --html
    """
    cmd = ["pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=fase3", "--cov=api", "--cov-report=term-missing"])

        if html:
            cmd.append("--cov-report=html")

    success = run_command(cmd, "Running ALL tests")

    if coverage and html and success:
        logger.info("\n📊 Coverage report generated:")
        logger.info("   Open: htmlcov/index.html")

    sys.exit(0 if success else 1)


@app.command()
def unit(verbose: bool = typer.Option(True, "--verbose", "-v")):
    """
    Run ONLY unit tests (fast, isolated).

    Examples:
        python run_tests.py unit
        python run_tests.py unit -v
    """
    cmd = ["pytest", "-m", "unit"]

    if verbose:
        cmd.append("-v")

    success = run_command(cmd, "Running UNIT tests")
    sys.exit(0 if success else 1)


@app.command()
def integration(verbose: bool = typer.Option(True, "--verbose", "-v")):
    """
    Run ONLY integration tests (slower, end-to-end).

    Examples:
        python run_tests.py integration
        python run_tests.py integration -v
    """
    cmd = ["pytest", "-m", "integration"]

    if verbose:
        cmd.append("-v")

    success = run_command(cmd, "Running INTEGRATION tests")
    sys.exit(0 if success else 1)


@app.command()
def api(verbose: bool = typer.Option(True, "--verbose", "-v")):
    """
    Run ONLY API tests.

    Examples:
        python run_tests.py api
        python run_tests.py api -v
    """
    cmd = ["pytest", "-m", "api"]

    if verbose:
        cmd.append("-v")

    success = run_command(cmd, "Running API tests")
    sys.exit(0 if success else 1)


@app.command()
def quick(verbose: bool = typer.Option(True, "--verbose", "-v")):
    """
    Run quick tests (skip slow tests).

    Examples:
        python run_tests.py quick
        python run_tests.py quick -v
    """
    cmd = ["pytest", "-m", "not slow"]

    if verbose:
        cmd.append("-v")

    success = run_command(cmd, "Running QUICK tests (no slow tests)")
    sys.exit(0 if success else 1)


@app.command()
def coverage(
    html: bool = typer.Option(True, "--html"),
    fail_under: int = typer.Option(70, "--fail-under", help="Minimum coverage %"),
):
    """
    Run tests with coverage report.

    Examples:
        python run_tests.py coverage
        python run_tests.py coverage --fail-under 80
        python run_tests.py coverage --no-html
    """
    cmd = [
        "pytest",
        "-v",
        "--cov=fase3",
        "--cov=api",
        "--cov-report=term-missing",
        f"--cov-fail-under={fail_under}",
    ]

    if html:
        cmd.append("--cov-report=html")

    success = run_command(cmd, f"Running tests with coverage (minimum {fail_under}%)")

    if html and success:
        logger.info("\n📊 Coverage report generated:")
        logger.info("   Open: htmlcov/index.html")
        logger.info(f"   Minimum coverage: {fail_under}%")

    sys.exit(0 if success else 1)


@app.command()
def file(
    filepath: str = typer.Argument(..., help="Path to test file"),
    verbose: bool = typer.Option(True, "--verbose", "-v"),
    show_output: bool = typer.Option(False, "--show-output", "-s"),
):
    """
    Run tests from a specific file.

    Examples:
        python run_tests.py file tests/unit/test_data_processor.py
        python run_tests.py file tests/integration/test_pipeline.py -s
    """
    cmd = ["pytest", filepath]

    if verbose:
        cmd.append("-v")

    if show_output:
        cmd.append("-s")

    success = run_command(cmd, f"Running tests from {filepath}")
    sys.exit(0 if success else 1)


@app.command()
def watch():
    """
    Watch for file changes and re-run tests (requires pytest-watch).

    Install: pip install pytest-watch

    Example:
        python run_tests.py watch
    """
    try:
        cmd = ["ptw", "--", "-v", "-m", "not slow"]
        logger.info("👀 Watching for changes...")
        logger.info("   Press Ctrl+C to stop")
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("\n✋ Stopped watching")
    except FileNotFoundError:
        logger.error("❌ pytest-watch not installed")
        logger.info("   Install with: pip install pytest-watch")
        sys.exit(1)


@app.command()
def ci():
    """
    Run tests in CI/CD mode (strict, with coverage).

    This is what CI/CD should run:
    - All tests
    - Coverage report (XML for CI)
    - Fail if coverage < 70%
    - Strict warnings

    Example:
        python run_tests.py ci
    """
    cmd = [
        "pytest",
        "-v",
        "--strict-warnings",
        "--cov=fase3",
        "--cov=api",
        "--cov-report=term",
        "--cov-report=xml",
        "--cov-fail-under=70",
        "--maxfail=1",  # Stop after first failure
    ]

    success = run_command(cmd, "Running CI/CD tests")

    if success:
        logger.success("\n✅ CI/CD tests passed!")
        logger.info("   Coverage report: coverage.xml")
    else:
        logger.error("\n❌ CI/CD tests failed!")

    sys.exit(0 if success else 1)


@app.command()
def clean():
    """
    Clean test artifacts (.pytest_cache, htmlcov, .coverage).

    Example:
        python run_tests.py clean
    """
    import shutil

    artifacts = [".pytest_cache", "htmlcov", ".coverage", "coverage.xml", ".coverage.*"]

    logger.info("🧹 Cleaning test artifacts...")

    for artifact in artifacts:
        path = Path(artifact)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
                logger.info(f"   Removed: {artifact}/")
            else:
                path.unlink()
                logger.info(f"   Removed: {artifact}")

    # Clean __pycache__
    for pycache in Path(".").rglob("__pycache__"):
        shutil.rmtree(pycache)
        logger.info(f"   Removed: {pycache}")

    logger.success("✅ Cleanup complete!")


@app.command()
def summary():
    """
    Show summary of available tests.

    Example:
        python run_tests.py summary
    """
    logger.info("\n📊 Test Summary\n")

    # Collect tests
    result = subprocess.run(
        ["pytest", "--collect-only", "-q"], capture_output=True, text=True
    )

    if result.returncode == 0:
        lines = result.stdout.strip().split("\n")

        # Parse summary
        total_tests = 0
        for line in lines:
            if " test" in line:
                logger.info(f"   {line}")
                # Extract number
                parts = line.split()
                if parts and parts[0].isdigit():
                    total_tests += int(parts[0])

        logger.info(f"\n📈 Total tests: {total_tests}")
        logger.info("\nRun with:")
        logger.info("   python run_tests.py all        # Run all tests")
        logger.info("   python run_tests.py unit       # Run unit tests")
        logger.info("   python run_tests.py integration # Run integration tests")
        logger.info("   python run_tests.py coverage   # Run with coverage")
    else:
        logger.error("Failed to collect tests")


if __name__ == "__main__":
    # If no command specified, run all tests
    if len(sys.argv) == 1:
        sys.argv.append("all")

    app()
