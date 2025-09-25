#!/usr/bin/env python3
"""Development setup script for Gas Hydraulics Analysis Suite."""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, check=True, shell=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Gas Hydraulics Analysis Suite - Development Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("Failed to install dependencies. Please check your pip installation.")
        return False
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running test suite"):
        print("Some tests failed. Please review the output above.")
        return False
    
    print("\n" + "=" * 50)
    print("✓ Development environment setup complete!")
    print("\nNext steps:")
    print("1. Install the plugin in QGIS by copying the 'gas_hydraulics' folder to your QGIS plugins directory")
    print("2. Enable the plugin in QGIS: Plugins > Manage and Install Plugins")
    print("3. Use the plugins from the QGIS interface")
    
    return True


if __name__ == "__main__":
    main()