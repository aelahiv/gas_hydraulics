# Gas Hydraulics Analysis Suite

A comprehensive QGIS plugin suite for gas utility load analysis, historical trends, and demand forecasting.

## Overview

This plugin suite provides three specialized tools for gas utility planning and analysis:

1. **Demand File Analysis** - Analyzes current year cumulative load breakdown per class and area
2. **Historical Analysis** - Examines historical load trends over 5-year periods
3. **Forecast Plugin** - Projects future loads for 5, 10, 15, and 20-year periods

## Features

- Spatial intersection analysis with configurable buffer zones
- Multi-class load calculations (Residential, Commercial, Industrial)
- Comprehensive audit and debugging capabilities
- Excel data integration with service point analysis
- Flexible factor code handling for different rate classes
- Growth projection modeling with priority area support
- **CSV Import/Export**: All tables support CSV import and export functionality
  - Results tables: Export analysis results to CSV files
  - Editable tables: Import data from CSV files with flexible header matching
  - Growth projections: Import/export forecast parameters and growth models

## Requirements

### Software Dependencies
- QGIS 3.16 or higher
- Python 3.8 or higher

### Python Packages
- pandas >= 1.3.0
- numpy >= 1.20.0
- openpyxl >= 3.0.0
- pytest >= 7.0.0 (for testing)

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd gas_hydraulics
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Copy the `gas_hydraulics` folder to your QGIS plugins directory:
   - Windows: `C:\Users\<username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\`
   - Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - macOS: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`

4. Enable the plugin in QGIS:
   - Open QGIS
   - Go to Plugins > Manage and Install Plugins
   - Find "Gas Hydraulics" in the Installed tab
   - Check the box to activate

## Usage

### Demand File Analysis

1. Prepare your data:
   - Excel file with "Service Point List" sheet containing service point data
   - QGIS polygon layer for analysis zones
   - QGIS line layer for pipe network

# Gas Hydraulics Analysis Suite

A QGIS plugin suite for gas utility load analysis, historical trends, and demand forecasting.

This repository contains a set of tools to support demand-file analysis, historical trend analysis,
forecasting, and a new utility to convert pipe-loaded models into node-loaded formats for downstream
consumption (for example, Synergi demand import scripts).

## Notable recent changes

- UI and validation: dialogs across the plugins (Forecast, Historical Analysis, Demand File Analysis,
  and Load Assignment) have been enhanced with clearer titles, descriptions, separators, and
  input validation to make the workflows easier to use and to provide immediate feedback on
  invalid inputs.
- Pipe→Node converter: a new utility was added at
  `gas_hydraulics/gas_hydraulics/pipe_to_node_converter.py`. It generates the following outputs
  (by default) when run in its GUI or invoked programmatically:
  - `node_ex.csv` — node exchange file
  - `cat_setup.csv` — category setup mapping file
  - `synergi_demand_script.dsf` — Synergi-style demand import script
  - `connections.csv` — node-to-pipe connection mapping
  Documentation for the converter is available at `docs/PIPE_TO_NODE_CONVERTER_GUIDE.md`.
- Emoji removal: repository text and code were cleaned to remove emoji characters for consistency.
  The removal was done by a scripted pass that created backups for safety. 80 files were modified
  and each modified file has a `.bak` backup next to it (for example `README.md.bak`). Keep these
  backups if you want to review or revert the automated edits.

## Overview

Main components:

1. Demand File Analysis — analyze cumulative loads from service point spreadsheets.
2. Historical Analysis — compute historical trends and 5-year period summaries.
3. Forecast Plugin — project future loads (5, 10, 15, 20 year horizons) with priority-area support.
4. PipeToNodeConverter — convert pipe-loaded models into node-based outputs and helper scripts.

## Requirements

### Software
- QGIS 3.16 or higher (plugins are designed to run inside QGIS)
- Python 3.8 or higher (for running standalone scripts, tests, and tools)

### Python packages
- pandas >= 1.3.0
- numpy >= 1.20.0
- openpyxl >= 3.0.0
- pytest >= 7.0.0 (for running test suite)

Install Python dependencies:

1. Create/activate your Python environment
2. Install packages:

   pip install -r requirements.txt

Note: QGIS Python environment provides the PyQt and QGIS bindings used by the plugin UI — run
GUI components from inside QGIS or ensure QGIS Python environment is accessible to your interpreter.

## Installation (QGIS plugin)

1. Copy the `gas_hydraulics` folder into your QGIS plugins directory (see QGIS docs for the exact
   location on your OS).
2. Launch QGIS, open Plugins > Manage and Install Plugins, and enable "Gas Hydraulics".

## PipeToNodeConverter (quick guide)

- File: `gas_hydraulics/gas_hydraulics/pipe_to_node_converter.py`
- Docs: `docs/PIPE_TO_NODE_CONVERTER_GUIDE.md`

This converter produces CSV files and a Synergi demand script to help move pipeloaded outputs to
node-loaded formats. The converter has a GUI dialog (recommended) but can be imported and used in
scripts; see the converter guide for examples and expected input formats.

## Running tests and basic validation

- Run unit tests:
  python -m pytest tests/ -v

- Quick syntax/compile checks for core modules (example):
  python -m py_compile gas_hydraulics/pipe_to_node_converter.py

## Backups and the emoji-removal pass

During a repository-wide cleanup, emoji characters were removed from source and doc files to
standardize text rendering across environments. The process created backup files for every file
it modified. If you want to inspect or revert an automated change, look for the corresponding
`.bak` file next to the modified file (for example `README.md.bak`). If you prefer to remove the
backups (after review) you can delete `*.bak` files, but we recommend keeping them until you've
verified all plugin behavior in your QGIS environment.

## Troubleshooting (high level)

- Zero or unexpected loads:
  - Verify factor code handling and that the Excel "Distribution Pipe" values match GIS pipe names.
  - Use the audit/logging panels included in the plugins to trace processing steps.

- No historical data found:
  - Ensure the Excel Install Date column contains valid dates and falls in your analysis period.

- Spatial intersection issues:
  - Confirm layer geometries and coordinate reference systems match.

## Contributing

1. Fork the repository
2. Add feature or bugfix on a new branch
3. Add tests for new behavior
4. Ensure tests pass and linting/syntax checks succeed
5. Submit a pull request

## License

This project is licensed under the GNU General Public License v3.0.

## Support

If you discover bugs or want enhancements, open an issue in the project tracker with a concise
description and example data when possible.
