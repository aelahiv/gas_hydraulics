# Gas Hydraulics Analysis Suite

A comprehensive QGIS plugin suite for gas utility load analysis, historical trends, and demand forecasting.

## Overview

This plugin suite provides a unified interface to four specialized gas hydraulics analysis tools:

1.  **Demand File Analysis**: Analyzes current year cumulative load breakdown per class and area.
2.  **Historical Analysis**: Examines historical load trends over 5-year periods.
3.  **Load Forecast**: Projects future loads using various forecasting methods.
4.  **Pipe to Node Converter**: Converts pipe-loaded hydraulic models to node-loaded models.

## Plugins

### Main Plugin (`gas_hydraulics_plugin.py`)

This is the main entry point for the plugin suite. It provides a GUI menu in QGIS to access the different analysis tools.

### Demand File Analysis (`demand_file_analysis.py`)

This plugin analyzes the current year's cumulative load breakdown per class (Residential, Commercial, Industrial) and per area.

- Reads a polygon layer for subzones, a pipe layer, and an Excel file with a service point list.
- Uses spatial intersection to find pipes within each zone.
- Calculates load based on factors like `Base Factor`, `Heat Factor`, and `HUC 3-Year Peak Demand`.
- Provides detailed audit logs and results visualization.

### Historical Analysis (`historical_analysis.py`)

This plugin provides historical load analysis over time, aggregated in 5-year periods.

- Uses the `Install Date` from the service point list to determine when loads came online.
- Aggregates loads by class and area for past periods to show growth trends.
- Can export a detailed CSV report with load by polygon, category, and year.

### Load Forecast (`forecast_plugin.py`)

This plugin performs comprehensive gas load forecasting. It can run in two modes:

-   **Basic Mode**: Uses a simpler forecasting model based on current loads, ultimate loads, and growth projections.
-   **Full Mode** (Enhanced): Requires `pandas`, `numpy`, and `scikit-learn`. This mode enables:
    -   Spatial analysis to calculate cumulative loads.
    -   Population-based residential forecasting using housing data.
    -   Regression-based commercial and industrial forecasting.

This plugin also includes a **Load Assignment** tool to assign forecasted loads to the pipe network.

### Pipe to Node Converter (`pipe_to_node_converter.py`)

This utility converts pipe-loaded hydraulic models into node-loaded models, suitable for use in other systems like Synergi.

- Identifies and validates connections between LTPS and GA pipes based on `Name` and `Diameter`.
- Extracts flow categories from pipe descriptions.
- Aggregates loads at terminal nodes.
- Generates output files, including:
    - `node_ex.csv`: Node loads aggregated by category.
    - `cat_setup.csv`: Flow category configuration.
    - `synergi_demand_script.dsf`: An import script for Synergi.
    - `connections.csv`: A list of validated pipe connections.

## Requirements

### Software
- QGIS 3.16 or higher
- Python 3.8 or higher

### Python Packages
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 1.0.0 (for Forecast Plugin's Full Mode)
- openpyxl >= 3.0.0
- pytest >= 7.0.0 (for testing)

You can install the required packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Installation

1.  Clone the repository.
2.  Install the required Python packages.
3.  Copy the `gas_hydraulics` folder into your QGIS plugins directory.
4.  Enable the "Gas Hydraulics" plugin in the QGIS Plugin Manager.

## Usage

Once installed and enabled in QGIS, the "Gas Hydraulics" menu will appear in the QGIS menu bar, providing access to the different analysis tools. Each tool has its own dialog for inputs and parameters.

## Testing

To run the unit tests for the project, use `pytest`:
```bash
pytest tests/
```
