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

2. Run the analysis:
   - Open the Demand File Analysis dialog
   - Select your Excel file and QGIS layers
   - Configure calculation parameters
   - Execute the analysis

3. Review results:
   - Audit dialog shows detailed processing steps
   - Results dialog displays load breakdowns by zone and class
   - New layer created with calculated load attributes

### Historical Analysis

1. Use the same data structure as Demand File Analysis
2. Specify the analysis period (start and end years)
3. Review historical load trends by 5-year periods
4. Analyze growth patterns across different zones

### Forecast Plugin

1. Input current loads by area and class
2. Define ultimate load targets
3. Configure growth projections and priority areas
4. Generate forecasts for multiple time horizons

## Data Requirements

### Excel Service Point List Format

Required columns:
- Factor Code: Rate class identifier
- Base Factor: Base load factor
- Heat Factor: Heating load factor
- HUC 3-Year Peak Demand: Historical peak demand
- Install Date: Service installation date
- Use Class: Customer class (RES, COM, IND, etc.)
- Distribution Pipe: Pipe identifier matching GIS layer

### QGIS Layer Requirements

**Zone Layer (Polygons):**
- Analysis areas/subzones
- Name field for zone identification

**Pipe Layer (Lines):**
- Gas distribution network
- FacNam1005 field containing facility names that match Excel Distribution Pipe values

## Load Calculation Formula

The plugin calculates loads using:
```
Load = LoadMultiplier × (BaseFactor + HeatFactorMultiplier × HeatFactor) + HUCPeakDemand
```

Default values:
- Load Multiplier: 1.07 (configurable range: 0.001-10.0)
- Heat Factor Multiplier: 56.8 (configurable range: 0.1-200.0)

Factor codes ['0', '1', '20'] retain their Base Factor and Heat Factor values. All other factor codes have these values zeroed, leaving only HUC 3-Year Peak Demand in the calculation.

## Testing

Run the test suite:
```
python -m pytest tests/ -v
```

Individual test files:
- `tests/test_demand_analysis.py` - Demand file analysis tests
- `tests/test_historical_analysis.py` - Historical analysis tests
- `tests/test_forecast_plugin.py` - Forecast plugin tests

## Project Structure

```
gas_hydraulics/
├── gas_hydraulics/              # Main plugin package
│   ├── __init__.py
│   ├── metadata.txt
│   ├── demand_file_analysis.py
│   ├── historical_analysis.py
│   └── forecast_plugin.py
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_demand_analysis.py
│   ├── test_historical_analysis.py
│   └── test_forecast_plugin.py
├── docs/                        # Documentation
├── requirements.txt
└── README.md
```

## Troubleshooting

### Common Issues

**Zero Load Results:**
- Verify factor code logic matches your utility's rate structure
- Check pipe name matching between Excel and GIS layers
- Use audit features to trace data processing steps

**Spatial Intersection Problems:**
- Ensure zone and pipe layers have valid geometries
- Adjust buffer pixel settings for intersection tolerance
- Verify coordinate reference systems match

**Excel Data Issues:**
- Check for required column names and data types
- Validate date formats in Install Date column
- Ensure Distribution Pipe values match GIS layer

### Debug Features

All plugins include comprehensive audit capabilities:
- Excel file structure validation
- Layer geometry and attribute analysis
- Step-by-step processing logs
- Spatial intersection diagnostics

## License

This project is licensed under the GNU General Public License v3.0.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For technical support or feature requests, please create an issue in the project repository.