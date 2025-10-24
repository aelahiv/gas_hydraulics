# Gas Hydraulics Plugin Usage Guide

## After Installation and Enabling the Plugin

### Method 1: Using the Plugin Menu (Recommended)

Once the plugin is enabled, you'll find the Gas Hydraulics tools in the QGIS interface:

1. **Access via Plugins Menu:**
   - Go to **Plugins → Gas Hydraulics** in the QGIS menu bar
   - You'll see four options:
     - `Demand File Analysis` - Analyze current year loads
     - `Historical Analysis` - View historical trends
     - `Load Forecast` - Generate future projections
     - `Gas Hydraulics Analysis` - Main dialog to select analysis type

2. **Access via Toolbar:**
   - Look for the Gas Hydraulics icon in the toolbar
   - Click it to open the main selection dialog

### Method 2: Using Python Console

If the menu items don't appear, use the Python Console:

1. Open **Plugins → Python Console**
2. Run this code to access the main plugin:

```python
# Get the main plugin interface
from gas_hydraulics.gas_hydraulics_plugin import GasHydraulicsPlugin
plugin = GasHydraulicsPlugin(iface)
plugin.initGui()
plugin.run()
```

## Step-by-Step Workflow

### 1. Prepare Your Data

Before using any analysis tool, ensure you have:

#### Required Layers:
- **Polygon Layer**: Study area subdivisions/zones
- **Vector Layer**: Pipe network with proper attributes

#### Required Excel File:
- File with 'Service Point List' sheet
- Required columns:
  - Service Point Number
  - Base Factor, Heat Factor, Factor Code
  - HUC 3-Year Peak Demand
  - Distribution Pipe, Use Class
  - Install Date, End Date
  - Elevation, Elevation Unit

### 2. Load Data in QGIS

```python
# Example of loading layers programmatically
# Add your polygon layer (zones)
zone_layer = iface.addVectorLayer("path/to/zones.shp", "Study Zones", "ogr")

# Add your pipe network layer
pipe_layer = iface.addVectorLayer("path/to/pipes.shp", "Pipe Network", "ogr")
```

### 3. Run Analysis Tools

#### A) Demand File Analysis (Current Year Loads)

**Purpose**: Calculate current year load breakdown by zone and class

**Steps**:
1. Select `Demand File Analysis` from menu
2. Choose your Excel file with service point data
3. Select zone and pipe layers
4. Set current year (default: 2025)
5. Review generated `zones_load` output layer

**Example Code**:
```python
from gas_hydraulics.demand_file_analysis import DemandFileAnalysisPlugin

plugin = DemandFileAnalysisPlugin(iface)
zone_loads = plugin.process_zone_loads(
    excel_file_path="C:/data/service_points.xlsx",
    zone_layer=zone_layer,
    pipe_layer=pipe_layer,
    current_year=2025
)
print(zone_loads)
```

#### B) Historical Analysis (Load Trends)

**Purpose**: Analyze load growth over 5-year periods

**Steps**:
1. Select `Historical Analysis` from menu
2. Choose Excel file with install dates
3. Set analysis time range (e.g., 2000-2025)
4. Review historical load tables and trends

**Example Code**:
```python
from gas_hydraulics.historical_analysis import HistoricalAnalysisPlugin

plugin = HistoricalAnalysisPlugin(iface)
historical_data = plugin.analyze_historical_loads_by_period(
    excel_file_path="C:/data/service_points.xlsx",
    zone_layer=zone_layer,
    pipe_layer=pipe_layer,
    start_year=2000,
    end_year=2025
)
```

#### C) Load Forecast (Future Projections)

**Purpose**: Generate 5/10/15/20 year load forecasts

**Steps**:
1. Select `Load Forecast` from menu
2. Input current loads by area and class
3. Input ultimate loads by area and class
4. Provide growth projection data
5. Configure priority zones (optional)
6. Review forecast tables and charts

**Example Code**:
```python
from gas_hydraulics.forecast_plugin import ForecastPlugin

# Define current and ultimate loads
current_loads = {
    "Zone A": {"residential": 1500, "commercial": 800, "industrial": 0},
    "Zone B": {"residential": 2200, "commercial": 1200, "industrial": 500}
}

ultimate_loads = {
    "Zone A": {"residential": 5000, "commercial": 2000, "industrial": 0},
    "Zone B": {"residential": 8000, "commercial": 3000, "industrial": 2000}
}

growth_projection = {year: 50 + year * 0.5 for year in range(2025, 2046)}

# Optional: Configure priority zones
priority_zones = [
    {
        'zone_name': 'Zone C',
        'priority_level': 'Zero Priority (0.0x)',
        'start_year': 2025,
        'end_year': 2035
    }
]

plugin = ForecastPlugin(iface)
forecasts = plugin.create_forecast_scenarios(
    current_loads, ultimate_loads, growth_projection, priority_zones
)
```

#### D) Load Assignment (Assign Forecast to Pipes)

**Purpose**: Assign forecasted loads from CSV to pipe infrastructure

**Steps**:
1. Generate a forecast CSV file (from Load Forecast tool)
2. Prepare pipe layer with Year field indicating construction period
3. Select `Load Assignment` from Forecast dialog
4. Choose forecast CSV file
5. Select polygon layer (zones/neighborhoods)
6. Select pipe layer
7. Configure settings:
   - Start year (e.g., 2025)
   - Aggregation window (default: 5 years)
8. Run assignment

**What Happens**:
- Pipes assigned to polygons by maximum overlap intersection
- Loads calculated using 5-year windows:
  - 2030 pipes: Sum of 2026-2030 increments
  - 2035 pipes: Sum of 2031-2035 increments
  - 2045 pipes: Sum of 2041-2045 increments
- CSV cumulative totals converted to per-year increments (baseline year = 0)
- All pipes updated with:
  - **LOAD**: Load value (GJ/d)
  - **DESC**: Polygon name + period description
  - **PROP**: "Proposed" status
  - **YEAR**: Construction year
  - **DATETIME**: Human-readable date (1-Nov-YYYY)
  - **SYN_DATE**: Synergi format date (YYYYMMDD)

**Example Code**:
```python
from gas_hydraulics.load_assignment import LoadAssignmentTool

tool = LoadAssignmentTool(iface)
tool.run_load_assignment(
    csv_file="load_forecast_2025.csv",
    polygon_layer=zone_layer,
    pipe_layer=pipe_layer,
    start_year=2025,
    aggregation_window=5
)
```

**Important Notes**:
- Pipes with no intersecting polygon receive "Unknown Area" description
- Pipes with no forecast load still get year, polygon, and Prop="Proposed"
- Priority zones with "Zero Priority" will not receive overflow loads
- Excess demand deferred to future years rather than forced into restricted areas

## Understanding the Results

### Demand File Analysis Output:
- **zones_load layer**: New layer with load summaries by class
- **Load by Class**: Residential, Industrial, Commercial totals per zone
- **Current Year Summary**: Total loads as of specified year

### Historical Analysis Output:
- **Period Tables**: Load data by 5-year periods (e.g., 2000-2004, 2005-2009)
- **Growth Rates**: Annual growth between periods
- **Cumulative Loads**: Total loads added by each period

### Forecast Output:
- **Projection Tables**: Future loads for 5, 10, 15, 20 year horizons
- **By Class and Zone**: Separate residential and non-residential forecasts
- **Capacity Analysis**: Shows when zones approach ultimate capacity

## Configuration Options

### Load Calculation Parameters:
```python
# Modify default parameters
plugin.load_multiplier = 1.07  # Default multiplier
plugin.heat_factor_multiplier = 56.8  # Heat factor multiplier
```

### Factor Code Treatment:
```python
# Customize which factor codes get zeroed
plugin.calculate_load(df, factor_codes_to_zero=['0', '1', '20', '10'])
```

### Forecast Parameters:
```python
# Adjust forecast settings
forecasts = plugin.create_forecast_scenarios(
    current_loads, 
    ultimate_loads, 
    growth_projection,
    forecast_years=[2030, 2035, 2040, 2045]  # Custom forecast years
)
```

## Troubleshooting

### Common Issues:

1. **Plugin Not Visible in Menu:**
   - Check that plugin is enabled in Plugin Manager
   - Use Python Console method as fallback

2. **Excel File Errors:**
   - Verify sheet name is exactly 'Service Point List'
   - Check all required columns are present

3. **Spatial Operations Not Working:**
   - Ensure zone and pipe layers are loaded
   - Check coordinate systems match

4. **Date Parsing Errors:**
   - Use standard date formats (YYYY-MM-DD)
   - Check for missing dates in Install Date column

### Debug Mode:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run plugin with debug output
plugin.run()
```

## Example Complete Workflow:

```python
# 1. Load required layers
zone_layer = iface.addVectorLayer("zones.shp", "Zones", "ogr")
pipe_layer = iface.addVectorLayer("pipes.shp", "Pipes", "ogr")

# 2. Run demand analysis
from gas_hydraulics.demand_file_analysis import DemandFileAnalysisPlugin
demand_plugin = DemandFileAnalysisPlugin(iface)
current_loads = demand_plugin.process_zone_loads("service_points.xlsx")

# 3. Run historical analysis
from gas_hydraulics.historical_analysis import HistoricalAnalysisPlugin
historical_plugin = HistoricalAnalysisPlugin(iface)
trends = historical_plugin.analyze_historical_loads_by_period("service_points.xlsx")

# 4. Run forecast
from gas_hydraulics.forecast_plugin import ForecastPlugin
forecast_plugin = ForecastPlugin(iface)
projections = forecast_plugin.create_forecast_scenarios(current_loads, ultimate_loads, growth_data)

# 5. Review results
print("Current Loads:", current_loads)
print("Historical Trends:", trends)
print("Future Projections:", projections)
```

This guide provides comprehensive instructions for using the Gas Hydraulics plugins in both interactive and programmatic modes.