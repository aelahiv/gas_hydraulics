# Forecasting Functions Reference

## Overview

The Forecasting module projects future gas demand using historical patterns, demographic data, and growth models. Supports both basic percentage-based growth and advanced demographic-driven forecasting.

## Core Forecast Functions

### run_basic_forecast

**Purpose:** Generate simple growth-based forecast using percentage increases.

**Parameters:**
- `current_loads` (dict): Current year loads by zone and use class
- `forecast_years` (list): Years to forecast (e.g., [2030, 2035, 2040])
- `growth_rates` (dict): Annual growth rates by use class
- `priority_zones` (list, optional): Zones with enhanced growth

**Current Loads Format:**
```python
{
    'zone_name': {
        'residential': float,
        'commercial': float,
        'industrial': float
    }
}
```

**Growth Rates Format:**
```python
{
    'residential': float,  # e.g., 0.02 for 2% annual growth
    'commercial': float,
    'industrial': float
}
```

**Returns:** dict - Forecasted loads:
```python
{
    'zone_name': {
        2030: {'residential': float, 'commercial': float, 'industrial': float},
        2035: {'residential': float, 'commercial': float, 'industrial': float},
        ...
    }
}
```

**Process:**
1. Validates current loads and growth rates
2. For each forecast year:
   - Calculates years from base year
   - Applies compound growth: Load * (1 + rate)^years
   - Applies priority zone multiplier if applicable
3. Returns year-by-year projections

**Notes:**
- Assumes constant compound growth
- Priority zones can have different growth rates
- Simple method suitable for short-term forecasts

---

### run_advanced_forecast

**Purpose:** Generate demographic-driven forecast using population and housing data.

**Parameters:**
- `current_loads` (dict): Current year loads by zone and use class
- `base_year` (int): Starting year for forecast
- `forecast_years` (list): Target forecast years
- `demographic_data` (dict): Population and housing projections
- `consumption_rates` (dict): Per-capita and per-dwelling rates
- `saturation_levels` (dict, optional): Maximum load saturation by zone

**Demographic Data Format:**
```python
{
    'zone_name': {
        2025: {'population': int, 'housing': int},
        2030: {'population': int, 'housing': int},
        ...
    }
}
```

**Consumption Rates Format:**
```python
{
    'residential_per_dwelling': float,  # GJ/day per dwelling
    'commercial_per_capita': float,     # GJ/day per capita
    'industrial_per_capita': float      # GJ/day per capita
}
```

**Returns:** dict - Detailed forecast with methodology:
```python
{
    'zone_name': {
        2030: {
            'residential': float,
            'commercial': float,
            'industrial': float,
            'population': int,
            'housing': int,
            'saturation_factor': float
        },
        ...
    }
}
```

**Process:**
1. Validates demographic data completeness
2. For each zone and forecast year:
   - Retrieves population and housing projections
   - Calculates raw demand using consumption rates
   - Applies saturation curve if defined
   - Constrains growth to realistic levels
3. Logs detailed calculation steps
4. Returns comprehensive results

**Notes:**
- More accurate for long-term forecasts
- Requires quality demographic data
- Accounts for market saturation
- Suitable for urban planning integration

---

### calculate_saturation

**Purpose:** Apply saturation curve to constrain load growth.

**Parameters:**
- `current_load` (float): Current load in GJ/day
- `projected_load` (float): Unconstrained projected load
- `saturation_load` (float): Maximum sustainable load
- `curve_factor` (float, optional): Saturation curve steepness (default: 0.5)

**Returns:** float - Adjusted load accounting for saturation

**Formula:**
```
saturation_factor = 1 - exp(-curve_factor * (saturation_load - current_load) / saturation_load)
adjusted_load = current_load + saturation_factor * (projected_load - current_load)
```

**Curve Behavior:**
- Rapid growth when far from saturation
- Gradual slowing as approaching saturation
- Asymptotic approach to saturation limit

**Notes:**
- Prevents unrealistic exponential growth
- Models market maturation
- Adjustable curve steepness for different scenarios

---

## Demographic Analysis Functions

### import_population_housing_csv

**Purpose:** Import population and housing projection data from CSV.

**Parameters:**
- `csv_file` (str): Path to CSV file

**Required CSV Columns:**
- Year: Projection year
- Population: Population count
- Housing: Housing unit count
- Zone (optional): Zone identifier for zone-specific data

**Returns:** dict - Parsed demographic data by zone and year

**CSV Format Example:**
```
Year,Population,Housing,Zone
2025,50000,20000,Downtown
2030,55000,22000,Downtown
2025,30000,12000,Suburb
```

**Process:**
1. Reads CSV with flexible header matching
2. Validates required columns present
3. Groups data by zone (if column exists)
4. Organizes as nested dictionary by zone and year
5. Validates data types and ranges

**Notes:**
- Case-insensitive column matching
- Supports zone-specific or overall data
- Interpolation available for missing years

---

### interpolate_demographic_data

**Purpose:** Fill gaps in demographic data using linear interpolation.

**Parameters:**
- `demographic_data` (dict): Sparse demographic data
- `start_year` (int): First year requiring data
- `end_year` (int): Last year requiring data

**Returns:** dict - Complete demographic data for all years

**Process:**
1. Identifies available data points
2. For each missing year:
   - Finds nearest earlier and later data points
   - Calculates linear interpolation
   - Fills missing value
3. Extrapolates beyond range if necessary (with warning)

**Interpolation Formula:**
```
value = value1 + (value2 - value1) * (year - year1) / (year2 - year1)
```

**Notes:**
- Linear interpolation between known points
- Logs extrapolation warnings
- Maintains integer values for population and housing

---

### validate_demographic_data

**Purpose:** Check demographic data for completeness and consistency.

**Parameters:**
- `demographic_data` (dict): Demographic data to validate
- `zones` (list): Expected zones
- `years` (list): Required years

**Returns:** dict - Validation results:
```python
{
    'valid': bool,
    'missing_zones': list,
    'missing_years': dict,  # Zone -> list of missing years
    'negative_values': list,
    'warnings': list
}
```

**Checks:**
1. All required zones present
2. All required years present for each zone
3. No negative population or housing values
4. Population and housing trends reasonable
5. No sudden unexplained jumps

**Notes:**
- Blocking errors prevent forecast
- Warnings indicate data quality concerns
- Helps identify data entry errors

---

## Growth Rate Functions

### calculate_historical_growth_rates

**Purpose:** Derive growth rates from historical load data.

**Parameters:**
- `historical_loads` (dict): Historical loads by zone, year, and class
- `n_years` (int, optional): Number of recent years to analyze (default: 5)

**Returns:** dict - Calculated growth rates:
```python
{
    'zone_name': {
        'residential': float,  # Annual growth rate
        'commercial': float,
        'industrial': float
    }
}
```

**Process:**
1. Extracts most recent n years of data
2. For each use class:
   - Calculates year-over-year changes
   - Averages the changes
   - Converts to percentage rate
3. Handles missing data gracefully
4. Logs calculated rates

**Calculation:**
```
rate = average((load_year_i - load_year_i-1) / load_year_i-1)
```

**Notes:**
- Uses recent history for relevance
- Handles missing years
- Returns 0.0 if insufficient data

---

### apply_growth_multiplier

**Purpose:** Adjust growth rates for priority zones or special cases.

**Parameters:**
- `base_rates` (dict): Standard growth rates by use class
- `multiplier` (float): Growth rate multiplier
- `zones` (list, optional): Zones to apply multiplier

**Returns:** dict - Adjusted growth rates

**Example:**
```python
# Increase residential growth by 50% in priority zones
adjusted = apply_growth_multiplier(
    base_rates={'residential': 0.02, 'commercial': 0.01, 'industrial': 0.005},
    multiplier=1.5,
    zones=['Downtown', 'NewDevelopment']
)
# Result: residential rate = 0.03 for specified zones
```

**Notes:**
- Multiplier > 1.0 increases growth
- Multiplier < 1.0 decreases growth
- Can apply to specific zones or all zones

---

## Priority Zone Functions

### identify_priority_zones

**Purpose:** Determine zones requiring enhanced growth projections.

**Parameters:**
- `zones` (list): All available zones
- `criteria` (dict): Selection criteria
- `current_loads` (dict, optional): Current load data

**Criteria Options:**
```python
{
    'names': list,              # Explicit zone names
    'min_load': float,          # Minimum current load
    'max_load': float,          # Maximum current load
    'growth_threshold': float,  # Minimum historical growth
    'development_status': str   # e.g., 'planned', 'under_construction'
}
```

**Returns:** list - Priority zone names

**Selection Methods:**
1. Explicit name list
2. Load-based (high or low current load)
3. Growth-based (high historical growth)
4. Development status
5. Combination of criteria

**Notes:**
- Multiple criteria can be combined
- Used to apply different forecast parameters
- Supports scenario planning

---

### set_priority_zone_parameters

**Purpose:** Configure special forecast parameters for priority zones.

**Parameters:**
- `zone_name` (str): Priority zone identifier
- `parameters` (dict): Special parameters for this zone

**Parameters Options:**
```python
{
    'growth_multiplier': float,
    'saturation_level': float,
    'consumption_rates': dict,
    'notes': str
}
```

**Returns:** bool - Configuration successful

**Notes:**
- Allows zone-specific customization
- Overrides default parameters
- Documented for reporting

---

## Consumption Rate Functions

### calculate_consumption_rates

**Purpose:** Derive consumption rates from historical data and demographics.

**Parameters:**
- `current_loads` (dict): Current loads by zone and use class
- `demographic_data` (dict): Current population and housing
- `method` (str, optional): Calculation method ('per_capita', 'per_dwelling', 'hybrid')

**Returns:** dict - Calculated consumption rates:
```python
{
    'residential_per_dwelling': float,
    'residential_per_capita': float,
    'commercial_per_capita': float,
    'commercial_per_employee': float,
    'industrial_per_capita': float
}
```

**Methods:**
- per_capita: Load / population
- per_dwelling: Load / housing units
- hybrid: Combination based on use class

**Process:**
1. Aggregates loads across zones
2. Sums demographic data
3. Calculates ratios
4. Validates reasonableness
5. Logs calculated rates

**Notes:**
- Residential typically per-dwelling
- Commercial and industrial per-capita or per-employee
- Rates should be relatively stable over time

---

### adjust_consumption_rates

**Purpose:** Modify consumption rates for efficiency improvements or policy changes.

**Parameters:**
- `base_rates` (dict): Current consumption rates
- `adjustments` (dict): Adjustment factors by use class and year

**Adjustments Format:**
```python
{
    2030: {
        'residential': 0.95,  # 5% efficiency improvement
        'commercial': 0.90,   # 10% efficiency improvement
        'industrial': 1.00    # No change
    },
    2040: {
        'residential': 0.90,
        'commercial': 0.85,
        'industrial': 0.95
    }
}
```

**Returns:** dict - Adjusted consumption rates by year

**Notes:**
- Models efficiency improvements over time
- Accounts for policy initiatives
- Conservative adjustments recommended

---

## Scenario Analysis Functions

### create_forecast_scenario

**Purpose:** Define complete forecast scenario with all parameters.

**Parameters:**
- `scenario_name` (str): Identifier for scenario
- `parameters` (dict): All scenario parameters

**Parameters Structure:**
```python
{
    'base_year': int,
    'forecast_years': list,
    'growth_rates': dict,
    'demographic_data': dict,
    'consumption_rates': dict,
    'saturation_levels': dict,
    'priority_zones': list,
    'description': str
}
```

**Returns:** dict - Complete scenario definition

**Usage:**
- Compare multiple scenarios
- Document assumptions
- Sensitivity analysis

---

### compare_scenarios

**Purpose:** Compare multiple forecast scenarios side-by-side.

**Parameters:**
- `scenarios` (list): List of scenario dictionaries
- `comparison_years` (list): Years to compare
- `zones` (list, optional): Zones to include in comparison

**Returns:** dict - Comparison results:
```python
{
    'year': {
        'zone_name': {
            'scenario1_name': {'residential': float, 'commercial': float, ...},
            'scenario2_name': {'residential': float, 'commercial': float, ...},
            'difference': {'residential': float, 'commercial': float, ...},
            'percent_difference': {'residential': float, 'commercial': float, ...}
        }
    }
}
```

**Process:**
1. Runs each scenario
2. Extracts results for comparison years
3. Calculates differences and percentages
4. Generates comparison report

**Notes:**
- Useful for sensitivity analysis
- Identifies key drivers
- Supports decision-making

---

## Utility Functions

### calculate_cumulative_loads_by_polygon

**Purpose:** Calculate historical cumulative loads for baseline.

**Parameters:**
- `demand_file` (str): Path to demand file
- `polygon_layer` (QgsVectorLayer): Zone polygons
- `pipe_layer` (QgsVectorLayer): Pipe network
- `start_year` (int): Beginning of analysis
- `end_year` (int): End of analysis (baseline year)

**Returns:** dict - Cumulative loads by polygon and year

**Process:**
1. Loads and processes demand file
2. Builds pipe-to-polygon mapping
3. Filters to valid date range
4. Calculates cumulative loads for each year
5. Organizes results by polygon

**Notes:**
- Provides baseline for forecasting
- Uses same methodology as historical analysis
- Ensures consistency between modules

---

### export_forecast_to_csv

**Purpose:** Export forecast results to CSV for reporting and analysis.

**Parameters:**
- `forecast_results` (dict): Forecast data
- `output_file` (str): Path for CSV output
- `format` (str, optional): 'wide' or 'long' format

**Wide Format:**
```
Zone,Year,Residential,Commercial,Industrial,Total
Zone1,2030,123.45,45.67,10.00,179.12
```

**Long Format:**
```
Zone,Year,UseClass,Load
Zone1,2030,Residential,123.45
Zone1,2030,Commercial,45.67
```

**Returns:** bool - Export successful

**Notes:**
- Wide format better for spreadsheet analysis
- Long format better for database import
- Includes metadata in header comments

---

### import_forecast_parameters

**Purpose:** Load forecast parameters from configuration file.

**Parameters:**
- `config_file` (str): Path to configuration file (JSON or YAML)

**Configuration Structure:**
```json
{
    "base_year": 2025,
    "forecast_years": [2030, 2035, 2040, 2045, 2050],
    "growth_rates": {
        "residential": 0.02,
        "commercial": 0.015,
        "industrial": 0.01
    },
    "consumption_rates": {
        "residential_per_dwelling": 0.15,
        "commercial_per_capita": 0.05
    }
}
```

**Returns:** dict - Parsed configuration parameters

**Notes:**
- Supports JSON and YAML formats
- Validates parameter types
- Provides defaults for missing values

---

## Validation Functions

### validate_forecast_inputs

**Purpose:** Comprehensive validation of all forecast inputs.

**Parameters:**
- `current_loads` (dict): Current year loads
- `demographic_data` (dict, optional): Population/housing data
- `growth_rates` (dict, optional): Growth rate assumptions
- `forecast_years` (list): Target forecast years

**Returns:** dict - Validation results:
```python
{
    'valid': bool,
    'errors': list,
    'warnings': list,
    'recommendations': list
}
```

**Checks:**
1. Current loads complete and positive
2. Demographic data covers forecast period
3. Growth rates reasonable (typically 0-10% annually)
4. Forecast years in chronological order
5. Saturation levels logical
6. Consumption rates reasonable

**Notes:**
- Prevents invalid forecast runs
- Provides actionable feedback
- Logs all validation issues

---

### check_forecast_reasonableness

**Purpose:** Validate forecast results for reasonableness.

**Parameters:**
- `forecast_results` (dict): Generated forecast
- `historical_data` (dict, optional): Historical loads for comparison

**Returns:** dict - Reasonableness assessment:
```python
{
    'reasonable': bool,
    'flags': list,  # Potential issues
    'statistics': dict
}
```

**Checks:**
1. No negative loads
2. Growth rates within expected ranges
3. No sudden unexplained jumps
4. Total system load reasonable
5. Consistency across zones
6. Alignment with demographic trends

**Notes:**
- Post-forecast validation
- Identifies potential errors
- Quality assurance step

---

## Reporting Functions

### generate_forecast_report

**Purpose:** Create comprehensive forecast report document.

**Parameters:**
- `forecast_results` (dict): Forecast data
- `scenario_info` (dict): Scenario parameters
- `output_file` (str): Path for report file
- `format` (str, optional): 'pdf', 'html', or 'markdown'

**Report Sections:**
1. Executive Summary
2. Methodology
3. Assumptions and Parameters
4. Results by Zone
5. Growth Trends
6. Comparison to Historical
7. Recommendations
8. Appendix (detailed data)

**Returns:** bool - Report generated successfully

**Notes:**
- Professional formatting
- Includes charts and tables
- Suitable for stakeholder presentation

---

## Examples

### Basic Forecast

```python
from gas_hydraulics.forecast_plugin import ForecastPlugin

plugin = ForecastPlugin()

current_loads = {
    'Zone1': {'residential': 100.0, 'commercial': 50.0, 'industrial': 10.0},
    'Zone2': {'residential': 150.0, 'commercial': 75.0, 'industrial': 15.0}
}

growth_rates = {
    'residential': 0.02,
    'commercial': 0.015,
    'industrial': 0.01
}

forecast = plugin.run_basic_forecast(
    current_loads=current_loads,
    forecast_years=[2030, 2035, 2040, 2045, 2050],
    growth_rates=growth_rates
)
```

### Advanced Forecast

```python
demographic_data = {
    'Zone1': {
        2025: {'population': 10000, 'housing': 4000},
        2030: {'population': 11000, 'housing': 4500}
    }
}

consumption_rates = {
    'residential_per_dwelling': 0.15,
    'commercial_per_capita': 0.05,
    'industrial_per_capita': 0.02
}

forecast = plugin.run_advanced_forecast(
    current_loads=current_loads,
    base_year=2025,
    forecast_years=[2030, 2035, 2040],
    demographic_data=demographic_data,
    consumption_rates=consumption_rates
)
```

---

## Performance Notes

**Optimization:**
- Demographic interpolation cached
- Growth calculations vectorized
- Scenario comparison parallelizable

**Memory:**
- Stores full forecast time series
- Demographic data retained for interpolation
- Scenario data can be substantial

**Scalability:**
- Tested with 100+ zones
- 30+ year forecast horizons
- Multiple scenarios simultaneously
