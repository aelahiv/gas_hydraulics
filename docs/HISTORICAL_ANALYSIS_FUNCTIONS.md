# Historical Analysis Functions Reference

## Overview

The Historical Analysis module provides functions for analyzing historical gas load data, tracking growth patterns over time, and generating reports by zone and use class.

## Core Analysis Functions

### create_historical_load_plots

**Purpose:** Generate visual plots showing cumulative load growth over time for each zone.

**Parameters:**
- `demand_file` (str): Path to Excel demand file
- `zone_layer` (QgsVectorLayer): Polygon layer defining analysis zones
- `pipe_layer` (QgsVectorLayer): Line layer containing gas distribution pipes
- `start_year` (int): Beginning year of analysis period
- `end_year` (int): Ending year of analysis period
- `output_dir` (str, optional): Directory for saving plot images

**Returns:** bool - True if plots created successfully, False otherwise

**Process:**
1. Loads demand file and calculates service point loads
2. Filters to analysis period (includes all installations up to end year)
3. Handles missing/invalid installation dates through proportional distribution
4. Builds pipe-to-zone mapping to prevent double counting
5. Calculates cumulative loads by year for each zone and use class
6. Generates plots with growth rate annotations
7. Saves plots as PNG files in output directory

**Output:** PNG files named `{zone_name}_historical_loads.png`

**Notes:**
- Growth rates shown as raw GJ/day per year (not percentages)
- Cumulative loads include all service points installed up to each year
- Bad dates distributed proportionally based on valid date patterns

---

### export_detailed_csv

**Purpose:** Export historical load data in two CSV formats for analysis and forecasting.

**Parameters:**
- `demand_file` (str): Path to Excel demand file
- `zone_layer` (QgsVectorLayer): Polygon layer defining analysis zones
- `pipe_layer` (QgsVectorLayer): Line layer containing gas distribution pipes
- `start_year` (int): Beginning year of analysis period
- `end_year` (int): Ending year of analysis period
- `output_file` (str): Base path for output CSV files

**Returns:** bool - True if export successful, False otherwise

**Output Files:**
1. Detailed CSV (`{output_file}_detailed.csv`):
   - Columns: Polygon, Year, Category, Cumulative_Load_GJ, New_Load_This_Year_GJ
   - One row per zone/year/category combination
   - Complete time series data

2. Basic Forecast CSV (`{output_file}_basic_forecast_format.csv`):
   - Columns: Zone, Residential, Commercial, Industrial
   - One row per zone
   - Current year (end_year) loads only

**Process:**
1. Processes demand file with load calculations
2. Distributes bad date loads across years
3. Builds pipe-to-zone mapping (no double counting)
4. Iterates through years calculating cumulative and incremental loads
5. Writes both CSV formats

**Notes:**
- Detailed CSV suitable for time series analysis
- Basic forecast CSV matches format required by forecasting module
- All loads in GJ/day units

---

### analyze_historical_loads_by_period

**Purpose:** Aggregate historical loads into 5-year periods for trend analysis.

**Parameters:**
- `demand_file` (str): Path to Excel demand file
- `zone_layer` (QgsVectorLayer): Polygon layer defining analysis zones
- `pipe_layer` (QgsVectorLayer): Line layer containing gas distribution pipes
- `start_year` (int): Beginning year of analysis (must be multiple of 5)
- `end_year` (int): Ending year of analysis (must be multiple of 5)

**Returns:** dict - Nested dictionary with structure:
```python
{
    'period_name': {  # e.g., '2015-2020'
        'zone_name': {
            'residential': float,  # Load in GJ/day
            'commercial': float,
            'industrial': float
        }
    }
}
```

**Process:**
1. Validates year ranges (must be multiples of 5)
2. Generates 5-year period endpoints
3. Builds pipe-to-zone mapping
4. For each period end year, calculates cumulative loads
5. Returns period-based aggregation

**Notes:**
- Useful for identifying long-term trends
- Cumulative totals represent loads existing at end of each period
- Periods are inclusive of end year

---

### build_pipe_to_zone_mapping

**Purpose:** Create spatial assignment of pipes to zones based on maximum overlap, preventing double counting.

**Parameters:**
- `zone_layer` (QgsVectorLayer): Polygon layer defining zones
- `pipe_layer` (QgsVectorLayer): Line layer containing pipes

**Returns:** tuple - (pipe_to_zone, zone_to_pipes)
- `pipe_to_zone` (dict): Maps pipe_name (str) to zone_name (str)
- `zone_to_pipes` (dict): Maps zone_name (str) to set of pipe_names

**Algorithm:**
1. For each pipe in pipe_layer:
   - Identify all zones that intersect the pipe
   - Calculate intersection length with each zone
   - Assign pipe to zone with maximum intersection length
2. Build reverse mapping from zones to their assigned pipes
3. Log pipes that intersect multiple zones

**Benefits:**
- Prevents double counting when zones overlap
- Ensures each service point counted exactly once
- Zone totals correctly sum to overall total
- Handles boundary-crossing pipes automatically

**Notes:**
- Replaces deprecated `get_pipes_in_zone()` method
- Build mapping once and reuse for efficiency
- Logs detailed assignment decisions for audit

---

## Data Processing Functions

### calculate_load

**Purpose:** Apply gas load calculation formula to service point data.

**Parameters:**
- `df` (pd.DataFrame): Service point data
- `factor_codes_to_keep` (list, optional): Factor codes to include (default: ['0', '1', '20'])

**Returns:** pd.DataFrame - Input DataFrame with added 'Load' column

**Formula:**
```
Load (GJ/day) = 1.07 * (Base Factor + 56.8 * Heat Factor) + HUC 3-Year Peak Demand
```

**Process:**
1. Filters factor codes (zeros out excluded codes)
2. Fills missing HUC peak demand with 0
3. Applies load formula
4. Logs total calculated load

**Notes:**
- Only factor codes 0, 1, and 20 included by default
- Other factor codes have Base Factor and Heat Factor set to 0
- Preserves all other DataFrame columns

---

### filter_by_use_class

**Purpose:** Categorize service points into residential, commercial, and industrial classes.

**Parameters:**
- `df` (pd.DataFrame): Service point data with 'Use Class' column

**Returns:** dict - Three DataFrames by category:
```python
{
    'residential': pd.DataFrame,  # APT, RES, HAPT, MAPT, MRES
    'commercial': pd.DataFrame,   # COMM, HCOM, MCOM
    'industrial': pd.DataFrame    # IND
}
```

**Classification:**
- Residential: APT, RES, HAPT, MAPT, MRES
- Commercial: COMM, HCOM, MCOM
- Industrial: IND

**Notes:**
- Service points not matching any category are excluded
- Logs count and load for each category
- Preserves all columns in filtered DataFrames

---

### handle_bad_dates

**Purpose:** Distribute service points with invalid dates across years based on valid date patterns.

**Parameters:**
- `df` (pd.DataFrame): Service point data
- `date_column` (str): Name of date column (default: 'Install Date')

**Returns:** pd.DataFrame - Data with bad dates distributed

**Process:**
1. Separates records with valid vs. invalid dates
2. Calculates year-by-year load distribution from valid dates
3. Computes proportion of load in each year
4. Creates duplicate records for bad date points (one per year)
5. Distributes load proportionally across years
6. Assigns synthetic mid-year dates (June 15)
7. Combines with valid date records

**Notes:**
- Preserves total load (sum of bad date loads unchanged)
- Provides temporal distribution for analysis
- Logs bad date count and affected load

---

## Utility Functions

### get_5year_periods

**Purpose:** Generate list of 5-year period endpoints for aggregation.

**Parameters:**
- `start_year` (int): Beginning year (must be multiple of 5)
- `end_year` (int): Ending year (must be multiple of 5)

**Returns:** list - Period endpoint years [start+4, start+9, ..., end]

**Example:**
```python
get_5year_periods(2000, 2020)
# Returns: [2004, 2009, 2014, 2019]
```

**Notes:**
- Validates years are multiples of 5
- Period names formed as "{start}-{end}" (e.g., "2000-2004")
- Used by period-based aggregation functions

---

### _get_zone_name

**Purpose:** Extract zone name from feature attributes with flexible field matching.

**Parameters:**
- `zone_feature` (QgsFeature): Zone feature to extract name from

**Returns:** str - Zone name or generated fallback

**Field Priority:**
1. 'Name', 'name', 'NAME'
2. 'zone_name', 'Zone_Name', 'ZONE_NAME'
3. 'zone', 'Zone', 'ZONE'
4. 'area_name', 'Area_Name', 'AREA_NAME'
5. 'id', 'ID', 'fid', 'FID', 'objectid', 'OBJECTID'
6. Fallback: 'Zone_{feature_id}'

**Notes:**
- Case-insensitive field matching
- Prioritizes descriptive names over IDs
- Logs which field was used for transparency

---

## Deprecated Functions

### get_pipes_in_zone

**Status:** DEPRECATED - Use `build_pipe_to_zone_mapping()` instead

**Purpose:** Find pipes intersecting a zone using buffered geometry.

**Warning:** May cause double counting if pipes intersect multiple zones.

**Parameters:**
- `zone_feature` (QgsFeature): Zone to check
- `pipe_layer` (QgsVectorLayer): Pipe layer
- `buffer_pixels` (int): Buffer size in pixels (default: 10)

**Returns:** list - Pipe names intersecting zone

**Notes:**
- Logs deprecation warning when called
- Still functional for backward compatibility
- Recommend migration to new mapping approach

---

## Data Validation

### Input Requirements

**Demand File:**
- Format: Excel (.xlsx, .xls)
- Sheet: "Service Point List"
- Required columns:
  - Distribution Pipe
  - Install Date
  - Use Class
  - Factor Code
  - Base Factor
  - Heat Factor
  - HUC 3-Year Peak Demand

**Zone Layer:**
- Type: Polygon
- Name field: One of Name, zone_name, zone, id, etc.
- Valid geometries required

**Pipe Layer:**
- Type: LineString
- Name field: Must match 'Distribution Pipe' in demand file
- Valid geometries required

### Common Issues

**Missing Loads:**
- Check pipe names match between demand file and spatial layer
- Verify coordinate systems match
- Ensure factor codes set correctly

**Incorrect Totals:**
- Confirm bad date distribution is appropriate
- Check year range includes all relevant data
- Verify no duplicate service points in demand file

**Slow Performance:**
- Use spatial indexing for large datasets
- Filter demand file before processing
- Simplify zone boundaries if complex

---

## Logging

All functions provide detailed logging:
- Processing steps and record counts
- Spatial join results and pipe assignments
- Load calculations and totals
- Warnings for data quality issues
- Error messages with context

Logs accessible via QGIS message log panel.

---

## Examples

### Generate Historical Plots

```python
from gas_hydraulics.historical_analysis import HistoricalAnalysisPlugin

plugin = HistoricalAnalysisPlugin()
success = plugin.create_historical_load_plots(
    demand_file='path/to/demand.xlsx',
    zone_layer=zone_layer,
    pipe_layer=pipe_layer,
    start_year=2000,
    end_year=2025,
    output_dir='path/to/output'
)
```

### Export Detailed CSV

```python
success = plugin.export_detailed_csv(
    demand_file='path/to/demand.xlsx',
    zone_layer=zone_layer,
    pipe_layer=pipe_layer,
    start_year=2000,
    end_year=2025,
    output_file='path/to/output/historical'
)
# Creates:
# - historical_detailed.csv
# - historical_basic_forecast_format.csv
```

### Period-Based Analysis

```python
results = plugin.analyze_historical_loads_by_period(
    demand_file='path/to/demand.xlsx',
    zone_layer=zone_layer,
    pipe_layer=pipe_layer,
    start_year=2000,
    end_year=2020
)

for period, zones in results.items():
    print(f"Period: {period}")
    for zone, loads in zones.items():
        total = sum(loads.values())
        print(f"  {zone}: {total:.2f} GJ/day")
```

---

## Performance Notes

**Optimization:**
- Pipe-to-zone mapping cached and reused
- DataFrame operations vectorized where possible
- Spatial indexing used automatically for large datasets

**Memory:**
- Demand file loaded into memory as DataFrame
- Spatial layers accessed through QGIS (disk-based)
- Plot generation may require significant memory for large datasets

**Scalability:**
- Tested with 10,000+ service points
- Hundreds of zones
- 40+ year time series
