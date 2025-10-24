# Demand File Analysis Functions Reference

## Overview

The Demand File Analysis module processes Excel demand files containing service point data, calculates gas loads, and performs spatial analysis to determine current load distribution by zone and use class.

## Core Analysis Functions

### analyze_demand_file

**Purpose:** Primary function to process demand file and generate load analysis results by zone.

**Parameters:**
- `excel_file` (str): Path to Excel demand file
- `zone_layer` (QgsVectorLayer, optional): Polygon layer defining analysis zones
- `pipe_layer` (QgsVectorLayer, optional): Line layer containing gas distribution pipes

**Returns:** dict - Load results by zone and use class:
```python
{
    'zone_name': {
        'residential': float,  # Load in GJ/day
        'commercial': float,
        'industrial': float
    }
}
```

**Process:**
1. Loads Excel file and validates required columns
2. Calculates load for each service point using formula
3. Filters by valid factor codes (0, 1, 20)
4. Categorizes service points by use class
5. If spatial layers provided:
   - Builds pipe-to-zone mapping to prevent double counting
   - Assigns service points to zones based on distribution pipe
   - Aggregates loads by zone and use class
6. If no spatial layers:
   - Returns overall totals by use class
7. Stores audit results for debugging

**Notes:**
- All service points included (no date filtering)
- Spatial assignment prevents double counting
- Results include detailed audit trail

---

### calculate_load

**Purpose:** Apply gas load calculation formula to service point DataFrame.

**Parameters:**
- `df` (pd.DataFrame): Service point data
- `factor_codes_to_keep` (list, optional): Factor codes to include (default: ['0', '1', '20'])

**Returns:** pd.DataFrame - Input DataFrame with added 'Load' column

**Formula:**
```
Load (GJ/day) = 1.07 * (Base Factor + 56.8 * Heat Factor) + HUC 3-Year Peak Demand
```

**Process:**
1. Converts factor codes to string for consistent comparison
2. Creates mask for factor codes NOT in keep list
3. Sets Base Factor and Heat Factor to 0 for excluded codes
4. Fills missing HUC 3-Year Peak Demand values with 0
5. Applies load formula to all rows
6. Logs total calculated load and count

**Constants:**
- Load multiplier: 1.07
- Heat factor multiplier: 56.8

**Notes:**
- Factor code filtering prevents incorrect load calculations
- Missing peak demand treated as zero (not null)
- Original DataFrame modified in place

---

### filter_by_use_class

**Purpose:** Separate service points into residential, commercial, and industrial categories.

**Parameters:**
- `df` (pd.DataFrame): Service point data with 'Use Class' column

**Returns:** dict - Three filtered DataFrames:
```python
{
    'residential': pd.DataFrame,  # Residential service points
    'commercial': pd.DataFrame,   # Commercial service points
    'industrial': pd.DataFrame    # Industrial service points
}
```

**Use Class Mapping:**
- Residential: APT, RES, HAPT, MAPT, MRES
- Commercial: COMM, HCOM, MCOM
- Industrial: IND

**Process:**
1. Creates boolean masks for each category
2. Filters DataFrame for each use class group
3. Logs count and total load for each category
4. Returns dictionary of filtered DataFrames

**Notes:**
- Case-sensitive use class matching
- Service points not matching any category excluded
- Maintains all columns in filtered results

---

### build_pipe_to_zone_mapping

**Purpose:** Create spatial assignment of pipes to zones preventing double counting.

**Parameters:**
- `zone_layer` (QgsVectorLayer): Polygon layer with zones
- `pipe_layer` (QgsVectorLayer): Line layer with pipes

**Returns:** tuple - (pipe_to_zone, zone_to_pipes)
- `pipe_to_zone` (dict): Maps pipe_name to zone_name
- `zone_to_pipes` (dict): Maps zone_name to set of pipe_names

**Algorithm:**
1. Iterate through all pipes in pipe layer
2. For each pipe:
   - Find all intersecting zones
   - Calculate intersection length with each zone
   - Assign to zone with maximum intersection
   - Log multi-zone intersections
3. Build reverse mapping: zone to pipes
4. Return both mappings

**Field Priority for Pipe Names:**
1. FacNam1005 (primary)
2. name, Name, NAME
3. pipe_name, Pipe_Name, PIPE_NAME
4. Fallback: Pipe_{feature_id}

**Field Priority for Zone Names:**
1. Name, name, NAME
2. zone_name, Zone_Name, ZONE_NAME
3. zone, Zone, ZONE
4. area_name, Area_Name, AREA_NAME
5. id, ID, fid, FID, objectid, OBJECTID
6. Fallback: Zone_{feature_id}

**Benefits:**
- Each pipe assigned to exactly one zone
- Prevents inflated totals from double counting
- Handles overlapping zones automatically
- Detailed logging of assignment decisions

---

## Data Processing Functions

### load_demand_file

**Purpose:** Read Excel demand file and validate structure.

**Parameters:**
- `excel_file` (str): Path to Excel file

**Returns:** pd.DataFrame - Service point data

**Validation:**
- File exists and readable
- Contains "Service Point List" sheet
- Required columns present:
  - Distribution Pipe
  - Use Class
  - Factor Code
  - Base Factor
  - Heat Factor
  - HUC 3-Year Peak Demand

**Process:**
1. Checks file existence
2. Attempts to read "Service Point List" sheet
3. Validates required columns
4. Logs record count and columns found
5. Returns DataFrame if valid

**Error Handling:**
- File not found: Logs error, returns empty DataFrame
- Sheet missing: Logs error, returns empty DataFrame
- Column missing: Logs error, returns empty DataFrame

---

### preview_data

**Purpose:** Display sample of demand file data for verification.

**Parameters:**
- `df` (pd.DataFrame): Service point data
- `n_rows` (int, optional): Number of rows to display (default: 10)

**Returns:** pd.DataFrame - First n rows of data

**Display Columns:**
- Distribution Pipe
- Use Class
- Factor Code
- Base Factor
- Heat Factor
- HUC 3-Year Peak Demand
- Load (if calculated)

**Notes:**
- Used for data verification before processing
- Helps identify data quality issues
- Truncates display for readability

---

## Spatial Analysis Functions

### get_pipes_in_zone (DEPRECATED)

**Status:** DEPRECATED - Use `build_pipe_to_zone_mapping()` instead

**Purpose:** Find pipes intersecting a zone using buffered geometry.

**Warning:** May cause double counting when pipes intersect multiple zones.

**Parameters:**
- `zone_feature` (QgsFeature): Zone feature to query
- `pipe_layer` (QgsVectorLayer): Pipe layer
- `buffer_pixels` (int, optional): Buffer size in pixels (default: 10)

**Returns:** list - Pipe names intersecting buffered zone

**Process:**
1. Gets zone geometry
2. Creates buffer around zone (buffer_pixels * 0.1 map units)
3. Checks each pipe for intersection with buffered zone
4. Collects pipe names that intersect
5. Logs spatial join statistics

**Notes:**
- Logs deprecation warning when called
- Replaced by `build_pipe_to_zone_mapping()`
- Still functional for backward compatibility

---

## Validation Functions

### validate_demand_file

**Purpose:** Comprehensive validation of demand file data quality.

**Parameters:**
- `df` (pd.DataFrame): Service point data

**Returns:** dict - Validation results:
```python
{
    'valid': bool,
    'errors': list,
    'warnings': list,
    'statistics': dict
}
```

**Checks:**
1. Required columns present
2. Numeric fields contain valid numbers
3. Factor codes are valid values
4. Use classes are recognized
5. Distribution pipes are not null
6. Date fields parseable (if present)

**Statistics:**
- Total records
- Valid vs. invalid counts per field
- Use class distribution
- Factor code distribution
- Load range and distribution

**Notes:**
- Non-blocking warnings for data quality
- Blocking errors prevent processing
- Detailed field-level diagnostics

---

## Audit and Debugging Functions

### get_audit_results

**Purpose:** Retrieve detailed audit trail from last analysis.

**Parameters:** None

**Returns:** dict - Audit information:
```python
{
    'excel_file': str,
    'total_records': int,
    'total_load': float,
    'factor_code_filtering': dict,
    'use_class_breakdown': dict,
    'processing_steps': list,
    'zone_details': list,
    'spatial_join_results': dict
}
```

**Information Captured:**
- Input file path
- Record counts at each processing step
- Load calculations and totals
- Spatial join statistics
- Zone-by-zone results
- Processing timestamps

**Usage:**
- Debugging data mismatches
- Understanding processing flow
- Generating reports
- Quality assurance

---

### export_audit_report

**Purpose:** Export audit trail to text file for documentation.

**Parameters:**
- `output_file` (str): Path for audit report file

**Returns:** bool - True if export successful

**Report Sections:**
1. File Information
2. Processing Steps
3. Load Calculations
4. Spatial Join Results
5. Zone Details
6. Data Quality Notes

**Format:** Plain text with sections and indentation

---

## Utility Functions

### check_coordinate_systems

**Purpose:** Verify zone and pipe layers use compatible coordinate systems.

**Parameters:**
- `zone_layer` (QgsVectorLayer): Zone layer
- `pipe_layer` (QgsVectorLayer): Pipe layer

**Returns:** dict - Coordinate system information:
```python
{
    'compatible': bool,
    'zone_crs': str,
    'pipe_crs': str,
    'recommendation': str
}
```

**Process:**
1. Gets CRS from each layer
2. Compares CRS definitions
3. Provides recommendation if mismatch

**Notes:**
- Mismatched CRS causes incorrect spatial joins
- Recommend reprojecting one layer to match other
- Critical check before spatial analysis

---

### get_field_names

**Purpose:** Extract available field names from layer for mapping.

**Parameters:**
- `layer` (QgsVectorLayer): Layer to examine

**Returns:** list - Field names in layer

**Usage:**
- Identifying available fields for name mapping
- Debugging field matching issues
- Dynamic field selection

---

## Integration Functions

### import_from_csv

**Purpose:** Import demand data from CSV file as alternative to Excel.

**Parameters:**
- `csv_file` (str): Path to CSV file

**Returns:** pd.DataFrame - Service point data

**Requirements:**
- CSV must contain same columns as Excel format
- First row treated as header
- Proper encoding for special characters

**Notes:**
- Alternative input format
- Same validation applied as Excel
- May require column name mapping

---

### export_results_to_csv

**Purpose:** Export analysis results to CSV for reporting.

**Parameters:**
- `results` (dict): Load results by zone
- `output_file` (str): Path for CSV file

**Output Format:**
```
Zone,Residential,Commercial,Industrial,Total
Zone1,123.45,45.67,0.00,169.12
Zone2,234.56,12.34,5.67,252.57
```

**Returns:** bool - True if export successful

**Notes:**
- Single row per zone
- Loads in GJ/day
- Total column calculated

---

## Data Quality Functions

### identify_orphan_pipes

**Purpose:** Find pipes in demand file not present in spatial layer.

**Parameters:**
- `df` (pd.DataFrame): Service point data
- `pipe_layer` (QgsVectorLayer): Spatial pipe layer

**Returns:** dict - Orphan pipe information:
```python
{
    'orphan_pipes': list,
    'orphan_count': int,
    'orphan_load': float,
    'service_points_affected': int
}
```

**Process:**
1. Gets unique pipes from demand file
2. Gets unique pipes from spatial layer
3. Identifies pipes in demand but not spatial
4. Calculates load on orphan pipes

**Notes:**
- Common cause of missing loads in zones
- Indicates data synchronization issues
- May require demand file or spatial layer update

---

### check_duplicate_service_points

**Purpose:** Identify potential duplicate service point records.

**Parameters:**
- `df` (pd.DataFrame): Service point data

**Returns:** pd.DataFrame - Suspected duplicates

**Criteria:**
- Same Distribution Pipe
- Same Use Class
- Same Base Factor and Heat Factor
- Same or similar HUC Peak Demand

**Notes:**
- Duplicates inflate load calculations
- May indicate data entry errors
- Review suspected duplicates manually

---

## Configuration

### Default Settings

```python
LOAD_MULTIPLIER = 1.07
HEAT_FACTOR_MULTIPLIER = 56.8
DEFAULT_FACTOR_CODES = ['0', '1', '20']
BUFFER_PIXELS = 10

RESIDENTIAL_CLASSES = ['APT', 'RES', 'HAPT', 'MAPT', 'MRES']
COMMERCIAL_CLASSES = ['COMM', 'HCOM', 'MCOM']
INDUSTRIAL_CLASSES = ['IND']
```

### Customization

Settings can be modified by passing parameters to functions or editing module constants.

---

## Examples

### Basic Analysis

```python
from gas_hydraulics.demand_file_analysis import DemandFileAnalysis

analyzer = DemandFileAnalysis()
results = analyzer.analyze_demand_file(
    excel_file='demand.xlsx',
    zone_layer=zone_layer,
    pipe_layer=pipe_layer
)

for zone, loads in results.items():
    total = sum(loads.values())
    print(f"{zone}: {total:.2f} GJ/day")
```

### Load Calculation Only

```python
import pandas as pd
from gas_hydraulics.demand_file_analysis import DemandFileAnalysis

analyzer = DemandFileAnalysis()
df = pd.read_excel('demand.xlsx', sheet_name='Service Point List')
df = analyzer.calculate_load(df)

print(f"Total Load: {df['Load'].sum():.2f} GJ/day")
```

### Custom Factor Codes

```python
df = analyzer.calculate_load(
    df,
    factor_codes_to_keep=['0', '1', '5', '20']
)
```

### Audit Review

```python
results = analyzer.analyze_demand_file(
    excel_file='demand.xlsx',
    zone_layer=zone_layer,
    pipe_layer=pipe_layer
)

audit = analyzer.get_audit_results()
print(f"Total Records: {audit['total_records']}")
print(f"Total Load: {audit['total_load']:.2f} GJ/day")

for zone in audit['zone_details']:
    print(f"{zone['name']}: {zone['pipe_count']} pipes")
```

---

## Performance Notes

**Optimization:**
- Spatial indexing for large datasets
- Pipe-to-zone mapping cached
- Vectorized DataFrame operations

**Memory:**
- Full demand file loaded into memory
- Spatial layers accessed on-demand
- Results stored in memory

**Scalability:**
- Tested with 10,000+ service points
- Hundreds of zones
- Complex spatial geometries
