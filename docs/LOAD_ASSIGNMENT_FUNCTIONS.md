# Load Assignment Functions Reference

## Overview

The Load Assignment module spatially assigns gas loads to network infrastructure for hydraulic modeling. It supports two primary modes:

1. **CSV-Based Forecast Assignment**: Assigns forecasted loads from CSV files to pipe infrastructure using spatial intersection and temporal windowing
2. **Point-to-Node Assignment**: Assigns loads from service points or zones to network nodes using distance-based methods

## CSV-Based Forecast Load Assignment

### assign_loads_to_pipes (from CSV forecast)

**Purpose:** Assign forecasted loads from a CSV file to pipe infrastructure based on spatial intersection with polygons and construction year timing.

**Key Features:**
- **Spatial Assignment**: Uses maximum overlap intersection between pipes and polygons
- **Temporal Windowing**: Applies 5-year aggregation windows (e.g., 2030 pipes receive sum of 2026-2030)
- **Cumulative Conversion**: Automatically converts cumulative forecast totals to per-year increments
- **Complete Coverage**: All pipes receive year, polygon name, and status (even with load = 0)
- **Priority Handling**: Respects zero-priority zones - excess demand deferred to future years

**Process Flow:**
1. **Parse CSV**: Read forecast data and detect format (cumulative vs incremental)
2. **Convert to Increments**: Transform cumulative totals to per-year increments (baseline year = 0)
3. **Normalize Area Names**: Match polygon names to CSV headers (case-insensitive, ignores parentheses)
4. **Spatial Assignment**: 
   - Build spatial index for polygons
   - Calculate intersection length for each pipe-polygon pair
   - Assign pipe to polygon with maximum overlap
5. **Group Pipes**: Organize by area and construction year
6. **Calculate Window Loads**:
   - For pipe year Y with start year S and window W:
   - Window start = S + ((Y - S) // W - 1) * W + 1 (skip baseline)
   - Window end = Y (inclusive)
   - Load = Sum of increments in window
7. **Distribute Loads**: Divide window total equally among pipes in group
8. **Update Attributes**: Set LOAD, DESC, PROP, YEAR, DATETIME, SYN_DATE for all pipes

**Parameters:**
- `csv_file_path` (str): Path to forecast CSV file
- `polygon_layer` (QgsVectorLayer): Area/zone polygons
- `pipe_layer` (QgsVectorLayer): Pipe infrastructure
- `start_year` (int): Forecast baseline year (default: 2025)
- `aggregation_window_years` (int): Window size (default: 5)
- `pipe_year_field` (str): Field with pipe construction year (default: 'Year')
- `polygon_name_field` (str): Field with polygon identifier (default: 'id')

**CSV Format:**
```csv
Load Forecast Results
Generated: 2025-10-23

SUBZONE: NBHD 1A (Quarry Ridge)

Year,DateTime,Residential (GJ/d),Commercial (GJ/d),Industrial (GJ/d),Total (GJ/d)
2025,2025-11-11,630.00,8.51,0.00,638.51
2026,2026-11-11,862.49,55.00,0.00,917.49
2027,2027-11-11,1050.72,55.00,0.00,1105.72
...

SUBZONE: NBHD 1B
...
```

**Output Fields Added/Updated:**
- `LOAD` (Double): Load value in GJ/d
- `DESC` (String): "{Polygon Name} - {Period} Year Load" or "{Polygon Name} - No Load"
- `PROP` (String): "Proposed" for all pipes
- `YEAR` (String): Construction year
- `DATETIME` (String): Human-readable date (1-Nov-YYYY)
- `SYN_DATE` (Integer): Synergi format date (YYYYMMDD)

**Window Calculation Examples:**
```
Start Year = 2025, Window = 5

Pipe Year 2030:
  Window = 2026-2030 (5 years)
  Load = increment_2026 + increment_2027 + increment_2028 + increment_2029 + increment_2030

Pipe Year 2035:
  Window = 2031-2035 (5 years)
  Load = increment_2031 + increment_2032 + increment_2033 + increment_2034 + increment_2035

Pipe Year 2045:
  Window = 2041-2045 (5 years)
  Load = increment_2041 + increment_2042 + increment_2043 + increment_2044 + increment_2045
```

**Special Cases:**
- **No Polygon Intersection**: Pipe receives "Unknown Area" description
- **No Load in Window**: Pipe receives 0 load but keeps year and polygon info
- **No Year Field**: Pipe receives empty year fields and "(No Year)" in description
- **Baseline Year (2025)**: Increment = 0 (existing load, not added)

**Returns:** None (modifies pipe_layer in place)

**Example:**
```python
from gas_hydraulics.load_assignment import LoadAssignmentTool

tool = LoadAssignmentTool(iface)
tool.run_load_assignment(
    csv_file="forecast_2025.csv",
    polygon_layer=zones_layer,
    pipe_layer=pipes_layer,
    start_year=2025,
    aggregation_window=5
)
```

---

## Node-Based Load Assignment

The Load Assignment module also spatially assigns calculated gas loads to network nodes for hydraulic modeling. Supports distance-based assignment, multiple node allocation, and export to hydraulic model formats.

## Core Assignment Functions

### assign_loads_to_nodes

**Purpose:** Assign gas loads from service points or zones to network nodes using spatial proximity.

**Parameters:**
- `load_data` (dict or pd.DataFrame): Load information by location
- `node_layer` (QgsVectorLayer): Network node locations
- `assignment_method` (str, optional): 'nearest', 'weighted', or 'multiple' (default: 'nearest')
- `distance_threshold` (float, optional): Maximum assignment distance in map units (default: 100)
- `coordinate_system` (str, optional): CRS for distance calculations

**Load Data Formats:**

Dictionary format:
```python
{
    'zone_name': {
        'residential': float,
        'commercial': float,
        'industrial': float
    }
}
```

DataFrame format:
```python
pd.DataFrame({
    'Node_ID': [1, 2, 3],
    'Load': [123.45, 234.56, 345.67],
    'X': [longitude],
    'Y': [latitude]
})
```

**Returns:** dict - Node assignments:
```python
{
    'node_id': {
        'total_load': float,
        'residential': float,
        'commercial': float,
        'industrial': float,
        'assigned_from': list,  # Source zones/service points
        'distance': float,
        'confidence': float
    }
}
```

**Assignment Methods:**

1. **Nearest:** Assigns full load to closest node
   - Simple and fast
   - May create unbalanced assignments

2. **Weighted:** Distributes load based on inverse distance
   - More balanced distribution
   - Considers multiple nearby nodes
   - Formula: weight = 1 / (distance^power)

3. **Multiple:** Assigns to all nodes within threshold
   - Comprehensive coverage
   - May result in load duplication
   - Requires normalization

**Process:**
1. Extracts node locations from layer
2. For each load location:
   - Calculates distance to all nodes
   - Applies assignment method
   - Allocates load proportionally
3. Aggregates loads by node
4. Validates total load conservation
5. Returns node assignments

**Notes:**
- Coordinate systems must match
- Distance threshold critical for accuracy
- Validates load conservation within tolerance

---

### assign_service_points_to_nodes

**Purpose:** Direct assignment of individual service points to nearest nodes.

**Parameters:**
- `service_points` (pd.DataFrame): Service point data with coordinates
- `node_layer` (QgsVectorLayer): Network nodes
- `max_distance` (float, optional): Maximum assignment distance (default: 50)

**Required DataFrame Columns:**
- X or Longitude: X coordinate
- Y or Latitude: Y coordinate
- Load: Load value in GJ/day
- Use Class: Service point category

**Returns:** pd.DataFrame - Service points with assigned nodes:
```python
pd.DataFrame({
    'Service_Point_ID': [1, 2, 3],
    'Assigned_Node': [101, 102, 101],
    'Distance': [10.5, 15.2, 8.3],
    'Load': [1.23, 2.34, 1.45],
    'Use_Class': ['RES', 'COMM', 'RES']
})
```

**Process:**
1. Validates service point coordinates
2. Builds spatial index for nodes
3. For each service point:
   - Finds nearest node using spatial index
   - Calculates distance
   - Assigns if within max_distance
4. Logs unassigned service points
5. Returns DataFrame with assignments

**Notes:**
- Faster than zone-based methods for large datasets
- Provides detailed assignment audit trail
- Identifies orphaned service points beyond threshold

---

### aggregate_loads_by_node

**Purpose:** Sum loads assigned to each node from multiple sources.

**Parameters:**
- `assignments` (list): List of assignment dictionaries or DataFrames
- `aggregation_method` (str, optional): 'sum', 'average', or 'weighted_average'

**Returns:** dict - Aggregated loads by node:
```python
{
    'node_id': {
        'total_load': float,
        'residential': float,
        'commercial': float,
        'industrial': float,
        'source_count': int,
        'sources': list
    }
}
```

**Aggregation Methods:**

1. **Sum:** Add all assigned loads
   - Default behavior
   - Appropriate when sources don't overlap

2. **Average:** Mean of assigned loads
   - Used when multiple estimates available
   - Reduces impact of outliers

3. **Weighted Average:** Average weighted by confidence or source quality
   - Most sophisticated approach
   - Requires confidence scores

**Process:**
1. Collects all assignments for each node
2. Applies aggregation method
3. Maintains use class breakdown
4. Tracks source information
5. Validates consistency

**Notes:**
- Critical for combining multiple data sources
- Prevents double counting
- Maintains load conservation

---

## Spatial Analysis Functions

### calculate_node_distances

**Purpose:** Compute distances between load locations and nodes.

**Parameters:**
- `load_locations` (list): List of (x, y) tuples for load sources
- `node_locations` (dict): Node IDs mapped to (x, y) coordinates
- `method` (str, optional): 'euclidean', 'manhattan', or 'geodesic'

**Returns:** dict - Distance matrix:
```python
{
    'load_location_index': {
        'node_id': float  # Distance in map units
    }
}
```

**Distance Methods:**

1. **Euclidean:** Straight-line distance
   - Formula: sqrt((x2-x1)^2 + (y2-y1)^2)
   - Appropriate for projected coordinate systems

2. **Manhattan:** Sum of absolute differences
   - Formula: |x2-x1| + |y2-y1|
   - Useful for grid-based networks

3. **Geodesic:** Great circle distance
   - Uses haversine formula
   - Required for geographic coordinates (lat/lon)

**Notes:**
- Coordinate system affects distance calculation
- Use geodesic for unprojected data
- Distance units match coordinate system units

---

### find_nearest_nodes

**Purpose:** Identify N nearest nodes to each load location.

**Parameters:**
- `load_location` (tuple): (x, y) coordinate
- `node_layer` (QgsVectorLayer): Network nodes
- `n` (int, optional): Number of nearest nodes to return (default: 1)
- `max_distance` (float, optional): Maximum search distance

**Returns:** list - Nearest nodes:
```python
[
    {'node_id': int, 'distance': float, 'coordinates': tuple},
    ...
]
```

**Process:**
1. Uses spatial index for efficiency
2. Queries nodes within search radius
3. Calculates distances to candidates
4. Sorts by distance
5. Returns top N within max_distance

**Notes:**
- Spatial indexing provides O(log n) performance
- Max distance prevents unreasonable assignments
- Returns empty list if no nodes within threshold

---

### validate_node_coverage

**Purpose:** Check if network nodes adequately cover load locations.

**Parameters:**
- `load_locations` (list): Load source coordinates
- `node_layer` (QgsVectorLayer): Network nodes
- `required_distance` (float): Maximum acceptable distance

**Returns:** dict - Coverage analysis:
```python
{
    'coverage_percent': float,
    'uncovered_locations': list,
    'coverage_gaps': list,  # Areas needing additional nodes
    'recommendations': list
}
```

**Process:**
1. For each load location:
   - Finds nearest node
   - Records distance
2. Identifies locations beyond required_distance
3. Clusters uncovered locations
4. Generates recommendations for node additions

**Notes:**
- Quality assurance for network adequacy
- Identifies infrastructure gaps
- Supports network planning

---

## Distance-Based Functions

### calculate_inverse_distance_weights

**Purpose:** Compute weights for load distribution based on inverse distance.

**Parameters:**
- `distances` (dict): Node IDs mapped to distances
- `power` (float, optional): Inverse distance power (default: 2)
- `normalize` (bool, optional): Scale weights to sum to 1.0 (default: True)

**Returns:** dict - Normalized weights:
```python
{
    'node_id': float  # Weight between 0 and 1
}
```

**Formula:**
```
weight_i = (1 / distance_i^power) / sum(1 / distance_j^power for all j)
```

**Power Parameter Effects:**
- power = 1: Linear inverse relationship
- power = 2: Inverse square (default, gives preference to closer nodes)
- power > 2: Strong preference for nearest nodes

**Notes:**
- Weights sum to 1.0 when normalized
- Handles zero distance (assigns weight = 1.0)
- Higher power concentrates load at nearest nodes

---

### apply_distance_decay

**Purpose:** Apply distance decay function to reduce load with distance.

**Parameters:**
- `load` (float): Original load value
- `distance` (float): Distance from source
- `decay_function` (str, optional): 'exponential', 'linear', or 'step'
- `decay_rate` (float, optional): Rate of decay

**Returns:** float - Adjusted load

**Decay Functions:**

1. **Exponential:** load * exp(-decay_rate * distance)
   - Smooth gradual decline
   - Most realistic for service areas

2. **Linear:** load * max(0, 1 - decay_rate * distance)
   - Constant decline rate
   - Simple to understand

3. **Step:** load if distance < threshold, else 0
   - All-or-nothing assignment
   - Clear service boundaries

**Notes:**
- Models service area effects
- Decay rate calibrated to local conditions
- Can represent market saturation with distance

---

## Export Functions

### export_to_synergi

**Purpose:** Export node load assignments in Synergi Gas format.

**Parameters:**
- `node_assignments` (dict): Load assignments by node
- `output_file` (str): Path for output file
- `model_info` (dict, optional): Additional model metadata

**Output Format:**
```
NODE,LOAD_GJ_DAY,RESIDENTIAL,COMMERCIAL,INDUSTRIAL
101,125.45,100.23,20.12,5.10
102,234.56,180.45,45.11,9.00
```

**Metadata Header:**
```
! Synergi Gas Load Assignment Export
! Generated: 2025-10-23
! Total Load: 1234.56 GJ/day
! Node Count: 150
```

**Returns:** bool - Export successful

**Process:**
1. Validates node assignments
2. Formats data per Synergi specifications
3. Writes header with metadata
4. Writes node data rows
5. Validates file written correctly

**Notes:**
- Compatible with Synergi Gas import
- Includes data validation
- Preserves use class breakdown

---

### export_to_csv

**Purpose:** Export node assignments to generic CSV format.

**Parameters:**
- `node_assignments` (dict): Load assignments
- `output_file` (str): Path for CSV file
- `include_details` (bool, optional): Include assignment details (default: False)

**Standard Format:**
```
Node_ID,Total_Load,Residential,Commercial,Industrial
101,125.45,100.23,20.12,5.10
```

**Detailed Format (if include_details=True):**
```
Node_ID,Total_Load,Residential,Commercial,Industrial,Source_Count,Sources,Avg_Distance
101,125.45,100.23,20.12,5.10,3,"Zone1,Zone2,Zone3",25.5
```

**Returns:** bool - Export successful

**Notes:**
- Flexible format for various applications
- Detailed mode for audit trail
- Compatible with spreadsheet software

---

### export_to_shapefile

**Purpose:** Create shapefile with node locations and assigned loads.

**Parameters:**
- `node_layer` (QgsVectorLayer): Original node layer
- `node_assignments` (dict): Load assignments
- `output_file` (str): Path for output shapefile

**Output Attributes:**
- Node_ID: Node identifier
- X, Y: Coordinates
- Total_Load: Total load in GJ/day
- Residential: Residential load
- Commercial: Commercial load
- Industrial: Industrial load
- Source_Count: Number of sources assigned
- Confidence: Assignment confidence score

**Returns:** bool - Export successful

**Process:**
1. Copies node layer structure
2. Adds load assignment fields
3. Populates attributes for each node
4. Writes shapefile with projection
5. Creates index file

**Notes:**
- Preserves spatial reference
- Suitable for GIS analysis
- Can be used for visualization

---

## Validation Functions

### validate_load_conservation

**Purpose:** Verify total assigned load matches input load.

**Parameters:**
- `input_loads` (dict): Original loads by source
- `assigned_loads` (dict): Loads assigned to nodes
- `tolerance` (float, optional): Acceptable difference percentage (default: 0.01)

**Returns:** dict - Validation results:
```python
{
    'conserved': bool,
    'input_total': float,
    'assigned_total': float,
    'difference': float,
    'difference_percent': float,
    'details': dict
}
```

**Process:**
1. Sums all input loads
2. Sums all assigned loads
3. Calculates difference
4. Compares to tolerance threshold
5. Identifies sources of discrepancy if present

**Notes:**
- Critical quality assurance check
- Small differences acceptable due to rounding
- Large differences indicate assignment errors

---

### check_node_load_capacity

**Purpose:** Validate assigned loads against node capacity constraints.

**Parameters:**
- `node_assignments` (dict): Assigned loads
- `node_capacities` (dict): Maximum capacity by node
- `utilization_threshold` (float, optional): Warning threshold (default: 0.8)

**Returns:** dict - Capacity analysis:
```python
{
    'within_capacity': bool,
    'overloaded_nodes': list,
    'high_utilization_nodes': list,
    'avg_utilization': float,
    'max_utilization': float
}
```

**Process:**
1. Compares assigned load to capacity for each node
2. Identifies overloaded nodes (load > capacity)
3. Flags high utilization (load > threshold * capacity)
4. Calculates utilization statistics
5. Generates recommendations

**Notes:**
- Prevents unrealistic assignments
- Supports capacity planning
- Identifies bottlenecks

---

### identify_unassigned_loads

**Purpose:** Find load sources not assigned to any node.

**Parameters:**
- `input_loads` (dict): All load sources
- `assignments` (dict): Node assignments with source tracking

**Returns:** dict - Unassigned load information:
```python
{
    'unassigned_count': int,
    'unassigned_load': float,
    'unassigned_sources': list,
    'reasons': dict  # Source -> reason mapping
}
```

**Reasons for Unassignment:**
- No nodes within distance threshold
- Invalid coordinates
- Zero or negative load
- Excluded by filters

**Notes:**
- Quality assurance for completeness
- Indicates data quality issues
- Supports troubleshooting

---

## Optimization Functions

### optimize_node_placement

**Purpose:** Suggest optimal locations for additional nodes to improve coverage.

**Parameters:**
- `unassigned_loads` (list): Loads not adequately covered
- `existing_nodes` (QgsVectorLayer): Current node network
- `constraints` (dict, optional): Placement constraints

**Returns:** list - Suggested node locations:
```python
[
    {
        'location': (x, y),
        'coverage': float,  # Load that would be covered
        'priority': int,
        'nearby_loads': list
    },
    ...
]
```

**Process:**
1. Clusters unassigned load locations
2. For each cluster:
   - Calculates centroid
   - Estimates coverage if node added
   - Applies placement constraints
3. Ranks suggestions by coverage and feasibility
4. Returns prioritized list

**Notes:**
- Supports network expansion planning
- Considers existing infrastructure
- Maximizes coverage per node added

---

### balance_node_loads

**Purpose:** Redistribute loads to balance utilization across nodes.

**Parameters:**
- `node_assignments` (dict): Current assignments
- `node_capacities` (dict): Node capacity limits
- `balance_method` (str, optional): 'equal', 'capacity_proportional'

**Returns:** dict - Rebalanced assignments

**Balance Methods:**

1. **Equal:** Distribute equally among candidate nodes
   - Simple and fair
   - May not account for capacity differences

2. **Capacity Proportional:** Distribute based on available capacity
   - More efficient utilization
   - Prevents overloading

**Process:**
1. Identifies overloaded and underutilized nodes
2. For overloaded nodes:
   - Identifies nearby alternative nodes
   - Redistributes load proportionally
3. Validates load conservation
4. Checks capacity constraints
5. Returns balanced assignments

**Notes:**
- Improves system efficiency
- Prevents bottlenecks
- May require network modifications

---

## Utility Functions

### build_spatial_index

**Purpose:** Create spatial index for efficient node queries.

**Parameters:**
- `node_layer` (QgsVectorLayer): Node layer to index

**Returns:** spatial index object

**Process:**
1. Extracts node geometries
2. Builds R-tree spatial index
3. Maps index IDs to node IDs
4. Returns index for queries

**Notes:**
- Dramatically improves performance for large datasets
- O(log n) query time vs O(n) without index
- Automatically used by nearest node functions

---

### get_node_attributes

**Purpose:** Extract node attributes from layer.

**Parameters:**
- `node_layer` (QgsVectorLayer): Node layer
- `attribute_names` (list, optional): Specific attributes to extract

**Returns:** dict - Node attributes:
```python
{
    'node_id': {
        'attribute1': value1,
        'attribute2': value2,
        ...
    }
}
```

**Common Attributes:**
- Node_ID: Unique identifier
- Elevation: Node elevation
- Pressure: Operating pressure
- Capacity: Maximum load capacity
- Type: Node type (customer, junction, etc.)

**Notes:**
- Flexible field name matching
- Case-insensitive
- Returns None for missing attributes

---

### create_assignment_map

**Purpose:** Generate visual representation of load assignments.

**Parameters:**
- `node_assignments` (dict): Load assignments
- `node_layer` (QgsVectorLayer): Node locations
- `output_file` (str, optional): Path for image file

**Output:** Visual map with:
- Nodes sized by total load
- Color-coded by use class breakdown
- Assignment lines from sources to nodes
- Legend and labels

**Returns:** bool - Map created successfully

**Notes:**
- Useful for visualization and verification
- Supports quality review
- Can be included in reports

---

## Examples

### Basic Assignment

```python
from gas_hydraulics.load_assignment import LoadAssignment

assigner = LoadAssignment()

loads = {
    'Zone1': {'residential': 100.0, 'commercial': 50.0, 'industrial': 10.0},
    'Zone2': {'residential': 150.0, 'commercial': 75.0, 'industrial': 15.0}
}

assignments = assigner.assign_loads_to_nodes(
    load_data=loads,
    node_layer=node_layer,
    assignment_method='weighted',
    distance_threshold=100.0
)
```

### Weighted Assignment

```python
assignments = assigner.assign_loads_to_nodes(
    load_data=loads,
    node_layer=node_layer,
    assignment_method='weighted',
    distance_threshold=200.0
)

# Validate conservation
validation = assigner.validate_load_conservation(
    input_loads=loads,
    assigned_loads=assignments,
    tolerance=0.01
)

print(f"Load conserved: {validation['conserved']}")
print(f"Difference: {validation['difference']:.2f} GJ/day")
```

### Export to Synergi

```python
success = assigner.export_to_synergi(
    node_assignments=assignments,
    output_file='path/to/output.txt',
    model_info={'project': 'ProjectName', 'date': '2025-10-23'}
)
```

---

## Performance Notes

**Optimization:**
- Spatial indexing critical for large networks
- Distance calculations vectorized where possible
- Assignment methods scale differently:
  - Nearest: O(n log m) with spatial index
  - Weighted: O(n * m) for n loads and m nodes within threshold
  - Multiple: O(n * m) but with higher constant factor

**Memory:**
- Node layer loaded on-demand
- Spatial index cached
- Assignment results stored in memory

**Scalability:**
- Tested with 1000+ nodes
- 10,000+ service points
- Multiple use class categories
