# Debugging Guide: 0 Load Results in Demand File Analysis

## Overview
When the Demand File Analysis plugin returns 0 load, it's usually due to one of several data issues. The enhanced plugin now includes comprehensive audit features to help identify exactly where the problem occurs.

## Step-by-Step Debugging Process

### 1. **Enable Detailed Logging**
The plugin now has extensive logging. When you run it in QGIS, check the Python console or QGIS log for detailed audit information.

### 2. **Use the New Audit Dialog**
The plugin now shows an **Audit Results** dialog before the regular results dialog. This dialog has 4 tabs:

#### **Excel Audit Tab**
- **File exists**: Confirms the Excel file is accessible
- **Target sheet exists**: Checks for "Service Point List" sheet
- **Row count**: Number of service points found
- **Missing columns**: Lists any required columns that are missing
- **Date range**: Min/max install dates found
- **Use classes**: Unique values in Use Class column
- **Factor codes**: Unique values in Factor Code column
- **Sample data**: First row of data for inspection

#### **Layer Audit Tab**
- **Zone Layer**: Feature count, field names, sample attributes
- **Pipe Layer**: Feature count, field names, sample attributes

#### **Processing Steps Tab**
- Step-by-step load calculations with totals at each stage
- Shows exactly where load values are lost

#### **Zone Details Tab**
- For each zone: intersecting pipe count, sample pipe names, loads by class

### 3. **Common Issues and Solutions**

#### **Issue 1: Wrong Sheet Name**
**Symptoms**: "Target sheet exists: False"
**Solution**: Ensure Excel file has sheet named exactly "Service Point List"

#### **Issue 2: Missing Required Columns**
**Symptoms**: "Missing columns: ['Factor Code', 'Base Factor', ...]"
**Required columns**:
- Factor Code
- Base Factor  
- Heat Factor
- HUC 3-Year Peak Demand
- Install Date
- Use Class
- Distribution Pipe

**Solution**: Add missing columns to Excel file or rename existing columns

#### **Issue 3: Zero Load Calculations**
**Symptoms**: "Calculated total load: 0.00"
**Causes**:
- All Factor Codes are 0, 1, or 20 (these zero out Base Factor and Heat Factor)
- Base Factor and Heat Factor columns contain all zeros
- Load formula issues

**Check**:
```
Load = 1.07 * (Base Factor + 56.8 * Heat Factor) + HUC 3-Year Peak Demand
```

#### **Issue 4: Date Filtering Removes All Data**
**Symptoms**: "After date filter: 0 points"
**Causes**:
- Install Date column has incorrect format
- All install dates are after the current year filter
- Null/empty install dates

**Solution**: Check date format (should be parseable by pandas) and date values

#### **Issue 5: Use Class Filtering Issues**
**Symptoms**: All use classes show "0 points, load: 0.00"
**Causes**:
- Use Class values don't match expected patterns
- Expected patterns: containing 'RES' or 'APT' for residential, 'COM' for commercial, 'IND' for industrial

**Solution**: Verify Use Class values match expected patterns

#### **Issue 6: No Spatial Intersection**
**Symptoms**: "Zone X: Found 0 intersecting pipes"
**Causes**:
- Layer coordinate systems don't match
- Buffer size too small
- Pipe/zone layers don't actually overlap
- Field name issues for pipe identification

**Solutions**:
- Check layer CRS match in QGIS
- Increase buffer size in plugin
- Verify layers visually overlap
- Check pipe layer field names in audit

#### **Issue 7: Pipe Name Matching Issues**
**Symptoms**: "Found X intersecting pipes" but still 0 load
**Causes**:
- Distribution Pipe values in Excel don't match pipe layer attribute values
- Case sensitivity issues
- Extra spaces or formatting differences

**Solution**: Compare "Sample pipe names" from Zone Details with Distribution Pipe values in Excel

### 4. **Manual Data Validation Steps**

#### **Check Excel Data Structure**:
```python
import pandas as pd
df = pd.read_excel('your_file.xlsx', sheet_name='Service Point List')
print("Columns:", df.columns.tolist())
print("Row count:", len(df))
print("Sample data:")
print(df.head())
```

#### **Check Load Calculation Manually**:
```python
# Test load calculation on sample row
base_factor = 100
heat_factor = 50
huc_demand = 10
load = 1.07 * (base_factor + 56.8 * heat_factor) + huc_demand
print(f"Expected load: {load}")
```

#### **Check Date Filtering**:
```python
df['Install Date'] = pd.to_datetime(df['Install Date'], errors='coerce')
print("Date range:", df['Install Date'].min(), "to", df['Install Date'].max())
print("Null dates:", df['Install Date'].isna().sum())
```

### 5. **Field Name Mapping**
The plugin now tries multiple field names for identification:

**Zone Names**: name, Name, NAME, zone_name, Zone_Name, ZONE_NAME, id, ID, fid, FID, objectid, OBJECTID

**Pipe Names**: name, Name, NAME, pipe_name, Pipe_Name, PIPE_NAME, id, ID, fid, FID, objectid, OBJECTID, pipe_id, PIPE_ID, facilityid, FACILITYID, facility_id, FACILITY_ID

### 6. **Debugging Workflow**

1. **Run the analysis** and wait for the Audit Results dialog
2. **Check Excel Audit tab** - ensure file structure is correct
3. **Check Processing Steps tab** - identify where loads become 0
4. **If spatial analysis**: Check Layer Audit and Zone Details tabs
5. **Compare pipe names** between Excel and QGIS layer
6. **Verify coordinate systems** match between layers
7. **Check data values manually** using Python scripts above

### 7. **Quick Validation Checklist**

- [ ] Excel file has "Service Point List" sheet
- [ ] All required columns present with correct names
- [ ] Base Factor and Heat Factor have non-zero values
- [ ] Factor Codes aren't all 0, 1, or 20
- [ ] Install Dates are valid and before current year
- [ ] Use Class values contain RES/APT, COM, or IND
- [ ] Distribution Pipe values match pipe layer attributes
- [ ] Zone and pipe layers have overlapping geometries
- [ ] Coordinate systems match between layers

The audit dialog will show you exactly which of these conditions is failing!