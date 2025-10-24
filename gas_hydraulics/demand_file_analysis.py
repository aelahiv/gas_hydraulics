"""Demand File Analysis Plugin for QGIS.

This plugin analyzes current year cumulative load breakdown per class and area.
It reads polygon layer (subzones), pipe layer, and Excel service point list.
Uses spatial intersection with 10-pixel buffer to find pipes per zone.
"""
from __future__ import annotations
import logging
from typing import Any, Optional, Dict, List
import pandas as pd
from pathlib import Path
import os

LOGGER = logging.getLogger(__name__)

# Import validation functions
try:
    from .data_validation import (
        DataValidator,
        validate_layer_data,
        show_validation_dialog,
        safe_int
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    LOGGER.warning("Data validation module not available")
    VALIDATION_AVAILABLE = False
    # Fallback safe_int
    def safe_int(value, default=None):
        try:
            return int(float(value)) if value is not None else default
        except (ValueError, TypeError):
            return default

class DemandFileAnalysisPlugin:
    def __init__(self, iface: Any = None):
        """Initialize the Demand File Analysis plugin."""
        self.iface = iface
        self.action = None
        
        # Default load calculation parameters
        self.load_multiplier = 1.07
        self.heat_factor_multiplier = 56.8
        
    def initGui(self):
        """Create GUI elements (only when running inside QGIS)."""
        try:
            from qgis.PyQt.QtWidgets import QAction
            self.action = QAction("Demand File Analysis", self.iface.mainWindow() if self.iface else None)
            self.action.triggered.connect(self.run)
            if self.iface:
                # Add to menu/toolbar in real implementation
                pass
        except Exception:
            LOGGER.debug("QGIS not available; skipping GUI setup")

    def unload(self):
        """Cleanup GUI items."""
        try:
            if self.action and self.iface:
                # Remove from menu/toolbar
                pass
        except Exception:
            LOGGER.debug("QGIS not available; skipping unload")

    def export_table_to_csv(self, table, default_filename="table_export"):
        """Export QTableWidget contents to CSV file."""
        try:
            from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
            
            # Get save file path
            file_path, _ = QFileDialog.getSaveFileName(
                self.iface.mainWindow() if self.iface else None,
                "Export Table to CSV",
                f"{default_filename}.csv",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Extract table data
            import csv
            
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write headers
                headers = []
                for col in range(table.columnCount()):
                    headers.append(table.horizontalHeaderItem(col).text())
                writer.writerow(headers)
                
                # Write data rows
                for row in range(table.rowCount()):
                    row_data = []
                    for col in range(table.columnCount()):
                        item = table.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)
            
            if self.iface:
                QMessageBox.information(
                    self.iface.mainWindow(),
                    "Export Successful",
                    f"Table exported to:\n{file_path}"
                )
            
            LOGGER.info(f"Table exported to CSV: {file_path}")
            
        except Exception as e:
            LOGGER.error(f"Error exporting table to CSV: {e}")
            if self.iface:
                from qgis.PyQt.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Export Error",
                    f"Failed to export table:\n{str(e)}"
                )

    def import_csv_to_table(self, table, title="Import CSV Data"):
        """Import CSV data into QTableWidget."""
        try:
            from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
            
            # Get CSV file path
            file_path, _ = QFileDialog.getOpenFileName(
                self.iface.mainWindow() if self.iface else None,
                title,
                "",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if not file_path:
                return False
            
            # Read CSV data
            import csv
            
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
            
            if not rows:
                if self.iface:
                    QMessageBox.warning(
                        self.iface.mainWindow(),
                        "Import Error",
                        "CSV file is empty."
                    )
                return False
            
            # Check if headers match
            csv_headers = rows[0]
            table_headers = []
            for col in range(table.columnCount()):
                header_item = table.horizontalHeaderItem(col)
                table_headers.append(header_item.text() if header_item else f"Column {col}")
            
            # Flexible header matching - allow partial matches
            header_mapping = {}
            for i, csv_header in enumerate(csv_headers):
                for j, table_header in enumerate(table_headers):
                    if csv_header.lower().strip() == table_header.lower().strip():
                        header_mapping[i] = j
                        break
            
            if not header_mapping:
                if self.iface:
                    QMessageBox.warning(
                        self.iface.mainWindow(),
                        "Import Error",
                        f"No matching headers found.\n\nCSV headers: {csv_headers}\nTable headers: {table_headers}"
                    )
                return False
            
            # Clear existing data and resize table
            data_rows = rows[1:]  # Skip header row
            table.setRowCount(len(data_rows))
            
            # Import data
            imported_rows = 0
            for row_idx, row_data in enumerate(data_rows):
                for csv_col, table_col in header_mapping.items():
                    if csv_col < len(row_data):
                        from qgis.PyQt.QtWidgets import QTableWidgetItem
                        table.setItem(row_idx, table_col, QTableWidgetItem(row_data[csv_col]))
                imported_rows += 1
            
            if self.iface:
                QMessageBox.information(
                    self.iface.mainWindow(),
                    "Import Successful",
                    f"Imported {imported_rows} rows from:\n{file_path}"
                )
            
            LOGGER.info(f"CSV data imported: {imported_rows} rows from {file_path}")
            return True
            
        except Exception as e:
            LOGGER.error(f"Error importing CSV to table: {e}")
            if self.iface:
                from qgis.PyQt.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Import Error",
                    f"Failed to import CSV:\n{str(e)}"
                )
            return False

    def calculate_load(self, df: pd.DataFrame, factor_codes_to_keep: List[str] = None) -> pd.DataFrame:
        """Calculate load for service points using the specified formula.
        
        Args:
            df: DataFrame with service point data
            factor_codes_to_keep: List of factor codes to keep Base Factor and Heat Factor (others get zeroed)
            
        Returns:
            DataFrame with calculated Load column
        """
        if factor_codes_to_keep is None:
            factor_codes_to_keep = ['0', '1', '20']  # Only these codes keep their factors
            
        df = df.copy()
        
        # Zero out Base Factor and Heat Factor for codes NOT in the keep list
        mask = ~df['Factor Code'].astype(str).isin(factor_codes_to_keep)
        df.loc[mask, 'Base Factor'] = 0
        df.loc[mask, 'Heat Factor'] = 0
        
        # Fill NaN values in HUC 3-Year Peak Demand with 0
        df['HUC 3-Year Peak Demand'] = df['HUC 3-Year Peak Demand'].fillna(0)
        
        # Calculate load using the formula
        df['Load'] = (self.load_multiplier * 
                     (df['Base Factor'] + self.heat_factor_multiplier * df['Heat Factor']) + 
                     df['HUC 3-Year Peak Demand'])
        
        return df

    def filter_by_use_class(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Filter DataFrame by use class categories.
        
        Returns:
            Dictionary with keys 'residential', 'industrial', 'commercial' and filtered DataFrames
        """
        df_res = df[df['Use Class'].str.contains('APT|RES', na=False)]
        df_ind = df[df['Use Class'].str.contains('IND', na=False)]
        df_comm = df[df['Use Class'].str.contains('COM', na=False)]
        return {
            'residential': df_res,
            'industrial': df_ind,
            'commercial': df_comm
        }

    def build_pipe_to_zone_mapping(self, zone_layer, pipe_layer):
        """
        Build a mapping of pipes to zones based on maximum overlap.
        Each pipe is assigned to exactly one zone (the one it overlaps most with).
        This prevents double counting when zones overlap or pipes cross zone boundaries.
        
        Args:
            zone_layer: The zone layer
            pipe_layer: The pipe layer
            
        Returns:
            Tuple of (pipe_to_zone dict, zone_to_pipes dict)
            - pipe_to_zone: Maps pipe_name -> zone_name
            - zone_to_pipes: Maps zone_name -> set of pipe_names
        """
        try:
            pipe_to_zone = {}  # Map pipe_name -> zone_name
            all_pipes_checked = set()
            
            LOGGER.info("Building pipe-to-zone mapping (one pipe = one zone, based on maximum overlap)...")
            
            for pipe_feature in pipe_layer.getFeatures():
                pipe_geom = pipe_feature.geometry()
                if pipe_geom.isEmpty():
                    continue
                
                # Get pipe name
                pipe_name = None
                possible_name_fields = ['FacNam1005', 'name', 'Name', 'NAME', 'pipe_name', 'Pipe_Name', 'PIPE_NAME']
                for field_name in possible_name_fields:
                    try:
                        pipe_name = pipe_feature.attribute(field_name)
                        if pipe_name is not None and str(pipe_name).strip():
                            break
                    except:
                        continue
                
                if pipe_name is None:
                    pipe_name = f"Pipe_{pipe_feature.id()}"
                
                pipe_name = str(pipe_name)
                all_pipes_checked.add(pipe_name)
                
                # Find which zones this pipe intersects
                max_overlap = 0
                best_zone = None
                zones_intersected = []
                
                for zone_feature in zone_layer.getFeatures():
                    zone_geom = zone_feature.geometry()
                    if zone_geom.isEmpty():
                        continue
                    
                    # Get zone name
                    zone_name = None
                    possible_name_fields = ['Name', 'name', 'NAME', 'zone_name', 'Zone_Name', 'ZONE_NAME',
                                          'zone', 'Zone', 'ZONE', 'area_name', 'Area_Name', 'AREA_NAME',
                                          'id', 'ID', 'fid', 'FID', 'objectid', 'OBJECTID']
                    for field_name in possible_name_fields:
                        try:
                            zone_name = zone_feature.attribute(field_name)
                            if zone_name is not None and str(zone_name).strip():
                                break
                        except:
                            continue
                    
                    if zone_name is None:
                        zone_name = f"Zone_{zone_feature.id()}"
                    
                    # Check intersection
                    if zone_geom.intersects(pipe_geom):
                        zones_intersected.append(zone_name)
                        # Calculate overlap (length of pipe within zone)
                        try:
                            intersection = zone_geom.intersection(pipe_geom)
                            overlap = intersection.length() if not intersection.isEmpty() else 0
                            
                            if overlap > max_overlap:
                                max_overlap = overlap
                                best_zone = zone_name
                        except:
                            # If intersection calculation fails, just note that it intersects
                            if best_zone is None:
                                best_zone = zone_name
                
                # Assign pipe to the zone with maximum overlap
                if best_zone is not None:
                    pipe_to_zone[pipe_name] = best_zone
                    # Log if pipe intersects multiple zones
                    if len(zones_intersected) > 1:
                        LOGGER.info(f"Pipe '{pipe_name}' intersects {len(zones_intersected)} zones: {zones_intersected}")
                        LOGGER.info(f"  -> Assigned to '{best_zone}' (maximum overlap: {max_overlap:.2f})")
            
            # Build reverse mapping: zone -> pipes
            zone_to_pipes = {}
            for pipe_name, zone_name in pipe_to_zone.items():
                if zone_name not in zone_to_pipes:
                    zone_to_pipes[zone_name] = set()
                zone_to_pipes[zone_name].add(pipe_name)
            
            LOGGER.info(f"Pipe-to-zone mapping complete:")
            LOGGER.info(f"  Total pipes checked: {len(all_pipes_checked)}")
            LOGGER.info(f"  Pipes assigned to zones: {len(pipe_to_zone)}")
            LOGGER.info(f"  Pipes not assigned: {len(all_pipes_checked) - len(pipe_to_zone)}")
            LOGGER.info(f"  Zones with pipes: {len(zone_to_pipes)}")
            for zone_name, pipes in zone_to_pipes.items():
                LOGGER.info(f"    {zone_name}: {len(pipes)} pipes")
            
            return pipe_to_zone, zone_to_pipes
            
        except Exception as e:
            LOGGER.error(f"Error building pipe-to-zone mapping: {e}")
            import traceback
            LOGGER.error(f"Traceback: {traceback.format_exc()}")
            return {}, {}

    def get_pipes_in_zone(self, zone_feature, pipe_layer, buffer_pixels: int = 10):
        """Find pipes that intersect with a zone polygon using a buffer.
        
        **DEPRECATED**: This method may double-count pipes if they intersect multiple zones.
        Use build_pipe_to_zone_mapping() instead to prevent double counting.
        
        Args:
            zone_feature: Zone polygon feature
            pipe_layer: QGIS vector layer with pipe data
            buffer_pixels: Buffer size in pixels to prevent undercounting
            
        Returns:
            List of pipe names that intersect the zone
        """
        LOGGER.warning("get_pipes_in_zone() is deprecated and may cause double counting.")
        LOGGER.warning("Consider using build_pipe_to_zone_mapping() instead.")
        
        try:
            # Check if we have QGIS available
            try:
                from qgis.core import QgsGeometry
                qgis_available = True
            except ImportError:
                qgis_available = False
                LOGGER.warning("QGIS not available for spatial operations")
                return []
            
            if not qgis_available:
                return []
            
            # Get zone geometry
            zone_geom = zone_feature.geometry()
            if zone_geom.isEmpty():
                LOGGER.warning("Zone geometry is empty")
                return []
            
            LOGGER.info(f"Zone geometry type: {zone_geom.wkbType()}")
            LOGGER.info(f"Zone bounds: {zone_geom.boundingBox().toString()}")
            
            # Create buffer (convert pixels to map units - this is approximate)
            # In a real implementation, you'd convert pixels to map units based on canvas scale
            buffer_distance = buffer_pixels * 0.1  # Approximate conversion
            buffered_geom = zone_geom.buffer(buffer_distance, 5)
            
            LOGGER.info(f"Created buffer of {buffer_distance} units")
            
            pipe_names = []
            intersecting_count = 0
            total_pipes = 0
            
            # Check each pipe feature for intersection
            for pipe_feature in pipe_layer.getFeatures():
                total_pipes += 1
                pipe_geom = pipe_feature.geometry()
                
                if pipe_geom.isEmpty():
                    continue
                
                # Check if pipe intersects buffered zone
                if buffered_geom.intersects(pipe_geom):
                    intersecting_count += 1
                    
                    # Try to get pipe name/identifier from various possible fields
                    pipe_name = None
                    # Your pipe layer uses FacNam1005 for facility names that match Excel Distribution Pipe
                    possible_name_fields = ['FacNam1005', 'name', 'Name', 'NAME', 'pipe_name', 'Pipe_Name', 'PIPE_NAME',
                                          'id', 'ID', 'fid', 'FID', 'objectid', 'OBJECTID', 'pipe_id', 'PIPE_ID',
                                          'facilityid', 'FACILITYID', 'facility_id', 'FACILITY_ID']
                    
                    for field_name in possible_name_fields:
                        try:
                            pipe_name = pipe_feature.attribute(field_name)
                            if pipe_name is not None and str(pipe_name).strip():
                                # Log which field we're using for debugging
                                if intersecting_count <= 5:  # Only log first few for debugging
                                    LOGGER.info(f"  Using field '{field_name}' for pipe name: {pipe_name}")
                                break
                        except:
                            continue
                    
                    if pipe_name is None:
                        pipe_name = f"Pipe_{pipe_feature.id()}"
                        LOGGER.warning(f"  No name field found, using fallback: {pipe_name}")
                    
                    pipe_names.append(str(pipe_name))
            
            LOGGER.info(f"Spatial intersection results:")
            LOGGER.info(f"  Total pipes checked: {total_pipes}")
            LOGGER.info(f"  Intersecting pipes: {intersecting_count}")
            LOGGER.info(f"  Pipe names found: {len(pipe_names)}")
            if len(pipe_names) > 0:
                LOGGER.info(f"  Sample pipe names: {pipe_names[:5]}")
            
            return pipe_names
            
        except Exception as e:
            LOGGER.error(f"Error finding pipes in zone: {e}")
            import traceback
            LOGGER.error(f"Traceback: {traceback.format_exc()}")
            return []

    def audit_excel_data(self, excel_file_path: str) -> Dict[str, Any]:
        """Audit Excel file to understand its structure and content."""
        audit_results = {
            'file_exists': False,
            'sheets_found': [],
            'target_sheet_exists': False,
            'columns_found': [],
            'missing_columns': [],
            'row_count': 0,
            'sample_data': None,
            'date_range': None,
            'use_classes': [],
            'factor_codes': []
        }
        
        try:
            import os
            if not os.path.exists(excel_file_path):
                LOGGER.error(f"Excel file not found: {excel_file_path}")
                return audit_results
            
            audit_results['file_exists'] = True
            
            # Read all sheet names
            excel_file = pd.ExcelFile(excel_file_path)
            audit_results['sheets_found'] = excel_file.sheet_names
            LOGGER.info(f"Sheets found: {audit_results['sheets_found']}")
            
            # Check for target sheet
            target_sheet = 'Service Point List'
            if target_sheet in excel_file.sheet_names:
                audit_results['target_sheet_exists'] = True
                
                # Read the target sheet
                df = pd.read_excel(excel_file_path, sheet_name=target_sheet)
                audit_results['row_count'] = len(df)
                audit_results['columns_found'] = list(df.columns)
                
                LOGGER.info(f"Found {audit_results['row_count']} rows in '{target_sheet}' sheet")
                LOGGER.info(f"Columns: {audit_results['columns_found']}")
                
                # Check for required columns
                required_columns = ['Factor Code', 'Base Factor', 'Heat Factor', 
                                  'HUC 3-Year Peak Demand', 'Install Date', 'Use Class', 'Distribution Pipe']
                audit_results['missing_columns'] = [col for col in required_columns if col not in df.columns]
                
                if audit_results['missing_columns']:
                    LOGGER.warning(f"Missing required columns: {audit_results['missing_columns']}")
                
                # Get sample data (first 5 rows)
                audit_results['sample_data'] = df.head().to_dict('records')
                
                # Analyze date range
                if 'Install Date' in df.columns:
                    df['Install Date'] = pd.to_datetime(df['Install Date'], errors='coerce')
                    valid_dates = df['Install Date'].dropna()
                    if not valid_dates.empty:
                        audit_results['date_range'] = {
                            'min_date': valid_dates.min().strftime('%Y-%m-%d'),
                            'max_date': valid_dates.max().strftime('%Y-%m-%d'),
                            'null_count': len(df) - len(valid_dates)
                        }
                
                # Get unique use classes and factor codes
                if 'Use Class' in df.columns:
                    audit_results['use_classes'] = df['Use Class'].unique().tolist()
                if 'Factor Code' in df.columns:
                    audit_results['factor_codes'] = df['Factor Code'].unique().tolist()
                
            else:
                LOGGER.error(f"Sheet '{target_sheet}' not found. Available sheets: {audit_results['sheets_found']}")
            
        except Exception as e:
            LOGGER.error(f"Error auditing Excel file: {e}")
        
        return audit_results

    def audit_layers(self, zone_layer, pipe_layer) -> Dict[str, Any]:
        """Audit QGIS layers to understand their structure."""
        audit_results = {
            'zone_layer': {'valid': False, 'feature_count': 0, 'fields': [], 'sample_attributes': []},
            'pipe_layer': {'valid': False, 'feature_count': 0, 'fields': [], 'sample_attributes': []}
        }
        
        try:
            # Audit zone layer
            if zone_layer:
                audit_results['zone_layer']['valid'] = True
                audit_results['zone_layer']['feature_count'] = zone_layer.featureCount()
                audit_results['zone_layer']['fields'] = [field.name() for field in zone_layer.fields()]
                
                # Get sample features
                features = list(zone_layer.getFeatures())[:3]  # First 3 features
                audit_results['zone_layer']['sample_attributes'] = [
                    {field.name(): feature.attribute(field.name()) for field in zone_layer.fields()}
                    for feature in features
                ]
                
                LOGGER.info(f"Zone layer: {audit_results['zone_layer']['feature_count']} features")
                LOGGER.info(f"Zone fields: {audit_results['zone_layer']['fields']}")
            
            # Audit pipe layer
            if pipe_layer:
                audit_results['pipe_layer']['valid'] = True
                audit_results['pipe_layer']['feature_count'] = pipe_layer.featureCount()
                audit_results['pipe_layer']['fields'] = [field.name() for field in pipe_layer.fields()]
                
                # Get sample features
                features = list(pipe_layer.getFeatures())[:3]  # First 3 features
                audit_results['pipe_layer']['sample_attributes'] = [
                    {field.name(): feature.attribute(field.name()) for field in pipe_layer.fields()}
                    for feature in features
                ]
                
                LOGGER.info(f"Pipe layer: {audit_results['pipe_layer']['feature_count']} features")
                LOGGER.info(f"Pipe fields: {audit_results['pipe_layer']['fields']}")
        
        except Exception as e:
            LOGGER.error(f"Error auditing layers: {e}")
        
        return audit_results

    def process_zone_loads(self, excel_file_path: str, zone_layer=None, pipe_layer=None, 
                          current_year: int = 2025) -> Dict[str, Dict[str, float]]:
        """Process loads for each zone by class with detailed audit logging.
        
        Args:
            excel_file_path: Path to Excel file with service point list
            zone_layer: QGIS polygon layer with subzones
            pipe_layer: QGIS vector layer with pipes
            current_year: Year to calculate current loads for
            
        Returns:
            Dictionary mapping zone names to load by class
        """
        LOGGER.info(f"=== STARTING DEMAND FILE ANALYSIS AUDIT ===")
        LOGGER.info(f"Excel file: {excel_file_path}")
        LOGGER.info(f"Current year filter: {current_year}")
        
        # Store audit results for dialog display
        self.audit_results = {
            'excel_audit': None,
            'layer_audit': None,
            'processing_steps': [],
            'zone_details': []
        }
        
        try:
            # Step 1: Audit Excel file
            LOGGER.info("Step 1: Auditing Excel file...")
            excel_audit = self.audit_excel_data(excel_file_path)
            self.audit_results['excel_audit'] = excel_audit
            
            if not excel_audit['file_exists']:
                LOGGER.error("Excel file does not exist!")
                return {}
            
            if not excel_audit['target_sheet_exists']:
                LOGGER.error("Target sheet 'Service Point List' not found!")
                return {}
                
            if excel_audit['missing_columns']:
                LOGGER.error(f"Missing required columns: {excel_audit['missing_columns']}")
                return {}
            
            # Step 2: Audit layers
            LOGGER.info("Step 2: Auditing QGIS layers...")
            layer_audit = self.audit_layers(zone_layer, pipe_layer)
            self.audit_results['layer_audit'] = layer_audit
            
            # Step 3: Read and process Excel data
            LOGGER.info("Step 3: Reading Excel data...")
            df = pd.read_excel(excel_file_path, sheet_name='Service Point List')
            LOGGER.info(f"Read {len(df)} service points from Excel")
            self.audit_results['processing_steps'].append(f"Loaded {len(df)} service points from Excel")
            
            # Step 4: Calculate loads
            LOGGER.info("Step 4: Calculating loads...")
            df_before_calc = df.copy()
            df = self.calculate_load(df)
            
            total_load = df['Load'].sum()
            LOGGER.info(f"Total calculated load: {total_load:.2f}")
            self.audit_results['processing_steps'].append(f"Calculated total load: {total_load:.2f}")
            
            # Step 5: Filter by date and handle bad dates
            LOGGER.info("Step 5: Filtering by install date...")
            df['Install Date'] = pd.to_datetime(df['Install Date'], errors='coerce')
            
            # Separate records with valid vs invalid dates
            valid_dates_mask = df['Install Date'].notna()
            df_with_dates = df[valid_dates_mask].copy()
            df_without_dates = df[~valid_dates_mask].copy()
            
            # Get date range from valid records
            if len(df_with_dates) > 0:
                min_year = df_with_dates['Install Date'].dt.year.min()
                max_year = df_with_dates['Install Date'].dt.year.max()
                year_span = max(1, max_year - min_year + 1)  # At least 1 year
                
                LOGGER.info(f"Valid date range: {min_year} to {max_year} ({year_span} years)")
                
                # For records with bad dates, distribute their load proportionally to valid date loads
                if len(df_without_dates) > 0:
                    bad_date_total_load = df_without_dates['Load'].sum()
                    LOGGER.info(f"Found {len(df_without_dates)} records with bad dates, total load: {bad_date_total_load:.2f}")
                    
                    # Calculate load by year from valid dates
                    df_with_dates_temp = df_with_dates.copy()
                    df_with_dates_temp['Install Year'] = df_with_dates_temp['Install Date'].dt.year
                    year_loads = df_with_dates_temp.groupby('Install Year')['Load'].sum()
                    
                    # Calculate total valid load
                    total_valid_load = year_loads.sum()
                    
                    # Calculate proportion for each year (aggressive: proportional to valid load)
                    year_proportions = year_loads / total_valid_load
                    
                    LOGGER.info(f"Distributing bad date loads proportionally to valid date loads:")
                    for year, proportion in year_proportions.items():
                        LOGGER.info(f"  Year {year}: {proportion*100:.1f}% (valid load: {year_loads[year]:.2f} GJ/d)")
                    
                    # Create records for each year with proportional load
                    distributed_records = []
                    for year, proportion in year_proportions.items():
                        for idx, row in df_without_dates.iterrows():
                            new_row = row.copy()
                            new_row['Install Date'] = pd.to_datetime(f'{year}-01-01')
                            new_row['Load'] = row['Load'] * proportion  # Distribute proportionally
                            distributed_records.append(new_row)
                    
                    df_distributed = pd.DataFrame(distributed_records)
                    df = pd.concat([df_with_dates, df_distributed], ignore_index=True)
                    LOGGER.info(f"After distributing bad date loads: {len(df)} total records")
                else:
                    df = df_with_dates
            else:
                LOGGER.warning("No valid dates found in data")
                df = df_with_dates  # Empty or all bad dates
            
            # Now filter by current year
            current_df = df[df['Install Date'].dt.year <= current_year]
            
            date_filtered_load = current_df['Load'].sum()
            LOGGER.info(f"After date filter ({current_year}): {len(current_df)} points, load: {date_filtered_load:.2f}")
            self.audit_results['processing_steps'].append(f"After date filter: {len(current_df)} points, load: {date_filtered_load:.2f}")
            
            # Step 6: Filter by use class
            LOGGER.info("Step 6: Filtering by use class...")
            class_dfs = self.filter_by_use_class(current_df)
            
            for class_name, class_df in class_dfs.items():
                class_load = class_df['Load'].sum()
                LOGGER.info(f"{class_name}: {len(class_df)} points, load: {class_load:.2f}")
                self.audit_results['processing_steps'].append(f"{class_name}: {len(class_df)} points, load: {class_load:.2f}")
            
            zone_loads = {}
            
            if zone_layer and pipe_layer:
                LOGGER.info("Step 7: Processing spatial intersection...")
                LOGGER.info("Using intersects with overlap-based assignment to prevent double counting...")
                
                # Build pipe-to-zone mapping once to prevent double counting
                pipe_to_zone, zone_to_pipes = self.build_pipe_to_zone_mapping(zone_layer, pipe_layer)
                
                # Process each zone
                for zone_feature in zone_layer.getFeatures():
                    # Get zone name
                    zone_name = None
                    possible_name_fields = ['Name', 'name', 'NAME', 'zone_name', 'Zone_Name', 'ZONE_NAME',
                                          'zone', 'Zone', 'ZONE', 'area_name', 'Area_Name', 'AREA_NAME',
                                          'id', 'ID', 'fid', 'FID', 'objectid', 'OBJECTID']
                    
                    for field_name in possible_name_fields:
                        try:
                            zone_name = zone_feature.attribute(field_name)
                            if zone_name is not None and str(zone_name).strip():
                                break
                        except:
                            continue
                    
                    if zone_name is None:
                        zone_name = f"Zone_{zone_feature.id()}"
                    
                    # Get pipes assigned to this zone (no double counting)
                    pipe_names = list(zone_to_pipes.get(zone_name, set()))
                    LOGGER.info(f"Zone {zone_name}: {len(pipe_names)} pipes (no double counting)")
                    
                    zone_detail = {
                        'name': zone_name,
                        'pipe_count': len(pipe_names),
                        'pipe_names': pipe_names[:10],  # First 10 pipe names
                        'loads_by_class': {}
                    }
                    
                    # Calculate loads by class for this zone
                    zone_class_loads = {}
                    for class_name, class_df in class_dfs.items():
                        if len(pipe_names) > 0:
                            # Filter service points by pipes in this zone
                            zone_class_df = class_df[class_df['Distribution Pipe'].isin(pipe_names)]
                            zone_load = zone_class_df['Load'].sum()
                            
                            # Enhanced debugging for pipe name matching
                            if len(zone_class_df) == 0 and len(class_df) > 0:
                                # No matches found - let's debug why
                                sample_excel_pipes = class_df['Distribution Pipe'].unique()[:5]
                                LOGGER.warning(f"Zone {zone_name} - {class_name}: No pipe name matches found!")
                                LOGGER.warning(f"  Sample Excel pipe names: {sample_excel_pipes.tolist()}")
                                LOGGER.warning(f"  Sample zone pipe names: {pipe_names[:5]}")
                                
                                # Check if any partial matches exist
                                partial_matches = []
                                for excel_pipe in sample_excel_pipes[:3]:
                                    for zone_pipe in pipe_names[:10]:
                                        if str(excel_pipe).lower() in str(zone_pipe).lower() or str(zone_pipe).lower() in str(excel_pipe).lower():
                                            partial_matches.append(f"{excel_pipe} ~ {zone_pipe}")
                                
                                if partial_matches:
                                    LOGGER.info(f"  Potential partial matches: {partial_matches}")
                                else:
                                    LOGGER.warning(f"  No partial matches found between Excel and zone pipe names")
                            
                            LOGGER.info(f"Zone {zone_name} - {class_name}: {len(zone_class_df)} points, load: {zone_load:.2f}")
                            zone_class_loads[class_name] = zone_load
                            zone_detail['loads_by_class'][class_name] = {
                                'point_count': len(zone_class_df),
                                'load': zone_load
                            }
                        else:
                            zone_class_loads[class_name] = 0.0
                            zone_detail['loads_by_class'][class_name] = {
                                'point_count': 0,
                                'load': 0.0
                            }
                    
                    zone_loads[zone_name] = zone_class_loads
                    self.audit_results['zone_details'].append(zone_detail)
            else:
                LOGGER.info("Step 7: No spatial layers provided, returning overall totals...")
                # If no spatial layers provided, return overall totals
                zone_loads['Total'] = {
                    class_name: class_df['Load'].sum()
                    for class_name, class_df in class_dfs.items()
                }
            
            LOGGER.info(f"=== ANALYSIS COMPLETE ===")
            LOGGER.info(f"Total zones processed: {len(zone_loads)}")
            
            return zone_loads
            
        except Exception as e:
            LOGGER.error(f"Error processing zone loads: {e}")
            import traceback
            LOGGER.error(f"Traceback: {traceback.format_exc()}")
            return {}

    def create_zones_load_layer(self, zone_loads: Dict[str, Dict[str, float]], 
                               original_zone_layer=None):
        """Create a new layer with zone polygons and load attributes.
        
        Args:
            zone_loads: Dictionary of zone loads by class
            original_zone_layer: Original zone layer to copy geometry from
        """
        try:
            # This would create a new QGIS layer with the zone polygons
            # and additional fields for load by class
            # In real implementation would use QGIS API to:
            # 1. Create new memory layer
            # 2. Copy geometry from original layer
            # 3. Add fields for residential_load, industrial_load, commercial_load
            # 4. Populate with calculated values
            # 5. Add to map canvas
            
            LOGGER.info(f"Creating zones_load layer with {len(zone_loads)} zones")
            for zone_name, loads in zone_loads.items():
                LOGGER.info(f"Zone {zone_name}: {loads}")
                
        except Exception as e:
            LOGGER.error(f"Error creating zones_load layer: {e}")

    def run(self):
        """Run the Demand File Analysis plugin."""
        try:
            LOGGER.info("Running Demand File Analysis")
            
            if self.iface:
                # Show GUI dialog to get user inputs
                success, inputs = self.show_input_dialog()
                if not success:
                    return
                
                # Validate inputs
                if VALIDATION_AVAILABLE:
                    LOGGER.info(" Validating demand file analysis inputs...")
                    validator = DataValidator()
                    
                    # Check Excel file
                    if not inputs.get('excel_file'):
                        validator.add_error("No Excel file specified")
                    elif not os.path.exists(inputs['excel_file']):
                        validator.add_error(f"Excel file not found: {inputs['excel_file']}")
                    else:
                        validator.add_info(f"Excel file: {os.path.basename(inputs['excel_file'])}")
                    
                    # Check current year
                    current_year = safe_int(inputs.get('current_year'), 2025)
                    if current_year < 1900 or current_year > 2100:
                        validator.add_warning(f"Unusual current year: {current_year}")
                    else:
                        validator.add_info(f"Analysis year: {current_year}")
                    
                    # Validate zone layer
                    if inputs.get('zone_layer'):
                        zone_validator = validate_layer_data(
                            inputs['zone_layer'],
                            "Zone Layer (Subzones)",
                            required_fields=None  # Will check for basic validity
                        )
                        validator.errors.extend(zone_validator.errors)
                        validator.warnings.extend(zone_validator.warnings)
                        validator.info.extend(zone_validator.info)
                    else:
                        validator.add_error("No zone layer selected")
                    
                    # Validate pipe layer
                    if inputs.get('pipe_layer'):
                        pipe_validator = validate_layer_data(
                            inputs['pipe_layer'],
                            "Pipe Layer",
                            required_fields=None
                        )
                        validator.errors.extend(pipe_validator.errors)
                        validator.warnings.extend(pipe_validator.warnings)
                        validator.info.extend(pipe_validator.info)
                    else:
                        validator.add_error("No pipe layer selected")
                    
                    # Show validation dialog
                    if not show_validation_dialog(validator,
                                                  title="Demand File Analysis - Data Validation",
                                                  parent=self.iface.mainWindow()):
                        LOGGER.info(" User cancelled after validation")
                        return
                
                # Process the data with user inputs
                zone_loads = self.process_zone_loads(
                    excel_file_path=inputs['excel_file'],
                    zone_layer=inputs['zone_layer'],
                    pipe_layer=inputs['pipe_layer'],
                    current_year=inputs['current_year']
                )
                
                if zone_loads:
                    # Create output layer
                    self.create_zones_load_layer(zone_loads, inputs['zone_layer'])
                    
                    # Show detailed audit results first
                    self.show_audit_dialog()
                    
                    # Show results dialog
                    self.show_results_dialog(zone_loads)
                else:
                    # Show audit dialog even if no loads found - this will help debug
                    self.show_audit_dialog()
                    
                    from qgis.PyQt.QtWidgets import QMessageBox
                    QMessageBox.warning(
                        self.iface.mainWindow(),
                        "Warning", 
                        "No load data could be processed. Check the audit results for details."
                    )
            else:
                print("Demand File Analysis plugin executed successfully")
                
        except Exception as e:
            LOGGER.error(f"Error running Demand File Analysis: {e}")
            if self.iface:
                from qgis.PyQt.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Error",
                    f"Error running analysis: {str(e)}"
                )

    def show_input_dialog(self):
        """Show dialog to get user inputs for the analysis."""
        try:
            from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                           QLineEdit, QPushButton, QComboBox, QSpinBox,
                                           QDoubleSpinBox, QFileDialog, QGroupBox, QFormLayout,
                                           QFrame)
            from qgis.PyQt.QtCore import Qt
            from qgis.PyQt.QtGui import QFont
            from qgis.core import QgsProject
            
            dialog = QDialog(self.iface.mainWindow())
            dialog.setWindowTitle("Demand File Analysis - Input Selection")
            dialog.setMinimumSize(650, 500)
            
            layout = QVBoxLayout()
            
            # Title
            title_label = QLabel(" Current Year Demand Analysis")
            title_font = QFont()
            title_font.setPointSize(13)
            title_font.setBold(True)
            title_label.setFont(title_font)
            layout.addWidget(title_label)
            
            # Description
            desc_label = QLabel(
                "Analyze current service point data to establish baseline loads:\n\n"
                " Processes service point data for a single reference year\n"
                "️ Aggregates current loads by geographic zones and customer categories\n"
                " Establishes baseline for forecasting and capacity planning\n\n"
                " This analysis provides the starting point for all future projections"
            )
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #666; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
            layout.addWidget(desc_label)
            
            # Separator
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            layout.addWidget(line)
            
            # File selection group
            file_group = QGroupBox(" Input Data Source")
            file_layout = QFormLayout()
            
            # Excel file selection
            excel_layout = QHBoxLayout()
            self.excel_line = QLineEdit()
            self.excel_line.setPlaceholderText("Select Excel file with current year service point data...")
            excel_browse = QPushButton(" Browse...")
            excel_browse.clicked.connect(self.browse_excel_file)
            excel_browse.setToolTip("Browse for Excel file containing current service point records")
            excel_layout.addWidget(self.excel_line)
            excel_layout.addWidget(excel_browse)
            
            excel_label = QLabel(" Service Point List (Excel):")
            excel_label.setToolTip(
                "Excel file containing current service point records\n"
                "Required columns: Location, Category, Load values\n"
                "Data should represent a single reference year"
            )
            file_layout.addRow(excel_label, excel_layout)
            
            file_group.setLayout(file_layout)
            layout.addWidget(file_group)
            
            # Layer selection group
            layer_group = QGroupBox("️ Spatial Layers")
            layer_layout = QFormLayout()
            
            # Zone layer selection
            self.zone_combo = QComboBox()
            self.populate_layer_combo(self.zone_combo, "Polygon")
            self.zone_combo.setToolTip(
                "Polygon layer defining analysis zones\n"
                "Service points will be aggregated within these zones"
            )
            layer_layout.addRow(" Zone Layer (Polygons):", self.zone_combo)
            
            # Pipe layer selection
            self.pipe_combo = QComboBox()
            self.populate_layer_combo(self.pipe_combo, "LineString")
            self.pipe_combo.setToolTip(
                "Pipe network layer for spatial analysis\n"
                "Used for infrastructure context and reporting"
            )
            layer_layout.addRow(" Pipe Layer (Lines):", self.pipe_combo)
            
            layer_group.setLayout(layer_layout)
            layout.addWidget(layer_group)
            
            # Parameters group
            param_group = QGroupBox("️ Calculation Parameters")
            param_layout = QFormLayout()
            
            # Load multiplier
            self.load_multiplier_spin = QDoubleSpinBox()
            self.load_multiplier_spin.setDecimals(3)
            self.load_multiplier_spin.setRange(0.001, 10.0)
            self.load_multiplier_spin.setValue(self.load_multiplier)
            self.load_multiplier_spin.setToolTip(
                "Multiplier applied to all load values\n"
                "Use for unit conversion or scaling (e.g., 1.0 = no change)"
            )
            param_layout.addRow(" Load Multiplier:", self.load_multiplier_spin)
            
            # Heat factor multiplier
            self.heat_multiplier_spin = QDoubleSpinBox()
            self.heat_multiplier_spin.setDecimals(1)
            self.heat_multiplier_spin.setRange(0.1, 200.0)
            self.heat_multiplier_spin.setValue(self.heat_factor_multiplier)
            self.heat_multiplier_spin.setToolTip(
                "Multiplier for heat factor calculations\n"
                "Adjusts peak demand estimation based on local conditions"
            )
            param_layout.addRow("️ Heat Factor Multiplier:", self.heat_multiplier_spin)
            
            # Current year
            self.year_spin = QSpinBox()
            self.year_spin.setRange(2000, 2050)
            self.year_spin.setValue(2025)
            self.year_spin.setToolTip(
                "Reference year for this analysis\n"
                "Should match the year of your service point data"
            )
            param_layout.addRow(" Current Year:", self.year_spin)
            
            param_group.setLayout(param_layout)
            layout.addWidget(param_group)
            
            # Buttons
            button_layout = QHBoxLayout()
            ok_button = QPushButton("Run Analysis")
            cancel_button = QPushButton("Cancel")
            
            ok_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)
            
            button_layout.addWidget(ok_button)
            button_layout.addWidget(cancel_button)
            layout.addLayout(button_layout)
            
            dialog.setLayout(layout)
            
            # Execute dialog
            if dialog.exec_() == QDialog.Accepted:
                # Update plugin parameters
                self.load_multiplier = self.load_multiplier_spin.value()
                self.heat_factor_multiplier = self.heat_multiplier_spin.value()
                
                # Get selected layers
                zone_layer = None
                pipe_layer = None
                
                if self.zone_combo.currentText() != "":
                    zone_layer = QgsProject.instance().mapLayersByName(self.zone_combo.currentText())[0]
                if self.pipe_combo.currentText() != "":
                    pipe_layer = QgsProject.instance().mapLayersByName(self.pipe_combo.currentText())[0]
                
                return True, {
                    'excel_file': self.excel_line.text(),
                    'zone_layer': zone_layer,
                    'pipe_layer': pipe_layer,
                    'current_year': self.year_spin.value()
                }
            else:
                return False, {}
                
        except Exception as e:
            LOGGER.error(f"Error showing input dialog: {e}")
            return False, {}

    def browse_excel_file(self):
        """Browse for Excel file."""
        try:
            from qgis.PyQt.QtWidgets import QFileDialog
            
            file_path, _ = QFileDialog.getOpenFileName(
                self.iface.mainWindow(),
                "Select Excel File",
                "",
                "Excel Files (*.xlsx *.xls);;All Files (*)"
            )
            
            if file_path:
                self.excel_line.setText(file_path)
                
        except Exception as e:
            LOGGER.error(f"Error browsing for Excel file: {e}")

    def populate_layer_combo(self, combo, geometry_type):
        """Populate combo box with layers of specified geometry type."""
        try:
            from qgis.core import QgsProject, QgsWkbTypes
            
            combo.clear()
            combo.addItem("")  # Empty option
            
            for layer in QgsProject.instance().mapLayers().values():
                if hasattr(layer, 'geometryType'):
                    if geometry_type == "Polygon" and layer.geometryType() == QgsWkbTypes.PolygonGeometry:
                        combo.addItem(layer.name())
                    elif geometry_type == "LineString" and layer.geometryType() == QgsWkbTypes.LineGeometry:
                        combo.addItem(layer.name())
                        
        except Exception as e:
            LOGGER.error(f"Error populating layer combo: {e}")

    def show_results_dialog(self, zone_loads):
        """Show results dialog with calculated loads."""
        try:
            from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QTextEdit, QPushButton,
                                           QTableWidget, QTableWidgetItem, QTabWidget, QWidget)
            
            dialog = QDialog(self.iface.mainWindow())
            dialog.setWindowTitle("Demand File Analysis - Results")
            dialog.setMinimumSize(600, 400)
            
            layout = QVBoxLayout()
            
            # Create tab widget
            tab_widget = QTabWidget()
            
            # Summary tab
            summary_text = QTextEdit()
            summary_text.setReadOnly(True)
            
            summary_content = "DEMAND FILE ANALYSIS RESULTS\n"
            summary_content += "=" * 50 + "\n\n"
            
            total_residential = 0
            total_commercial = 0
            total_industrial = 0
            
            for zone_name, loads in zone_loads.items():
                summary_content += f"Zone: {zone_name}\n"
                summary_content += f"  Residential: {loads.get('residential', 0):.2f} GJ/d\n"
                summary_content += f"  Commercial:  {loads.get('commercial', 0):.2f} GJ/d\n"
                summary_content += f"  Industrial:  {loads.get('industrial', 0):.2f} GJ/d\n"
                summary_content += f"  Total:       {sum(loads.values()):.2f} GJ/d\n\n"
                
                total_residential += loads.get('residential', 0)
                total_commercial += loads.get('commercial', 0)
                total_industrial += loads.get('industrial', 0)
            
            summary_content += "OVERALL TOTALS:\n"
            summary_content += f"  Total Residential: {total_residential:.2f} GJ/d\n"
            summary_content += f"  Total Commercial:  {total_commercial:.2f} GJ/d\n"
            summary_content += f"  Total Industrial:  {total_industrial:.2f} GJ/d\n"
            summary_content += f"  GRAND TOTAL:       {total_residential + total_commercial + total_industrial:.2f} GJ/d\n"
            
            summary_text.setPlainText(summary_content)
            tab_widget.addTab(summary_text, "Summary")
            
            # Table tab
            table = QTableWidget()
            table.setRowCount(len(zone_loads))
            table.setColumnCount(4)
            table.setHorizontalHeaderLabels(["Zone", "Residential", "Commercial", "Industrial"])
            
            for row, (zone_name, loads) in enumerate(zone_loads.items()):
                table.setItem(row, 0, QTableWidgetItem(zone_name))
                table.setItem(row, 1, QTableWidgetItem(f"{loads.get('residential', 0):.2f}"))
                table.setItem(row, 2, QTableWidgetItem(f"{loads.get('commercial', 0):.2f}"))
                table.setItem(row, 3, QTableWidgetItem(f"{loads.get('industrial', 0):.2f}"))
            
            table.resizeColumnsToContents()
            
            # Create table tab with export button
            table_widget = QWidget()
            table_layout = QVBoxLayout()
            
            # Export button
            export_button = QPushButton("Export Table to CSV")
            export_button.clicked.connect(lambda: self.export_table_to_csv(table, "demand_analysis_results"))
            
            table_layout.addWidget(export_button)
            table_layout.addWidget(table)
            table_widget.setLayout(table_layout)
            
            tab_widget.addTab(table_widget, "Table")
            
            layout.addWidget(tab_widget)
            
            # Close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)
            
            dialog.setLayout(layout)
            dialog.exec_()
            
        except Exception as e:
            LOGGER.error(f"Error showing results dialog: {e}")

    def show_audit_dialog(self):
        """Show detailed audit dialog with step-by-step analysis results."""
        try:
            from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QTextEdit, QPushButton,
                                           QTableWidget, QTableWidgetItem, QTabWidget, QLabel)
            
            dialog = QDialog(self.iface.mainWindow())
            dialog.setWindowTitle("Demand File Analysis - Audit Results")
            dialog.setMinimumSize(800, 600)
            
            layout = QVBoxLayout()
            
            # Create tab widget
            tab_widget = QTabWidget()
            
            # Excel audit tab
            excel_tab = QTextEdit()
            excel_tab.setReadOnly(True)
            excel_content = "EXCEL FILE AUDIT RESULTS\n"
            excel_content += "=" * 50 + "\n\n"
            
            if hasattr(self, 'audit_results') and self.audit_results.get('excel_audit'):
                excel_audit = self.audit_results['excel_audit']
                
                excel_content += f"File exists: {excel_audit['file_exists']}\n"
                excel_content += f"Target sheet exists: {excel_audit['target_sheet_exists']}\n"
                excel_content += f"Row count: {excel_audit['row_count']}\n"
                excel_content += f"Sheets found: {excel_audit['sheets_found']}\n"
                excel_content += f"Missing columns: {excel_audit['missing_columns']}\n\n"
                
                if excel_audit['date_range']:
                    excel_content += f"Date range: {excel_audit['date_range']['min_date']} to {excel_audit['date_range']['max_date']}\n"
                    excel_content += f"Null dates: {excel_audit['date_range']['null_count']}\n"
                
                excel_content += f"\nUse classes found: {excel_audit['use_classes']}\n"
                excel_content += f"Factor codes found: {excel_audit['factor_codes']}\n\n"
                
                excel_content += "COLUMNS FOUND:\n"
                for col in excel_audit['columns_found']:
                    excel_content += f"  - {col}\n"
                
                if excel_audit['sample_data']:
                    excel_content += f"\nSAMPLE DATA (first row):\n"
                    sample = excel_audit['sample_data'][0]
                    for key, value in sample.items():
                        excel_content += f"  {key}: {value}\n"
            else:
                excel_content += "No Excel audit data available\n"
            
            excel_tab.setPlainText(excel_content)
            tab_widget.addTab(excel_tab, "Excel Audit")
            
            # Layer audit tab
            layer_tab = QTextEdit()
            layer_tab.setReadOnly(True)
            layer_content = "QGIS LAYERS AUDIT RESULTS\n"
            layer_content += "=" * 50 + "\n\n"
            
            if hasattr(self, 'audit_results') and self.audit_results.get('layer_audit'):
                layer_audit = self.audit_results['layer_audit']
                
                layer_content += "ZONE LAYER:\n"
                zone_info = layer_audit['zone_layer']
                layer_content += f"  Valid: {zone_info['valid']}\n"
                layer_content += f"  Feature count: {zone_info['feature_count']}\n"
                layer_content += f"  Fields: {zone_info['fields']}\n"
                
                if zone_info['sample_attributes']:
                    layer_content += f"  Sample attributes:\n"
                    for i, attr in enumerate(zone_info['sample_attributes'][:2], 1):
                        layer_content += f"    Feature {i}: {attr}\n"
                
                layer_content += "\nPIPE LAYER:\n"
                pipe_info = layer_audit['pipe_layer']
                layer_content += f"  Valid: {pipe_info['valid']}\n"
                layer_content += f"  Feature count: {pipe_info['feature_count']}\n"
                layer_content += f"  Fields: {pipe_info['fields']}\n"
                
                if pipe_info['sample_attributes']:
                    layer_content += f"  Sample attributes:\n"
                    for i, attr in enumerate(pipe_info['sample_attributes'][:2], 1):
                        layer_content += f"    Feature {i}: {attr}\n"
            else:
                layer_content += "No layer audit data available\n"
            
            layer_tab.setPlainText(layer_content)
            tab_widget.addTab(layer_tab, "Layer Audit")
            
            # Processing steps tab
            process_tab = QTextEdit()
            process_tab.setReadOnly(True)
            process_content = "PROCESSING STEPS\n"
            process_content += "=" * 50 + "\n\n"
            
            if hasattr(self, 'audit_results') and self.audit_results.get('processing_steps'):
                for i, step in enumerate(self.audit_results['processing_steps'], 1):
                    process_content += f"{i}. {step}\n"
            else:
                process_content += "No processing steps recorded\n"
            
            process_tab.setPlainText(process_content)
            tab_widget.addTab(process_tab, "Processing Steps")
            
            # Zone details tab
            zone_tab = QTextEdit()
            zone_tab.setReadOnly(True)
            zone_content = "ZONE ANALYSIS DETAILS\n"
            zone_content += "=" * 50 + "\n\n"
            
            if hasattr(self, 'audit_results') and self.audit_results.get('zone_details'):
                for zone_detail in self.audit_results['zone_details']:
                    zone_content += f"Zone: {zone_detail['name']}\n"
                    zone_content += f"  Intersecting pipes: {zone_detail['pipe_count']}\n"
                    if zone_detail['pipe_names']:
                        zone_content += f"  Sample pipe names: {zone_detail['pipe_names'][:5]}\n"
                    
                    zone_content += f"  Loads by class:\n"
                    for class_name, class_info in zone_detail['loads_by_class'].items():
                        zone_content += f"    {class_name}: {class_info['point_count']} points, {class_info['load']:.2f} load\n"
                    zone_content += "\n"
            else:
                zone_content += "No zone details available\n"
            
            zone_tab.setPlainText(zone_content)
            tab_widget.addTab(zone_tab, "Zone Details")
            
            layout.addWidget(tab_widget)
            
            # Close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)
            
            dialog.setLayout(layout)
            dialog.exec_()
            
        except Exception as e:
            LOGGER.error(f"Error showing audit dialog: {e}")