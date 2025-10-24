"""Enhanced Forecast Plugin for QGIS.

This plugin performs comprehensive gas load forecasting using:
- Spatial analysis to calculate cumulative loads by polygon and class
- Population-based residential forecasting with housing ratios
- Regression-based commercial/industrial forecasting
- Historical load analysis for trend projection
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

# Import validation module
from .data_validation import (
    DataValidator,
    validate_forecast_inputs,
    validate_layer_data,
    validate_csv_forecast_data,
    show_validation_dialog,
    safe_int,
    safe_float
)

LOGGER = logging.getLogger(__name__)

class ForecastPlugin:
    def __init__(self, iface: Any = None):
        """Initialize the Forecast plugin."""
        self.iface = iface
        self.action = None
        self.pop_housing_data = None  # Store imported population/housing data
        
        # Check enhanced features availability on startup
        self.enhanced_features_available = self._check_enhanced_features()
        self._log_startup_status()
        
    def _check_enhanced_features(self) -> bool:
        """Check if enhanced features are available."""
        try:
            import pandas
            import numpy
            import sklearn
            return True
        except ImportError:
            return False
            
    def _log_startup_status(self):
        """Log the plugin startup status."""
        if self.enhanced_features_available:
            LOGGER.info(" Gas Hydraulics Forecasting Plugin - FULL FUNCTIONALITY")
            LOGGER.info("    Enhanced forecasting features available")
            LOGGER.info("    Population-based residential forecasting")
            LOGGER.info("    Regression-based commercial/industrial forecasting")
            LOGGER.info("    Spatial load analysis capabilities")
        else:
            LOGGER.warning("️  Gas Hydraulics Forecasting Plugin - BASIC MODE")
            LOGGER.warning("   • Enhanced features not available")
            LOGGER.warning("   • Install pandas, numpy, scikit-learn for full functionality")
            LOGGER.warning("   • Run setup_dev.py to install missing dependencies")
        
    def initGui(self):
        """Create GUI elements (only when running inside QGIS)."""
        try:
            from qgis.PyQt.QtWidgets import QAction
            self.action = QAction("Load Forecast", self.iface.mainWindow() if self.iface else None)
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
                    header_item = table.horizontalHeaderItem(col)
                    headers.append(header_item.text() if header_item else f"Column {col}")
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
            
            # Flexible header matching - allow partial matches and recognize Name column variants
            header_mapping = {}
            
            # First, try exact matches
            for i, csv_header in enumerate(csv_headers):
                for j, table_header in enumerate(table_headers):
                    if csv_header.lower().strip() == table_header.lower().strip():
                        header_mapping[i] = j
                        break
            
            # Then, handle special case for area/name columns - prioritize "Name" column variants
            area_name_variants = ['Name', 'name', 'NAME', 'Area_Name', 'area_name', 'AREA_NAME',
                                'Zone_Name', 'zone_name', 'ZONE_NAME', 'Zone', 'zone', 'ZONE']
            
            for i, csv_header in enumerate(csv_headers):
                if i not in header_mapping:  # Only if not already mapped
                    csv_header_clean = csv_header.lower().strip()
                    
                    # Check if this CSV column is a name/area variant
                    if csv_header_clean in [v.lower() for v in area_name_variants]:
                        # Map to the first column if it's labeled "Area" or similar
                        for j, table_header in enumerate(table_headers):
                            table_header_clean = table_header.lower().strip()
                            if table_header_clean in ['area', 'name', 'zone', 'area_name', 'zone_name']:
                                header_mapping[i] = j
                                LOGGER.info(f"Mapped CSV column '{csv_header}' to table column '{table_header}'")
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

    def calculate_cumulative_loads_by_polygon(self, excel_file_path: str, polygon_layer: Any = None, 
                                            pipe_layer: Any = None, start_year: int = 2000, 
                                            end_year: int = 2025) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Calculate cumulative loads by polygon by class by year using spatial analysis.
        
        Args:
            excel_file_path: Path to Excel file with service point data
            polygon_layer: QGIS layer with polygons  
            pipe_layer: QGIS layer with pipe lines
            start_year: Start year for analysis
            end_year: End year for analysis
            
        Returns:
            Dict with polygon as key, year as second key, and class loads as values
            Format: {polygon: {year: {class: cumulative_load}}}
        """
        try:
            import pandas as pd
            
            LOGGER.info(f"Calculating cumulative loads by polygon for {start_year}-{end_year}")
            
            # Load and process Excel data
            df = pd.read_excel(excel_file_path)
            LOGGER.info(f"Loaded Excel file: {len(df)} rows")
            
            if df.empty:
                LOGGER.error("Excel file is empty")
                return {}
            
            # Calculate loads using historical analysis methods
            from .historical_analysis import HistoricalAnalysisPlugin
            hist_plugin = HistoricalAnalysisPlugin()
            df = hist_plugin.calculate_load(df)
            
            # Convert Install Date and filter by years
            df['Install Date'] = pd.to_datetime(df['Install Date'], errors='coerce')
            df = df.dropna(subset=['Install Date'])
            df['Install Year'] = df['Install Date'].dt.year
            
            # Filter to analysis period
            period_df = df[(df['Install Year'] >= start_year) & (df['Install Year'] <= end_year)]
            LOGGER.info(f"Filtered to analysis period: {len(period_df)} rows")
            
            if period_df.empty:
                LOGGER.error("No data in analysis period")
                return {}
            
            # Filter by use class
            class_dfs = hist_plugin.filter_by_use_class(period_df)
            
            # Prepare results
            results = {}
            
            if polygon_layer and pipe_layer:
                LOGGER.info("Processing polygons for cumulative load calculation...")
                LOGGER.info("Using intersects with overlap-based assignment to prevent double counting...")
                
                # Build pipe-to-polygon mapping once to prevent double counting
                pipe_to_polygon, polygon_to_pipes = self._build_pipe_to_polygon_mapping(polygon_layer, pipe_layer)
                
                for polygon_feature in polygon_layer.getFeatures():
                    # Get polygon name
                    polygon_name = self._get_polygon_name(polygon_feature)
                    
                    # Get pipes assigned to this polygon (no double counting)
                    pipe_names = list(polygon_to_pipes.get(polygon_name, set()))
                    LOGGER.info(f"Polygon {polygon_name}: {len(pipe_names)} pipes (no double counting)")
                    
                    # Initialize polygon results
                    results[polygon_name] = {}
                    
                    # Process each year
                    for year in range(start_year, end_year + 1):
                        year_loads = {}
                        
                        for class_name, class_df in class_dfs.items():
                            # Get cumulative load up to this year for this polygon
                            polygon_year_df = class_df[
                                (class_df['Distribution Pipe'].isin(pipe_names)) &
                                (class_df['Install Year'] <= year)
                            ]
                            
                            cumulative_load = polygon_year_df['Load'].sum()
                            year_loads[class_name] = round(cumulative_load, 2)
                        
                        results[polygon_name][year] = year_loads
            else:
                LOGGER.info("No spatial layers - creating overall totals")
                # If no spatial layers, create overall totals
                results['Overall'] = {}
                
                for year in range(start_year, end_year + 1):
                    year_loads = {}
                    for class_name, class_df in class_dfs.items():
                        year_df = class_df[class_df['Install Year'] <= year]
                        cumulative_load = year_df['Load'].sum()
                        year_loads[class_name] = round(cumulative_load, 2)
                    
                    results['Overall'][year] = year_loads
            
            LOGGER.info(f"Calculated cumulative loads for {len(results)} polygons")
            return results
            
        except Exception as e:
            LOGGER.error(f"Error calculating cumulative loads by polygon: {e}")
            return {}
    
    def _get_polygon_name(self, polygon_feature):
        """Get polygon name from feature attributes, prioritizing Name column."""
        polygon_name = None
        possible_name_fields = ['Name', 'name', 'NAME', 'zone_name', 'Zone_Name', 'ZONE_NAME',
                              'zone', 'Zone', 'ZONE', 'area_name', 'Area_Name', 'AREA_NAME',
                              'id', 'ID', 'fid', 'FID', 'objectid', 'OBJECTID']
        
        for field_name in possible_name_fields:
            try:
                polygon_name = polygon_feature.attribute(field_name)
                if polygon_name is not None and str(polygon_name).strip():
                    break
            except:
                continue
        
        if polygon_name is None:
            polygon_name = f"Polygon_{polygon_feature.id()}"
            
        return str(polygon_name)
    
    def _build_pipe_to_polygon_mapping(self, polygon_layer, pipe_layer):
        """
        Build a mapping of pipes to polygons based on maximum overlap.
        Each pipe is assigned to exactly one polygon (the one it overlaps most with).
        This prevents double counting when polygons overlap or pipes cross boundaries.
        
        Args:
            polygon_layer: The polygon layer
            pipe_layer: The pipe layer
            
        Returns:
            Tuple of (pipe_to_polygon dict, polygon_to_pipes dict)
            - pipe_to_polygon: Maps pipe_name -> polygon_name
            - polygon_to_pipes: Maps polygon_name -> set of pipe_names
        """
        try:
            from qgis.core import QgsGeometry
            
            pipe_to_polygon = {}
            all_pipes_checked = set()
            
            LOGGER.info("Building pipe-to-polygon mapping (one pipe = one polygon, based on maximum overlap)...")
            
            for pipe_feature in pipe_layer.getFeatures():
                pipe_geom = pipe_feature.geometry()
                if pipe_geom.isEmpty():
                    continue
                
                # Get pipe name
                pipe_name = None
                name_fields = ['Name', 'name', 'NAME', 'pipe_name', 'Pipe_Name', 'id', 'ID']
                for field_name in name_fields:
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
                
                # Find which polygons this pipe intersects
                max_overlap = 0
                best_polygon = None
                polygons_intersected = []
                
                for polygon_feature in polygon_layer.getFeatures():
                    polygon_geom = polygon_feature.geometry()
                    if polygon_geom.isEmpty():
                        continue
                    
                    polygon_name = self._get_polygon_name(polygon_feature)
                    
                    # Check intersection
                    if polygon_geom.intersects(pipe_geom):
                        polygons_intersected.append(polygon_name)
                        try:
                            intersection = polygon_geom.intersection(pipe_geom)
                            overlap = intersection.length() if not intersection.isEmpty() else 0
                            
                            if overlap > max_overlap:
                                max_overlap = overlap
                                best_polygon = polygon_name
                        except:
                            if best_polygon is None:
                                best_polygon = polygon_name
                
                # Assign pipe to the polygon with maximum overlap
                if best_polygon is not None:
                    pipe_to_polygon[pipe_name] = best_polygon
                    if len(polygons_intersected) > 1:
                        LOGGER.info(f"Pipe '{pipe_name}' intersects {len(polygons_intersected)} polygons: {polygons_intersected}")
                        LOGGER.info(f"  -> Assigned to '{best_polygon}' (maximum overlap: {max_overlap:.2f})")
            
            # Build reverse mapping: polygon -> pipes
            polygon_to_pipes = {}
            for pipe_name, polygon_name in pipe_to_polygon.items():
                if polygon_name not in polygon_to_pipes:
                    polygon_to_pipes[polygon_name] = set()
                polygon_to_pipes[polygon_name].add(pipe_name)
            
            LOGGER.info(f"Pipe-to-polygon mapping complete:")
            LOGGER.info(f"  Total pipes checked: {len(all_pipes_checked)}")
            LOGGER.info(f"  Pipes assigned to polygons: {len(pipe_to_polygon)}")
            LOGGER.info(f"  Pipes not assigned: {len(all_pipes_checked) - len(pipe_to_polygon)}")
            LOGGER.info(f"  Polygons with pipes: {len(polygon_to_pipes)}")
            
            return pipe_to_polygon, polygon_to_pipes
            
        except Exception as e:
            LOGGER.error(f"Error building pipe-to-polygon mapping: {e}")
            import traceback
            LOGGER.error(f"Traceback: {traceback.format_exc()}")
            return {}, {}
    
    def _get_pipes_in_polygon(self, polygon_feature, pipe_layer):
        """Get list of pipe names that intersect with the polygon.
        
        **DEPRECATED**: This method may double-count pipes if they intersect multiple polygons.
        Use _build_pipe_to_polygon_mapping() instead to prevent double counting.
        """
        LOGGER.warning("_get_pipes_in_polygon() is deprecated and may cause double counting.")
        LOGGER.warning("Consider using _build_pipe_to_polygon_mapping() instead.")
        
        try:
            from qgis.core import QgsGeometry
            
            polygon_geom = polygon_feature.geometry()
            pipe_names = []
            
            for pipe_feature in pipe_layer.getFeatures():
                pipe_geom = pipe_feature.geometry()
                
                if polygon_geom.intersects(pipe_geom):
                    # Try to get pipe name/ID
                    pipe_name = None
                    name_fields = ['Name', 'name', 'NAME', 'pipe_name', 'Pipe_Name', 'id', 'ID']
                    
                    for field_name in name_fields:
                        try:
                            pipe_name = pipe_feature.attribute(field_name)
                            if pipe_name is not None:
                                break
                        except:
                            continue
                    
                    if pipe_name is None:
                        pipe_name = f"Pipe_{pipe_feature.id()}"
                    
                    pipe_names.append(str(pipe_name))
            
            return pipe_names
            
        except Exception as e:
            LOGGER.error(f"Error finding pipes in polygon: {e}")
            return []

    def import_population_housing_csv(self):
        """Import CSV with population and housing data by year."""
        try:
            import pandas as pd
            
            if not self.iface:
                return None
                
            from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
            
            # Get CSV file path
            file_path, _ = QFileDialog.getOpenFileName(
                self.iface.mainWindow(),
                "Import Population and Housing CSV",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if not file_path:
                return None
            
            # Read CSV data
            df = pd.read_csv(file_path)
            LOGGER.info(f"Loaded population/housing CSV: {len(df)} rows")
            
            # Validate required columns (case insensitive)
            required_cols = ['year', 'population', 'housing']
            df_lower = df.columns.str.lower()
            
            missing_cols = []
            for col in required_cols:
                if col not in df_lower.tolist():
                    missing_cols.append(col)
            
            if missing_cols:
                QMessageBox.warning(
                    self.iface.mainWindow(),
                    "Missing Columns",
                    f"CSV must contain columns: {required_cols}\nMissing: {missing_cols}"
                )
                return None
            
            # Standardize column names
            col_mapping = {}
            for i, col in enumerate(df.columns):
                if col.lower() == 'year':
                    col_mapping[col] = 'Year'
                elif col.lower() == 'population':
                    col_mapping[col] = 'Population'
                elif col.lower() == 'housing':
                    col_mapping[col] = 'Housing'
            
            df = df.rename(columns=col_mapping)[['Year', 'Population', 'Housing']]
            
            # Remove rows with missing data
            original_len = len(df)
            df = df.dropna()
            
            if len(df) != original_len:
                LOGGER.warning(f"Removed {original_len - len(df)} rows with missing data")
            
            # Convert to numeric
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df['Population'] = pd.to_numeric(df['Population'], errors='coerce')
            df['Housing'] = pd.to_numeric(df['Housing'], errors='coerce')
            
            # Remove invalid data
            df = df.dropna()
            
            if df.empty:
                QMessageBox.warning(
                    self.iface.mainWindow(),
                    "No Valid Data",
                    "No valid numeric data found in CSV file."
                )
                return None
            
            # Sort by year
            df = df.sort_values('Year')
            
            LOGGER.info(f"Successfully processed {len(df)} years of population/housing data")
            LOGGER.info(f"Year range: {df['Year'].min():.0f} to {df['Year'].max():.0f}")
            
            return df
            
        except Exception as e:
            LOGGER.error(f"Error importing population/housing CSV: {e}")
            if self.iface:
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Import Error",
                    f"Failed to import CSV:\n{str(e)}"
                )
            return None

    def calculate_persons_per_home(self, pop_housing_df) -> Dict[int, float]:
        """Calculate persons per home ratio for years with both population and housing data."""
        try:
            if pop_housing_df is None or len(pop_housing_df) == 0:
                return {}
            
            ratios = {}
            
            for _, row in pop_housing_df.iterrows():
                year = int(row['Year'])
                population = row['Population']
                housing = row['Housing']
                
                if housing > 0:
                    ratio = population / housing
                    ratios[year] = ratio
                    LOGGER.info(f"Year {year}: {population:,.0f} people / {housing:,.0f} homes = {ratio:.2f} persons/home")
                else:
                    LOGGER.warning(f"Year {year}: Zero housing units, skipping ratio calculation")
            
            if ratios:
                avg_ratio = sum(ratios.values()) / len(ratios)
                LOGGER.info(f"Average persons per home: {avg_ratio:.2f}")
            
            return ratios
            
        except Exception as e:
            LOGGER.error(f"Error calculating persons per home: {e}")
            return {}

    def forecast_residential_loads(self, historical_loads: Dict[str, Dict[int, Dict[str, float]]], 
                                  pop_housing_df, persons_per_home: Dict[int, float],
                                  forecast_start_year: int) -> Dict[str, Dict[int, float]]:
        """Forecast residential loads based on population projections and housing ratios."""
        try:
            LOGGER.info(f"Starting residential forecasting from year {forecast_start_year}")
            
            if pop_housing_df is None or pop_housing_df.empty:
                LOGGER.error("No population/housing data available")
                return {}
            
            if not persons_per_home:
                LOGGER.error("No persons per home ratios calculated")
                return {}
            
            # Calculate average persons per home ratio
            avg_ratio = sum(persons_per_home.values()) / len(persons_per_home)
            LOGGER.info(f"Using average persons per home ratio: {avg_ratio:.2f}")
            
            forecast_results = {}
            
            # Get years available for forecasting (beyond start year)
            future_years = pop_housing_df[pop_housing_df['Year'] >= forecast_start_year]['Year'].tolist()
            LOGGER.info(f"Forecasting for years: {future_years}")
            
            for polygon_name, polygon_data in historical_loads.items():
                forecast_results[polygon_name] = {}
                
                # Get historical residential loads for this polygon
                historical_res_loads = []
                historical_years = []
                
                for year, year_data in polygon_data.items():
                    if year < forecast_start_year and 'residential' in year_data:
                        historical_res_loads.append(year_data['residential'])
                        historical_years.append(year)
                
                if not historical_res_loads:
                    LOGGER.warning(f"No historical residential data for {polygon_name}")
                    continue
                
                # Calculate load per housing unit from historical data
                # Find overlapping years between historical data and population/housing data
                overlapping_years = []
                load_per_housing = []
                
                for year in historical_years:
                    pop_row = pop_housing_df[pop_housing_df['Year'] == year]
                    if not pop_row.empty:
                        housing_units = pop_row.iloc[0]['Housing']
                        if housing_units > 0:
                            res_load = polygon_data[year]['residential']
                            load_per_unit = res_load / housing_units
                            load_per_housing.append(load_per_unit)
                            overlapping_years.append(year)
                
                if not load_per_housing:
                    # Use average load from historical data
                    avg_load = sum(historical_res_loads) / len(historical_res_loads)
                    LOGGER.warning(f"No overlapping years for {polygon_name}, using average load: {avg_load:.2f}")
                    
                    for year in future_years:
                        forecast_results[polygon_name][int(year)] = avg_load
                else:
                    # Use average load per housing unit
                    avg_load_per_housing = sum(load_per_housing) / len(load_per_housing)
                    LOGGER.info(f"{polygon_name}: Average load per housing unit: {avg_load_per_housing:.4f}")
                    
                    # Project future loads
                    for year in future_years:
                        pop_row = pop_housing_df[pop_housing_df['Year'] == year]
                        if not pop_row.empty:
                            housing_units = pop_row.iloc[0]['Housing']
                            projected_load = avg_load_per_housing * housing_units
                            forecast_results[polygon_name][int(year)] = round(projected_load, 2)
                            LOGGER.info(f"{polygon_name} {int(year)}: {housing_units:,.0f} units × {avg_load_per_housing:.4f} = {projected_load:.2f}")
            
            return forecast_results
            
        except Exception as e:
            LOGGER.error(f"Error in residential forecasting: {e}")
            return {}

    def forecast_commercial_industrial_loads(self, historical_loads: Dict[str, Dict[int, Dict[str, float]]], 
                                           pop_housing_df, 
                                           forecast_start_year: int) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Forecast commercial and industrial loads using regression analysis against population."""
        try:
            # Import required libraries
            import numpy as np
            try:
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score
            except ImportError:
                LOGGER.error("scikit-learn not available - cannot perform regression analysis")
                return {}
            
            LOGGER.info(f"Starting commercial/industrial forecasting from year {forecast_start_year}")
            
            if pop_housing_df is None or len(pop_housing_df) == 0:
                LOGGER.error("No population data available")
                return {}
            
            forecast_results = {}
            
            # Get years available for forecasting (beyond start year)
            future_years = pop_housing_df[pop_housing_df['Year'] >= forecast_start_year]['Year'].tolist()
            LOGGER.info(f"Forecasting for years: {future_years}")
            
            for polygon_name, polygon_data in historical_loads.items():
                forecast_results[polygon_name] = {}
                
                # Process commercial and industrial separately
                for load_class in ['commercial', 'industrial']:
                    LOGGER.info(f"Processing {load_class} loads for {polygon_name}")
                    
                    # Collect historical data
                    historical_data = []
                    population_data = []
                    
                    for year, year_data in polygon_data.items():
                        if year < forecast_start_year and load_class in year_data:
                            # Find matching population data
                            pop_row = pop_housing_df[pop_housing_df['Year'] == year]
                            if not pop_row.empty:
                                historical_data.append(year_data[load_class])
                                population_data.append(pop_row.iloc[0]['Population'])
                    
                    if len(historical_data) < 2:
                        LOGGER.warning(f"Insufficient data for {load_class} regression in {polygon_name}")
                        # Use average historical load
                        if historical_data:
                            avg_load = historical_data[0]
                        else:
                            avg_load = 0
                        
                        for year in future_years:
                            if int(year) not in forecast_results[polygon_name]:
                                forecast_results[polygon_name][int(year)] = {}
                            forecast_results[polygon_name][int(year)][load_class] = avg_load
                        continue
                    
                    # Perform linear regression
                    X = np.array(population_data).reshape(-1, 1)
                    y = np.array(historical_data)
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Calculate R-squared
                    r2 = r2_score(y, model.predict(X))
                    LOGGER.info(f"{polygon_name} {load_class}: R² = {r2:.3f}, "
                              f"slope = {model.coef_[0]:.6f}, intercept = {model.intercept_:.2f}")
                    
                    # Project future loads
                    for year in future_years:
                        pop_row = pop_housing_df[pop_housing_df['Year'] == year]
                        if not pop_row.empty:
                            future_population = pop_row.iloc[0]['Population']
                            predicted_load = model.predict([[future_population]])[0]
                            
                            # Ensure non-negative loads
                            predicted_load = max(0, predicted_load)
                            
                            if int(year) not in forecast_results[polygon_name]:
                                forecast_results[polygon_name][int(year)] = {}
                            
                            forecast_results[polygon_name][int(year)][load_class] = round(predicted_load, 2)
                            LOGGER.info(f"{polygon_name} {int(year)} {load_class}: "
                                      f"pop={future_population:,.0f} → load={predicted_load:.2f}")
            
            return forecast_results
            
        except Exception as e:
            LOGGER.error(f"Error in commercial/industrial forecasting: {e}")
            return {}

    def project_units(self, area_dicts: Dict[str, Dict[int, float]], 
                     growth_dict: Dict[int, float], start_year: int, end_year: int, 
                     areas: List[str], case: str = 'fixed', threshold: float = 0.85,
                     priority_zones: List[Dict] = None) -> Dict[int, List[float]]:
        """Project load units over time using the forecasting template logic.
        
        Args:
            area_dicts: Dictionary mapping area names to {year: load, 'ultimate': ultimate_load}
            growth_dict: Dictionary mapping years to growth values
            start_year: Starting year for projection
            end_year: Ending year for projection
            areas: List of area names
            case: Projection case (fixed, variable, etc.)
            threshold: Development threshold for reduction factor
            priority_zones: List of priority zone configurations with zone_name, priority_level, start_year, end_year
            
        Returns:
            Dictionary mapping years to list of projected loads by area
        """
        LOGGER.info(" DEBUG: === project_units called ===")
        LOGGER.info(f" DEBUG: area_dicts = {area_dicts}")
        LOGGER.info(f" DEBUG: growth_dict = {growth_dict}")
        LOGGER.info(f" DEBUG: growth_dict types = {[(k, type(v)) for k, v in growth_dict.items()]}")
        LOGGER.info(f" DEBUG: start_year = {start_year}, end_year = {end_year}")
        LOGGER.info(f" DEBUG: areas = {areas}")
        res_dict = {start_year: [area_dicts[area][start_year] for area in areas]}
        
        ultimate_units = {area: area_dicts[area].get('ultimate', area_dicts[area][start_year]) for area in areas}
        initial_supply = [l - u for l, u in zip(ultimate_units.values(), res_dict[start_year])]
        factor_dict = {start_year: [u / s if s != 0 else 1 for u, s in zip(ultimate_units.values(), initial_supply)]}
        
        # Process priority zones configuration
        if priority_zones is None:
            priority_zones = []

        years_range = range(start_year, end_year + 1)
        # Ensure growth values are numeric to prevent concatenation errors
        total_units_changes = []
        for year in years_range:
            growth_value = growth_dict.get(year, 0)
            # Convert to float if it's numeric, otherwise use 0
            if isinstance(growth_value, (int, float)):
                total_units_changes.append(float(growth_value))
            elif isinstance(growth_value, (list, tuple)) and len(growth_value) > 0:
                # If it's a list/tuple, take the first numeric value
                first_val = growth_value[0] if isinstance(growth_value[0], (int, float)) else 0
                total_units_changes.append(float(first_val))
                LOGGER.warning(f"Growth dict contains non-numeric value for year {year}: {growth_value}. Using {first_val}")
            else:
                total_units_changes.append(0.0)
                LOGGER.warning(f"Growth dict contains invalid value for year {year}: {growth_value}. Using 0")
        
        for year_idx, year in enumerate(range(start_year + 1, end_year + 1)):
            # year_idx is 0-based from (start_year+1), but total_units_changes is indexed from start_year
            # So we need to offset by 1 to get the right growth value
            growth_idx = year_idx + 1
            
            # Calculate development percentage for each area
            development_pct = [res_dict[year-1][i] / list(ultimate_units.values())[i] 
                             if list(ultimate_units.values())[i] != 0 else 0
                             for i in range(len(areas))]
            
            # Apply reduction factor for areas over threshold
            reduction_factors = [0.25 if pct >= threshold else 1.0 for pct in development_pct]
            
            total_unit_inv = sum(factor_dict[year - 1])
            if total_unit_inv == 0:
                unit_inv = [0 for _ in factor_dict[year - 1]]
            else:
                unit_inv = [u / total_unit_inv for u in factor_dict[year - 1]]
            
            # Modify unit investment based on reduction factors
            unit_inv = [inv * red for inv, red in zip(unit_inv, reduction_factors)]
            
            # Normalize unit investments to sum to 1
            total_modified_inv = sum(unit_inv)
            if total_modified_inv > 0:
                unit_inv = [u / total_modified_inv for u in unit_inv]
            
            # Apply priority zone adjustments
            priority_applied = False
            for i, area in enumerate(areas):
                for zone_config in priority_zones:
                    zone_name = zone_config.get('zone_name', '')
                    priority_level = zone_config.get('priority_level', 'Normal (1.0x)')
                    zone_start_year = zone_config.get('start_year', start_year)
                    zone_end_year = zone_config.get('end_year', end_year)
                    
                    # Check if area matches zone and year is in range
                    if (area == zone_name and 
                        zone_start_year <= year <= zone_end_year):
                        
                        # Apply priority multiplier based on level
                        if 'High Priority (2.5x)' in priority_level:
                            unit_inv[i] *= 2.5
                        elif 'Priority (2.0x)' in priority_level:
                            unit_inv[i] *= 2.0
                        elif 'Low Priority (0.5x)' in priority_level:
                            unit_inv[i] *= 0.5
                        elif 'Depriority (0.25x)' in priority_level:
                            unit_inv[i] *= 0.25
                        elif 'Zero Priority (0.0x)' in priority_level:
                            unit_inv[i] = 0.0
                        # Normal (1.0x) doesn't change the value
                        
                        priority_applied = True
                        break  # Only apply the first matching zone
            
            # Re-normalize if any priority adjustments were made
            if priority_applied:
                total_modified_inv = sum(unit_inv)
                if total_modified_inv > 0:
                    unit_inv = [u / total_modified_inv for u in unit_inv]
            
            # Use numpy if available, otherwise use list comprehension
            try:
                import numpy as np
                units = np.array(unit_inv) * total_units_changes[growth_idx] + res_dict[year - 1]
                units = units.tolist()
            except ImportError:
                # Fallback without numpy
                units = [unit_inv[i] * total_units_changes[growth_idx] + res_dict[year - 1][i] 
                        for i in range(len(unit_inv))]
            
            # Handle excess over ultimate capacity
            try:
                import numpy as np
                ultimate_values = list(ultimate_units.values())
                diff = np.array(ultimate_values) - units
                if any(diff < 0):
                    negative_indices = np.where(diff < 0)[0].tolist()
                else:
                    negative_indices = []
            except ImportError:
                # Fallback without numpy
                ultimate_values = list(ultimate_units.values())
                diff = [ultimate_values[i] - units[i] for i in range(len(units))]
                negative_indices = [i for i, val in enumerate(diff) if val < 0]
            
            # Process negative indices (areas over ultimate capacity)
            # Track unallocated load that needs to be pushed to next year
            unallocated_load = 0.0
            
            for idx in negative_indices:
                    excess_load = -diff[idx]
                    units[idx] = list(ultimate_units.values())[idx]
                    
                    # Find remaining areas that are not at capacity AND have non-zero priority
                    # (unit_inv > 0 means the area can accept load based on priority)
                    remaining_indices = [i for i in range(len(units)) 
                                       if i not in negative_indices and unit_inv[i] > 0]
                    remaining_unit_inv = [unit_inv[i] for i in remaining_indices]
                    total_remaining_inv = sum(remaining_unit_inv)
                    
                    if total_remaining_inv > 0:
                        # Redistribute excess to eligible areas
                        for i in remaining_indices:
                            units[i] += (unit_inv[i] / total_remaining_inv) * excess_load
                    else:
                        # No eligible areas to receive excess load
                        # This happens when all non-full areas have zero priority
                        # Track this load to be pushed to next year
                        unallocated_load += excess_load
                        LOGGER.warning(f"Year {year}: Cannot allocate {excess_load:.2f} units from {areas[idx]} "
                                     f"(all remaining areas either at capacity or have zero priority). "
                                     f"Load will be deferred to next year.")
            
            # If there's unallocated load, reduce the total growth for this year
            # It will naturally appear in the next year's growth calculation
            if unallocated_load > 0:
                LOGGER.info(f"Year {year}: Deferring {unallocated_load:.2f} units to next year")

            # Adjustment factor to match total growth
            # Account for unallocated load when calculating target growth
            target_growth = total_units_changes[growth_idx] - unallocated_load
            
            total_units_added_actual = sum(units) - sum(res_dict[year - 1])
            if total_units_added_actual != 0:
                if total_units_added_actual != target_growth:
                    adjustment_factor = target_growth / total_units_added_actual
                    # Apply adjustment but prevent negative values
                    proposed_units = [u * adjustment_factor for u in units]
                    
                    # Check if adjustment would cause negative values
                    if any(u < 0 for u in proposed_units):
                        # Don't apply full adjustment - distribute growth proportionally instead
                        LOGGER.warning(f"Year {year}: Adjustment factor {adjustment_factor:.2f} would cause negative values. Using proportional distribution.")
                        # Distribute the target growth proportionally to areas not at capacity with non-zero priority
                        available_indices = [i for i in range(len(units)) 
                                           if units[i] < list(ultimate_units.values())[i] and unit_inv[i] > 0]
                        if available_indices:
                            available_inv = [unit_inv[i] for i in available_indices]
                            total_available_inv = sum(available_inv)
                            if total_available_inv > 0:
                                for i in available_indices:
                                    growth_share = (unit_inv[i] / total_available_inv) * target_growth
                                    units[i] = res_dict[year - 1][i] + growth_share
                            else:
                                # No capacity in eligible areas - defer to next year
                                LOGGER.warning(f"Year {year}: No capacity in non-zero priority areas. "
                                             f"Deferring {target_growth:.2f} units to next year.")
                                units = res_dict[year - 1]
                        else:
                            # All areas either at capacity or have zero priority - defer to next year
                            LOGGER.warning(f"Year {year}: All areas at capacity or zero priority. "
                                         f"Deferring {target_growth:.2f} units to next year.")
                            units = res_dict[year - 1]
                    else:
                        units = proposed_units
            else:
                adjustment_factor = 1
                units = [u * adjustment_factor for u in units]

            # Final safety check: ensure no values are negative
            units = [max(0, u) for u in units]
            
            # Final safety check: ensure no values exceed ultimate
            for i in range(len(units)):
                if units[i] > list(ultimate_units.values())[i]:
                    LOGGER.warning(f"Year {year}, Area {areas[i]}: Load {units[i]:.2f} exceeds ultimate {list(ultimate_units.values())[i]:.2f}. Capping.")
                    units[i] = list(ultimate_units.values())[i]

            res_dict[year] = units
            supply = [l - u for u, l in zip(units, list(ultimate_units.values()))]
            factor_dict[year] = [u / s if s != 0 else 1 for u, s in zip(list(ultimate_units.values()), supply)]
            factor_dict[year] = [max(0, min(f, 5)) for f in factor_dict[year]]
            
            # Ensure no negative or infinite values
            try:
                import numpy as np
                has_invalid = any(u < 0 or np.isinf(u) for u in res_dict[year])
            except ImportError:
                import math
                has_invalid = any(u < 0 or math.isinf(u) for u in res_dict[year])
            
            if has_invalid:
                res_dict[year] = list(ultimate_units.values())
            if sum(res_dict[year]) >= sum(list(ultimate_units.values())):
                res_dict[year] = list(ultimate_units.values())
                
        return res_dict

    def run_enhanced_forecasting(self, inputs):
        """Run enhanced forecasting with spatial data and population/housing analysis."""
        try:
            LOGGER.info(" Starting enhanced forecasting with spatial analysis")
            print(" Starting enhanced forecasting with spatial analysis")
            
            # Load population and housing data
            population_data = self.load_csv_data(
                inputs['population_file'], 
                inputs['population_fields']
            ) if inputs['population_file'] else {}
            
            housing_data = self.load_csv_data(
                inputs['housing_file'],
                inputs['housing_fields'] 
            ) if inputs['housing_file'] else {}
            
            LOGGER.info(f"Loaded population data for {len(population_data)} areas")
            LOGGER.info(f"Loaded housing data for {len(housing_data)} areas")
            print(f"Loaded population data for {len(population_data)} areas")
            print(f"Loaded housing data for {len(housing_data)} areas")
            
            # Get spatial layer information
            spatial_areas = self.get_spatial_areas(inputs['polygon_layer'])
            LOGGER.info(f"Found {len(spatial_areas)} spatial areas")
            print(f"Found {len(spatial_areas)} spatial areas")
            
            # Get current loads from polygon layer
            current_loads = self.get_current_loads_from_layer(inputs['polygon_layer'])
            LOGGER.info(f"Extracted current loads for {len(current_loads)} areas")
            print(f"Extracted current loads for {len(current_loads)} areas")
            
            # Check if aggregation mode is enabled
            aggregate_polygons = inputs.get('aggregate_polygons', False)
            assume_linear = inputs.get('assume_linear', False)
            
            if aggregate_polygons:
                LOGGER.info(" Using aggregated polygon regression analysis")
                print(" Using aggregated polygon regression analysis")
                forecast_results = self.perform_aggregated_regression_forecasting(
                    population_data,
                    housing_data,
                    spatial_areas,
                    current_loads,
                    inputs['forecast_start_year'],
                    inputs['forecast_end_year'],
                    inputs['growth_rates'],
                    inputs.get('priority_zones', []),
                    assume_linear=assume_linear
                )
            else:
                LOGGER.info(" Using individual polygon regression analysis")
                print(" Using individual polygon regression analysis")
                forecast_results = self.perform_regression_forecasting(
                    population_data,
                    housing_data,
                    spatial_areas,
                    current_loads,
                    inputs['forecast_start_year'],
                    inputs['forecast_end_year'],
                    inputs['growth_rates'],
                    inputs.get('priority_zones', []),
                    assume_linear=assume_linear
                )
            
            LOGGER.info(" Enhanced forecasting completed successfully")
            print(" Enhanced forecasting completed successfully")
            
            return forecast_results
            
        except Exception as e:
            LOGGER.error(f"Error in enhanced forecasting: {e}")
            print(f"Error in enhanced forecasting: {e}")
            import traceback
            LOGGER.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def run_basic_forecasting(self, inputs):
        """Run basic forecasting with simple inputs."""
        try:
            LOGGER.info(" Running basic forecasting")
            print(" Running basic forecasting")
            
            # Extract start and end years from forecast_years list
            forecast_years = inputs['forecast_years']
            start_year = min(forecast_years) if forecast_years else None
            end_year = max(forecast_years) if forecast_years else None
            
            LOGGER.info(f"Basic mode - extracted start_year = {start_year}, end_year = {end_year}")
            
            forecast_results = self.create_forecast_scenarios(
                inputs['current_loads'], 
                inputs['ultimate_loads'], 
                inputs['growth_projection'],
                start_year,
                end_year,
                priority_zones=inputs.get('priority_zones', [])
            )
            
            # Validate forecast results
            self.validate_forecast_results(
                forecast_results,
                inputs['current_loads'],
                inputs['ultimate_loads'],
                inputs['growth_projection'],
                start_year
            )
            
            LOGGER.info(" Basic forecasting completed successfully")
            print(" Basic forecasting completed successfully")
            
            return forecast_results
            
        except Exception as e:
            LOGGER.error(f"Error in basic forecasting: {e}")
            print(f"Error in basic forecasting: {e}")
            import traceback
            LOGGER.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def load_csv_data(self, file_path, field_mapping):
        """Load and parse CSV data."""
        try:
            import pandas as pd
            
            df = pd.read_csv(file_path)
            LOGGER.info(f"Loaded CSV with {len(df)} rows")
            
            # Rename columns according to field mapping
            area_field = field_mapping['area']
            year_field = field_mapping['year'] 
            value_field = field_mapping['value']
            
            # Group by area and create time series
            data_by_area = {}
            for area in df[area_field].unique():
                area_data = df[df[area_field] == area]
                time_series = {}
                for _, row in area_data.iterrows():
                    year = int(row[year_field])
                    value = float(row[value_field])
                    time_series[year] = value
                data_by_area[area] = time_series
                
            return data_by_area
            
        except Exception as e:
            LOGGER.error(f"Error loading CSV data: {e}")
            return {}

    def get_spatial_areas(self, layer_name):
        """Extract area information from spatial layer."""
        try:
            from qgis.core import QgsProject
            
            project = QgsProject.instance()
            layer = None
            
            # Find the layer by name
            for layer_id, layer_obj in project.mapLayers().items():
                if layer_obj.name() == layer_name:
                    layer = layer_obj
                    break
            
            if not layer:
                LOGGER.warning(f"Layer '{layer_name}' not found")
                return []
            
            areas = []
            for feature in layer.getFeatures():
                # Get area name (try common field names)
                area_name = None
                for field_name in ['name', 'Name', 'NAME', 'area', 'Area', 'AREA', 'id', 'ID']:
                    if field_name in [field.name() for field in feature.fields()]:
                        area_name = feature[field_name]
                        break
                
                if area_name:
                    areas.append(str(area_name))
            
            return areas
            
        except Exception as e:
            LOGGER.error(f"Error getting spatial areas: {e}")
            return []

    def get_current_loads_from_layer(self, layer_name):
        """Extract current load data from polygon layer by area and class.
        
        Returns dict like: {area_name: {'residential': load, 'commercial': load, 'industrial': load}}
        """
        try:
            from qgis.core import QgsProject
            
            project = QgsProject.instance()
            layer = None
            
            # Find the layer by name
            for layer_id, layer_obj in project.mapLayers().items():
                if layer_obj.name() == layer_name:
                    layer = layer_obj
                    break
            
            if not layer:
                LOGGER.warning(f"Layer '{layer_name}' not found for load extraction")
                return {}
            
            loads = {}
            field_names = [field.name() for field in layer.fields()]
            LOGGER.info(f"Available fields in layer: {field_names}")
            
            # Determine which load fields exist
            load_field_map = {}
            possible_names = {
                'residential': ['residential', 'Residential', 'RESIDENTIAL', 'res', 'Res', 'RES'],
                'commercial': ['commercial', 'Commercial', 'COMMERCIAL', 'com', 'Com', 'COM'],
                'industrial': ['industrial', 'Industrial', 'INDUSTRIAL', 'ind', 'Ind', 'IND']
            }
            
            for load_class, possible_field_names in possible_names.items():
                for field_name in possible_field_names:
                    if field_name in field_names:
                        load_field_map[load_class] = field_name
                        break
            
            LOGGER.info(f"Found load fields: {load_field_map}")
            
            for feature in layer.getFeatures():
                # Get area name
                area_name = None
                for field_name in ['name', 'Name', 'NAME', 'area', 'Area', 'AREA', 'id', 'ID']:
                    if field_name in field_names:
                        area_name = feature[field_name]
                        break
                
                if not area_name:
                    continue
                
                area_name = str(area_name)
                loads[area_name] = {}
                
                # Extract loads for each class
                for load_class, field_name in load_field_map.items():
                    try:
                        load_value = feature[field_name]
                        loads[area_name][load_class] = float(load_value) if load_value is not None else 0.0
                    except (ValueError, TypeError):
                        loads[area_name][load_class] = 0.0
                
                # If no load fields found, initialize to zero
                for load_class in ['residential', 'commercial', 'industrial']:
                    if load_class not in loads[area_name]:
                        loads[area_name][load_class] = 0.0
            
            LOGGER.info(f"Extracted loads for {len(loads)} areas")
            return loads
            
        except Exception as e:
            LOGGER.error(f"Error extracting current loads: {e}")
            import traceback
            LOGGER.error(f"Full traceback: {traceback.format_exc()}")
            return {}

    def perform_regression_forecasting(self, population_data, housing_data, spatial_areas, 
                                     current_loads, start_year, end_year, growth_rates, priority_zones=None,
                                     assume_linear=False):
        """Perform regression-based forecasting using population and housing data.
        
        Args:
            population_data: Population data by area
            housing_data: Housing data by area
            spatial_areas: List of spatial area names
            start_year: Start year for forecast
            end_year: End year for forecast
            growth_rates: Growth rates by class
            priority_zones: List of priority zone configurations
            assume_linear: If True, use only population for linear regression (ignore housing)
        """
        try:
            from sklearn.linear_model import LinearRegression
            import numpy as np
            
            results = {
                'residential': {},
                'commercial': {},
                'industrial': {}
            }
            
            forecast_years = list(range(start_year, end_year + 1))
            
            for area in spatial_areas:
                LOGGER.info(f"Processing area: {area}")
                
                # Get historical data for this area
                pop_history = population_data.get(area, {})
                housing_history = housing_data.get(area, {})
                area_current_loads = current_loads.get(area, {})
                
                if not pop_history and not housing_history:
                    LOGGER.warning(f"No data available for area {area}")
                    continue
                
                # Residential forecasting based on population/housing
                if pop_history or housing_history:
                    # Use only population if assume_linear is True, otherwise use housing if available
                    if assume_linear:
                        base_data = pop_history
                        LOGGER.info(f"Using linear regression (population only) for {area} residential")
                    else:
                        base_data = housing_history if housing_history else pop_history
                    
                    if len(base_data) >= 2:  # Need at least 2 points for regression
                        years = list(base_data.keys())
                        values = list(base_data.values())
                        
                        # Fit linear regression
                        X = np.array(years).reshape(-1, 1)
                        y = np.array(values)
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        # Project future years
                        future_years = np.array(forecast_years).reshape(-1, 1)
                        projected_base = model.predict(future_years)
                        
                        # Calculate residential load factor from current data
                        current_res_load = area_current_loads.get('residential', 0.0)
                        if base_data and current_res_load > 0:
                            most_recent_year = max(base_data.keys())
                            most_recent_value = base_data[most_recent_year]
                            residential_factor = current_res_load / most_recent_value if most_recent_value > 0 else 2.5
                            unit_type = "housing unit" if (not assume_linear and housing_history) else "person"
                            LOGGER.info(f"{area} residential: Calculated factor = {residential_factor:.3f} GJ/{unit_type} (from {current_res_load:.1f} GJ / {most_recent_value} {unit_type}s)")
                        else:
                            residential_factor = 2.5  # Default fallback
                        
                        for i, year in enumerate(forecast_years):
                            if year not in results['residential']:
                                results['residential'][year] = {}
                            
                            base_load = max(0, projected_base[i] * residential_factor)
                            # Apply growth rate
                            growth_factor = (1 + growth_rates['residential']) ** (year - start_year)
                            final_load = base_load * growth_factor
                            
                            results['residential'][year][area] = final_load
                
                # Commercial forecasting (simplified - based on population with different factor)
                if pop_history:
                    base_data = pop_history
                    if len(base_data) >= 2:
                        years = list(base_data.keys())
                        values = list(base_data.values())
                        
                        X = np.array(years).reshape(-1, 1)
                        y = np.array(values)
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        future_years = np.array(forecast_years).reshape(-1, 1)
                        projected_pop = model.predict(future_years)
                        
                        # Calculate commercial factor from current data
                        current_com_load = area_current_loads.get('commercial', 0.0)
                        if pop_history and current_com_load > 0:
                            most_recent_year = max(pop_history.keys())
                            most_recent_pop = pop_history[most_recent_year]
                            commercial_factor = current_com_load / most_recent_pop if most_recent_pop > 0 else 1.8
                            LOGGER.info(f"{area} commercial: Calculated factor = {commercial_factor:.3f} GJ/person (from {current_com_load:.1f} GJ / {most_recent_pop} people)")
                        else:
                            commercial_factor = 1.8  # Default fallback
                        
                        for i, year in enumerate(forecast_years):
                            if year not in results['commercial']:
                                results['commercial'][year] = {}
                            
                            base_load = max(0, projected_pop[i] * commercial_factor)
                            growth_factor = (1 + growth_rates['commercial']) ** (year - start_year)
                            final_load = base_load * growth_factor
                            
                            results['commercial'][year][area] = final_load
            
            # Add some industrial data (simplified)
            for year in forecast_years:
                if year not in results['industrial']:
                    results['industrial'][year] = {}
                # Add minimal industrial load for demonstration
                for area in spatial_areas[:2]:  # Only first 2 areas get industrial
                    results['industrial'][year][area] = 500.0 * ((year - start_year + 1) / 10)
            
            return results
            
        except Exception as e:
            LOGGER.error(f"Error in regression forecasting: {e}")
            import traceback
            LOGGER.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def perform_aggregated_regression_forecasting(self, population_data, housing_data, spatial_areas, 
                                                current_loads, start_year, end_year, growth_rates, priority_zones=None,
                                                assume_linear=False):
        """Perform regression-based forecasting using combined polygon data.
        
        This method aggregates all polygon data, performs regression analysis on the combined dataset
        to determine total load growth by class, then uses the projection function from the template
        to distribute the forecasted loads across sub-polygons.
        
        Args:
            population_data: Population data by area
            housing_data: Housing data by area  
            spatial_areas: List of spatial area names
            start_year: Start year for forecast
            end_year: End year for forecast
            growth_rates: Growth rates by class
            priority_zones: List of priority zone configurations
            assume_linear: If True, use only population for linear regression (ignore housing)
            
        Returns:
            Dictionary with forecast results by class, year, and area
        """
        try:
            from sklearn.linear_model import LinearRegression
            import numpy as np
            
            LOGGER.info(" Starting aggregated regression analysis")
            
            # Combine all area data into aggregated time series
            combined_population = {}
            combined_housing = {}
            
            # Aggregate population data across all areas
            all_years = set()
            for area_data in population_data.values():
                all_years.update(area_data.keys())
            for area_data in housing_data.values():
                all_years.update(area_data.keys())
            
            all_years = sorted(all_years)
            LOGGER.info(f"Found years spanning: {min(all_years) if all_years else 'None'} to {max(all_years) if all_years else 'None'}")
            
            # Sum population and housing across all areas by year
            for year in all_years:
                total_pop = sum(area_data.get(year, 0) for area_data in population_data.values())
                total_housing = sum(area_data.get(year, 0) for area_data in housing_data.values())
                
                if total_pop > 0:
                    combined_population[year] = total_pop
                if total_housing > 0:
                    combined_housing[year] = total_housing
            
            LOGGER.info(f"Combined population data: {len(combined_population)} years")
            LOGGER.info(f"Combined housing data: {len(combined_housing)} years")
            
            # Perform regression analysis on aggregated data
            forecast_years = list(range(start_year, end_year + 1))
            total_growth_by_class = {}
            
            # Calculate total current loads across all areas
            total_current_loads = {'residential': 0.0, 'commercial': 0.0, 'industrial': 0.0}
            for area_loads in current_loads.values():
                for load_class in ['residential', 'commercial', 'industrial']:
                    total_current_loads[load_class] += area_loads.get(load_class, 0.0)
            
            LOGGER.info(f"Total current loads: {total_current_loads}")
            
            # Calculate total growth patterns by class using regression
            for load_class in ['residential', 'commercial', 'industrial']:
                LOGGER.info(f"Analyzing {load_class} growth pattern")
                
                # Calculate load factor from aggregated current data
                current_total_load = total_current_loads[load_class]
                
                if assume_linear:
                    # Use only population for all load classes (linear regression)
                    base_data = combined_population
                    # Calculate GJ per person from most recent aggregated data
                    if combined_population and current_total_load > 0:
                        most_recent_year = max(combined_population.keys())
                        most_recent_pop = combined_population[most_recent_year]
                        load_factor = current_total_load / most_recent_pop if most_recent_pop > 0 else (2.5 if load_class == 'residential' else 1.8 if load_class == 'commercial' else 0.8)
                        LOGGER.info(f"{load_class}: Calculated aggregated factor = {load_factor:.3f} GJ/person (from {current_total_load:.1f} GJ / {most_recent_pop} people)")
                    else:
                        load_factor = 2.5 if load_class == 'residential' else 1.8 if load_class == 'commercial' else 0.8
                    LOGGER.info(f"Using linear regression (population only) for {load_class}")
                else:
                    # Use housing data for residential if available, otherwise population
                    if load_class == 'residential':
                        base_data = combined_housing if combined_housing else combined_population
                        if base_data and current_total_load > 0:
                            most_recent_year = max(base_data.keys())
                            most_recent_value = base_data[most_recent_year]
                            load_factor = current_total_load / most_recent_value if most_recent_value > 0 else 2.5
                            unit_type = "housing unit" if combined_housing else "person"
                            LOGGER.info(f"{load_class}: Calculated aggregated factor = {load_factor:.3f} GJ/{unit_type} (from {current_total_load:.1f} GJ / {most_recent_value} {unit_type}s)")
                        else:
                            load_factor = 2.5
                    elif load_class == 'commercial':
                        base_data = combined_population
                        if combined_population and current_total_load > 0:
                            most_recent_year = max(combined_population.keys())
                            most_recent_pop = combined_population[most_recent_year]
                            load_factor = current_total_load / most_recent_pop if most_recent_pop > 0 else 1.8
                            LOGGER.info(f"{load_class}: Calculated aggregated factor = {load_factor:.3f} GJ/person (from {current_total_load:.1f} GJ / {most_recent_pop} people)")
                        else:
                            load_factor = 1.8
                    else:  # industrial
                        base_data = combined_population
                        if combined_population and current_total_load > 0:
                            most_recent_year = max(combined_population.keys())
                            most_recent_pop = combined_population[most_recent_year]
                            load_factor = current_total_load / most_recent_pop if most_recent_pop > 0 else 0.8
                            LOGGER.info(f"{load_class}: Calculated aggregated factor = {load_factor:.3f} GJ/person (from {current_total_load:.1f} GJ / {most_recent_pop} people)")
                        else:
                            load_factor = 0.8
                
                if not base_data or len(base_data) < 2:
                    LOGGER.warning(f"Insufficient data for {load_class} regression")
                    # Use simple growth rate
                    total_growth_by_class[load_class] = {
                        year: 1000 * (1 + growth_rates.get(load_class, 0.02)) ** (year - start_year)
                        for year in forecast_years
                    }
                    continue
                
                # Perform regression on aggregated data
                years = list(base_data.keys())
                values = [base_data[year] * load_factor for year in years]
                
                if len(years) >= 2:
                    X = np.array(years).reshape(-1, 1)
                    y = np.array(values)
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Calculate R-squared
                    r2 = model.score(X, y)
                    LOGGER.info(f"{load_class} regression: R² = {r2:.3f}, slope = {model.coef_[0]:.2f}")
                    
                    # Project total loads for forecast years
                    future_X = np.array(forecast_years).reshape(-1, 1)
                    projected_totals = model.predict(future_X)
                    
                    # Apply growth rates
                    total_growth_by_class[load_class] = {}
                    for i, year in enumerate(forecast_years):
                        base_load = max(0, projected_totals[i])
                        growth_factor = (1 + growth_rates.get(load_class, 0.02)) ** (year - start_year)
                        total_growth_by_class[load_class][year] = base_load * growth_factor
                        
                        LOGGER.info(f"{load_class} {year}: Total projected load = {total_growth_by_class[load_class][year]:.2f}")
            
            # Now distribute the total projected loads across areas using the projection function
            LOGGER.info(" Distributing aggregated forecasts across polygons")
            
            results = {
                'residential': {},
                'commercial': {},
                'industrial': {}
            }
            
            for load_class in ['residential', 'commercial', 'industrial']:
                if load_class not in total_growth_by_class:
                    continue
                
                # Create area dictionaries for projection function
                area_dicts = {}
                for area in spatial_areas:
                    # Estimate current load for this area (simplified)
                    area_pop = sum(population_data.get(area, {}).values()) / max(1, len(population_data.get(area, {})))
                    area_housing = sum(housing_data.get(area, {}).values()) / max(1, len(housing_data.get(area, {})))
                    
                    if load_class == 'residential':
                        current_load = max(1, area_housing * 2.5 if area_housing else area_pop * 1.5)
                        ultimate_load = current_load * 3  # Assume 3x current as ultimate
                    elif load_class == 'commercial':
                        current_load = max(1, area_pop * 1.2)
                        ultimate_load = current_load * 2.5
                    else:  # industrial
                        current_load = max(1, area_pop * 0.6)
                        ultimate_load = current_load * 2
                    
                    area_dicts[area] = {
                        start_year: current_load,
                        'ultimate': ultimate_load
                    }
                
                # Calculate growth increments from total projections
                growth_dict = {}
                prev_total = sum(area_dict[start_year] for area_dict in area_dicts.values())
                
                for year in forecast_years:
                    if year == start_year:
                        growth_dict[year] = 0
                    else:
                        current_total = total_growth_by_class[load_class][year]
                        growth_increment = current_total - prev_total
                        growth_dict[year] = max(0, growth_increment)
                        prev_total = current_total
                
                LOGGER.info(f"Calling project_units for {load_class} with growth_dict sample: {dict(list(growth_dict.items())[:3])}")
                
                # Use the projection function to distribute across areas
                projection_results = self.project_units(
                    area_dicts,
                    growth_dict,
                    start_year,
                    end_year,
                    list(area_dicts.keys()),
                    priority_zones=priority_zones or []
                )
                
                # Convert projection results to the expected format
                for year in forecast_years:
                    if year not in results[load_class]:
                        results[load_class][year] = {}
                    
                    if year in projection_results:
                        for i, area in enumerate(area_dicts.keys()):
                            results[load_class][year][area] = projection_results[year][i]
            
            LOGGER.info(" Aggregated regression forecasting completed")
            return results
            
        except Exception as e:
            LOGGER.error(f"Error in aggregated regression forecasting: {e}")
            import traceback
            LOGGER.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def create_forecast_scenarios(self, current_loads: Dict[str, Dict[str, float]], 
                                ultimate_loads: Dict[str, Dict[str, float]], 
                                growth_projection: Dict[int, float],
                                start_year: int = None,
                                end_year: int = None,
                                priority_zones: List[Dict] = None) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Create forecast scenarios for residential and commercial/industrial loads.
        
        Args:
            current_loads: Current loads by area and class
            ultimate_loads: Ultimate loads by area and class  
            growth_projection: Annual growth projections
            start_year: Starting year for forecast (default: current year)
            end_year: Ending year for forecast (default: start_year + 20)
            
        Returns:
            Dictionary with forecast results by class, year, and area
        """
        if start_year is None:
            current_year = datetime.now().year
            start_year = current_year
        if end_year is None:
            end_year = start_year + 20
            
        forecast_years = list(range(start_year, end_year + 1))
        
        # Separate loads by class
        area_dicts_res = {}
        area_dicts_comm = {}
        area_dicts_ind = {}
        
        for area in current_loads.keys():
            # Residential loads (including apartments)
            res_current = current_loads[area].get('residential', 0) + current_loads[area].get('apartments', 0)
            res_ultimate = ultimate_loads[area].get('residential', 0) + ultimate_loads[area].get('apartments', 0)
            
            if res_ultimate > 0:
                area_dicts_res[area] = {start_year: res_current, 'ultimate': res_ultimate}
            
            # Commercial loads
            comm_current = current_loads[area].get('commercial', 0)
            comm_ultimate = ultimate_loads[area].get('commercial', 0)
            
            if comm_ultimate > 0:
                area_dicts_comm[area] = {start_year: comm_current, 'ultimate': comm_ultimate}
                
            # Industrial loads  
            ind_current = current_loads[area].get('industrial', 0)
            ind_ultimate = ultimate_loads[area].get('industrial', 0)
            
            if ind_ultimate > 0:
                area_dicts_ind[area] = {start_year: ind_current, 'ultimate': ind_ultimate}
        
        # Project loads
        max_forecast_year = max(forecast_years)
        
        results = {
            'residential': {},
            'commercial': {},
            'industrial': {}
        }
        
        if area_dicts_res:
            # Extract residential growth values for project_units
            LOGGER.info(" DEBUG: Processing residential loads...")
            res_growth = {}
            for year, growth_data in growth_projection.items():
                if isinstance(growth_data, dict):
                    res_growth[year] = growth_data.get('residential', 0)
                else:
                    res_growth[year] = growth_data
            
            LOGGER.info(f" DEBUG: area_dicts_res = {area_dicts_res}")
            LOGGER.info(f" DEBUG: res_growth sample = {dict(list(res_growth.items())[:3])}")
            LOGGER.info(f" DEBUG: res_growth types = {[(k, type(v)) for k, v in list(res_growth.items())[:3]]}")
            
            LOGGER.info(" DEBUG: Calling project_units for residential...")
            try:
                res_projection = self.project_units(
                    area_dicts_res, res_growth, start_year, max_forecast_year, 
                    list(area_dicts_res.keys()), priority_zones=priority_zones or []
                )
                LOGGER.info(" DEBUG: Residential project_units completed successfully")
            except Exception as e:
                LOGGER.error(f" DEBUG: Error in residential project_units: {e}")
                LOGGER.error(f" DEBUG: Error type: {type(e).__name__}")
                import traceback
                LOGGER.error(f" DEBUG: Full traceback: {traceback.format_exc()}")
                raise
            # Extract forecast years
            for year in forecast_years:
                if year in res_projection:
                    results['residential'][year] = {
                        area: load for area, load in zip(list(area_dicts_res.keys()), res_projection[year])
                    }
        
        if area_dicts_comm:
            # Extract commercial growth values for project_units  
            LOGGER.info(" DEBUG: Processing commercial loads...")
            comm_growth = {}
            for year, growth_data in growth_projection.items():
                if isinstance(growth_data, dict):
                    comm_growth[year] = growth_data.get('commercial', 0)
                else:
                    comm_growth[year] = growth_data
            
            LOGGER.info(f" DEBUG: area_dicts_comm = {area_dicts_comm}")
            LOGGER.info(f" DEBUG: comm_growth sample = {dict(list(comm_growth.items())[:3])}")
            
            LOGGER.info(" DEBUG: Calling project_units for commercial...")
            try:
                comm_projection = self.project_units(
                    area_dicts_comm, comm_growth, start_year, max_forecast_year,
                    list(area_dicts_comm.keys()), priority_zones=priority_zones or []
                )
                LOGGER.info(" DEBUG: Commercial project_units completed successfully")
            except Exception as e:
                LOGGER.error(f" DEBUG: Error in commercial project_units: {e}")
                LOGGER.error(f" DEBUG: Error type: {type(e).__name__}")
                import traceback
                LOGGER.error(f" DEBUG: Full traceback: {traceback.format_exc()}")
                raise
            # Extract forecast years
            for year in forecast_years:
                if year in comm_projection:
                    results['commercial'][year] = {
                        area: load for area, load in zip(list(area_dicts_comm.keys()), comm_projection[year])
                    }
                    
        if area_dicts_ind:
            # Extract industrial growth values for project_units  
            LOGGER.info(" DEBUG: Processing industrial loads...")
            ind_growth = {}
            for year, growth_data in growth_projection.items():
                if isinstance(growth_data, dict):
                    ind_growth[year] = growth_data.get('industrial', 0)
                else:
                    ind_growth[year] = growth_data
            
            LOGGER.info(f" DEBUG: area_dicts_ind = {area_dicts_ind}")
            LOGGER.info(f" DEBUG: ind_growth sample = {dict(list(ind_growth.items())[:3])}")
            
            LOGGER.info(" DEBUG: Calling project_units for industrial...")
            try:
                ind_projection = self.project_units(
                    area_dicts_ind, ind_growth, start_year, max_forecast_year,
                    list(area_dicts_ind.keys()), priority_zones=priority_zones or []
                )
                LOGGER.info(" DEBUG: Industrial project_units completed successfully")
            except Exception as e:
                LOGGER.error(f" DEBUG: Error in industrial project_units: {e}")
                LOGGER.error(f" DEBUG: Error type: {type(e).__name__}")
                import traceback
                LOGGER.error(f" DEBUG: Full traceback: {traceback.format_exc()}")
                raise
            # Extract forecast years
            for year in forecast_years:
                if year in ind_projection:
                    results['industrial'][year] = {
                        area: load for area, load in zip(list(area_dicts_ind.keys()), ind_projection[year])
                    }
        
        return results

    def validate_forecast_results(self, forecast_results: Dict[str, Dict[int, Dict[str, float]]],
                                  current_loads: Dict[str, Dict[str, float]],
                                  ultimate_loads: Dict[str, Dict[str, float]],
                                  growth_projection: Dict[int, float],
                                  start_year: int) -> bool:
        """Validate forecast results for consistency and constraints.
        
        Checks:
        1. Start load + CSV growth = final forecast load for each year
        2. No area exceeds its ultimate capacity
        3. No negative loads
        
        Args:
            forecast_results: Forecast results by class, year, area
            current_loads: Current loads by area and class
            ultimate_loads: Ultimate loads by area and class
            growth_projection: Annual growth projections from CSV
            start_year: Starting year
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        validation_passed = True
        
        LOGGER.info("\n" + "="*80)
        LOGGER.info("VALIDATING FORECAST RESULTS")
        LOGGER.info("="*80)
        
        # Get all areas
        all_areas = set()
        for load_class in ['residential', 'commercial', 'industrial']:
            if load_class in forecast_results:
                for year_data in forecast_results[load_class].values():
                    all_areas.update(year_data.keys())
        
        # Check 1: Verify total growth matches CSV input
        LOGGER.info("\n1. Checking that forecast matches CSV growth projections:")
        for year in sorted(forecast_results.get('residential', {}).keys()):
            if year == start_year:
                continue  # Skip start year
                
            # Calculate actual total load for this year
            year_total = 0
            prev_year_total = 0
            
            for load_class in ['residential', 'commercial', 'industrial']:
                if load_class in forecast_results:
                    if year in forecast_results[load_class]:
                        year_total += sum(forecast_results[load_class][year].values())
                    if year - 1 in forecast_results[load_class]:
                        prev_year_total += sum(forecast_results[load_class][year - 1].values())
            
            actual_growth = year_total - prev_year_total
            
            # Get expected growth from CSV
            expected_growth = 0
            if isinstance(growth_projection.get(year), dict):
                for load_class in ['residential', 'commercial', 'industrial']:
                    expected_growth += growth_projection[year].get(load_class, 0)
            else:
                expected_growth = growth_projection.get(year, 0)
            
            diff = abs(actual_growth - expected_growth)
            if diff > 0.01:  # Allow small rounding errors
                LOGGER.warning(f"   Year {year}: Expected growth {expected_growth:.2f}, Actual growth {actual_growth:.2f}, Diff {diff:.2f}")
                validation_passed = False
            else:
                LOGGER.info(f"   Year {year}: Growth matches ({actual_growth:.2f} GJ/d) ✓")
        
        # Check 2: Verify no area exceeds ultimate capacity
        LOGGER.info("\n2. Checking that no area exceeds ultimate capacity:")
        for load_class in ['residential', 'commercial', 'industrial']:
            if load_class not in forecast_results:
                continue
                
            for year, area_loads in forecast_results[load_class].items():
                for area, load in area_loads.items():
                    # Get ultimate for this area and class
                    ultimate = 0
                    if area in ultimate_loads:
                        if load_class == 'residential':
                            ultimate = ultimate_loads[area].get('residential', 0) + ultimate_loads[area].get('apartments', 0)
                        else:
                            ultimate = ultimate_loads[area].get(load_class, 0)
                    
                    if load > ultimate + 0.01:  # Allow small rounding errors
                        LOGGER.error(f"   {area} ({load_class}) Year {year}: Load {load:.2f} EXCEEDS ultimate {ultimate:.2f}!")
                        validation_passed = False
                    elif year == max(forecast_results[load_class].keys()) and ultimate > 0:
                        pct = (load / ultimate) * 100
                        LOGGER.info(f"   {area} ({load_class}): {load:.2f} / {ultimate:.2f} ({pct:.1f}%) ✓")
        
        # Check 3: Verify no negative loads
        LOGGER.info("\n3. Checking for negative loads:")
        found_negative = False
        for load_class in ['residential', 'commercial', 'industrial']:
            if load_class not in forecast_results:
                continue
                
            for year, area_loads in forecast_results[load_class].items():
                for area, load in area_loads.items():
                    if load < 0:
                        LOGGER.error(f"   {area} ({load_class}) Year {year}: NEGATIVE load {load:.2f}!")
                        validation_passed = False
                        found_negative = True
        
        if not found_negative:
            LOGGER.info("   No negative loads found ✓")
        
        LOGGER.info("\n" + "="*80)
        if validation_passed:
            LOGGER.info("VALIDATION PASSED ✓")
        else:
            LOGGER.error("VALIDATION FAILED - See warnings/errors above")
        LOGGER.info("="*80 + "\n")
        
        return validation_passed

    def create_forecast_output(self, forecast_results: Dict[str, Dict[int, Dict[str, float]]]):
        """Create output layer/table with forecast results."""
        try:
            # This would create a new layer or table showing:
            # - Forecast loads by area and class for each forecast year
            # - Growth rates and trends
            # - Summary statistics
            
            LOGGER.info("Creating forecast output")
            
            for load_class, yearly_data in forecast_results.items():
                LOGGER.info(f"\n{load_class.title()} Load Forecasts:")
                for year, area_loads in yearly_data.items():
                    total_load = sum(area_loads.values())
                    LOGGER.info(f"  {year}: Total {total_load:.2f}")
                    for area, load in area_loads.items():
                        LOGGER.info(f"    {area}: {load:.2f}")
                        
        except Exception as e:
            LOGGER.error(f"Error creating forecast output: {e}")

    def run(self):
        """Run the Load Forecast plugin."""
        try:
            # Let user choose between basic, full mode, or load assignment
            if self.iface:
                selected_mode = self.show_mode_selection_dialog()
                if selected_mode is None:
                    return  # User cancelled
                
                # Check if load assignment was selected (mode ID = 2)
                if selected_mode == 2:
                    # Run load assignment tool
                    LOGGER.info(" Running Load Assignment Tool")
                    print(" Running Load Assignment Tool")
                    self.run_load_assignment()
                    return
                
                # Check if full mode was selected (mode ID = 1)
                use_full_mode = (selected_mode == 1) and self.enhanced_features_available
            else:
                # Non-GUI mode: use whatever is available
                use_full_mode = self.enhanced_features_available
            
            # Show current functionality status
            if use_full_mode:
                LOGGER.info(" Running Load Forecast - FULL MODE")
                print(" Running Load Forecast - FULL MODE")
            else:
                LOGGER.info(" Running Load Forecast - BASIC MODE")
                print(" Running Load Forecast - BASIC MODE")
            
            LOGGER.info(" DEBUG: Starting forecast plugin execution")
            print(" DEBUG: Starting forecast plugin execution")
            
            if self.iface:
                if use_full_mode:
                    # Enhanced mode - use spatial analysis with layers
                    LOGGER.info(" Starting FULL MODE with spatial analysis")
                    print(" Starting FULL MODE with spatial analysis")
                    success, inputs = self.show_enhanced_input_dialog()
                    if not success:
                        return
                    
                    # Validate layers if present
                    validator = DataValidator()
                    LOGGER.info(" Validating enhanced forecast inputs...")
                    
                    # Validate input parameters
                    inputs_validator = validate_forecast_inputs(inputs)
                    validator.errors.extend(inputs_validator.errors)
                    validator.warnings.extend(inputs_validator.warnings)
                    validator.info.extend(inputs_validator.info)
                    
                    # Validate population layer
                    if 'population_layer' in inputs and inputs['population_layer']:
                        layer_validator = validate_layer_data(
                            inputs['population_layer'],
                            "Population Layer",
                            required_fields=['population', 'year']
                        )
                        validator.errors.extend(layer_validator.errors)
                        validator.warnings.extend(layer_validator.warnings)
                        validator.info.extend(layer_validator.info)
                    
                    # Validate housing layer
                    if 'housing_layer' in inputs and inputs['housing_layer']:
                        layer_validator = validate_layer_data(
                            inputs['housing_layer'],
                            "Housing Layer",
                            required_fields=['housing_units', 'year']
                        )
                        validator.errors.extend(layer_validator.errors)
                        validator.warnings.extend(layer_validator.warnings)
                        validator.info.extend(layer_validator.info)
                    
                    # Validate spatial layer
                    if 'spatial_layer' in inputs and inputs['spatial_layer']:
                        layer_validator = validate_layer_data(
                            inputs['spatial_layer'],
                            "Spatial Areas Layer",
                            required_fields=['area_name']
                        )
                        validator.errors.extend(layer_validator.errors)
                        validator.warnings.extend(layer_validator.warnings)
                        validator.info.extend(layer_validator.info)
                    
                    # Show validation dialog
                    if not show_validation_dialog(validator,
                                                  title="Enhanced Forecast - Data Validation",
                                                  parent=self.iface.mainWindow()):
                        LOGGER.info(" User cancelled after validation")
                        return
                    
                    # Process with enhanced spatial forecasting
                    forecast_results = self.run_enhanced_forecasting(inputs)
                else:
                    # Basic mode - use simple input dialog
                    LOGGER.info(" Starting BASIC MODE with simple inputs")
                    print(" Starting BASIC MODE with simple inputs")
                    success, inputs = self.show_input_dialog()
                    if not success:
                        return
                    
                    # Validate inputs before processing
                    LOGGER.info(" Validating basic forecast inputs...")
                    validator = validate_forecast_inputs(inputs)
                    
                    # Show validation dialog
                    if not show_validation_dialog(validator, 
                                                  title="Basic Forecast - Data Validation",
                                                  parent=self.iface.mainWindow()):
                        LOGGER.info(" User cancelled after validation")
                        return
                    
                    # Process with basic forecasting
                    forecast_results = self.run_basic_forecasting(inputs)
                
                # Show results
                if forecast_results:
                    self.show_results_dialog(forecast_results)
                else:
                    from qgis.PyQt.QtWidgets import QMessageBox
                    QMessageBox.warning(
                        self.iface.mainWindow(),
                        "Warning", 
                        "No forecast could be generated. Please check your inputs."
                    )
                return
                
                # This old code should not be reached anymore
                from qgis.PyQt.QtWidgets import QMessageBox
                if use_full_mode:
                    QMessageBox.information(
                        self.iface.mainWindow(),
                        " Gas Hydraulics - Full Functionality",
                        "Plugin running with FULL FUNCTIONALITY!\n\n"
                        " All dependencies available (pandas, numpy, scikit-learn)\n\n"
                        "Enhanced capabilities include:\n"
                        "• Spatial load analysis by polygon\n"
                        "• Population-based residential forecasting\n"
                        "• Regression-based commercial/industrial forecasting\n"
                        "• Advanced statistical modeling\n\n"
                        "All features are ready to use!"
                    )
                else:
                    # Show installation guidance
                    reply = QMessageBox.question(
                        self.iface.mainWindow(),
                        "️ Gas Hydraulics - Basic Mode",
                        "Plugin running in BASIC MODE (limited functionality)\n\n"
                        " Enhanced forecasting features require additional packages:\n"
                        "• pandas (for data processing)\n"
                        "• numpy (for numerical calculations)\n"
                        "• scikit-learn (for regression analysis)\n\n"
                        "Basic forecasting functionality is still available.\n\n"
                        "Would you like installation instructions for full features?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    
                    if reply == QMessageBox.Yes:
                        QMessageBox.information(
                            self.iface.mainWindow(),
                            "Installation Instructions",
                            "To enable enhanced forecasting features:\n\n"
                            "1. Open Command Prompt/Terminal as Administrator\n"
                            "2. Run these commands:\n"
                            "   pip install pandas>=1.3.0\n"
                            "   pip install numpy>=1.20.0\n"
                            "   pip install scikit-learn>=1.0.0\n\n"
                            "3. Restart QGIS\n\n"
                            "Alternative: Run setup_dev.py from the plugin folder"
                        )
                
                # Create forecasts with user inputs using original method
                LOGGER.info(" DEBUG: BASIC MODE - Input data prepared")
                print(" DEBUG: BASIC MODE - Input data prepared")
                LOGGER.info(f" DEBUG: BASIC MODE - current_loads = {inputs['current_loads']}")
                print(f" DEBUG: BASIC MODE - current_loads = {inputs['current_loads']}")
                LOGGER.info(f" DEBUG: BASIC MODE - ultimate_loads = {inputs['ultimate_loads']}")
                print(f" DEBUG: BASIC MODE - ultimate_loads = {inputs['ultimate_loads']}")
                LOGGER.info(f" DEBUG: BASIC MODE - growth_projection sample = {dict(list(inputs['growth_projection'].items())[:5])}")
                print(f" DEBUG: BASIC MODE - growth_projection sample = {dict(list(inputs['growth_projection'].items())[:5])}")
                LOGGER.info(f" DEBUG: BASIC MODE - growth_projection types = {[(k, type(v)) for k, v in list(inputs['growth_projection'].items())[:5]]}")
                
                LOGGER.info(" DEBUG: BASIC MODE - Calling create_forecast_scenarios...")
                print(" DEBUG: BASIC MODE - Calling create_forecast_scenarios...")
                
                # Extract start and end years from forecast_years list
                forecast_years = inputs['forecast_years']
                start_year = min(forecast_years) if forecast_years else None
                end_year = max(forecast_years) if forecast_years else None
                
                LOGGER.info(f" DEBUG: BASIC MODE - forecast_years = {forecast_years}")
                LOGGER.info(f" DEBUG: BASIC MODE - extracted start_year = {start_year}, end_year = {end_year}")
                print(f" DEBUG: BASIC MODE - forecast_years = {forecast_years}")
                print(f" DEBUG: BASIC MODE - extracted start_year = {start_year}, end_year = {end_year}")
                
                try:
                    forecast_results = self.create_forecast_scenarios(
                        inputs['current_loads'], 
                        inputs['ultimate_loads'], 
                        inputs['growth_projection'],
                        start_year,
                        end_year,
                        priority_zones=[]
                    )
                    LOGGER.info(" DEBUG: BASIC MODE - create_forecast_scenarios completed successfully")
                    print(" DEBUG: BASIC MODE - create_forecast_scenarios completed successfully")
                except Exception as e:
                    LOGGER.error(f" DEBUG: BASIC MODE - Error in create_forecast_scenarios: {e}")
                    print(f" DEBUG: BASIC MODE - Error in create_forecast_scenarios: {e}")
                    LOGGER.error(f" DEBUG: BASIC MODE - Error type: {type(e).__name__}")
                    print(f" DEBUG: BASIC MODE - Error type: {type(e).__name__}")
                    import traceback
                    LOGGER.error(f" DEBUG: BASIC MODE - Full traceback: {traceback.format_exc()}")
                    print(f" DEBUG: BASIC MODE - Full traceback: {traceback.format_exc()}")
                    raise
                
                if forecast_results:
                    # Show results dialog
                    self.show_results_dialog(forecast_results)
                else:
                    from qgis.PyQt.QtWidgets import QMessageBox
                    QMessageBox.warning(
                        self.iface.mainWindow(),
                        "Warning", 
                        "No forecast could be generated. Please check your inputs."
                    )
            else:
                # Non-QGIS execution with sample data
                print("Load Forecast plugin executed successfully")
                
                # Sample data for demonstration
                current_loads = {
                    "Arbour Hills": {"residential": 195, "commercial": 312, "industrial": 0},
                    "Meadow View": {"residential": 1468, "commercial": 375, "industrial": 0},
                    "South East": {"residential": 2690, "commercial": 650, "industrial": 0}
                }
                
                ultimate_loads = {
                    "Arbour Hills": {"residential": 1854, "commercial": 2350, "industrial": 0},
                    "Meadow View": {"residential": 4309, "commercial": 445, "industrial": 0},
                    "South East": {"residential": 11060, "commercial": 1281, "industrial": 1783}
                }
                
                # Sample growth projection (annual growth)
                growth_projection = {year: 100.0 + year * 2.0 for year in range(2025, 2046)}
                
                LOGGER.info(" DEBUG: Input data prepared")
                LOGGER.info(f" DEBUG: current_loads = {current_loads}")
                LOGGER.info(f" DEBUG: ultimate_loads = {ultimate_loads}")
                LOGGER.info(f" DEBUG: growth_projection sample = {dict(list(growth_projection.items())[:5])}")
                LOGGER.info(f" DEBUG: growth_projection types = {[(k, type(v)) for k, v in list(growth_projection.items())[:5]]}")
                
                # Create forecasts
                LOGGER.info(" DEBUG: Calling create_forecast_scenarios...")
                try:
                    forecast_results = self.create_forecast_scenarios(
                        current_loads, ultimate_loads, growth_projection,
                        priority_zones=[]
                    )
                    LOGGER.info(" DEBUG: create_forecast_scenarios completed successfully")
                except Exception as e:
                    LOGGER.error(f" DEBUG: Error in create_forecast_scenarios: {e}")
                    LOGGER.error(f" DEBUG: Error type: {type(e).__name__}")
                    import traceback
                    LOGGER.error(f" DEBUG: Full traceback: {traceback.format_exc()}")
                    raise
                
                # Create output
                self.create_forecast_output(forecast_results)
                
        except Exception as e:
            LOGGER.error(f" DEBUG: MAIN EXCEPTION CAUGHT - Error running Load Forecast: {e}")
            LOGGER.error(f" DEBUG: Error type: {type(e).__name__}")
            import traceback
            LOGGER.error(f" DEBUG: Full traceback: {traceback.format_exc()}")
            
            # Print to console as well in case QGIS logging isn't visible
            print(f" DEBUG: MAIN EXCEPTION - Error running Load Forecast: {e}")
            print(f" DEBUG: Error type: {type(e).__name__}")
            print(f" DEBUG: Full traceback: {traceback.format_exc()}")
            
            if self.iface:
                from qgis.PyQt.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Error",
                    f"Error running forecast: {str(e)}"
                )

    def show_mode_selection_dialog(self):
        """Show dialog to let user select between Basic and Full mode, or Load Assignment."""
        try:
            from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                           QPushButton, QRadioButton, QButtonGroup)
            from qgis.PyQt.QtCore import Qt
            
            LOGGER.info("Creating mode selection dialog...")
            
            dialog = QDialog(self.iface.mainWindow() if self.iface else None)
            dialog.setWindowTitle("Load Forecast - Mode Selection")
            # Increased height to 350 to accommodate 3 radio buttons comfortably
            dialog.setFixedSize(450, 350)
            
            layout = QVBoxLayout()
            
            # Title
            title_label = QLabel("Select Tool")
            title_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
            layout.addWidget(title_label)
            
            # Mode options
            self.mode_group = QButtonGroup()
            
            # Full mode option
            full_radio = QRadioButton("Full Mode Forecasting (Enhanced Features)")
            if self.enhanced_features_available:
                full_radio.setEnabled(True)
                full_radio.setToolTip("Uses advanced forecasting with numpy, pandas, and scikit-learn")
            else:
                full_radio.setEnabled(False)
                full_radio.setToolTip("Not available - missing required packages (numpy, pandas, scikit-learn)")
            self.mode_group.addButton(full_radio, 1)
            layout.addWidget(full_radio)
            
            # Basic mode option
            basic_radio = QRadioButton("Basic Mode Forecasting")
            basic_radio.setEnabled(True)
            basic_radio.setToolTip("Uses standard forecasting algorithms without external dependencies")
            basic_radio.setChecked(True)  # Default selection
            self.mode_group.addButton(basic_radio, 0)
            layout.addWidget(basic_radio)
            
            # Load assignment option
            load_radio = QRadioButton("Load Assignment (Assign Forecast to Pipes)")
            load_radio.setEnabled(True)
            load_radio.setToolTip("Assign forecasted loads to pipe network based on polygon intersections")
            self.mode_group.addButton(load_radio, 2)
            layout.addWidget(load_radio)
            
            # Info text
            if not self.enhanced_features_available:
                info_label = QLabel("ℹ️ To enable Full Mode, install: numpy, pandas, scikit-learn")
                info_label.setStyleSheet("color: #666; font-size: 10px; margin-top: 10px;")
                layout.addWidget(info_label)
            
            layout.addSpacing(20)
            
            # Buttons
            button_layout = QHBoxLayout()
            ok_button = QPushButton("OK")
            cancel_button = QPushButton("Cancel")
            
            ok_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)
            
            button_layout.addStretch()
            button_layout.addWidget(ok_button)
            button_layout.addWidget(cancel_button)
            layout.addLayout(button_layout)
            
            dialog.setLayout(layout)
            
            LOGGER.info("Showing mode selection dialog...")
            result = dialog.exec_()
            LOGGER.info(f"Dialog result: {result}, QDialog.Accepted: {QDialog.Accepted}")
            
            if result == QDialog.Accepted:
                # Return the actual button ID: 0=Basic, 1=Full, 2=Load Assignment
                checked_id = self.mode_group.checkedId()
                LOGGER.info(f"User selected mode ID: {checked_id}")
                return checked_id
            else:
                LOGGER.info("User cancelled dialog")
                return None  # User cancelled
                
        except Exception as e:
            LOGGER.error(f"Error showing mode selection dialog: {e}")
            # Fallback to basic mode
            return False

    def show_enhanced_input_dialog(self):
        """Show enhanced dialog for full mode with layer selection and spatial data."""
        try:
            from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                           QPushButton, QComboBox, QTabWidget, QWidget,
                                           QFormLayout, QSpinBox, QFileDialog, QLineEdit,
                                           QTextEdit, QGroupBox, QFrame)
            from qgis.PyQt.QtCore import Qt
            from qgis.PyQt.QtGui import QFont
            from qgis.core import QgsProject, QgsVectorLayer
            
            dialog = QDialog(self.iface.mainWindow())
            dialog.setWindowTitle("Load Forecast - Enhanced Mode Setup")
            dialog.setMinimumSize(850, 750)
            
            layout = QVBoxLayout()
            
            # Title
            title_label = QLabel(" Enhanced Spatial Forecasting Setup")
            title_font = QFont()
            title_font.setPointSize(14)
            title_font.setBold(True)
            title_label.setFont(title_font)
            layout.addWidget(title_label)
            
            # Subtitle description
            subtitle_label = QLabel(
                "Configure spatial analysis parameters for advanced load forecasting with "
                "population and housing data integration."
            )
            subtitle_label.setWordWrap(True)
            subtitle_label.setStyleSheet("color: #666; margin-bottom: 15px;")
            layout.addWidget(subtitle_label)
            
            # Create tab widget
            tab_widget = QTabWidget()
            
            # Tab 1: Layer Selection
            layer_tab = QWidget()
            layer_main_layout = QVBoxLayout()
            
            # Layer tab title
            layer_title = QLabel("️ QGIS Layer Selection")
            layer_title_font = QFont()
            layer_title_font.setBold(True)
            layer_title_font.setPointSize(11)
            layer_title.setFont(layer_title_font)
            layer_main_layout.addWidget(layer_title)
            
            # Layer tab description
            layer_desc = QLabel(
                "Select the QGIS layers for spatial analysis:\n\n"
                " Service Area Polygons: Define geographic zones for load aggregation\n"
                " Pipe Network Layer: Gas distribution infrastructure for spatial intersection\n\n"
                " Tip: Ensure layers are loaded in QGIS and have valid geometries."
            )
            layer_desc.setWordWrap(True)
            layer_desc.setStyleSheet("color: #666; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
            layer_main_layout.addWidget(layer_desc)
            
            # Separator
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            layer_main_layout.addWidget(line)
            
            layer_layout = QFormLayout()
            
            # Get available layers
            project = QgsProject.instance()
            vector_layers = [layer for layer in project.mapLayers().values() 
                           if isinstance(layer, QgsVectorLayer)]
            layer_names = [layer.name() for layer in vector_layers]
            
            # Polygon layer selection
            self.polygon_layer_combo = QComboBox()
            self.polygon_layer_combo.addItems(layer_names)
            self.polygon_layer_combo.setToolTip(
                "Select the polygon layer containing service areas/zones\n"
                "This layer defines the geographic boundaries for load aggregation"
            )
            layer_layout.addRow(" Service Area Polygons:", self.polygon_layer_combo)
            
            # Pipe layer selection  
            self.pipe_layer_combo = QComboBox()
            self.pipe_layer_combo.addItems(layer_names)
            self.pipe_layer_combo.setToolTip(
                "Select the pipe network layer (LineString geometry)\n"
                "Used for spatial intersection analysis with service areas"
            )
            layer_layout.addRow(" Pipe Network Layer:", self.pipe_layer_combo)
            
            layer_main_layout.addLayout(layer_layout)
            layer_main_layout.addStretch()
            layer_tab.setLayout(layer_main_layout)
            tab_widget.addTab(layer_tab, "️ Layers")
            
            # Tab 2: Population & Housing Data
            data_tab = QWidget()
            data_main_layout = QVBoxLayout()
            
            # Data tab title
            data_title = QLabel(" Demographic Data Sources")
            data_title_font = QFont()
            data_title_font.setBold(True)
            data_title_font.setPointSize(11)
            data_title.setFont(data_title_font)
            data_main_layout.addWidget(data_title)
            
            # Data tab description
            data_desc = QLabel(
                "Provide a SINGLE CSV file with combined population and housing data for regression-based forecasting.\n\n"
                " REQUIRED CSV FORMAT (case-insensitive headers):\n"
                "   Year, Population, Housing\n"
                "   2015, 125000, 48500\n"
                "   2016, 128500, 49800\n"
                "   2017, 132000, 51200\n"
                "   ...\n\n"
                " Column Definitions:\n"
                "   • Year: Numeric year values (e.g., 2015, 2016, 2017...)\n"
                "   • Population: Total population count for that year\n"
                "   • Housing: Total housing unit count for that year\n\n"
                " How It Works:\n"
                "   Residential: Calculated as avg_load_per_housing_unit × future_housing_units\n"
                "   Commercial/Industrial: Linear regression against Population (historical loads vs population)\n"
                "   The system builds regression models (load = slope × population + intercept) and reports R² values\n\n"
                " Tip: Need at least 2 years of historical data for regression. More data = better predictions!"
            )
            data_desc.setWordWrap(True)
            data_desc.setStyleSheet("color: #666; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
            data_main_layout.addWidget(data_desc)
            
            # Separator
            line2 = QFrame()
            line2.setFrameShape(QFrame.HLine)
            line2.setFrameShadow(QFrame.Sunken)
            data_main_layout.addWidget(line2)
            
            # Combined Population & Housing CSV group
            data_group = QGroupBox(" Population & Housing Data File")
            data_layout = QFormLayout()
            
            self.pop_housing_file_edit = QLineEdit()
            self.pop_housing_file_edit.setPlaceholderText("Select CSV file with Year, Population, Housing columns...")
            pop_housing_browse_btn = QPushButton(" Browse...")
            pop_housing_browse_btn.clicked.connect(
                lambda: self.browse_file(self.pop_housing_file_edit, "Population & Housing Data CSV")
            )
            pop_housing_browse_btn.setToolTip(
                "Browse for combined population and housing data CSV file\n"
                "Must have columns: Year, Population, Housing (case-insensitive)"
            )
            file_layout = QHBoxLayout()
            file_layout.addWidget(self.pop_housing_file_edit)
            file_layout.addWidget(pop_housing_browse_btn)
            data_layout.addRow(" CSV File:", file_layout)
            
            # Add import button
            import_btn = QPushButton(" Import & Preview Data")
            import_btn.clicked.connect(self.import_and_preview_pop_housing)
            import_btn.setToolTip(
                "Import the CSV file and preview the data\n"
                "This validates the format and shows you what data will be used"
            )
            data_layout.addRow("", import_btn)
            
            # Add a preview label
            self.pop_housing_preview_label = QLabel("No data loaded yet.")
            self.pop_housing_preview_label.setWordWrap(True)
            self.pop_housing_preview_label.setStyleSheet("color: #666; font-style: italic;")
            data_layout.addRow(" Data Preview:", self.pop_housing_preview_label)
            
            data_group.setLayout(data_layout)
            data_main_layout.addWidget(data_group)
            
            data_main_layout.addStretch()
            data_tab.setLayout(data_main_layout)
            tab_widget.addTab(data_tab, " Data Sources")
            
            # Tab 3: Priority Zones
            priority_tab = QWidget()
            priority_main_layout = QVBoxLayout()
            
            # Priority tab title
            priority_title = QLabel(" Enhanced Priority Zones Configuration")
            priority_title_font = QFont()
            priority_title_font.setBold(True)
            priority_title_font.setPointSize(11)
            priority_title.setFont(priority_title_font)
            priority_main_layout.addWidget(priority_title)
            
            # Priority zones explanation
            priority_desc = QLabel(
                "Configure zone-specific growth multipliers for enhanced spatial forecasting:\n\n"
                " Use Cases:\n"
                "   • New Development Areas: Prioritize zones with planned infrastructure\n"
                "   • Urban Infill: Focus growth on high-density redevelopment areas\n"
                "   • Master Plan Implementation: Align with city planning documents\n"
                "   • Restricted Zones: Deprioritize areas with limited development potential\n\n"
                " Priority Levels & Effects:\n"
                "   • 2.5× Priority: Major growth centers, new subdivisions (250% of base growth)\n"
                "   • 2.0× Priority: High-priority development zones (200% of base growth)\n"
                "   • 1.0× Normal: Standard growth areas (100% - default for all zones)\n"
                "   • 0.5× Depriority: Limited growth areas, conservation zones (50% of base growth)\n"
                "   • 0.25× Depriority: Minimal growth, protected/restricted areas (25% of base growth)\n"
                "   • 0.0× Zero: No growth zones, frozen/excluded areas (0% - no new loads)\n\n"
                "⏰ Time Periods: Different priorities can be set for different forecast years"
            )
            priority_desc.setWordWrap(True)
            priority_desc.setStyleSheet("color: #666; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
            priority_main_layout.addWidget(priority_desc)
            
            # Separator
            line3 = QFrame()
            line3.setFrameShape(QFrame.HLine)
            line3.setFrameShadow(QFrame.Sunken)
            priority_main_layout.addWidget(line3)
            
            # Priority zones table
            from qgis.PyQt.QtWidgets import QTableWidget, QTableWidgetItem, QComboBox, QHeaderView
            
            self.priority_zones_table = QTableWidget()
            self.priority_zones_table.setColumnCount(4)
            self.priority_zones_table.setHorizontalHeaderLabels([
                "️ Zone Name", " Priority Level", " Start Year", " End Year"
            ])
            self.priority_zones_table.setToolTip(
                "Configure zone-specific growth multipliers\n"
                "Zone Name: Must match area names in your polygon layer\n"
                "Priority Level: 2.5×, 2.0×, 1.0× (normal), 0.5×, 0.25×, or 0.0×\n"
                "Years: Time period when this priority applies"
            )
            
            # Make table fill available space
            header = self.priority_zones_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
            
            self.priority_zones_table.setMinimumHeight(200)
            priority_main_layout.addWidget(self.priority_zones_table)
            
            # Buttons for managing priority zones
            priority_buttons_layout = QHBoxLayout()
            
            add_priority_btn = QPushButton(" Add Priority Zone")
            add_priority_btn.clicked.connect(self.add_priority_zone_row)
            add_priority_btn.setToolTip("Add a new priority zone configuration")
            priority_buttons_layout.addWidget(add_priority_btn)
            
            remove_priority_btn = QPushButton(" Remove Selected")
            remove_priority_btn.clicked.connect(self.remove_priority_zone_row)
            remove_priority_btn.setToolTip("Remove the selected priority zone")
            priority_buttons_layout.addWidget(remove_priority_btn)
            
            priority_buttons_layout.addStretch()
            priority_main_layout.addLayout(priority_buttons_layout)
            
            # Add some default zones for demonstration
            self.populate_default_priority_zones()
            
            priority_main_layout.addStretch()
            priority_tab.setLayout(priority_main_layout)
            tab_widget.addTab(priority_tab, " Priority Zones")
            
            # Tab 4: Forecast Parameters
            params_tab = QWidget()
            params_main_layout = QVBoxLayout()
            
            # Params tab title
            params_title = QLabel("️ Enhanced Forecast Parameters")
            params_title_font = QFont()
            params_title_font.setBold(True)
            params_title_font.setPointSize(11)
            params_title.setFont(params_title_font)
            params_main_layout.addWidget(params_title)
            
            # Params tab description
            params_desc = QLabel(
                "Configure forecast period and base growth rates for enhanced spatial analysis:\n\n"
                " Forecast Period: Start and end years for the projection\n"
                " Base Growth Rates: Default annual growth percentages for each customer class\n"
                " These rates are modified by priority zones and demographic data\n\n"
                " Tip: Base rates are applied to zones without specific priority settings"
            )
            params_desc.setWordWrap(True)
            params_desc.setStyleSheet("color: #666; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
            params_main_layout.addWidget(params_desc)
            
            # Separator
            line4 = QFrame()
            line4.setFrameShape(QFrame.HLine)
            line4.setFrameShadow(QFrame.Sunken)
            params_main_layout.addWidget(line4)
            
            params_layout = QFormLayout()
            
            self.forecast_start_year = QSpinBox()
            self.forecast_start_year.setRange(2020, 2060)
            self.forecast_start_year.setValue(2025)
            self.forecast_start_year.setToolTip(
                "Year when the forecast begins\n"
                "This should typically be the current year or the year of your baseline data"
            )
            params_layout.addRow(" Forecast Start Year:", self.forecast_start_year)
            
            self.forecast_end_year = QSpinBox()
            self.forecast_end_year.setRange(2025, 2070)
            self.forecast_end_year.setValue(2050)
            self.forecast_end_year.setToolTip(
                "Final year of the forecast projection\n"
                "Common planning horizons: 20 years (master plan), 50 years (long-term)"
            )
            params_layout.addRow(" Forecast End Year:", self.forecast_end_year)
            
            self.growth_rate_residential = QSpinBox()
            self.growth_rate_residential.setRange(-10, 20)
            self.growth_rate_residential.setValue(2)
            self.growth_rate_residential.setSuffix("%")
            self.growth_rate_residential.setToolTip(
                "Annual percentage growth rate for residential loads\n"
                "Typical range: 1-3% for established areas, 3-5% for growing areas\n"
                "Modified by priority zones and demographic trends"
            )
            params_layout.addRow(" Base Residential Growth Rate:", self.growth_rate_residential)
            
            self.growth_rate_commercial = QSpinBox()
            self.growth_rate_commercial.setRange(-10, 20)
            self.growth_rate_commercial.setValue(3)
            self.growth_rate_commercial.setSuffix("%")
            self.growth_rate_commercial.setToolTip(
                "Annual percentage growth rate for commercial loads\n"
                "Typical range: 2-4% for established areas, 4-6% for growing areas\n"
                "Modified by priority zones and demographic trends"
            )
            params_layout.addRow(" Base Commercial Growth Rate:", self.growth_rate_commercial)
            
            # Add aggregation toggle for areas with insufficient data
            from qgis.PyQt.QtWidgets import QCheckBox
            self.aggregate_polygons_checkbox = QCheckBox()
            self.aggregate_polygons_checkbox.setChecked(False)
            self.aggregate_polygons_checkbox.setToolTip(
                "When checked, combines all polygon data for regression analysis if individual areas have insufficient data.\n"
                "Useful when some areas have zero or minimal development. The combined regression results are then\n"
                "distributed across individual polygons using the projection function.\n\n"
                " Use when: Many zones have sparse or zero data\n"
                " Avoid when: Each zone has sufficient historical data for individual regression"
            )
            params_layout.addRow(" Aggregate Polygons for Regression:", self.aggregate_polygons_checkbox)
            
            # Add linear regression flag
            self.assume_linear_checkbox = QCheckBox()
            self.assume_linear_checkbox.setChecked(False)
            self.assume_linear_checkbox.setToolTip(
                "When checked, uses simple linear regression between population and loads (ignores housing data).\n"
                "Performs linear regression for each load class:\n"
                "  • Residential Load = f(Population)\n"
                "  • Commercial Load = f(Population)\n"
                "  • Industrial Load = f(Population)\n\n"
                " Use when: Housing data is unavailable or unreliable\n"
                " Result: Simpler model based only on population trends"
            )
            params_layout.addRow(" Assume Linear (Population → Load):", self.assume_linear_checkbox)
            
            params_main_layout.addLayout(params_layout)
            params_main_layout.addStretch()
            params_tab.setLayout(params_main_layout)
            tab_widget.addTab(params_tab, "️ Parameters")
            
            layout.addWidget(tab_widget)
            
            # Buttons
            button_layout = QHBoxLayout()
            ok_button = QPushButton("Run Enhanced Forecast")
            cancel_button = QPushButton("Cancel")
            
            ok_button.clicked.connect(dialog.accept)  
            cancel_button.clicked.connect(dialog.reject)
            
            button_layout.addStretch()
            button_layout.addWidget(ok_button)
            button_layout.addWidget(cancel_button)
            layout.addLayout(button_layout)
            
            dialog.setLayout(layout)
            
            if dialog.exec_() == QDialog.Accepted:
                # Collect inputs
                inputs = {
                    'polygon_layer': self.polygon_layer_combo.currentText(),
                    'pipe_layer': self.pipe_layer_combo.currentText(),
                    'population_file': self.pop_housing_file_edit.text(),
                    'population_fields': {
                        'area': '',  # Not used in enhanced mode
                        'year': '',  # Not used in enhanced mode
                        'value': ''  # Not used in enhanced mode
                    },
                    'housing_file': self.pop_housing_file_edit.text(),  # Same file for both
                    'housing_fields': {
                        'area': '',  # Not used in enhanced mode
                        'year': '',  # Not used in enhanced mode
                        'value': ''  # Not used in enhanced mode
                    },
                    'forecast_start_year': self.forecast_start_year.value(),
                    'forecast_end_year': self.forecast_end_year.value(),
                    'growth_rates': {
                        'residential': self.growth_rate_residential.value() / 100.0,
                        'commercial': self.growth_rate_commercial.value() / 100.0
                    },
                    'aggregate_polygons': self.aggregate_polygons_checkbox.isChecked(),
                    'assume_linear': self.assume_linear_checkbox.isChecked(),
                    'priority_zones': self.collect_priority_zones_data()
                }
                return True, inputs
            else:
                return False, None
                
        except Exception as e:
            LOGGER.error(f"Error showing enhanced input dialog: {e}")
            return False, None

    def browse_file(self, line_edit, title):
        """Helper method to browse for files."""
        try:
            from qgis.PyQt.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getOpenFileName(
                self.iface.mainWindow(),
                title,
                "",
                "CSV Files (*.csv);;All Files (*)"
            )
            if file_path:
                line_edit.setText(file_path)
        except Exception as e:
            LOGGER.error(f"Error browsing file: {e}")

    def import_and_preview_pop_housing(self):
        """Import and preview the population/housing CSV data."""
        try:
            import pandas as pd
            from qgis.PyQt.QtWidgets import QMessageBox
            
            file_path = self.pop_housing_file_edit.text().strip()
            
            if not file_path:
                QMessageBox.warning(
                    self.iface.mainWindow(),
                    "No File Selected",
                    "Please select a CSV file first using the Browse button."
                )
                return
            
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Validate columns (case insensitive)
            required_cols = ['year', 'population', 'housing']
            df_lower = df.columns.str.lower()
            
            missing_cols = []
            for col in required_cols:
                if col not in df_lower.tolist():
                    missing_cols.append(col)
            
            if missing_cols:
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Invalid CSV Format",
                    f"CSV must contain these columns (case-insensitive):\n"
                    f"  • Year\n"
                    f"  • Population\n"
                    f"  • Housing\n\n"
                    f"Missing columns: {', '.join(missing_cols)}\n\n"
                    f"Your CSV has: {', '.join(df.columns.tolist())}"
                )
                self.pop_housing_preview_label.setText("Error: Invalid CSV format.")
                self.pop_housing_preview_label.setStyleSheet("color: red; font-style: italic;")
                return
            
            # Standardize column names
            col_mapping = {}
            for col in df.columns:
                if col.lower() == 'year':
                    col_mapping[col] = 'Year'
                elif col.lower() == 'population':
                    col_mapping[col] = 'Population'
                elif col.lower() == 'housing':
                    col_mapping[col] = 'Housing'
            
            df = df.rename(columns=col_mapping)[['Year', 'Population', 'Housing']]
            
            # Remove rows with missing data
            df = df.dropna()
            
            # Convert to numeric
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df['Population'] = pd.to_numeric(df['Population'], errors='coerce')
            df['Housing'] = pd.to_numeric(df['Housing'], errors='coerce')
            
            # Remove invalid data
            df = df.dropna()
            
            if df.empty:
                QMessageBox.warning(
                    self.iface.mainWindow(),
                    "No Valid Data",
                    "No valid numeric data found in CSV file after cleaning."
                )
                self.pop_housing_preview_label.setText("Error: No valid data.")
                self.pop_housing_preview_label.setStyleSheet("color: red; font-style: italic;")
                return
            
            # Sort by year
            df = df.sort_values('Year')
            
            # Store the dataframe for later use
            self.pop_housing_data = df
            
            # Create preview message
            year_range = f"{int(df['Year'].min())} to {int(df['Year'].max())}"
            pop_range = f"{int(df['Population'].min()):,} to {int(df['Population'].max()):,}"
            housing_range = f"{int(df['Housing'].min()):,} to {int(df['Housing'].max()):,}"
            
            preview_text = (
                f"Successfully loaded {len(df)} years of data!\n"
                f"Year range: {year_range}\n"
                f"Population range: {pop_range}\n"
                f"Housing range: {housing_range}\n"
                f"This data will be used for regression-based forecasting."
            )
            
            self.pop_housing_preview_label.setText(preview_text)
            self.pop_housing_preview_label.setStyleSheet("color: green; font-style: normal;")
            
            LOGGER.info(f"Successfully imported population/housing data: {len(df)} rows")
            LOGGER.info(f"Year range: {year_range}")
            
            QMessageBox.information(
                self.iface.mainWindow(),
                "Data Import Successful",
                preview_text
            )
            
        except FileNotFoundError:
            QMessageBox.critical(
                self.iface.mainWindow(),
                "File Not Found",
                f"The file does not exist:\n{file_path}"
            )
            self.pop_housing_preview_label.setText("Error: File not found.")
            self.pop_housing_preview_label.setStyleSheet("color: red; font-style: italic;")
        except Exception as e:
            LOGGER.error(f"Error importing population/housing CSV: {e}")
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Import Error",
                f"Failed to import CSV:\n{str(e)}"
            )
            self.pop_housing_preview_label.setText(f"Error: {str(e)}")
            self.pop_housing_preview_label.setStyleSheet("color: red; font-style: italic;")

    def populate_default_priority_zones(self):
        """Populate the priority zones table with some default examples."""
        try:
            from qgis.PyQt.QtWidgets import QTableWidgetItem, QComboBox, QSpinBox
            
            # Add some example priority zones
            default_zones = [
                ("Kensington", "High Priority", 2025, 2030),
                ("West Industrial", "Priority", 2025, 2035),
                ("Meadow View", "Priority", 2025, 2030),
                ("Northern Lights", "Depriority", 2025, 2040)
            ]
            
            self.priority_zones_table.setRowCount(len(default_zones))
            
            for row, (zone_name, priority_level, start_year, end_year) in enumerate(default_zones):
                # Zone name
                zone_item = QTableWidgetItem(zone_name)
                self.priority_zones_table.setItem(row, 0, zone_item)
                
                # Priority level dropdown
                priority_combo = QComboBox()
                priority_combo.addItems([
                    "High Priority (2.5x)",
                    "Priority (2.0x)", 
                    "Normal (1.0x)",
                    "Low Priority (0.5x)",
                    "Depriority (0.25x)",
                    "Zero Priority (0.0x)"
                ])
                # Set the appropriate selection
                if priority_level == "High Priority":
                    priority_combo.setCurrentIndex(0)
                elif priority_level == "Priority":
                    priority_combo.setCurrentIndex(1)
                elif priority_level == "Depriority":
                    priority_combo.setCurrentIndex(4)
                else:
                    priority_combo.setCurrentIndex(2)
                
                self.priority_zones_table.setCellWidget(row, 1, priority_combo)
                
                # Start year
                start_spin = QSpinBox()
                start_spin.setRange(2020, 2070)
                start_spin.setValue(start_year)
                self.priority_zones_table.setCellWidget(row, 2, start_spin)
                
                # End year
                end_spin = QSpinBox()
                end_spin.setRange(2020, 2070)
                end_spin.setValue(end_year)
                self.priority_zones_table.setCellWidget(row, 3, end_spin)
                
        except Exception as e:
            LOGGER.error(f"Error populating default priority zones: {e}")

    def add_priority_zone_row(self):
        """Add a new row to the priority zones table."""
        try:
            from qgis.PyQt.QtWidgets import QTableWidgetItem, QComboBox, QSpinBox
            
            current_row_count = self.priority_zones_table.rowCount()
            self.priority_zones_table.setRowCount(current_row_count + 1)
            
            # Zone name (empty)
            zone_item = QTableWidgetItem("New Zone")
            self.priority_zones_table.setItem(current_row_count, 0, zone_item)
            
            # Priority level dropdown
            priority_combo = QComboBox()
            priority_combo.addItems([
                "High Priority (2.5x)",
                "Priority (2.0x)", 
                "Normal (1.0x)",
                "Low Priority (0.5x)",
                "Depriority (0.25x)",
                "Zero Priority (0.0x)"
            ])
            priority_combo.setCurrentIndex(2)  # Default to Normal
            self.priority_zones_table.setCellWidget(current_row_count, 1, priority_combo)
            
            # Start year
            start_spin = QSpinBox()
            start_spin.setRange(2020, 2070)
            start_spin.setValue(2025)
            self.priority_zones_table.setCellWidget(current_row_count, 2, start_spin)
            
            # End year
            end_spin = QSpinBox()
            end_spin.setRange(2020, 2070)
            end_spin.setValue(2030)
            self.priority_zones_table.setCellWidget(current_row_count, 3, end_spin)
            
        except Exception as e:
            LOGGER.error(f"Error adding priority zone row: {e}")

    def remove_priority_zone_row(self):
        """Remove the selected row from the priority zones table."""
        try:
            current_row = self.priority_zones_table.currentRow()
            if current_row >= 0:
                self.priority_zones_table.removeRow(current_row)
        except Exception as e:
            LOGGER.error(f"Error removing priority zone row: {e}")

    def collect_priority_zones_data(self):
        """Collect priority zones configuration from the table."""
        try:
            priority_zones = []
            for row in range(self.priority_zones_table.rowCount()):
                # Get zone name
                zone_item = self.priority_zones_table.item(row, 0)
                zone_name = zone_item.text() if zone_item else ""
                
                # Get priority level
                priority_widget = self.priority_zones_table.cellWidget(row, 1)
                priority_level = priority_widget.currentText() if priority_widget else "Normal (1.0x)"
                
                # Get start year
                start_widget = self.priority_zones_table.cellWidget(row, 2)
                start_year = start_widget.value() if start_widget else 2025
                
                # Get end year
                end_widget = self.priority_zones_table.cellWidget(row, 3)
                end_year = end_widget.value() if end_widget else 2030
                
                if zone_name.strip():  # Only add non-empty zone names
                    priority_zones.append({
                        'zone_name': zone_name.strip(),
                        'priority_level': priority_level,
                        'start_year': start_year,
                        'end_year': end_year
                    })
            
            return priority_zones
        except Exception as e:
            LOGGER.error(f"Error collecting priority zones data: {e}")
            return []

    def show_input_dialog(self):
        """Show dialog to get user inputs for the forecast."""
        try:
            from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                           QLineEdit, QPushButton, QComboBox, QSpinBox,
                                           QDoubleSpinBox, QFileDialog, QGroupBox, QFormLayout,
                                           QTableWidget, QTableWidgetItem, QTabWidget,
                                           QCheckBox)
            from qgis.PyQt.QtCore import Qt
            
            dialog = QDialog(self.iface.mainWindow())
            dialog.setWindowTitle("Load Forecast - Input Configuration")
            dialog.setMinimumSize(700, 600)
            
            layout = QVBoxLayout()
            
            # Create tab widget for different input sections
            tab_widget = QTabWidget()
            
            # Parameters tab
            param_widget = self.create_parameters_tab()
            tab_widget.addTab(param_widget, "Parameters")
            
            # Current loads tab
            current_widget = self.create_loads_input_tab("current")
            tab_widget.addTab(current_widget, "Current Loads")
            
            # Ultimate loads tab
            ultimate_widget = self.create_loads_input_tab("ultimate")
            tab_widget.addTab(ultimate_widget, "Ultimate Loads")
            
            # Growth projection tab
            growth_widget = self.create_growth_projection_tab()
            tab_widget.addTab(growth_widget, "Growth Projection")
            
            # Priority zones tab
            priority_widget = self.create_basic_priority_zones_tab()
            tab_widget.addTab(priority_widget, "Priority Zones")
            
            layout.addWidget(tab_widget)

            # Buttons
            button_layout = QHBoxLayout()
            ok_button = QPushButton("Run Forecast")
            cancel_button = QPushButton("Cancel")
            
            ok_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)
            
            button_layout.addWidget(ok_button)
            button_layout.addWidget(cancel_button)
            layout.addLayout(button_layout)
            
            dialog.setLayout(layout)
            
            # Execute dialog
            if dialog.exec_() == QDialog.Accepted:
                return True, self.collect_forecast_inputs()
            else:
                return False, {}
                
        except Exception as e:
            LOGGER.error(f"Error showing forecast input dialog: {e}")
            return False, {}

    def create_parameters_tab(self):
        """Create the parameters input tab."""
        try:
            from qgis.PyQt.QtWidgets import (QWidget, QFormLayout, QSpinBox, QCheckBox,
                                           QVBoxLayout, QLabel, QFrame)
            from qgis.PyQt.QtGui import QFont
            
            widget = QWidget()
            layout = QVBoxLayout()
            
            # Title
            title_label = QLabel(" Forecast Period Configuration")
            title_font = QFont()
            title_font.setPointSize(12)
            title_font.setBold(True)
            title_label.setFont(title_font)
            layout.addWidget(title_label)
            
            # Description
            desc_label = QLabel(
                "Select the forecast years for your analysis. These years will be used to project "
                "load growth from current loads to ultimate loads.\n\n"
                " Tip: Select at least 2 years for meaningful forecasting. Typically, "
                "5-year intervals are used."
            )
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #666; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
            layout.addWidget(desc_label)
            
            # Separator
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            layout.addWidget(line)
            
            form_layout = QFormLayout()
            
            # Forecast years selection
            years_label = QLabel("Select forecast years:")
            years_label_font = QFont()
            years_label_font.setBold(True)
            years_label.setFont(years_label_font)
            layout.addWidget(years_label)
            
            # Checkboxes for forecast years
            self.year_checkboxes = {}
            base_year = 2025
            for offset in [0, 5, 10, 15, 20]:
                year = base_year + offset
                if offset == 0:
                    checkbox = QCheckBox(f"{year} (Start year - current loads baseline)")
                else:
                    checkbox = QCheckBox(f"{year} ({offset} years from start)")
                checkbox.setChecked(True)
                checkbox.setToolTip(f"Include year {year} in the forecast analysis")
                self.year_checkboxes[year] = checkbox
                form_layout.addWidget(checkbox)
            
            layout.addLayout(form_layout)
            layout.addStretch()
            widget.setLayout(layout)
            
            return widget
            
        except Exception as e:
            LOGGER.error(f"Error creating parameters tab: {e}")
            return None

    def create_loads_input_tab(self, load_type):
        """Create a tab for current or ultimate loads input."""
        try:
            from qgis.PyQt.QtWidgets import (QWidget, QVBoxLayout, QTableWidget, 
                                           QTableWidgetItem, QPushButton, QHBoxLayout, QLabel, QFrame)
            from qgis.PyQt.QtGui import QFont
            
            widget = QWidget()
            layout = QVBoxLayout()
            
            # Title and description based on load type
            if load_type == "current":
                title_label = QLabel(" Current Loads (Baseline)")
                desc_text = (
                    "Enter current load values for each area and category (in appropriate units, e.g., m³/h or CFH).\n\n"
                    " These values represent the existing baseline loads that will be used as the starting point "
                    "for forecasting. Click 'Add Row' to add more areas, 'Remove Row' to delete selected areas."
                )
            else:
                title_label = QLabel(" Ultimate Loads (Build-Out)")
                desc_text = (
                    "Enter ultimate (build-out) load values for each area and category.\n\n"
                    " These values represent the expected loads when the area is fully developed. "
                    "The forecast will project growth from current loads toward these ultimate values. "
                    "Ultimate loads should be ≥ current loads for each area."
                )
            
            title_font = QFont()
            title_font.setPointSize(12)
            title_font.setBold(True)
            title_label.setFont(title_font)
            layout.addWidget(title_label)
            
            desc_label = QLabel(desc_text)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #666; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
            layout.addWidget(desc_label)
            
            # Separator
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            layout.addWidget(line)
            
            # Sample areas for the table
            areas = ["Arbour Hills", "Meadow View", "South East", "North East", "Kensington"]
            
            # Create table
            table = QTableWidget()
            table.setRowCount(len(areas))
            table.setColumnCount(4)
            table.setHorizontalHeaderLabels(["Area", "Residential", "Commercial", "Industrial"])
            
            # Sample values
            sample_values = {
                "current": {
                    "Arbour Hills": [195, 312, 0],
                    "Meadow View": [1468, 375, 0],
                    "South East": [2690, 650, 0],
                    "North East": [817, 534, 3588],
                    "Kensington": [260, 327, 0]
                },
                "ultimate": {
                    "Arbour Hills": [1854, 2350, 0],
                    "Meadow View": [4309, 445, 0],
                    "South East": [11060, 1281, 1783],
                    "North East": [6586, 694, 3588],
                    "Kensington": [2519, 382, 0]
                }
            }
            
            for row, area in enumerate(areas):
                table.setItem(row, 0, QTableWidgetItem(area))
                values = sample_values[load_type].get(area, [0, 0, 0])
                for col, value in enumerate(values, 1):
                    table.setItem(row, col, QTableWidgetItem(str(value)))
            
            table.resizeColumnsToContents()
            
            # Store reference to table
            if load_type == "current":
                self.current_loads_table = table
            else:
                self.ultimate_loads_table = table
            
            layout.addWidget(table)
            
            # Buttons for adding/removing rows and CSV import/export
            button_layout = QHBoxLayout()
            add_button = QPushButton("Add Area")
            remove_button = QPushButton("Remove Area")
            import_button = QPushButton("Import CSV")
            export_button = QPushButton("Export CSV")
            
            add_button.clicked.connect(lambda: self.add_area_row(table))
            remove_button.clicked.connect(lambda: self.remove_area_row(table))
            import_button.clicked.connect(lambda: self.import_csv_to_table(table, f"Import {load_type.title()} Loads CSV"))
            export_button.clicked.connect(lambda: self.export_table_to_csv(table, f"forecast_{load_type}_loads"))
            
            button_layout.addWidget(add_button)
            button_layout.addWidget(remove_button)
            button_layout.addWidget(import_button)
            button_layout.addWidget(export_button)
            layout.addLayout(button_layout)
            
            widget.setLayout(layout)
            return widget
            
        except Exception as e:
            LOGGER.error(f"Error creating loads input tab: {e}")
            return None

    def create_growth_projection_tab(self):
        """Create the growth projection input tab."""
        try:
            from qgis.PyQt.QtWidgets import (QWidget, QVBoxLayout, QTableWidget, 
                                           QTableWidgetItem, QLabel, QPushButton,
                                           QHBoxLayout, QDoubleSpinBox, QFrame)
            from qgis.PyQt.QtGui import QFont
            
            widget = QWidget()
            layout = QVBoxLayout()
            
            # Title
            title_label = QLabel(" Growth Projection Parameters")
            title_font = QFont()
            title_font.setPointSize(12)
            title_font.setBold(True)
            title_label.setFont(title_font)
            layout.addWidget(title_label)
            
            # Description
            desc_label = QLabel(
                "Configure how loads will grow over time. You can set either a simple growth rate "
                "or customize growth rates for each zone and category.\n\n"
                " Linear Growth: Growth increases steadily each year (base + increment × year)\n"
                " Exponential Growth: Use the table below to set percentage rates per zone"
            )
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #666; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
            layout.addWidget(desc_label)
            
            # Separator
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            layout.addWidget(line)
            
            # Instructions
            instructions = QLabel(" Simple Linear Growth Model:")
            instructions_font = QFont()
            instructions_font.setBold(True)
            instructions.setFont(instructions_font)
            layout.addWidget(instructions)
            
            # Simple growth parameters
            params_layout = QHBoxLayout()
            
            # Base growth
            base_label = QLabel("Base Annual Growth:")
            base_label.setToolTip("Starting annual growth value in GJ/d")
            self.base_growth_spin = QDoubleSpinBox()
            self.base_growth_spin.setRange(0, 1000)
            self.base_growth_spin.setValue(100)
            self.base_growth_spin.setSuffix(" GJ/d")
            self.base_growth_spin.setToolTip("Base load growth per year (e.g., 100 GJ/d per year)")
            
            # Growth increment
            increment_label = QLabel("Annual Increment:")
            increment_label.setToolTip("How much the growth rate increases each year")
            self.growth_increment_spin = QDoubleSpinBox()
            self.growth_increment_spin.setRange(0, 100)
            self.growth_increment_spin.setValue(2.0)
            self.growth_increment_spin.setSuffix(" GJ/d/year")
            self.growth_increment_spin.setToolTip("Additional growth added each year (acceleration)")
            
            params_layout.addWidget(base_label)
            params_layout.addWidget(self.base_growth_spin)
            params_layout.addWidget(increment_label)
            params_layout.addWidget(self.growth_increment_spin)
            
            layout.addLayout(params_layout)
            
            # Generate button
            generate_button = QPushButton("Generate Growth Table")
            generate_button.clicked.connect(self.generate_growth_table)
            layout.addWidget(generate_button)
            
            # Growth table
            self.growth_table = QTableWidget()
            self.growth_table.setColumnCount(2)
            self.growth_table.setHorizontalHeaderLabels(["Year", "Growth (GJ/d)"])
            
            layout.addWidget(self.growth_table)
            
            # CSV import/export buttons for growth table
            growth_button_layout = QHBoxLayout()
            import_growth_button = QPushButton("Import Growth CSV")
            export_growth_button = QPushButton("Export Growth CSV")
            
            import_growth_button.clicked.connect(lambda: self.import_csv_to_table(self.growth_table, "Import Growth Projections CSV"))
            export_growth_button.clicked.connect(lambda: self.export_table_to_csv(self.growth_table, "forecast_growth_projections"))
            
            growth_button_layout.addWidget(import_growth_button)
            growth_button_layout.addWidget(export_growth_button)
            layout.addLayout(growth_button_layout)
            
            # Generate initial table
            self.generate_growth_table()
            
            widget.setLayout(layout)
            return widget
            
        except Exception as e:
            LOGGER.error(f"Error creating growth projection tab: {e}")
            return None

    def generate_growth_table(self):
        """Generate growth projection table based on parameters."""
        try:
            from qgis.PyQt.QtWidgets import QTableWidgetItem
            
            base_growth = self.base_growth_spin.value()
            increment = self.growth_increment_spin.value()
            
            years = list(range(2025, 2046))  # 2025 to 2045
            self.growth_table.setRowCount(len(years))
            
            for row, year in enumerate(years):
                growth_value = base_growth + (year - 2025) * increment
                self.growth_table.setItem(row, 0, QTableWidgetItem(str(year)))
                self.growth_table.setItem(row, 1, QTableWidgetItem(f"{growth_value:.1f}"))
            
            self.growth_table.resizeColumnsToContents()
            
        except Exception as e:
            LOGGER.error(f"Error generating growth table: {e}")

    def add_area_row(self, table):
        """Add a new row to the loads table."""
        try:
            from qgis.PyQt.QtWidgets import QTableWidgetItem
            
            row_count = table.rowCount()
            table.setRowCount(row_count + 1)
            table.setItem(row_count, 0, QTableWidgetItem(f"New Area {row_count + 1}"))
            for col in range(1, 4):
                table.setItem(row_count, col, QTableWidgetItem("0"))
        except Exception as e:
            LOGGER.error(f"Error adding area row: {e}")

    def remove_area_row(self, table):
        """Remove the last row from the loads table."""
        try:
            row_count = table.rowCount()
            if row_count > 0:
                table.setRowCount(row_count - 1)
        except Exception as e:
            LOGGER.error(f"Error removing area row: {e}")

    def create_basic_priority_zones_tab(self):
        """Create a priority zones tab for basic mode."""
        try:
            from qgis.PyQt.QtWidgets import (QWidget, QVBoxLayout, QTableWidget,
                                           QTableWidgetItem, QPushButton, QHBoxLayout,
                                           QComboBox, QSpinBox, QLabel, QFrame)
            from qgis.PyQt.QtGui import QFont
            
            widget = QWidget()
            layout = QVBoxLayout()
            
            # Title
            title_label = QLabel(" Priority Zones Configuration")
            title_font = QFont()
            title_font.setPointSize(12)
            title_font.setBold(True)
            title_label.setFont(title_font)
            layout.addWidget(title_label)
            
            # Explanation
            explanation = QLabel(
                "Configure priority zones to control load allocation during forecasting. "
                "Priority zones receive enhanced load allocation, while depriority zones receive reduced allocation.\n\n"
                " Use Cases:\n"
                "• High Priority: Rapidly developing areas needing accelerated infrastructure\n"
                "• Priority: Growing areas with higher demand expectations\n"
                "• Normal: Standard growth areas (default)\n"
                "• Low Priority: Slower growing or transitioning areas\n"
                "• Depriority: Areas with declining loads or pending redevelopment\n"
                "• Zero Priority: Frozen zones with no new growth allowed\n\n"
                "Priority levels affect how loads are distributed:\n"
                "• High Priority (2.5×): Zone gets 2.5 times normal allocation\n"
                "• Priority (2.0×): Zone gets double normal allocation\n"
                "• Normal (1.0×): Standard allocation (baseline)\n"
                "• Low Priority (0.5×): Zone gets half normal allocation\n"
                "• Depriority (0.25×): Zone gets quarter normal allocation\n"
                "• Zero Priority (0.0×): Zone gets no new loads (frozen)\n\n"
                " Set start/end years to apply priority only during specific periods."
            )
            explanation.setWordWrap(True)
            explanation.setStyleSheet("color: #666; background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
            layout.addWidget(explanation)
            
            # Separator
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            layout.addWidget(line)
            
            # Table for priority zones
            table_label = QLabel("Priority Zone Assignments:")
            table_label_font = QFont()
            table_label_font.setBold(True)
            table_label.setFont(table_label_font)
            layout.addWidget(table_label)
            
            self.basic_priority_zones_table = QTableWidget()
            self.basic_priority_zones_table.setColumnCount(4)
            self.basic_priority_zones_table.setHorizontalHeaderLabels([
                "Zone Name", "Priority Level", "Start Year", "End Year"
            ])
            self.basic_priority_zones_table.setMinimumHeight(200)
            self.basic_priority_zones_table.setToolTip(
                "Add zones with priority adjustments. Leave empty for no priority zones."
            )
            layout.addWidget(self.basic_priority_zones_table)
            
            # Buttons
            button_layout = QHBoxLayout()
            add_button = QPushButton(" Add Zone")
            add_button.setToolTip("Add a new priority zone row")
            remove_button = QPushButton(" Remove Selected")
            remove_button.setToolTip("Remove the selected priority zone")
            populate_button = QPushButton(" Load Default Zones")
            populate_button.setToolTip("Populate with common zone names from current loads")
            
            add_button.clicked.connect(self.add_basic_priority_zone_row)
            remove_button.clicked.connect(self.remove_basic_priority_zone_row)
            populate_button.clicked.connect(self.populate_basic_default_priority_zones)
            
            button_layout.addWidget(add_button)
            button_layout.addWidget(remove_button)
            button_layout.addWidget(populate_button)
            button_layout.addStretch()
            
            layout.addLayout(button_layout)
            
            # Populate with default zones
            self.populate_basic_default_priority_zones()
            
            widget.setLayout(layout)
            return widget
            
        except Exception as e:
            LOGGER.error(f"Error creating basic priority zones tab: {e}")
            return None

    def populate_basic_default_priority_zones(self):
        """Populate the basic priority zones table with default examples."""
        try:
            from qgis.PyQt.QtWidgets import QTableWidgetItem, QComboBox, QSpinBox
            
            # Clear existing rows
            self.basic_priority_zones_table.setRowCount(0)
            
            # Add some example priority zones matching basic mode areas
            default_zones = [
                ("Kensington", "High Priority (2.5x)", 2025, 2030),
                ("Meadow View", "Priority (2.0x)", 2025, 2035),
                ("Arbour Hills", "Priority (2.0x)", 2025, 2030),
                ("North East", "Depriority (0.25x)", 2025, 2040)
            ]
            
            self.basic_priority_zones_table.setRowCount(len(default_zones))
            
            for row, (zone_name, priority_level, start_year, end_year) in enumerate(default_zones):
                # Zone name
                zone_item = QTableWidgetItem(zone_name)
                self.basic_priority_zones_table.setItem(row, 0, zone_item)
                
                # Priority level dropdown
                priority_combo = QComboBox()
                priority_combo.addItems([
                    "High Priority (2.5x)",
                    "Priority (2.0x)", 
                    "Normal (1.0x)",
                    "Low Priority (0.5x)",
                    "Depriority (0.25x)",
                    "Zero Priority (0.0x)"
                ])
                # Set the appropriate selection
                if priority_level == "High Priority (2.5x)":
                    priority_combo.setCurrentIndex(0)
                elif priority_level == "Priority (2.0x)":
                    priority_combo.setCurrentIndex(1)
                elif priority_level == "Depriority (0.25x)":
                    priority_combo.setCurrentIndex(4)
                else:
                    priority_combo.setCurrentIndex(2)
                
                self.basic_priority_zones_table.setCellWidget(row, 1, priority_combo)
                
                # Start year
                start_spin = QSpinBox()
                start_spin.setRange(2020, 2070)
                start_spin.setValue(start_year)
                self.basic_priority_zones_table.setCellWidget(row, 2, start_spin)
                
                # End year
                end_spin = QSpinBox()
                end_spin.setRange(2020, 2070)
                end_spin.setValue(end_year)
                self.basic_priority_zones_table.setCellWidget(row, 3, end_spin)
                
        except Exception as e:
            LOGGER.error(f"Error populating basic default priority zones: {e}")

    def add_basic_priority_zone_row(self):
        """Add a new row to the basic priority zones table."""
        try:
            from qgis.PyQt.QtWidgets import QTableWidgetItem, QComboBox, QSpinBox
            
            current_row_count = self.basic_priority_zones_table.rowCount()
            self.basic_priority_zones_table.setRowCount(current_row_count + 1)
            
            # Zone name (empty)
            zone_item = QTableWidgetItem("New Zone")
            self.basic_priority_zones_table.setItem(current_row_count, 0, zone_item)
            
            # Priority level dropdown
            priority_combo = QComboBox()
            priority_combo.addItems([
                "High Priority (2.5x)",
                "Priority (2.0x)", 
                "Normal (1.0x)",
                "Low Priority (0.5x)",
                "Depriority (0.25x)",
                "Zero Priority (0.0x)"
            ])
            priority_combo.setCurrentIndex(2)  # Default to Normal
            self.basic_priority_zones_table.setCellWidget(current_row_count, 1, priority_combo)
            
            # Start year
            start_spin = QSpinBox()
            start_spin.setRange(2020, 2070)
            start_spin.setValue(2025)
            self.basic_priority_zones_table.setCellWidget(current_row_count, 2, start_spin)
            
            # End year
            end_spin = QSpinBox()
            end_spin.setRange(2020, 2070)
            end_spin.setValue(2030)
            self.basic_priority_zones_table.setCellWidget(current_row_count, 3, end_spin)
            
        except Exception as e:
            LOGGER.error(f"Error adding basic priority zone row: {e}")

    def remove_basic_priority_zone_row(self):
        """Remove the selected row from the basic priority zones table."""
        try:
            current_row = self.basic_priority_zones_table.currentRow()
            if current_row >= 0:
                self.basic_priority_zones_table.removeRow(current_row)
        except Exception as e:
            LOGGER.error(f"Error removing basic priority zone row: {e}")

    def collect_basic_priority_zones_data(self):
        """Collect priority zones configuration from the basic table."""
        try:
            priority_zones = []
            for row in range(self.basic_priority_zones_table.rowCount()):
                # Get zone name
                zone_item = self.basic_priority_zones_table.item(row, 0)
                zone_name = zone_item.text() if zone_item else ""
                
                # Get priority level
                priority_widget = self.basic_priority_zones_table.cellWidget(row, 1)
                priority_level = priority_widget.currentText() if priority_widget else "Normal (1.0x)"
                
                # Get start year
                start_widget = self.basic_priority_zones_table.cellWidget(row, 2)
                start_year = start_widget.value() if start_widget else 2025
                
                # Get end year
                end_widget = self.basic_priority_zones_table.cellWidget(row, 3)
                end_year = end_widget.value() if end_widget else 2030
                
                if zone_name.strip():  # Only add non-empty zone names
                    priority_zones.append({
                        'zone_name': zone_name.strip(),
                        'priority_level': priority_level,
                        'start_year': start_year,
                        'end_year': end_year
                    })
            
            return priority_zones
        except Exception as e:
            LOGGER.error(f"Error collecting basic priority zones data: {e}")
            return []

    def collect_forecast_inputs(self):
        """Collect all inputs from the dialog."""
        try:
            # Collect forecast years
            forecast_years = []
            for year, checkbox in self.year_checkboxes.items():
                if checkbox.isChecked():
                    forecast_years.append(year)
            
            # Collect current loads
            current_loads = {}
            for row in range(self.current_loads_table.rowCount()):
                area_item = self.current_loads_table.item(row, 0)
                if area_item:
                    area = area_item.text()
                    current_loads[area] = {
                        'residential': float(self.current_loads_table.item(row, 1).text() or 0),
                        'commercial': float(self.current_loads_table.item(row, 2).text() or 0),
                        'industrial': float(self.current_loads_table.item(row, 3).text() or 0)
                    }
            
            # Collect ultimate loads
            ultimate_loads = {}
            for row in range(self.ultimate_loads_table.rowCount()):
                area_item = self.ultimate_loads_table.item(row, 0)
                if area_item:
                    area = area_item.text()
                    ultimate_loads[area] = {
                        'residential': float(self.ultimate_loads_table.item(row, 1).text() or 0),
                        'commercial': float(self.ultimate_loads_table.item(row, 2).text() or 0),
                        'industrial': float(self.ultimate_loads_table.item(row, 3).text() or 0)
                    }
            
            # Collect growth projection
            growth_projection = {}
            for row in range(self.growth_table.rowCount()):
                year_item = self.growth_table.item(row, 0)
                growth_item = self.growth_table.item(row, 1)
                if year_item and growth_item:
                    year = int(year_item.text())
                    growth = float(growth_item.text())
                    growth_projection[year] = growth
            
            return {
                'forecast_years': forecast_years,
                'current_loads': current_loads,
                'ultimate_loads': ultimate_loads,
                'growth_projection': growth_projection,
                'priority_zones': self.collect_basic_priority_zones_data()
            }
            
        except Exception as e:
            LOGGER.error(f"Error collecting forecast inputs: {e}")
            return {}

    def show_results_dialog(self, forecast_results):
        """Show results dialog with forecast data."""
        try:
            from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QTextEdit, QPushButton,
                                           QTableWidget, QTableWidgetItem, QTabWidget)
            
            dialog = QDialog(self.iface.mainWindow())
            dialog.setWindowTitle("Load Forecast - Results")
            dialog.setMinimumSize(800, 600)
            
            layout = QVBoxLayout()
            
            # Create tab widget
            tab_widget = QTabWidget()
            
            # Summary tab
            summary_text = QTextEdit()
            summary_text.setReadOnly(True)
            
            summary_content = "LOAD FORECAST RESULTS\n"
            summary_content += "=" * 50 + "\n\n"
            
            for load_class, yearly_data in forecast_results.items():
                summary_content += f"{load_class.replace('_', ' ').title()} Loads:\n"
                summary_content += "-" * 40 + "\n"
                
                for year, area_loads in yearly_data.items():
                    total_load = sum(area_loads.values())
                    summary_content += f"  {year}: Total {total_load:.2f} GJ/d\n"
                    for area, load in area_loads.items():
                        summary_content += f"    {area}: {load:.2f} GJ/d\n"
                    summary_content += "\n"
                
                summary_content += "\n"
            
            summary_text.setPlainText(summary_content)
            tab_widget.addTab(summary_text, "Summary")
            
            # Create table for each load class and year
            for load_class, yearly_data in forecast_results.items():
                for year, area_loads in yearly_data.items():
                    table = QTableWidget()
                    table.setRowCount(len(area_loads))
                    table.setColumnCount(2)
                    table.setHorizontalHeaderLabels(["Area", f"Load {year} (GJ/d)"])
                    
                    for row, (area, load) in enumerate(area_loads.items()):
                        table.setItem(row, 0, QTableWidgetItem(area))
                        table.setItem(row, 1, QTableWidgetItem(f"{load:.2f}"))
                    
                    table.resizeColumnsToContents()
                    tab_name = f"{load_class.title()} {year}"
                    tab_widget.addTab(table, tab_name)
            
            layout.addWidget(tab_widget)
            
            # Buttons
            from qgis.PyQt.QtWidgets import QHBoxLayout
            button_layout = QHBoxLayout()
            
            export_csv_button = QPushButton("Export to CSV")
            export_csv_button.clicked.connect(lambda: self.export_forecast_to_csv(forecast_results))
            export_csv_button.setToolTip("Export detailed forecast by area and year (original format)")
            
            export_summary_button = QPushButton("Export Summary CSV")
            export_summary_button.clicked.connect(lambda: self.export_forecast_summary_csv(forecast_results))
            export_summary_button.setToolTip("Export summary with absolute and incremental loads by area")
            
            export_reforecast_button = QPushButton("Export for Reforecaster")
            export_reforecast_button.clicked.connect(lambda: self.export_forecast_for_reforecaster(forecast_results))
            export_reforecast_button.setToolTip("Export in simple Area,Year,Load format for use with Reforecaster tool")
            
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            
            button_layout.addWidget(export_csv_button)
            button_layout.addWidget(export_summary_button)
            button_layout.addWidget(export_reforecast_button)
            button_layout.addStretch()
            button_layout.addWidget(close_button)
            
            layout.addLayout(button_layout)
            
            dialog.setLayout(layout)
            dialog.exec_()
            
        except Exception as e:
            LOGGER.error(f"Error showing forecast results dialog: {e}")

    def run_load_assignment(self):
        """Run the load assignment tool to assign forecasted loads to pipes."""
        from .load_assignment import LoadAssignmentTool
        
        tool = LoadAssignmentTool(self.iface)
        tool.run()

    def export_forecast_to_csv(self, forecast_results):
        """Export forecast results to CSV with separate tables for each subzone."""
        try:
            from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
            import csv
            from datetime import datetime
            
            # Get save location from user
            default_filename = f"load_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            file_path, _ = QFileDialog.getSaveFileName(
                self.iface.mainWindow(),
                "Save Forecast Results",
                default_filename,
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if not file_path:
                return  # User cancelled
            
            # Collect all areas (subzones) across all load classes
            all_areas = set()
            all_years = set()
            
            for load_class, yearly_data in forecast_results.items():
                for year, area_loads in yearly_data.items():
                    all_years.add(year)
                    all_areas.update(area_loads.keys())
            
            all_areas = sorted(all_areas)
            all_years = sorted(all_years)
            
            LOGGER.info(f"Exporting forecast results to {file_path}")
            LOGGER.info(f"Areas: {all_areas}")
            LOGGER.info(f"Years: {all_years}")
            
            # Write CSV file
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header information
                writer.writerow(['Load Forecast Results'])
                writer.writerow([f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
                writer.writerow([])  # Empty row
                
                # Create a table for each subzone (area)
                for area in all_areas:
                    writer.writerow([f'SUBZONE: {area}'])
                    writer.writerow([])  # Empty row
                    
                    # Write table header with date column
                    header = ['Year', 'DateTime', 'Residential (GJ/d)', 'Commercial (GJ/d)', 'Industrial (GJ/d)', 'Total (GJ/d)']
                    writer.writerow(header)
                    
                    # Write data rows for each year
                    for year in all_years:
                        residential = 0
                        commercial = 0  
                        industrial = 0
                        
                        # Get loads for this area and year from each load class
                        if 'residential' in forecast_results and year in forecast_results['residential']:
                            residential = forecast_results['residential'][year].get(area, 0)
                        
                        if 'commercial' in forecast_results and year in forecast_results['commercial']:
                            commercial = forecast_results['commercial'][year].get(area, 0)
                            
                        if 'industrial' in forecast_results and year in forecast_results['industrial']:
                            industrial = forecast_results['industrial'][year].get(area, 0)
                        
                        total = residential + commercial + industrial
                        
                        # Format year as human-readable string: "2025"
                        year_str = str(year)
                        
                        # Format datetime - use format that works with Synergi
                        # Based on testing: Python datetime objects work when exported to shapefile
                        # For CSV, we'll use the ISO format which should be recognized
                        from datetime import datetime
                        datetime_obj = datetime(year, 11, 11)
                        # Convert to string for CSV: ISO format YYYY-MM-DD
                        datetime_str = datetime_obj.strftime("%Y-%m-%d")
                        
                        # Write row with date columns
                        row = [
                            year_str,  # Human-readable year
                            datetime_str,  # ISO datetime format: YYYY-MM-DD HH:MM:SS
                            f"{residential:.2f}",
                            f"{commercial:.2f}", 
                            f"{industrial:.2f}",
                            f"{total:.2f}"
                        ]
                        writer.writerow(row)
                    
                    writer.writerow([])  # Empty row between tables
                    writer.writerow([])  # Additional spacing
                
                # Add summary table with totals across all areas
                writer.writerow(['SUMMARY - ALL SUBZONES COMBINED'])
                writer.writerow([])
                writer.writerow(['Year', 'Total Residential (GJ/d)', 'Total Commercial (GJ/d)', 'Total Industrial (GJ/d)', 'Grand Total (GJ/d)'])
                
                for year in all_years:
                    total_residential = 0
                    total_commercial = 0
                    total_industrial = 0
                    
                    # Sum across all areas for each load class
                    if 'residential' in forecast_results and year in forecast_results['residential']:
                        total_residential = sum(forecast_results['residential'][year].values())
                    
                    if 'commercial' in forecast_results and year in forecast_results['commercial']:
                        total_commercial = sum(forecast_results['commercial'][year].values())
                        
                    if 'industrial' in forecast_results and year in forecast_results['industrial']:
                        total_industrial = sum(forecast_results['industrial'][year].values())
                    
                    grand_total = total_residential + total_commercial + total_industrial
                    
                    summary_row = [
                        year,
                        f"{total_residential:.2f}",
                        f"{total_commercial:.2f}",
                        f"{total_industrial:.2f}",
                        f"{grand_total:.2f}"
                    ]
                    writer.writerow(summary_row)
            
            # Show success message
            QMessageBox.information(
                self.iface.mainWindow(),
                "Export Successful",
                f"Forecast results exported successfully to:\n{file_path}\n\n"
                f"Exported {len(all_areas)} subzones with data for {len(all_years)} years."
            )
            
            LOGGER.info(f" Forecast results exported successfully to {file_path}")
            
        except Exception as e:
            LOGGER.error(f"Error exporting forecast to CSV: {e}")
            import traceback
            LOGGER.error(f"Full traceback: {traceback.format_exc()}")
            
            if self.iface:
                from qgis.PyQt.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Export Error",
                    f"Error exporting forecast results:\n{str(e)}"
                )
    
    def export_forecast_summary_csv(self, forecast_results):
        """
        Export forecast results in summary pivot table format with absolute and incremental loads.
        
        Format:
        - Table 1: Load Forecast (GJ/d) with years as rows, areas as columns, TOTAL column
        - Table 2: Incremental Load (GJ/d) showing period-over-period changes
        
        Args:
            forecast_results: Dictionary containing forecast data
                             Format: {load_class: {year: {area: load_value}}}
        """
        try:
            from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
            import csv
            
            # Prompt for save location
            file_path, _ = QFileDialog.getSaveFileName(
                self.iface.mainWindow(),
                "Export Summary Forecast to CSV",
                "",
                "CSV Files (*.csv)"
            )
            
            if not file_path:
                return  # User cancelled
            
            # Extract all unique years and areas
            all_years = set()
            all_areas = set()
            
            for load_class in ['residential', 'commercial', 'industrial']:
                if load_class in forecast_results:
                    for year, areas_dict in forecast_results[load_class].items():
                        all_years.add(year)
                        all_areas.update(areas_dict.keys())
            
            all_years = sorted(list(all_years))
            all_areas = sorted(list(all_areas))
            
            if not all_years or not all_areas:
                QMessageBox.warning(
                    self.iface.mainWindow(),
                    "No Data",
                    "No forecast data available to export."
                )
                return
            
            # Calculate total loads per area per year (sum all load classes)
            # Structure: {year: {area: total_load}}
            total_loads = {}
            
            for year in all_years:
                total_loads[year] = {}
                
                for area in all_areas:
                    area_total = 0.0
                    
                    # Sum residential, commercial, industrial for this area/year
                    for load_class in ['residential', 'commercial', 'industrial']:
                        if load_class in forecast_results and year in forecast_results[load_class]:
                            area_total += forecast_results[load_class][year].get(area, 0.0)
                    
                    total_loads[year][area] = area_total
            
            # Write CSV file
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # === TABLE 1: Load Forecast (GJ/d) ===
                writer.writerow(['Load Forecast (GJ/d)'])
                writer.writerow([])
                
                # Header row: Year, Area1, Area2, ..., TOTAL
                header = ['Year'] + all_areas + ['TOTAL']
                writer.writerow(header)
                
                # Data rows: one per year
                for year in all_years:
                    row = [str(year)]
                    
                    # Add load for each area
                    row_total = 0.0
                    for area in all_areas:
                        load = total_loads[year].get(area, 0.0)
                        row.append(f"{load:.2f}")
                        row_total += load
                    
                    # Add TOTAL column
                    row.append(f"{row_total:.2f}")
                    writer.writerow(row)
                
                # Empty rows for spacing
                writer.writerow([])
                writer.writerow([])
                
                # === TABLE 2: Incremental Load (GJ/d) ===
                writer.writerow(['Incremental Load (GJ/d)'])
                writer.writerow([])
                
                # Same header
                writer.writerow(header)
                
                # Calculate incremental loads (current_year - previous_year)
                for i, year in enumerate(all_years):
                    if i == 0:
                        # First year: no previous year, show zeros or skip
                        row = [str(year)]
                        for area in all_areas:
                            row.append("0.00")
                        row.append("0.00")
                        writer.writerow(row)
                    else:
                        prev_year = all_years[i-1]
                        row = [str(year)]
                        
                        # Calculate increment for each area
                        row_total = 0.0
                        for area in all_areas:
                            current_load = total_loads[year].get(area, 0.0)
                            previous_load = total_loads[prev_year].get(area, 0.0)
                            increment = current_load - previous_load
                            row.append(f"{increment:.2f}")
                            row_total += increment
                        
                        # Add TOTAL column
                        row.append(f"{row_total:.2f}")
                        writer.writerow(row)
            
            # Show success message
            QMessageBox.information(
                self.iface.mainWindow(),
                "Export Successful",
                f"Summary forecast exported successfully to:\n{file_path}\n\n"
                f"Exported data for {len(all_areas)} areas and {len(all_years)} years.\n"
                f"Includes absolute loads and incremental loads tables."
            )
            
            LOGGER.info(f"Summary forecast exported to {file_path}")
            
        except Exception as e:
            LOGGER.error(f"Error exporting summary forecast to CSV: {e}")
            import traceback
            LOGGER.error(f"Full traceback: {traceback.format_exc()}")
            
            if self.iface:
                from qgis.PyQt.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Export Error",
                    f"Error exporting summary forecast:\n{str(e)}"
                )
    
    def export_forecast_for_reforecaster(self, forecast_results):
        """
        Export forecast results in simple format for the Reforecaster tool.
        
        Format: Area, Year, Load
        Where Load is the total load (residential + commercial + industrial) for that area and year.
        
        This format is required by the pipe_to_node_converter's reforecast mode.
        
        Args:
            forecast_results: Dictionary containing forecast data
                             Format: {load_class: {year: {area: load_value}}}
        """
        try:
            from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
            import csv
            from datetime import datetime
            
            # Prompt for save location
            default_filename = f"reforecast_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            file_path, _ = QFileDialog.getSaveFileName(
                self.iface.mainWindow(),
                "Export for Reforecaster",
                default_filename,
                "CSV Files (*.csv)"
            )
            
            if not file_path:
                return  # User cancelled
            
            # Extract all unique years and areas
            all_years = set()
            all_areas = set()
            
            for load_class in ['residential', 'commercial', 'industrial']:
                if load_class in forecast_results:
                    for year, areas_dict in forecast_results[load_class].items():
                        all_years.add(year)
                        all_areas.update(areas_dict.keys())
            
            all_years = sorted(list(all_years))
            all_areas = sorted(list(all_areas))
            
            if not all_years or not all_areas:
                QMessageBox.warning(
                    self.iface.mainWindow(),
                    "No Data",
                    "No forecast data available to export."
                )
                return
            
            # Convert calendar years to period years for reforecaster
            # Reforecaster expects "Year" to be periods (5, 10, 15...) not calendar years (2025, 2026...)
            start_year = min(all_years)
            
            # Calculate total loads per area per year (sum all load classes)
            # Structure: {(area, period_year): total_load}
            total_loads = {}
            period_years = set()
            
            for year in all_years:
                period_year = year - start_year  # Convert to period (0, 1, 2, 3...)
                period_years.add(period_year)
                
                for area in all_areas:
                    area_total = 0.0
                    
                    # Sum residential, commercial, industrial for this area/year
                    for load_class in ['residential', 'commercial', 'industrial']:
                        if load_class in forecast_results and year in forecast_results[load_class]:
                            area_total += forecast_results[load_class][year].get(area, 0.0)
                    
                    total_loads[(area, period_year)] = area_total
            
            period_years = sorted(list(period_years))
            
            # Write CSV file in simple format with period years
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header row
                writer.writerow(['Area', 'Year', 'Load'])
                
                # Data rows: one per area-period combination
                for area in all_areas:
                    for period_year in period_years:
                        load = total_loads.get((area, period_year), 0.0)
                        writer.writerow([area, period_year, f"{load:.2f}"])
            
            # Show success message
            QMessageBox.information(
                self.iface.mainWindow(),
                "Export Successful",
                f"Reforecast data exported successfully to:\n{file_path}\n\n"
                f"Format: Area, Year (period), Load\n"
                f"Calendar years {min(all_years)}-{max(all_years)} converted to periods {min(period_years)}-{max(period_years)}\n"
                f"Exported {len(all_areas)} areas × {len(period_years)} periods = {len(all_areas) * len(period_years)} rows\n\n"
                f"This file can be used with the Pipe-to-Node Converter's Reforecast mode."
            )
            
            LOGGER.info(f"Reforecast data exported to {file_path}")
            LOGGER.info(f"  Areas: {all_areas}")
            LOGGER.info(f"  Years: {all_years}")
            
        except Exception as e:
            LOGGER.error(f"Error exporting reforecast data to CSV: {e}")
            import traceback
            LOGGER.error(f"Full traceback: {traceback.format_exc()}")
            
            if self.iface:
                from qgis.PyQt.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Export Error",
                    f"Error exporting reforecast data:\n{str(e)}"
                )