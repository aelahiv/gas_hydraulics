"""Historical Analysis Plugin for QGIS.

This plugin provides historical load analysis over time by 5-year periods.
Uses Install Date from service point list to determine when loads came online.
Aggregates by class and area for past periods.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List
import pandas as pd
from datetime import datetime
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

class HistoricalAnalysisPlugin:
    def __init__(self, iface: Any = None):
        """Initialize the Historical Analysis plugin."""
        self.iface = iface
        self.action = None
        
        # Default load calculation parameters
        self.load_multiplier = 1.07
        self.heat_factor_multiplier = 56.8
        
    def initGui(self):
        """Create GUI elements (only when running inside QGIS)."""
        try:
            from qgis.PyQt.QtWidgets import QAction
            self.action = QAction("Historical Analysis", self.iface.mainWindow() if self.iface else None)
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

    def create_historical_load_plots(self, excel_file_path: str, zone_layer: Any = None,
                                    pipe_layer: Any = None, start_year: int = 2000,
                                    end_year: int = 2025, output_dir: str = None):
        """Create plots showing historical loads by category for each area.
        
        Creates one plot per area showing residential (blue), commercial (red), 
        and industrial (purple) loads over time, with average growth rates for 
        the last 5 years displayed on each plot.
        
        Args:
            excel_file_path: Path to Excel file with service point data
            zone_layer: QGIS layer with zone polygons
            pipe_layer: QGIS layer with pipe lines
            start_year: Start year for analysis
            end_year: End year for analysis
            output_dir: Directory to save plots (if None, prompts user)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for file output
            
            LOGGER.info("Creating historical load plots...")
            
            # Load and process Excel data
            # Try to read 'Service Point List' sheet first, fall back to first sheet
            try:
                df = pd.read_excel(excel_file_path, sheet_name='Service Point List')
                LOGGER.info("Reading from 'Service Point List' sheet")
            except:
                df = pd.read_excel(excel_file_path)
                LOGGER.info("Reading from default sheet")
            
            LOGGER.info(f"Loaded Excel file: {len(df)} rows")
            
            if df.empty:
                LOGGER.error("Excel file is empty")
                return False
            
            # Calculate loads
            df = self.calculate_load(df)
            
            # Convert Install Date and handle bad dates
            df['Install Date'] = pd.to_datetime(df['Install Date'], errors='coerce')
            
            # Separate records with valid vs invalid dates
            valid_dates_mask = df['Install Date'].notna()
            df_with_dates = df[valid_dates_mask].copy()
            df_without_dates = df[~valid_dates_mask].copy()
            
            # For rows with invalid/missing dates, distribute their load proportionally to valid date loads
            if len(df_without_dates) > 0:
                bad_date_total_load = df_without_dates['Load'].sum()
                LOGGER.info(f"Found {len(df_without_dates)} records with bad dates, total load: {bad_date_total_load:.2f}")
                
                # Extract year from valid dates and calculate load by year
                df_with_dates['Install Year'] = df_with_dates['Install Date'].dt.year
                year_loads_all = df_with_dates.groupby('Install Year')['Load'].sum()
                
                LOGGER.info(f"Valid date range: {year_loads_all.index.min()} to {year_loads_all.index.max()}")
                
                # Filter to analysis period only for distribution (but calculate proportions from ALL years with data)
                year_loads = year_loads_all[(year_loads_all.index >= start_year) & (year_loads_all.index <= end_year)]
                
                if len(year_loads) == 0:
                    LOGGER.warning("No valid dates in analysis period - cannot distribute bad date loads")
                    df = df_with_dates
                else:
                    # Calculate total valid load in period
                    total_valid_load = year_loads.sum()
                    
                    # Calculate proportion for each year (aggressive: proportional to valid load)
                    year_proportions = year_loads / total_valid_load
                    
                    LOGGER.info(f"Distributing bad date loads proportionally to valid date loads in analysis period {start_year}-{end_year}:")
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
            
            df['Install Year'] = df['Install Date'].dt.year
            
            # Include all service points installed up to end_year (not just between start_year and end_year)
            # For cumulative load calculations, we need ALL installations up to end_year
            period_df = df[df['Install Year'] <= end_year]
            LOGGER.info(f"Filtered to installations <= {end_year}: {len(period_df)} rows, {period_df['Load'].sum():.2f} GJ/d")
            
            if period_df.empty:
                LOGGER.error("No data in analysis period")
                return False
            
            # Filter by use class
            class_dfs = self.filter_by_use_class(period_df)
            
            # Get output directory if not provided
            if not output_dir:
                if self.iface:
                    from qgis.PyQt.QtWidgets import QFileDialog
                    output_dir = QFileDialog.getExistingDirectory(
                        self.iface.mainWindow(),
                        "Select Directory for Plots",
                        "",
                        QFileDialog.ShowDirsOnly
                    )
                    if not output_dir:
                        LOGGER.info("User cancelled plot creation")
                        return False
                else:
                    output_dir = "."
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Collect data by zone
            zone_data = {}
            
            if zone_layer and pipe_layer:
                LOGGER.info("Processing zones for plot creation...")
                
                # Build pipe-to-zone mapping once to prevent double counting
                pipe_to_zone, zone_to_pipes = self.build_pipe_to_zone_mapping(zone_layer, pipe_layer)
                
                for zone_feature in zone_layer.getFeatures():
                    zone_name = self._get_zone_name(zone_feature)
                    pipe_names = list(zone_to_pipes.get(zone_name, set()))
                    
                    if not pipe_names:
                        LOGGER.warning(f"No pipes assigned to zone {zone_name}, skipping")
                        continue
                    
                    LOGGER.info(f"Zone {zone_name}: {len(pipe_names)} pipes (no double counting)")
                    
                    # Initialize data structure for this zone
                    zone_data[zone_name] = {
                        'years': list(range(start_year, end_year + 1)),
                        'residential': [],
                        'commercial': [],
                        'industrial': []
                    }
                    
                    # Calculate cumulative load for each year and category
                    for year in range(start_year, end_year + 1):
                        for class_name in ['residential', 'commercial', 'industrial']:
                            class_df = class_dfs[class_name]
                            zone_year_df = class_df[
                                (class_df['Distribution Pipe'].isin(pipe_names)) &
                                (class_df['Install Year'] <= year)
                            ]
                            cumulative_load = zone_year_df['Load'].sum()
                            zone_data[zone_name][class_name].append(cumulative_load)
            else:
                LOGGER.info("No spatial layers - creating overall plot")
                zone_name = "Overall"
                zone_data[zone_name] = {
                    'years': list(range(start_year, end_year + 1)),
                    'residential': [],
                    'commercial': [],
                    'industrial': []
                }
                
                for year in range(start_year, end_year + 1):
                    for class_name in ['residential', 'commercial', 'industrial']:
                        class_df = class_dfs[class_name]
                        year_df = class_df[class_df['Install Year'] <= year]
                        cumulative_load = year_df['Load'].sum()
                        zone_data[zone_name][class_name].append(cumulative_load)
            
            # Create plots
            plots_created = 0
            for zone_name, data in zone_data.items():
                try:
                    # Calculate average load growth for last 5 years for each category (raw GJ/d per year)
                    growth_rates = {}
                    for category in ['residential', 'commercial', 'industrial']:
                        loads = data[category]
                        if len(loads) >= 6:  # Need at least 6 years to calculate 5-year growth
                            last_5_years = loads[-6:]  # Last 6 values (to calculate 5 differences)
                            # Calculate year-over-year load increases
                            yearly_growth = []
                            for i in range(1, len(last_5_years)):
                                growth = last_5_years[i] - last_5_years[i-1]  # Raw load increase
                                yearly_growth.append(growth)
                            avg_growth = sum(yearly_growth) / len(yearly_growth) if yearly_growth else 0
                        else:
                            avg_growth = 0
                        growth_rates[category] = avg_growth
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(12, 7))
                    
                    # Plot each category
                    ax.plot(data['years'], data['residential'], 
                           color='blue', linewidth=2.5, marker='o', 
                           markersize=4, label='Residential')
                    ax.plot(data['years'], data['commercial'], 
                           color='red', linewidth=2.5, marker='s', 
                           markersize=4, label='Commercial')
                    ax.plot(data['years'], data['industrial'], 
                           color='purple', linewidth=2.5, marker='^', 
                           markersize=4, label='Industrial')
                    
                    # Formatting
                    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Cumulative Load (GJ/d)', fontsize=12, fontweight='bold')
                    ax.set_title(f'Historical Load Analysis - {zone_name}', 
                                fontsize=14, fontweight='bold', pad=20)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
                    
                    # Add growth rate annotations
                    annotation_text = 'Average Load Growth (Last 5 Years):\n'
                    annotation_text += f"Residential: {growth_rates['residential']:.2f} GJ/d per year\n"
                    annotation_text += f"Commercial: {growth_rates['commercial']:.2f} GJ/d per year\n"
                    annotation_text += f"Industrial: {growth_rates['industrial']:.2f} GJ/d per year"
                    
                    # Place annotation in upper right
                    ax.text(0.98, 0.97, annotation_text,
                           transform=ax.transAxes,
                           fontsize=9,
                           verticalalignment='top',
                           horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    
                    # Tight layout
                    plt.tight_layout()
                    
                    # Save plot
                    safe_zone_name = "".join(c for c in zone_name if c.isalnum() or c in (' ', '_', '-')).strip()
                    safe_zone_name = safe_zone_name.replace(' ', '_')
                    output_path = os.path.join(output_dir, f'historical_loads_{safe_zone_name}.png')
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                    plots_created += 1
                    LOGGER.info(f"Created plot for {zone_name}: {output_path}")
                    
                except Exception as e:
                    LOGGER.error(f"Error creating plot for {zone_name}: {e}")
                    import traceback
                    LOGGER.error(traceback.format_exc())
            
            # Show success message
            if plots_created > 0 and self.iface:
                from qgis.PyQt.QtWidgets import QMessageBox
                QMessageBox.information(
                    self.iface.mainWindow(),
                    "Plots Created",
                    f"Successfully created {plots_created} historical load plot(s) in:\n{output_dir}"
                )
                LOGGER.info(f"Successfully created {plots_created} plots")
                return True
            elif plots_created > 0:
                LOGGER.info(f"Successfully created {plots_created} plots in {output_dir}")
                return True
            else:
                if self.iface:
                    from qgis.PyQt.QtWidgets import QMessageBox
                    QMessageBox.warning(
                        self.iface.mainWindow(),
                        "No Plots Created",
                        "No plots were created. Check the log for details."
                    )
                return False
                
        except ImportError as e:
            LOGGER.error(f"matplotlib not available: {e}")
            if self.iface:
                from qgis.PyQt.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Missing Dependency",
                    "matplotlib is required for plotting but not installed.\n\n"
                    "Install it with: pip install matplotlib"
                )
            return False
        except Exception as e:
            LOGGER.error(f"Error creating plots: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            if self.iface:
                from qgis.PyQt.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Plot Creation Error",
                    f"Failed to create plots:\n{str(e)}"
                )
            return False

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
        """Filter DataFrame by use class categories."""
        df_res = df[df['Use Class'].str.contains('APT|RES', na=False)]
        df_ind = df[df['Use Class'].str.contains('IND', na=False)]
        df_comm = df[df['Use Class'].str.contains('COM', na=False)]
        
        return {
            'residential': df_res,
            'industrial': df_ind,
            'commercial': df_comm
        }

    def get_5year_periods(self, start_year: int, end_year: int) -> List[int]:
        """Generate list of 5-year period end years."""
        periods = []
        current = start_year
        while current <= end_year:
            periods.append(current)
            current += 5
        # Add final year if it doesn't align with 5-year periods
        if periods[-1] != end_year:
            periods.append(end_year)
        return periods

    def export_detailed_csv(self, excel_file_path: str, zone_layer: Any = None, 
                           pipe_layer: Any = None, start_year: int = 2000, 
                           end_year: int = 2025) -> str:
        """Export comprehensive CSV with load by polygon by category by year.
        
        Args:
            excel_file_path: Path to Excel file with service point data
            zone_layer: QGIS layer with zone polygons  
            pipe_layer: QGIS layer with pipe lines
            start_year: Start year for analysis
            end_year: End year for analysis
            
        Returns:
            Path to exported CSV file or empty string if failed
        """
        try:
            LOGGER.info(f"Starting detailed CSV export for {start_year}-{end_year}")
            
            # Load and process Excel data
            # Try to read 'Service Point List' sheet first, fall back to first sheet
            try:
                df = pd.read_excel(excel_file_path, sheet_name='Service Point List')
                LOGGER.info("Reading from 'Service Point List' sheet")
            except:
                df = pd.read_excel(excel_file_path)
                LOGGER.info("Reading from default sheet")
            
            original_count = len(df)
            LOGGER.info(f"Loaded Excel file: {original_count} rows")
            
            if df.empty:
                LOGGER.error("Excel file is empty")
                return ""
            
            # Calculate loads for each service point
            df = self.calculate_load(df)
            LOGGER.info(f"Calculated loads for {len(df)} service points")
            
            # Convert Install Date and handle bad dates
            df['Install Date'] = pd.to_datetime(df['Install Date'], errors='coerce')
            
            # Separate records with valid vs invalid dates
            valid_dates_mask = df['Install Date'].notna()
            df_with_dates = df[valid_dates_mask].copy()
            df_without_dates = df[~valid_dates_mask].copy()
            
            # For rows with invalid/missing dates, distribute their load proportionally to valid date loads
            if len(df_without_dates) > 0:
                bad_date_total_load = df_without_dates['Load'].sum()
                LOGGER.info(f"Found {len(df_without_dates)} records with bad dates, total load: {bad_date_total_load:.2f}")
                
                # Extract year from valid dates and calculate load by year
                df_with_dates_temp = df_with_dates.copy()
                df_with_dates_temp['Install Year'] = df_with_dates_temp['Install Date'].dt.year
                year_loads_all = df_with_dates_temp.groupby('Install Year')['Load'].sum()
                
                LOGGER.info(f"Valid date range: {year_loads_all.index.min()} to {year_loads_all.index.max()}")
                
                # Filter to analysis period only for distribution
                year_loads = year_loads_all[(year_loads_all.index >= start_year) & (year_loads_all.index <= end_year)]
                
                if len(year_loads) == 0:
                    LOGGER.warning("No valid dates in analysis period - cannot distribute bad date loads")
                    df = df_with_dates
                else:
                    # Calculate total valid load in period
                    total_valid_load = year_loads.sum()
                    
                    # Calculate proportion for each year (aggressive: proportional to valid load)
                    year_proportions = year_loads / total_valid_load
                    
                    LOGGER.info(f"Distributing bad date loads proportionally to valid date loads in analysis period {start_year}-{end_year}:")
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
            
            df['Install Year'] = df['Install Date'].dt.year
            
            # Filter to end year only (include all installations up to end_year, regardless of start_year)
            # For cumulative loads, we need all service points installed before end_year
            period_df = df[df['Install Year'] <= end_year]
            LOGGER.info(f"Filtered to installations <= {end_year}: {len(period_df)} rows, {period_df['Load'].sum():.2f} GJ/d")
            
            if period_df.empty:
                LOGGER.error("No data in analysis period")
                return ""
            
            # Filter by use class
            class_dfs = self.filter_by_use_class(period_df)
            
            # Prepare detailed results
            detailed_results = []
            
            if zone_layer and pipe_layer:
                LOGGER.info("Processing zones for detailed CSV export...")
                LOGGER.info("Using intersects with overlap-based assignment to prevent double counting...")
                
                # First pass: assign each pipe to the zone it overlaps most with
                pipe_to_zone = {}  # Map pipe_name -> zone_name
                all_pipes_checked = set()
                
                for pipe_feature in pipe_layer.getFeatures():
                    pipe_geom = pipe_feature.geometry()
                    if pipe_geom.isEmpty():
                        continue
                    
                    # Get pipe name
                    pipe_name = None
                    possible_name_fields = ['FacNam1005', 'name', 'Name', 'NAME', 'pipe_name', 'Pipe_Name']
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
                    
                    for zone_feature in zone_layer.getFeatures():
                        zone_geom = zone_feature.geometry()
                        if zone_geom.isEmpty():
                            continue
                        
                        # Check intersection
                        if zone_geom.intersects(pipe_geom):
                            # Calculate overlap (length of pipe within zone)
                            try:
                                intersection = zone_geom.intersection(pipe_geom)
                                overlap = intersection.length() if not intersection.isEmpty() else 0
                                
                                if overlap > max_overlap:
                                    max_overlap = overlap
                                    best_zone = self._get_zone_name(zone_feature)
                            except:
                                # If intersection fails, just note that it intersects
                                if best_zone is None:
                                    best_zone = self._get_zone_name(zone_feature)
                    
                    # Assign pipe to the zone with maximum overlap
                    if best_zone is not None:
                        pipe_to_zone[pipe_name] = best_zone
                
                LOGGER.info(f"Pipe-to-zone assignment complete:")
                LOGGER.info(f"  Total pipes checked: {len(all_pipes_checked)}")
                LOGGER.info(f"  Pipes assigned to zones: {len(pipe_to_zone)}")
                LOGGER.info(f"  Pipes not assigned: {len(all_pipes_checked) - len(pipe_to_zone)}")
                
                # Second pass: aggregate loads by zone
                zone_pipes = {}  # Map zone_name -> set of pipe names
                for pipe_name, zone_name in pipe_to_zone.items():
                    if zone_name not in zone_pipes:
                        zone_pipes[zone_name] = set()
                    zone_pipes[zone_name].add(pipe_name)
                
                for zone_feature in zone_layer.getFeatures():
                    # Get zone name
                    zone_name = self._get_zone_name(zone_feature)
                    pipe_names = zone_pipes.get(zone_name, set())
                    LOGGER.info(f"Zone {zone_name}: Assigned {len(pipe_names)} pipes (no double counting)")
                    
                    # Process each year and class combination
                    for year in range(start_year, end_year + 1):
                        for class_name, class_df in class_dfs.items():
                            # Get cumulative load up to this year for this zone
                            zone_year_df = class_df[
                                (class_df['Distribution Pipe'].isin(pipe_names)) &
                                (class_df['Install Year'] <= year)
                            ]
                            
                            cumulative_load = zone_year_df['Load'].sum()
                            
                            # Add to results
                            detailed_results.append({
                                'Polygon': zone_name,
                                'Year': year,
                                'Category': class_name.title(),
                                'Cumulative_Load_GJ': round(cumulative_load, 2),
                                'New_Load_This_Year_GJ': 0  # Will calculate below
                            })
            else:
                LOGGER.info("No spatial layers - creating overall totals")
                # If no spatial layers, create overall totals
                for year in range(start_year, end_year + 1):
                    for class_name, class_df in class_dfs.items():
                        year_df = class_df[class_df['Install Year'] <= year]
                        cumulative_load = year_df['Load'].sum()
                        
                        detailed_results.append({
                            'Polygon': 'Overall',
                            'Year': year,
                            'Category': class_name.title(),
                            'Cumulative_Load_GJ': round(cumulative_load, 2),
                            'New_Load_This_Year_GJ': 0
                        })
            
            # Calculate new loads for each year (difference from previous year)
            results_df = pd.DataFrame(detailed_results)
            results_df = results_df.sort_values(['Polygon', 'Category', 'Year'])
            
            for polygon in results_df['Polygon'].unique():
                for category in results_df['Category'].unique():
                    mask = (results_df['Polygon'] == polygon) & (results_df['Category'] == category)
                    polygon_category_data = results_df[mask].copy()
                    
                    if len(polygon_category_data) > 0:
                        # Calculate new load as difference from previous year
                        polygon_category_data['New_Load_This_Year_GJ'] = polygon_category_data['Cumulative_Load_GJ'].diff().fillna(polygon_category_data['Cumulative_Load_GJ'])
                        results_df.loc[mask, 'New_Load_This_Year_GJ'] = polygon_category_data['New_Load_This_Year_GJ']
            
            # Export to CSV
            if self.iface:
                from qgis.PyQt.QtWidgets import QFileDialog
                csv_path, _ = QFileDialog.getSaveFileName(
                    self.iface.mainWindow(),
                    "Export Detailed Historical Analysis",
                    f"historical_analysis_{start_year}_{end_year}.csv",
                    "CSV Files (*.csv)"
                )
            else:
                csv_path = f"historical_analysis_{start_year}_{end_year}.csv"
            
            if csv_path:
                # Export detailed CSV
                results_df.to_csv(csv_path, index=False)
                LOGGER.info(f"Exported detailed CSV to: {csv_path}")
                
                # Create basic forecasting format CSV (current year loads by zone)
                # Format: Zone, Residential, Commercial, Industrial
                basic_forecast_path = csv_path.replace('.csv', '_basic_forecast_format.csv')
                
                # Get the most recent year's loads
                latest_year = end_year
                latest_year_data = results_df[results_df['Year'] == latest_year]
                
                # Pivot to get one row per zone with columns for each category
                basic_df_data = []
                for polygon in latest_year_data['Polygon'].unique():
                    polygon_data = latest_year_data[latest_year_data['Polygon'] == polygon]
                    
                    row = {'Zone': polygon}
                    for category in ['Residential', 'Commercial', 'Industrial']:
                        category_data = polygon_data[polygon_data['Category'] == category]
                        load = category_data['Cumulative_Load_GJ'].values[0] if len(category_data) > 0 else 0.0
                        row[category] = round(load, 2)
                    
                    basic_df_data.append(row)
                
                basic_df = pd.DataFrame(basic_df_data)
                basic_df.to_csv(basic_forecast_path, index=False)
                LOGGER.info(f"Exported basic forecast format CSV to: {basic_forecast_path}")
                LOGGER.info(f"  Format: Zone, Residential, Commercial, Industrial (current year {latest_year} loads)")
                
                return csv_path
            
            return ""
            
        except Exception as e:
            LOGGER.error(f"Error creating detailed CSV export: {e}")
            return ""
    
    def _get_zone_name(self, zone_feature):
        """Get zone name from feature attributes, prioritizing Name column."""
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
            
        return str(zone_name)

    def analyze_historical_loads_by_period(self, excel_file_path: str, zone_layer=None, 
                                         pipe_layer=None, start_year: int = 2000, 
                                         end_year: int = 2025) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Analyze historical loads by 5-year periods.
        
        Args:
            excel_file_path: Path to Excel file with service point list
            zone_layer: QGIS polygon layer with subzones
            pipe_layer: QGIS vector layer with pipes
            start_year: Start year for analysis
            end_year: End year for analysis
            
        Returns:
            Dictionary mapping periods to zones to load by class
            Format: {period: {zone: {class: load}}}
        """
        try:
            LOGGER.info(f"=== STARTING HISTORICAL ANALYSIS ===")
            LOGGER.info(f"Excel file: {excel_file_path}")
            LOGGER.info(f"Analysis period: {start_year} to {end_year}")
            
            # Read Excel file
            df = pd.read_excel(excel_file_path, sheet_name='Service Point List')
            LOGGER.info(f"Loaded {len(df)} service points from Excel")
            
            if df.empty:
                LOGGER.error("Excel file contains no data")
                return {}
            
            # Calculate loads
            df = self.calculate_load(df)
            total_load = df['Load'].sum()
            LOGGER.info(f"Total calculated load: {total_load:.2f}")
            
            if total_load == 0:
                LOGGER.warning("All calculated loads are zero - check factor code logic")
            
            # Parse install dates
            LOGGER.info("Processing install dates...")
            original_count = len(df)
            df['Install Date'] = pd.to_datetime(df['Install Date'], errors='coerce')
            
            # Count invalid dates
            invalid_dates = df['Install Date'].isna().sum()
            LOGGER.info(f"Found {invalid_dates} invalid/missing install dates out of {original_count}")
            
            if invalid_dates == original_count:
                LOGGER.error("No valid install dates found - all data would be dropped")
                return {}
            
            # Separate records with valid vs invalid dates
            valid_dates_mask = df['Install Date'].notna()
            df_with_dates = df[valid_dates_mask].copy()
            df_without_dates = df[~valid_dates_mask].copy()
            
            # For rows with invalid/missing dates, distribute their load proportionally to valid date loads
            if len(df_without_dates) > 0:
                bad_date_total_load = df_without_dates['Load'].sum()
                LOGGER.info(f"Found {len(df_without_dates)} records with bad dates, total load: {bad_date_total_load:.2f}")
                
                # Extract year from valid dates and calculate load by year
                df_with_dates_temp = df_with_dates.copy()
                df_with_dates_temp['Install Year'] = df_with_dates_temp['Install Date'].dt.year
                year_loads_all = df_with_dates_temp.groupby('Install Year')['Load'].sum()
                
                LOGGER.info(f"Valid date range: {year_loads_all.index.min()} to {year_loads_all.index.max()}")
                
                # Filter to analysis period only for distribution
                year_loads = year_loads_all[(year_loads_all.index >= start_year) & (year_loads_all.index <= end_year)]
                
                if len(year_loads) == 0:
                    LOGGER.warning("No valid dates in analysis period - cannot distribute bad date loads")
                    df = df_with_dates
                else:
                    # Calculate total valid load in period
                    total_valid_load = year_loads.sum()
                    
                    # Calculate proportion for each year (aggressive: proportional to valid load)
                    year_proportions = year_loads / total_valid_load
                    
                    LOGGER.info(f"Distributing bad date loads proportionally to valid date loads in analysis period {start_year}-{end_year}:")
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
            
            df['Install Year'] = df['Install Date'].dt.year
            
            # Show date range in data
            min_year = df['Install Year'].min()
            max_year = df['Install Year'].max()
            LOGGER.info(f"Data date range: {min_year} to {max_year}")
            
            # Include all service points installed up to end_year for cumulative calculations
            # We still generate periods between start_year and end_year, but include ALL installations
            period_df = df[df['Install Year'] <= end_year]
            LOGGER.info(f"Filtered to installations <= {end_year}: {len(period_df)} points, {period_df['Load'].sum():.2f} GJ/d")
            
            if period_df.empty:
                LOGGER.error(f"No data found with installations <= {end_year}")
                LOGGER.info(f"Available data spans {min_year}-{max_year}")
                return {}
            
            # Get 5-year periods
            periods = self.get_5year_periods(start_year, end_year)
            LOGGER.info(f"Analysis periods: {periods}")
            
            # Filter by use class
            class_dfs = self.filter_by_use_class(period_df)
            
            for class_name, class_df in class_dfs.items():
                class_load = class_df['Load'].sum()
                LOGGER.info(f"{class_name}: {len(class_df)} points, load: {class_load:.2f}")
            
            # Initialize results
            period_results = {}
            
            for period_end in periods:
                period_start = period_end - 4
                period_name = f"{period_start}-{period_end}"
                LOGGER.info(f"Processing period: {period_name}")
                
                # Get cumulative data up to end of period (loads that came online by this period)
                period_data = {}
                
                if zone_layer and pipe_layer:
                    LOGGER.info("Processing spatial zones...")
                    
                    # Build pipe-to-zone mapping once to prevent double counting
                    pipe_to_zone, zone_to_pipes = self.build_pipe_to_zone_mapping(zone_layer, pipe_layer)
                    
                    # Process each zone
                    for zone_feature in zone_layer.getFeatures():
                        zone_name = self._get_zone_name(zone_feature)
                        pipe_names = list(zone_to_pipes.get(zone_name, set()))
                        LOGGER.info(f"Zone {zone_name}: {len(pipe_names)} pipes (no double counting)")
                        
                        # Calculate cumulative loads by class for this zone up to period end
                        zone_class_loads = {}
                        for class_name, class_df in class_dfs.items():
                            # Filter by zone pipes and install year <= period_end
                            zone_period_df = class_df[
                                (class_df['Distribution Pipe'].isin(pipe_names)) &
                                (class_df['Install Year'] <= period_end)
                            ]
                            zone_load = zone_period_df['Load'].sum()
                            zone_class_loads[class_name] = zone_load
                            
                            if len(zone_period_df) > 0:
                                LOGGER.info(f"  {class_name}: {len(zone_period_df)} points, load: {zone_load:.2f}")
                        
                        period_data[zone_name] = zone_class_loads
                else:
                    LOGGER.info("No spatial layers - using overall totals")
                    # If no spatial layers, return overall totals
                    overall_loads = {}
                    for class_name, class_df in class_dfs.items():
                        period_class_df = class_df[class_df['Install Year'] <= period_end]
                        class_load = period_class_df['Load'].sum()
                        overall_loads[class_name] = class_load
                        LOGGER.info(f"  {class_name} (cumulative to {period_end}): {len(period_class_df)} points, load: {class_load:.2f}")
                    period_data['Total'] = overall_loads
                
                period_results[period_name] = period_data
            
            LOGGER.info(f"=== HISTORICAL ANALYSIS COMPLETE ===")
            LOGGER.info(f"Generated {len(period_results)} periods")
            
            return period_results
            
        except Exception as e:
            LOGGER.error(f"Error analyzing historical loads: {e}")
            import traceback
            LOGGER.error(f"Traceback: {traceback.format_exc()}")
            return {}

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
                    
                    zone_name = self._get_zone_name(zone_feature)
                    
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

    def create_historical_analysis_output(self, period_results: Dict[str, Dict[str, Dict[str, float]]]):
        """Create output showing historical load trends."""
        try:
            # This would create tables/charts showing:
            # - Load growth over time by zone and class
            # - Cumulative load by period
            # - Growth rates between periods
            
            LOGGER.info("Creating historical analysis output")
            for period, zones in period_results.items():
                LOGGER.info(f"Period {period}:")
                for zone, loads in zones.items():
                    total_load = sum(loads.values())
                    LOGGER.info(f"  {zone}: Total {total_load:.2f} ({loads})")
                    
        except Exception as e:
            LOGGER.error(f"Error creating historical output: {e}")

    def run(self):
        """Run the Historical Analysis plugin."""
        try:
            LOGGER.info("Running Historical Analysis")
            
            if self.iface:
                # Show GUI dialog to get user inputs
                success, inputs = self.show_input_dialog()
                if not success:
                    return
                
                # Validate inputs
                if VALIDATION_AVAILABLE:
                    LOGGER.info(" Validating historical analysis inputs...")
                    validator = DataValidator()
                    
                    # Check Excel file
                    if not inputs.get('excel_file'):
                        validator.add_error("No Excel file specified")
                    elif not os.path.exists(inputs['excel_file']):
                        validator.add_error(f"Excel file not found: {inputs['excel_file']}")
                    else:
                        validator.add_info(f"Excel file: {os.path.basename(inputs['excel_file'])}")
                    
                    # Check years
                    start_year = safe_int(inputs.get('start_year'), 2000)
                    end_year = safe_int(inputs.get('end_year'), 2025)
                    
                    if start_year >= end_year:
                        validator.add_error(f"Start year ({start_year}) must be before end year ({end_year})")
                    else:
                        year_span = end_year - start_year
                        validator.add_info(f"Analysis period: {year_span} years ({start_year} - {end_year})")
                        
                        if year_span > 50:
                            validator.add_warning(f"Long analysis period: {year_span} years - may take longer to process")
                    
                    # Validate zone layer
                    if inputs.get('zone_layer'):
                        zone_validator = validate_layer_data(
                            inputs['zone_layer'],
                            "Zone Layer",
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
                                                  title="Historical Analysis - Data Validation",
                                                  parent=self.iface.mainWindow()):
                        LOGGER.info(" User cancelled after validation")
                        return
                
                # Check if user wants detailed CSV export
                if inputs.get('export_csv', False):
                    # Export detailed CSV
                    csv_path = self.export_detailed_csv(
                        excel_file_path=inputs['excel_file'],
                        zone_layer=inputs['zone_layer'],
                        pipe_layer=inputs['pipe_layer'],
                        start_year=inputs['start_year'],
                        end_year=inputs['end_year']
                    )
                    
                    if csv_path:
                        from qgis.PyQt.QtWidgets import QMessageBox
                        basic_forecast_path = csv_path.replace('.csv', '_basic_forecast_format.csv')
                        QMessageBox.information(
                            self.iface.mainWindow(),
                            "CSV Export Complete",
                            f"Detailed historical analysis exported to:\n{csv_path}\n\n"
                            f"Basic forecast format exported to:\n{basic_forecast_path}\n\n"
                            f"Detailed CSV contains load data by polygon, category, and year from {inputs['start_year']} to {inputs['end_year']}.\n\n"
                            f"Basic forecast CSV contains current year ({inputs['end_year']}) loads by zone in format: Zone, Residential, Commercial, Industrial"
                        )
                    else:
                        from qgis.PyQt.QtWidgets import QMessageBox
                        QMessageBox.warning(
                            self.iface.mainWindow(),
                            "CSV Export Failed",
                            "Failed to export CSV. Check the log for details."
                        )
                
                # Check if user wants to create plots
                if inputs.get('create_plots', False):
                    # Create historical load plots
                    self.create_historical_load_plots(
                        excel_file_path=inputs['excel_file'],
                        zone_layer=inputs['zone_layer'],
                        pipe_layer=inputs['pipe_layer'],
                        start_year=inputs['start_year'],
                        end_year=inputs['end_year']
                    )
                
                # Run standard analysis if not just exporting
                if not inputs.get('export_csv', False) or not inputs.get('create_plots', False):
                    # Process the data with user inputs for standard analysis
                    historical_data = self.analyze_historical_loads_by_period(
                        excel_file_path=inputs['excel_file'],
                        zone_layer=inputs['zone_layer'],
                        pipe_layer=inputs['pipe_layer'],
                        start_year=inputs['start_year'],
                        end_year=inputs['end_year']
                    )
                    
                    if historical_data:
                        # Show results dialog
                        self.show_results_dialog(historical_data)
                    else:
                        from qgis.PyQt.QtWidgets import QMessageBox
                        QMessageBox.warning(
                            self.iface.mainWindow(),
                            "No Historical Data Found", 
                            "No historical data could be processed.\n\n"
                            "Common causes:\n"
                            "• Install dates are missing or invalid in Excel file\n"
                        "• No data found in the selected year range\n"
                        "• All loads calculated as zero (check factor codes)\n"
                        "• Missing required columns in Excel file\n\n"
                        "Check the QGIS log panel for detailed debugging information."
                    )
            else:
                print("Historical Analysis plugin executed successfully")
                
        except Exception as e:
            LOGGER.error(f"Error running Historical Analysis: {e}")
            if self.iface:
                from qgis.PyQt.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Error",
                    f"Error running analysis: {str(e)}"
                )

    def show_input_dialog(self):
        """Show dialog to get user inputs for the historical analysis."""
        try:
            from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                           QLineEdit, QPushButton, QComboBox, QSpinBox,
                                           QDoubleSpinBox, QFileDialog, QGroupBox, QFormLayout,
                                           QFrame)
            from qgis.PyQt.QtCore import Qt
            from qgis.PyQt.QtGui import QFont
            from qgis.core import QgsProject
            
            dialog = QDialog(self.iface.mainWindow())
            dialog.setWindowTitle("Historical Analysis - Input Selection")
            dialog.setMinimumSize(650, 550)
            
            layout = QVBoxLayout()
            
            # Title
            title_label = QLabel(" Historical Load Analysis")
            title_font = QFont()
            title_font.setPointSize(13)
            title_font.setBold(True)
            title_label.setFont(title_font)
            layout.addWidget(title_label)
            
            # Description
            desc_label = QLabel(
                "Analyze historical service point data to understand past load growth trends:\n\n"
                " Processes multiple years of customer data to calculate actual load changes\n"
                "️ Aggregates results by geographic zones and customer categories\n"
                " Generates trend analysis for informed future projections\n\n"
                " This analysis helps validate and refine forecasting parameters"
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
            self.excel_line.setPlaceholderText("Select Excel file with historical service point data...")
            excel_browse = QPushButton(" Browse...")
            excel_browse.clicked.connect(self.browse_excel_file)
            excel_browse.setToolTip("Browse for Excel file containing historical service point records")
            excel_layout.addWidget(self.excel_line)
            excel_layout.addWidget(excel_browse)
            
            excel_label = QLabel(" Service Point List (Excel):")
            excel_label.setToolTip(
                "Excel file containing historical service point records\n"
                "Required columns: Location, Category, Date/Year, Load values"
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
            param_group = QGroupBox("️ Analysis Parameters")
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
            
            # Start year
            self.start_year_spin = QSpinBox()
            self.start_year_spin.setRange(1990, 2030)
            self.start_year_spin.setValue(2000)
            self.start_year_spin.setToolTip(
                "First year to include in the analysis\n"
                "Earlier years will be filtered out"
            )
            param_layout.addRow(" Start Year:", self.start_year_spin)
            
            # End year
            self.end_year_spin = QSpinBox()
            self.end_year_spin.setRange(2000, 2050)
            self.end_year_spin.setValue(2025)
            self.end_year_spin.setToolTip(
                "Last year to include in the analysis\n"
                "Later years will be filtered out"
            )
            param_layout.addRow(" End Year:", self.end_year_spin)
            
            param_group.setLayout(param_layout)
            layout.addWidget(param_group)
            
            # Export options group
            export_group = QGroupBox(" Output Options")
            export_layout = QFormLayout()
            
            from qgis.PyQt.QtWidgets import QCheckBox
            self.csv_export_checkbox = QCheckBox("Export detailed CSV report")
            self.csv_export_checkbox.setToolTip(
                "Generate comprehensive CSV file with:\n"
                "• Load by polygon\n"
                "• Load by customer category\n"
                "• Load by year\n"
                "Useful for external analysis and visualization"
            )
            export_layout.addRow(self.csv_export_checkbox)
            
            self.plots_checkbox = QCheckBox("Create historical load plots")
            self.plots_checkbox.setToolTip(
                "Generate plots showing historical loads over time:\n"
                "• One plot per area/zone\n"
                "• Residential loads in blue\n"
                "• Commercial loads in red\n"
                "• Industrial loads in purple\n"
                "• Includes 5-year average growth rates\n"
                "Requires matplotlib to be installed"
            )
            export_layout.addRow(self.plots_checkbox)
            
            export_group.setLayout(export_layout)
            layout.addWidget(export_group)
            
            # Buttons
            button_layout = QHBoxLayout()
            ok_button = QPushButton("Run Analysis")
            csv_button = QPushButton("Export CSV Only")
            cancel_button = QPushButton("Cancel")
            
            ok_button.clicked.connect(dialog.accept)
            csv_button.clicked.connect(lambda: self._set_csv_export_and_accept(dialog))
            cancel_button.clicked.connect(dialog.reject)
            
            button_layout.addWidget(ok_button)
            button_layout.addWidget(csv_button)
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
                    'start_year': self.start_year_spin.value(),
                    'end_year': self.end_year_spin.value(),
                    'export_csv': getattr(self, '_csv_export_requested', False) or self.csv_export_checkbox.isChecked(),
                    'create_plots': self.plots_checkbox.isChecked()
                }
            else:
                return False, {}
                
        except Exception as e:
            LOGGER.error(f"Error showing input dialog: {e}")
            return False, {}

    def _set_csv_export_and_accept(self, dialog):
        """Set CSV export flag and accept dialog."""
        self._csv_export_requested = True
        dialog.accept()

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

    def show_results_dialog(self, historical_data):
        """Show results dialog with historical analysis."""
        try:
            from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QTextEdit, QPushButton,
                                           QTableWidget, QTableWidgetItem, QTabWidget)
            
            dialog = QDialog(self.iface.mainWindow())
            dialog.setWindowTitle("Historical Analysis - Results")
            dialog.setMinimumSize(700, 500)
            
            layout = QVBoxLayout()
            
            # Create tab widget
            tab_widget = QTabWidget()
            
            # Summary tab
            summary_text = QTextEdit()
            summary_text.setReadOnly(True)
            
            summary_content = "HISTORICAL ANALYSIS RESULTS\n"
            summary_content += "=" * 50 + "\n\n"
            
            for period, zones in historical_data.items():
                summary_content += f"Period: {period}\n"
                summary_content += "-" * 30 + "\n"
                
                for zone_name, loads in zones.items():
                    total_load = sum(loads.values())
                    summary_content += f"  {zone_name}:\n"
                    summary_content += f"    Residential: {loads.get('residential', 0):.2f} GJ/d\n"
                    summary_content += f"    Commercial:  {loads.get('commercial', 0):.2f} GJ/d\n"
                    summary_content += f"    Industrial:  {loads.get('industrial', 0):.2f} GJ/d\n"
                    summary_content += f"    Total:       {total_load:.2f} GJ/d\n\n"
                
                summary_content += "\n"
            
            summary_text.setPlainText(summary_content)
            tab_widget.addTab(summary_text, "Summary")
            
            # Create table for each period
            for period, zones in historical_data.items():
                table = QTableWidget()
                table.setRowCount(len(zones))
                table.setColumnCount(5)
                table.setHorizontalHeaderLabels(["Zone", "Residential", "Commercial", "Industrial", "Total"])
                
                for row, (zone_name, loads) in enumerate(zones.items()):
                    total_load = sum(loads.values())
                    table.setItem(row, 0, QTableWidgetItem(zone_name))
                    table.setItem(row, 1, QTableWidgetItem(f"{loads.get('residential', 0):.2f}"))
                    table.setItem(row, 2, QTableWidgetItem(f"{loads.get('commercial', 0):.2f}"))
                    table.setItem(row, 3, QTableWidgetItem(f"{loads.get('industrial', 0):.2f}"))
                    table.setItem(row, 4, QTableWidgetItem(f"{total_load:.2f}"))
                
                table.resizeColumnsToContents()
                
                # Create table tab with export button
                from qgis.PyQt.QtWidgets import QWidget
                table_widget = QWidget()
                table_layout = QVBoxLayout()
                
                # Export button
                export_button = QPushButton(f"Export {period} to CSV")
                export_button.clicked.connect(lambda checked, t=table, p=period: self.export_table_to_csv(t, f"historical_analysis_{p}"))
                
                table_layout.addWidget(export_button)
                table_layout.addWidget(table)
                table_widget.setLayout(table_layout)
                
                tab_widget.addTab(table_widget, period)
            
            layout.addWidget(tab_widget)
            
            # Close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)
            
            dialog.setLayout(layout)
            dialog.exec_()
            
        except Exception as e:
            LOGGER.error(f"Error showing results dialog: {e}")