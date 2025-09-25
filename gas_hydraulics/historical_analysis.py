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

LOGGER = logging.getLogger(__name__)

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
        current = start_year + 4  # First 5-year period ends 4 years after start
        while current <= end_year:
            periods.append(current)
            current += 5
        # Add final year if it doesn't align with 5-year periods
        if periods[-1] != end_year:
            periods.append(end_year)
        return periods

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
            # Read Excel file
            df = pd.read_excel(excel_file_path, sheet_name='Service Point List')
            
            # Calculate loads
            df = self.calculate_load(df)
            
            # Parse install dates
            df['Install Date'] = pd.to_datetime(df['Install Date'], errors='coerce')
            df = df.dropna(subset=['Install Date'])  # Remove entries without install dates
            df['Install Year'] = df['Install Date'].dt.year
            
            # Filter to analysis period
            df = df[(df['Install Year'] >= start_year) & (df['Install Year'] <= end_year)]
            
            # Get 5-year periods
            periods = self.get_5year_periods(start_year, end_year)
            
            # Filter by use class
            class_dfs = self.filter_by_use_class(df)
            
            # Initialize results
            period_results = {}
            
            for period_end in periods:
                period_start = period_end - 4
                period_name = f"{period_start}-{period_end}"
                
                # Get cumulative data up to end of period (loads that came online by this period)
                period_data = {}
                
                if zone_layer and pipe_layer:
                    # Process each zone
                    for zone_feature in zone_layer.getFeatures():
                        zone_name = zone_feature.attribute('name')
                        
                        # Find pipes in this zone (placeholder implementation)
                        pipe_names = self.get_pipes_in_zone(zone_feature, pipe_layer)
                        
                        # Calculate cumulative loads by class for this zone up to period end
                        zone_class_loads = {}
                        for class_name, class_df in class_dfs.items():
                            # Filter by zone pipes and install year <= period_end
                            zone_period_df = class_df[
                                (class_df['Distribution Pipe'].isin(pipe_names)) &
                                (class_df['Install Year'] <= period_end)
                            ]
                            zone_class_loads[class_name] = zone_period_df['Load'].sum()
                        
                        period_data[zone_name] = zone_class_loads
                else:
                    # If no spatial layers, return overall totals
                    overall_loads = {}
                    for class_name, class_df in class_dfs.items():
                        period_df = class_df[class_df['Install Year'] <= period_end]
                        overall_loads[class_name] = period_df['Load'].sum()
                    period_data['Total'] = overall_loads
                
                period_results[period_name] = period_data
            
            return period_results
            
        except Exception as e:
            LOGGER.error(f"Error analyzing historical loads: {e}")
            return {}

    def get_pipes_in_zone(self, zone_feature, pipe_layer, buffer_pixels: int = 10):
        """Find pipes that intersect with a zone polygon using a buffer.
        
        Args:
            zone_feature: Zone polygon feature
            pipe_layer: QGIS vector layer with pipe data
            buffer_pixels: Buffer size in pixels to prevent undercounting
            
        Returns:
            List of pipe names that intersect the zone
        """
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
                
                # Process the data with user inputs
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
                        "Warning", 
                        "No historical data could be processed. Please check your inputs."
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
                                           QDoubleSpinBox, QFileDialog, QGroupBox, QFormLayout)
            from qgis.PyQt.QtCore import Qt
            from qgis.core import QgsProject
            
            dialog = QDialog(self.iface.mainWindow())
            dialog.setWindowTitle("Historical Analysis - Input Selection")
            dialog.setMinimumSize(500, 400)
            
            layout = QVBoxLayout()
            
            # File selection group
            file_group = QGroupBox("Input Files")
            file_layout = QFormLayout()
            
            # Excel file selection
            excel_layout = QHBoxLayout()
            self.excel_line = QLineEdit()
            excel_browse = QPushButton("Browse...")
            excel_browse.clicked.connect(self.browse_excel_file)
            excel_layout.addWidget(self.excel_line)
            excel_layout.addWidget(excel_browse)
            file_layout.addRow("Excel File (Service Point List):", excel_layout)
            
            file_group.setLayout(file_layout)
            layout.addWidget(file_group)
            
            # Layer selection group
            layer_group = QGroupBox("QGIS Layers")
            layer_layout = QFormLayout()
            
            # Zone layer selection
            self.zone_combo = QComboBox()
            self.populate_layer_combo(self.zone_combo, "Polygon")
            layer_layout.addRow("Zone Layer (Polygons):", self.zone_combo)
            
            # Pipe layer selection
            self.pipe_combo = QComboBox()
            self.populate_layer_combo(self.pipe_combo, "LineString")
            layer_layout.addRow("Pipe Layer (Lines):", self.pipe_combo)
            
            layer_group.setLayout(layer_layout)
            layout.addWidget(layer_group)
            
            # Parameters group
            param_group = QGroupBox("Analysis Parameters")
            param_layout = QFormLayout()
            
            # Load multiplier
            self.load_multiplier_spin = QDoubleSpinBox()
            self.load_multiplier_spin.setDecimals(3)
            self.load_multiplier_spin.setRange(0.001, 10.0)
            self.load_multiplier_spin.setValue(self.load_multiplier)
            param_layout.addRow("Load Multiplier:", self.load_multiplier_spin)
            
            # Heat factor multiplier
            self.heat_multiplier_spin = QDoubleSpinBox()
            self.heat_multiplier_spin.setDecimals(1)
            self.heat_multiplier_spin.setRange(0.1, 200.0)
            self.heat_multiplier_spin.setValue(self.heat_factor_multiplier)
            param_layout.addRow("Heat Factor Multiplier:", self.heat_multiplier_spin)
            
            # Start year
            self.start_year_spin = QSpinBox()
            self.start_year_spin.setRange(1990, 2030)
            self.start_year_spin.setValue(2000)
            param_layout.addRow("Start Year:", self.start_year_spin)
            
            # End year
            self.end_year_spin = QSpinBox()
            self.end_year_spin.setRange(2000, 2050)
            self.end_year_spin.setValue(2025)
            param_layout.addRow("End Year:", self.end_year_spin)
            
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
                    'start_year': self.start_year_spin.value(),
                    'end_year': self.end_year_spin.value()
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
                tab_widget.addTab(table, period)
            
            layout.addWidget(tab_widget)
            
            # Close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)
            
            dialog.setLayout(layout)
            dialog.exec_()
            
        except Exception as e:
            LOGGER.error(f"Error showing results dialog: {e}")