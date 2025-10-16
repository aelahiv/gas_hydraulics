"""Load Assignment Module for QGIS Gas Hydraulics Plugin.

This module handles the assignment of forecasted loads to pipe infrastructure
based on spatial intersection with area polygons and construction year timing.
"""

import logging
from collections import defaultdict

# Import validation functions from centralized module
from .data_validation import (
    DataValidator,
    validate_layer_data,
    validate_csv_forecast_data,
    show_validation_dialog,
    safe_int,
    safe_float,
    safe_str
)

LOGGER = logging.getLogger(__name__)


def find_field_name_case_insensitive(layer, target_field_name):
    """Find actual field name in layer matching target (case-insensitive).
    
    Args:
        layer: QGIS vector layer
        target_field_name: Field name to find (will be compared case-insensitively)
        
    Returns:
        str: Actual field name if found, None otherwise
    """
    target_upper = target_field_name.upper()
    for field in layer.fields():
        if field.name().upper() == target_upper:
            return field.name()
    return None


class LoadAssignmentTool:
    """Tool for assigning forecasted loads to pipes."""
    
    def __init__(self, iface=None):
        """Initialize the Load Assignment tool.
        
        Args:
            iface: QGIS interface object (can be None for testing)
        """
        self.iface = iface
        self.selected_csv_path = None
        self.csv_path_label = None
        self.pipe_layer_combo = None
        self.polygon_layer_combo = None
        self.start_year_spin = None
    
    def run(self):
        """Run the load assignment tool with GUI."""
        try:
            from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, 
                                            QLabel, QPushButton, QFileDialog, 
                                            QComboBox, QSpinBox, QMessageBox,
                                            QGroupBox, QFormLayout, QFrame)
            from qgis.PyQt.QtGui import QFont
            from qgis.core import QgsProject, QgsVectorLayer, QgsFeature, QgsField
            from qgis.PyQt.QtCore import QVariant
            
            # Create dialog
            dialog = QDialog(self.iface.mainWindow() if self.iface else None)
            dialog.setWindowTitle("Load Assignment - Assign Forecast Loads to Pipes")
            dialog.setMinimumWidth(700)
            
            layout = QVBoxLayout()
            
            # Title
            title_label = QLabel(" Load Assignment to Infrastructure")
            title_font = QFont()
            title_font.setPointSize(13)
            title_font.setBold(True)
            title_label.setFont(title_font)
            layout.addWidget(title_label)
            
            # Description
            desc_label = QLabel(
                "Assign forecast loads from polygons to pipe segments:\n\n"
                " Takes CSV forecast output and distributes loads to infrastructure\n"
                "️ Uses spatial intersection between zones and pipe network\n"
                "️ Calculates load per pipe segment based on area coverage\n\n"
                " Validation Preview: After selecting inputs, review validation summary before processing"
            )
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #666; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
            layout.addWidget(desc_label)
            
            # Separator
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            layout.addWidget(line)
            
            # CSV File Selection
            csv_group = QGroupBox(" 1. Select Forecast CSV File")
            csv_layout = QHBoxLayout()
            self.csv_path_label = QLabel("No file selected")
            self.csv_path_label.setStyleSheet("color: #888; font-style: italic;")
            csv_browse_btn = QPushButton(" Browse...")
            csv_browse_btn.clicked.connect(lambda: self.browse_csv())
            csv_browse_btn.setToolTip(
                "Select the CSV file output from forecast plugin\n"
                "File should contain columns: Area, Year, Residential, Commercial, etc."
            )
            csv_layout.addWidget(self.csv_path_label, 1)
            csv_layout.addWidget(csv_browse_btn)
            csv_group.setLayout(csv_layout)
            layout.addWidget(csv_group)
            
            # Layer Selection
            layer_group = QGroupBox("️ 2. Select Spatial Layers")
            layer_layout = QFormLayout()
            
            # Pipe layer selection
            pipe_label = QLabel(" Pipe Network (Lines):")
            pipe_label.setToolTip("Line layer representing gas distribution pipes")
            self.pipe_layer_combo = QComboBox()
            self.populate_line_layers(self.pipe_layer_combo)
            self.pipe_layer_combo.setToolTip(
                "Select the pipe network line layer\n"
                "Loads will be assigned to pipe segments based on spatial intersection"
            )
            layer_layout.addRow(pipe_label, self.pipe_layer_combo)
            
            # Polygon layer selection
            polygon_label = QLabel(" Service Areas (Polygons):")
            polygon_label.setToolTip("Polygon layer defining service area zones")
            self.polygon_layer_combo = QComboBox()
            self.populate_polygon_layers(self.polygon_layer_combo)
            self.polygon_layer_combo.setToolTip(
                "Select the polygon layer containing service areas\n"
                "Must match the 'Area' names in your forecast CSV"
            )
            layer_layout.addRow(polygon_label, self.polygon_layer_combo)
            
            layer_group.setLayout(layer_layout)
            layout.addWidget(layer_group)
            
            # Start Year
            year_group = QGroupBox(" 3. Set Start Year")
            year_layout = QHBoxLayout()
            year_description = QLabel("Forecast starting year:")
            year_description.setToolTip("Should match the start year from your forecast")
            self.start_year_spin = QSpinBox()
            self.start_year_spin.setMinimum(2000)
            self.start_year_spin.setMaximum(2100)
            self.start_year_spin.setValue(2025)
            self.start_year_spin.setToolTip(
                "Base year for the forecast period\n"
                "Subsequent years will be calculated as offsets (Y0, Y5, Y10, etc.)"
            )
            year_layout.addWidget(year_description)
            year_layout.addWidget(self.start_year_spin)
            year_layout.addStretch()
            year_group.setLayout(year_layout)
            layout.addWidget(year_group)
            
            # Buttons
            button_layout = QHBoxLayout()
            run_btn = QPushButton(" Assign Loads to Pipes")
            run_btn.clicked.connect(lambda: self.execute_assignment(dialog))
            run_btn.setToolTip("Process the forecast CSV and assign loads to pipe segments")
            cancel_btn = QPushButton(" Cancel")
            cancel_btn.clicked.connect(dialog.reject)
            button_layout.addStretch()
            button_layout.addWidget(run_btn)
            button_layout.addWidget(cancel_btn)
            layout.addLayout(button_layout)
            
            dialog.setLayout(layout)
            dialog.exec_()
            
        except Exception as e:
            LOGGER.error(f"Error in load assignment dialog: {e}")
            import traceback
            LOGGER.error(f"Traceback: {traceback.format_exc()}")
            if self.iface:
                from qgis.PyQt.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Error",
                    f"Error opening load assignment dialog:\n{str(e)}"
                )
    
    def browse_csv(self):
        """Browse for forecast CSV file."""
        try:
            from qgis.PyQt.QtWidgets import QFileDialog
            
            file_path, _ = QFileDialog.getOpenFileName(
                self.iface.mainWindow() if self.iface else None,
                "Select Forecast CSV File",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if file_path:
                self.csv_path_label.setText(file_path)
                self.selected_csv_path = file_path
                
        except Exception as e:
            LOGGER.error(f"Error browsing for CSV: {e}")
    
    def populate_line_layers(self, combo):
        """Populate combo box with line layers from QGIS project.
        
        Args:
            combo: QComboBox to populate
        """
        try:
            from qgis.core import QgsProject, QgsWkbTypes
            
            combo.clear()
            project = QgsProject.instance()
            
            for layer in project.mapLayers().values():
                if layer.type() == 0:  # Vector layer
                    if layer.geometryType() == QgsWkbTypes.LineGeometry:
                        combo.addItem(layer.name(), layer)
                        
        except Exception as e:
            LOGGER.error(f"Error populating line layers: {e}")
    
    def populate_polygon_layers(self, combo):
        """Populate combo box with polygon layers from QGIS project.
        
        Args:
            combo: QComboBox to populate
        """
        try:
            from qgis.core import QgsProject, QgsWkbTypes
            
            combo.clear()
            project = QgsProject.instance()
            
            for layer in project.mapLayers().values():
                if layer.type() == 0:  # Vector layer
                    if layer.geometryType() == QgsWkbTypes.PolygonGeometry:
                        combo.addItem(layer.name(), layer)
                        
        except Exception as e:
            LOGGER.error(f"Error populating polygon layers: {e}")
    
    def execute_assignment(self, dialog):
        """Execute the load assignment process.
        
        Args:
            dialog: Parent dialog to close on success
        """
        try:
            from qgis.PyQt.QtWidgets import QMessageBox
            from qgis.core import QgsField, QgsFeature, QgsSpatialIndex, QgsGeometry
            from qgis.PyQt.QtCore import QVariant
            import csv
            from datetime import datetime
            
            # Validate inputs
            if not hasattr(self, 'selected_csv_path') or not self.selected_csv_path:
                QMessageBox.warning(
                    dialog,
                    "Missing Input",
                    "Please select a forecast CSV file."
                )
                return
            
            pipe_layer = self.pipe_layer_combo.currentData()
            polygon_layer = self.polygon_layer_combo.currentData()
            
            if not pipe_layer or not polygon_layer:
                QMessageBox.warning(
                    dialog,
                    "Missing Layers",
                    "Please select both pipe and area layers."
                )
                return
            
            start_year = self.start_year_spin.value()
            
            # Check if pipe layer has YEAR column (case-insensitive)
            year_field = find_field_name_case_insensitive(pipe_layer, 'YEAR')
            if not year_field:
                QMessageBox.warning(
                    dialog,
                    "Missing Column",
                    f"Pipe layer must have a 'YEAR' column (case-insensitive).\n\n"
                    f"Available fields: {', '.join([f.name() for f in pipe_layer.fields()])}"
                )
                return
            
            # Read forecast CSV
            LOGGER.info(f"Reading forecast CSV: {self.selected_csv_path}")
            forecast_data = self.parse_csv(self.selected_csv_path)
            
            if not forecast_data:
                QMessageBox.warning(
                    dialog,
                    "CSV Error",
                    "Could not parse forecast CSV file or no data found."
                )
                return
            
            # Get polygon name field (case-insensitive)
            polygon_fields = {field.name().upper(): field.name() for field in polygon_layer.fields()}
            name_field = None
            for field_name in ['NAME', 'AREA', 'ID']:
                if field_name in polygon_fields:
                    name_field = polygon_fields[field_name]
                    break
            
            if not name_field:
                QMessageBox.warning(
                    dialog,
                    "Missing Field",
                    "Polygon layer must have a name field (NAME, AREA, or ID - case-insensitive)."
                )
                return
            
            # === DATA VALIDATION AND PREVIEW ===
            LOGGER.info("="*70)
            LOGGER.info("DATA VALIDATION AND PREVIEW")
            LOGGER.info("="*70)
            
            # Show data preview dialog
            preview_result = self.show_data_preview(
                pipe_layer, 
                polygon_layer, 
                forecast_data, 
                year_field, 
                name_field,
                start_year
            )
            
            if not preview_result:
                LOGGER.info("User cancelled after reviewing data preview")
                return
            
            LOGGER.info("User confirmed data preview, proceeding with assignment...")
            
            # Execute the assignment
            success, message = self.assign_loads_to_pipes(
                pipe_layer, 
                polygon_layer, 
                forecast_data, 
                start_year, 
                name_field
            )
            
            if success:
                LOGGER.info(" Load assignment completed successfully")
                QMessageBox.information(dialog, "Success", message)
                dialog.accept()
            else:
                LOGGER.error(f"Load assignment failed: {message}")
                QMessageBox.critical(dialog, "Error", message)
            
        except Exception as e:
            LOGGER.error(f"Error executing load assignment: {e}")
            import traceback
            LOGGER.error(f"Traceback: {traceback.format_exc()}")
            
            if 'pipe_layer' in locals() and hasattr(locals()['pipe_layer'], 'isEditable'):
                if locals()['pipe_layer'].isEditable():
                    locals()['pipe_layer'].rollBack()
            
            from qgis.PyQt.QtWidgets import QMessageBox
            QMessageBox.critical(
                dialog,
                "Error",
                f"Error during load assignment:\n{str(e)}"
            )
    
    def show_data_preview(self, pipe_layer, polygon_layer, forecast_data, year_field, name_field, start_year):
        """Show a preview of the data to be processed with validation.
        
        Args:
            pipe_layer: Pipe vector layer
            polygon_layer: Polygon vector layer  
            forecast_data: Parsed forecast data from CSV
            year_field: Name of year field in pipe layer
            name_field: Name of name field in polygon layer
            start_year: Base year for calculations
            
        Returns:
            bool: True if user confirms, False if cancelled
        """
        from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QTextEdit, QPushButton,
                                         QHBoxLayout, QLabel, QTabWidget, QWidget,
                                         QTableWidget, QTableWidgetItem, QHeaderView)
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtGui import QFont
        
        # Collect data statistics
        LOGGER.info("Collecting data statistics for preview...")
        
        # Pipe layer stats
        pipe_count = pipe_layer.featureCount()
        pipe_fields = [f.name() for f in pipe_layer.fields()]
        pipe_years = set()
        pipe_year_counts = {}
        
        for pipe in pipe_layer.getFeatures():
            year_raw = pipe[year_field]
            year = safe_int(year_raw)
            if year:
                pipe_years.add(year)
                pipe_year_counts[year] = pipe_year_counts.get(year, 0) + 1
        
        # Polygon layer stats
        polygon_count = polygon_layer.featureCount()
        polygon_fields = [f.name() for f in polygon_layer.fields()]
        polygon_names = set()
        
        for polygon in polygon_layer.getFeatures():
            name_raw = polygon[name_field]
            name = safe_str(name_raw).strip()
            if name:
                polygon_names.add(name)
        
        # CSV forecast data stats
        csv_areas = set(forecast_data.keys())
        csv_years = set()
        for area_data in forecast_data.values():
            csv_years.update(area_data.keys())
        
        # Matching analysis
        matching_areas = polygon_names & csv_areas
        missing_in_csv = polygon_names - csv_areas
        missing_in_polygons = csv_areas - polygon_names
        overlapping_years = pipe_years & csv_years
        
        # Create preview dialog
        dialog = QDialog(self.iface.mainWindow() if self.iface else None)
        dialog.setWindowTitle("Data Validation Preview")
        dialog.setMinimumSize(900, 700)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(" Data Validation and Preview")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Create tabs
        tabs = QTabWidget()
        
        # === TAB 1: Summary ===
        summary_widget = QWidget()
        summary_layout = QVBoxLayout()
        summary_text = QTextEdit()
        summary_text.setReadOnly(True)
        summary_text.setFont(QFont("Courier New", 10))
        
        summary = []
        summary.append("="*80)
        summary.append("DATA SUMMARY")
        summary.append("="*80)
        summary.append("")
        
        summary.append(" PIPE LAYER:")
        summary.append(f"  • Total pipes: {pipe_count}")
        summary.append(f"  • Year field: '{year_field}'")
        summary.append(f"  • Year range: {min(pipe_years) if pipe_years else 'N/A'} - {max(pipe_years) if pipe_years else 'N/A'}")
        summary.append(f"  • Unique years: {len(pipe_years)}")
        summary.append(f"  • Fields: {', '.join(pipe_fields)}")
        summary.append("")
        
        summary.append(" POLYGON LAYER:")
        summary.append(f"  • Total polygons: {polygon_count}")
        summary.append(f"  • Name field: '{name_field}'")
        summary.append(f"  • Unique areas: {len(polygon_names)}")
        summary.append(f"  • Area names: {', '.join(sorted(polygon_names)) if polygon_names else 'None'}")
        summary.append(f"  • Fields: {', '.join(polygon_fields)}")
        summary.append("")
        
        summary.append(" FORECAST CSV:")
        summary.append(f"  • Areas in CSV: {len(csv_areas)}")
        summary.append(f"  • Area names: {', '.join(sorted(csv_areas)) if csv_areas else 'None'}")
        summary.append(f"  • Year range: {min(csv_years) if csv_years else 'N/A'} - {max(csv_years) if csv_years else 'N/A'}")
        summary.append(f"  • Unique years: {len(csv_years)}")
        summary.append(f"  • Start year: {start_year}")
        summary.append("")
        
        summary.append("="*80)
        summary.append("MATCHING ANALYSIS")
        summary.append("="*80)
        summary.append("")
        
        if matching_areas:
            summary.append(f" MATCHED AREAS ({len(matching_areas)}):")
            for area in sorted(matching_areas):
                summary.append(f"  • {area}")
            summary.append("")
        else:
            summary.append(" NO MATCHING AREAS FOUND!")
            summary.append("")
        
        if missing_in_csv:
            summary.append(f"️  POLYGONS NOT IN CSV ({len(missing_in_csv)}):")
            for area in sorted(missing_in_csv):
                summary.append(f"  • {area}")
            summary.append("")
        
        if missing_in_polygons:
            summary.append(f"️  CSV AREAS NOT IN POLYGONS ({len(missing_in_polygons)}):")
            for area in sorted(missing_in_polygons):
                summary.append(f"  • {area}")
            summary.append("")
        
        if overlapping_years:
            summary.append(f" OVERLAPPING YEARS ({len(overlapping_years)}):")
            summary.append(f"  {', '.join(map(str, sorted(overlapping_years)))}")
            summary.append("")
        else:
            summary.append(" NO OVERLAPPING YEARS BETWEEN PIPES AND CSV!")
            summary.append("")
        
        summary.append("="*80)
        summary.append("COMPATIBILITY CHECK")
        summary.append("="*80)
        summary.append("")
        
        issues = []
        if not matching_areas:
            issues.append(" CRITICAL: No matching areas between polygons and CSV")
        if not overlapping_years:
            issues.append(" CRITICAL: No overlapping years between pipes and CSV")
        if len(matching_areas) < len(polygon_names) * 0.5:
            issues.append("️  WARNING: Less than 50% of polygons match CSV areas")
        
        if issues:
            for issue in issues:
                summary.append(issue)
            summary.append("")
            summary.append(" Issues detected - review data before proceeding!")
        else:
            summary.append(" Data looks compatible!")
            summary.append(f" {len(matching_areas)} areas will be processed")
            summary.append(f" {len(overlapping_years)} years will be matched")
        
        summary_text.setText("\n".join(summary))
        summary_layout.addWidget(summary_text)
        summary_widget.setLayout(summary_layout)
        tabs.addTab(summary_widget, " Summary")
        
        # === TAB 2: Pipe Years ===
        pipe_years_widget = QWidget()
        pipe_years_layout = QVBoxLayout()
        pipe_years_table = QTableWidget()
        pipe_years_table.setColumnCount(2)
        pipe_years_table.setHorizontalHeaderLabels(["Year", "Pipe Count"])
        pipe_years_table.setRowCount(len(pipe_year_counts))
        
        for i, (year, count) in enumerate(sorted(pipe_year_counts.items())):
            pipe_years_table.setItem(i, 0, QTableWidgetItem(str(year)))
            pipe_years_table.setItem(i, 1, QTableWidgetItem(str(count)))
        
        pipe_years_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        pipe_years_layout.addWidget(QLabel(f"Pipes by Construction Year (Total: {pipe_count})"))
        pipe_years_layout.addWidget(pipe_years_table)
        pipe_years_widget.setLayout(pipe_years_layout)
        tabs.addTab(pipe_years_widget, " Pipe Years")
        
        # === TAB 3: Forecast Data ===
        forecast_widget = QWidget()
        forecast_layout = QVBoxLayout()
        forecast_table = QTableWidget()
        
        # Create table with areas as rows, years as columns
        sorted_years = sorted(csv_years) if csv_years else []
        sorted_areas = sorted(csv_areas) if csv_areas else []
        
        forecast_table.setColumnCount(len(sorted_years) + 1)
        forecast_table.setRowCount(len(sorted_areas))
        forecast_table.setHorizontalHeaderLabels(["Area"] + [str(y) for y in sorted_years])
        
        for i, area in enumerate(sorted_areas):
            forecast_table.setItem(i, 0, QTableWidgetItem(area))
            for j, year in enumerate(sorted_years):
                load = forecast_data.get(area, {}).get(year, 0.0)
                forecast_table.setItem(i, j + 1, QTableWidgetItem(f"{load:.2f}"))
        
        forecast_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        forecast_layout.addWidget(QLabel(f"Forecast Loads from CSV ({len(sorted_areas)} areas, {len(sorted_years)} years)"))
        forecast_layout.addWidget(forecast_table)
        forecast_widget.setLayout(forecast_layout)
        tabs.addTab(forecast_widget, " Forecast Data")
        
        # === TAB 4: Matching Details ===
        matching_widget = QWidget()
        matching_layout = QVBoxLayout()
        matching_text = QTextEdit()
        matching_text.setReadOnly(True)
        matching_text.setFont(QFont("Courier New", 10))
        
        matching_details = []
        matching_details.append("AREA NAME MATCHING")
        matching_details.append("="*80)
        matching_details.append("")
        matching_details.append(f"Polygon Layer uses field: '{name_field}'")
        matching_details.append(f"CSV Area Names (case-sensitive):")
        matching_details.append("")
        
        for area in sorted(csv_areas):
            match_status = "" if area in polygon_names else ""
            matching_details.append(f"{match_status} '{area}'")
        
        matching_details.append("")
        matching_details.append("TIPS FOR FIXING MISMATCHES:")
        matching_details.append("  • Check spelling and capitalization")
        matching_details.append("  • Check for extra spaces or special characters")
        matching_details.append("  • Ensure CSV area names match polygon layer exactly")
        matching_details.append("  • Use 'NAME', 'AREA', or 'ID' field in polygon layer")
        
        matching_text.setText("\n".join(matching_details))
        matching_layout.addWidget(matching_text)
        matching_widget.setLayout(matching_layout)
        tabs.addTab(matching_widget, " Matching Details")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        info_label = QLabel("Review the data above. Click 'Proceed' to continue or 'Cancel' to abort.")
        info_label.setWordWrap(True)
        button_layout.addWidget(info_label)
        button_layout.addStretch()
        
        proceed_button = QPushButton(" Proceed with Assignment")
        cancel_button = QPushButton(" Cancel")
        
        proceed_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(proceed_button)
        
        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        
        # Log to console as well
        LOGGER.info("\n".join(summary))
        
        # Show dialog
        result = dialog.exec_()
        return result == QDialog.Accepted
    
    def _export_synergi_shapefile(self, qgis_layer):
        """Export QGIS layer to Synergi-compatible shapefile using GeoPandas.
        
        GeoPandas preserves datetime objects correctly (Format 11/12 from testing),
        whereas QGIS's export converts them in ways that break Synergi compatibility.
        
        Args:
            qgis_layer: QGIS vector layer with YEAR field
            
        Returns:
            str: Path to exported Synergi-compatible shapefile
        """
        import geopandas as gpd
        import pandas as pd
        from datetime import datetime
        import tempfile
        import os
        
        # Export QGIS layer to temporary shapefile first
        temp_dir = tempfile.gettempdir()
        temp_shp = os.path.join(temp_dir, "qgis_temp_export.shp")
        
        from qgis.core import QgsVectorFileWriter, QgsCoordinateReferenceSystem
        error = QgsVectorFileWriter.writeAsVectorFormat(
            qgis_layer,
            temp_shp,
            "utf-8",
            qgis_layer.crs(),
            "ESRI Shapefile"
        )
        
        if error[0] != QgsVectorFileWriter.NoError:
            raise Exception(f"Failed to export temporary shapefile: {error}")
        
        # Read with GeoPandas
        gdf = gpd.read_file(temp_shp)
        
        # Remove existing DATETIME field (it's a string from QGIS)
        if 'DATETIME' in gdf.columns:
            gdf = gdf.drop(columns=['DATETIME'])
        
        # Remove SYN_DATE if it exists
        if 'SYN_DATE' in gdf.columns:
            gdf = gdf.drop(columns=['SYN_DATE'])
        
        # Create DATETIME as Excel serial date (Format 14 - confirmed working!)
        # CRITICAL: Store as STRING (Character field), not numeric
        # Format 14 test used Character field with serial number as text: '45971'
        # Excel serial = days since 1899-12-30 (Excel's epoch)
        if 'YEAR' in gdf.columns or 'Year' in gdf.columns:
            year_col = 'YEAR' if 'YEAR' in gdf.columns else 'Year'
            
            def year_to_excel_serial_string(year_str):
                """Convert year to Excel serial date STRING for November 11th."""
                if pd.isna(year_str) or not str(year_str).strip():
                    return ""
                try:
                    year = int(year_str)
                    # November 11th
                    target_date = pd.Timestamp(year, 11, 11)
                    # Excel epoch (serial 1 = 1900-01-01, but use 1899-12-30 due to Excel bug)
                    epoch = pd.Timestamp(1899, 12, 30)
                    serial = (target_date - epoch).days
                    return str(serial)  # Return as STRING, not float!
                except:
                    return ""
            
            gdf['DATETIME'] = gdf[year_col].apply(year_to_excel_serial_string)
        
        # Determine output path
        layer_source = qgis_layer.source()
        if layer_source and layer_source.endswith('.shp'):
            base_path = layer_source.replace('.shp', '')
            output_path = f"{base_path}_synergi.shp"
        else:
            # Use same directory as temp file
            layer_name = qgis_layer.name().replace(' ', '_')
            output_path = os.path.join(os.path.dirname(temp_shp), f"{layer_name}_synergi.shp")
        
        # Write using GeoPandas (preserves datetime objects correctly!)
        gdf.to_file(output_path)
        
        # Clean up temp files
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            temp_file = temp_shp.replace('.shp', ext)
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        
        return output_path
    
    def assign_loads_to_pipes(self, pipe_layer, polygon_layer, forecast_data, start_year, name_field):
        """Assign loads to pipes based on spatial intersection and forecast data.
        
        Args:
            pipe_layer: Vector layer containing pipe lines with YEAR column
            polygon_layer: Vector layer containing area polygons with name field
            forecast_data: Dictionary of {area: {year: load}}
            start_year: Base year for period calculation
            name_field: Field name in polygon layer containing area names
            
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            from qgis.core import QgsField, QgsSpatialIndex
            from qgis.PyQt.QtCore import QVariant, QDate
            
            # Start editing pipe layer
            pipe_layer.startEditing()
            
            # Add required fields if they don't exist (case-insensitive check)
            existing_fields_map = {field.name().upper(): field.name() for field in pipe_layer.fields()}
            
            if 'DESC' not in existing_fields_map:
                pipe_layer.addAttribute(QgsField('DESC', QVariant.String))
            if 'LOAD' not in existing_fields_map:
                pipe_layer.addAttribute(QgsField('LOAD', QVariant.Double))
            if 'PROP' not in existing_fields_map:
                pipe_layer.addAttribute(QgsField('PROP', QVariant.String))
            if 'YEAR' not in existing_fields_map:
                pipe_layer.addAttribute(QgsField('YEAR', QVariant.String))
            if 'DATETIME' not in existing_fields_map:
                # Human-readable date for QGIS display
                pipe_layer.addAttribute(QgsField('DATETIME', QVariant.String, len=15))
            if 'SYN_DATE' not in existing_fields_map:
                # Integer date for Synergi (YYYYMMDD format - QGIS won't convert integers)
                pipe_layer.addAttribute(QgsField('SYN_DATE', QVariant.Int))
            
            pipe_layer.updateFields()
            
            # Get actual field names (case-insensitive)
            year_field = find_field_name_case_insensitive(pipe_layer, 'YEAR')
            desc_field = find_field_name_case_insensitive(pipe_layer, 'DESC')
            load_field = find_field_name_case_insensitive(pipe_layer, 'LOAD')
            prop_field = find_field_name_case_insensitive(pipe_layer, 'PROP')
            datetime_field = find_field_name_case_insensitive(pipe_layer, 'DATETIME')
            syn_date_field = find_field_name_case_insensitive(pipe_layer, 'SYN_DATE')
            
            # Create spatial index for polygons
            LOGGER.info("Creating spatial index for polygons...")
            spatial_index = QgsSpatialIndex(polygon_layer.getFeatures())
            
            # Group pipes by polygon intersection and year
            LOGGER.info("Grouping pipes by polygon intersection and year...")
            pipe_groups = defaultdict(lambda: defaultdict(list))
            pipe_polygon_map = {}  # Map pipe ID to polygon name
            
            for pipe in pipe_layer.getFeatures():
                pipe_geom = pipe.geometry()
                pipe_year_raw = pipe[year_field]
                
                # Safely convert year to integer (handles strings like "2020")
                pipe_year = safe_int(pipe_year_raw)
                
                if pipe_year is None:
                    LOGGER.warning(f"Pipe {pipe.id()} has invalid year value: '{pipe_year_raw}', skipping")
                    continue
                
                # Find intersecting polygons
                intersecting_ids = spatial_index.intersects(pipe_geom.boundingBox())
                
                for poly_id in intersecting_ids:
                    polygon = polygon_layer.getFeature(poly_id)
                    poly_geom = polygon.geometry()
                    
                    if pipe_geom.intersects(poly_geom):
                        poly_name_raw = polygon[name_field]
                        # Safely convert to string (handles None, numbers, etc.)
                        poly_name = safe_str(poly_name_raw).strip()
                        
                        if not poly_name:
                            LOGGER.warning(f"Polygon {poly_id} has empty name, skipping")
                            continue
                        
                        pipe_groups[poly_name][pipe_year].append(pipe.id())
                        pipe_polygon_map[pipe.id()] = poly_name
                        break  # Use first intersecting polygon
            
            LOGGER.info(f"Found {len(pipe_groups)} polygon groups")
            
            # Calculate and assign loads
            LOGGER.info("Calculating and assigning loads...")
            
            field_indices = {
                'DESC': pipe_layer.fields().indexFromName(desc_field),
                'LOAD': pipe_layer.fields().indexFromName(load_field),
                'PROP': pipe_layer.fields().indexFromName(prop_field),
                'YEAR': pipe_layer.fields().indexFromName(year_field),
                'DATETIME': pipe_layer.fields().indexFromName(datetime_field),
                'SYN_DATE': pipe_layer.fields().indexFromName(syn_date_field)
            }
            
            for poly_name, year_pipes in pipe_groups.items():
                for pipe_year, pipe_ids in year_pipes.items():
                    # Calculate period (how many years from start)
                    # Both pipe_year and start_year should be int, but ensure they are
                    period_years = safe_int(pipe_year, 0) - safe_int(start_year, 0)
                    
                    # Get load for this polygon and period
                    load_raw = forecast_data.get(poly_name, {}).get(pipe_year, 0.0)
                    load = safe_float(load_raw, 0.0)
                    
                    if load > 0 and len(pipe_ids) > 0:
                        # Divide load equally among pipes
                        load_per_pipe = load / len(pipe_ids)
                        
                        # Generate description
                        desc = f"{poly_name} - {period_years} Year Load"
                        
                        # Generate human-readable year string
                        year_str = str(pipe_year)
                        
                        # Create two date representations:
                        # 1. DATETIME: Human-readable string for QGIS display (11-Nov-2025)
                        datetime_str = f"11-Nov-{pipe_year}"
                        
                        # 2. SYN_DATE: Integer in YYYYMMDD format for Synergi
                        #    QGIS won't convert integers, so Synergi will receive raw value
                        syn_date_int = (pipe_year * 10000) + 1111  # November 11th = YYYYMMDD format
                        
                        # Update each pipe
                        for pipe_id in pipe_ids:
                            pipe_layer.changeAttributeValue(pipe_id, field_indices['DESC'], desc)
                            pipe_layer.changeAttributeValue(pipe_id, field_indices['LOAD'], load_per_pipe)
                            pipe_layer.changeAttributeValue(pipe_id, field_indices['PROP'], 'Proposed')
                            pipe_layer.changeAttributeValue(pipe_id, field_indices['YEAR'], year_str)
                            pipe_layer.changeAttributeValue(pipe_id, field_indices['DATETIME'], datetime_str)
                            pipe_layer.changeAttributeValue(pipe_id, field_indices['SYN_DATE'], syn_date_int)
            
            # Commit changes
            pipe_layer.commitChanges()
            
            # Refresh layer
            pipe_layer.triggerRepaint()
            
            # Export Synergi-compatible shapefile using GeoPandas
            synergi_export_msg = ""
            try:
                synergi_path = self._export_synergi_shapefile(pipe_layer)
                synergi_export_msg = f"\n\n✓ Synergi-ready shapefile exported:\n  {synergi_path}\n  (Use this file for Synergi import, not the QGIS layer)"
            except Exception as e:
                LOGGER.warning(f"Could not export Synergi shapefile: {e}")
                synergi_export_msg = f"\n\n⚠ Could not auto-export Synergi file: {e}\n  You can manually export using: python fix_datetime_for_synergi.py"
            
            message = (
                f"Load assignment completed successfully!\n\n"
                f"Processed {len(pipe_polygon_map)} pipes across {len(pipe_groups)} areas.\n\n"
                f"Added/Updated fields:\n"
                f"  DESC = Description\n"
                f"  LOAD = Load value (GJ/d)\n"
                f"  PROP = Status ('Proposed')\n"
                f"  YEAR = Human-readable year (e.g., '2025')\n"
                f"  DATETIME = Human-readable date (11-Nov-YYYY)\n"
                f"  SYN_DATE = Integer date for Synergi (YYYYMMDD)\n"
                f"{synergi_export_msg}"
            )
            
            return True, message
            
        except Exception as e:
            LOGGER.error(f"Error in assign_loads_to_pipes: {e}")
            import traceback
            LOGGER.error(f"Traceback: {traceback.format_exc()}")
            
            # Rollback changes on error
            if pipe_layer.isEditable():
                pipe_layer.rollBack()
            
            return False, f"Error during load assignment:\n{str(e)}"
    
    def parse_csv(self, csv_path):
        """Parse forecast CSV and extract loads by area and year.
        
        Args:
            csv_path: Path to forecast CSV file
            
        Returns:
            dict: Dictionary of {area: {year: load}}
        """
        try:
            import csv
            
            forecast_data = {}
            current_area = None
            in_summary = False
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                
                for row in reader:
                    if not row or not row[0]:
                        continue
                    
                    # Check if we've reached the summary section
                    if 'SUMMARY' in row[0]:
                        in_summary = True
                        current_area = None
                        continue
                    
                    # Check if this is a subzone header
                    if 'SUBZONE' in row[0] or 'AREA' in row[0]:
                        in_summary = False
                        # Extract area name from header like "SUBZONE: Kensington"
                        parts = row[0].split(':')
                        if len(parts) > 1:
                            current_area = parts[1].strip()
                            forecast_data[current_area] = {}
                        continue
                    
                    # Skip header rows (case-insensitive)
                    if row[0].upper() in ['YEAR']:
                        continue
                    
                    # Try to parse data row (only if we're in a subzone, not summary)
                    # Use safe_int to handle both numeric and string year values
                    if current_area and not in_summary:
                        try:
                            # Try to convert year - handles "2020", 2020, 2020.0
                            year = safe_int(row[0])
                            if year is None:
                                continue
                            
                            # Total load is in the last column - handle strings
                            total_load = safe_float(row[-1], 0.0)
                            
                            if total_load > 0:  # Only store non-zero loads
                                forecast_data[current_area][year] = total_load
                        except (IndexError, AttributeError) as e:
                            LOGGER.debug(f"Skipping row: {row}, error: {e}")
                            continue
            
            LOGGER.info(f"Parsed forecast data for {len(forecast_data)} areas")
            return forecast_data
            
        except Exception as e:
            LOGGER.error(f"Error parsing forecast CSV: {e}")
            return {}
