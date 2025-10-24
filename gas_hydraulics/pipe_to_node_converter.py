"""
Pipe-to-Node Load Converter Plugin
===================================

This plugin converts pipe-loaded hydraulic models to node-loaded models by:
1. Identifying pipe intersections and validating connections between LTPS and GA pipes
2. Initializing flow categories for each ASP from pipe descriptions
3. Identifying terminal nodes and summing incoming loads
4. Producing demand files for node loading with correct category assignments
5. Generating Synergi demand scripts (.dsf files)

Author: Gas Hydraulics Team
Version: 1.0.0
"""

import os
import csv
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None


# Column mapping from Synergi field names to internal names
COLUMN_MAPPING = {
    'FacNam1005': 'Name',              # Facility Name
    'FacTyp1020': 'Type',              # Facility Type
    'PipeLo2010': 'Load',              # Pipe Load (GJ/d)
    'FacAct1024': 'Activity',          # Facility Activity
    'FacDes1010': 'Description',       # Facility Description
    'FacFlo1029': 'Flow',              # Facility Flow
    'FacFro1006': 'From-Node',         # From Node Name
    'FacToN1007': 'To-Node',           # To Node Name
    'FacSer1023': 'Service1',          # Service field 1
    'FacSer1017': 'Service2',          # Service field 2
    'FacSta1009': 'Status',            # Facility Status
    'PipeDi2003': 'Diameter',          # Pipe Diameter (mm)
}

# Reverse mapping for output
REVERSE_COLUMN_MAPPING = {v: k for k, v in COLUMN_MAPPING.items()}


class PipeToNodeConverter:
    """Main plugin class for pipe-to-node load conversion."""
    
    def __init__(self, iface=None):
        """Initialize the plugin."""
        self.iface = iface
        
        # Default parameters
        self.diameter_threshold = 900  # mm - threshold for LTPS vs GA pipes
        self.forecast_is_cumulative = True  # Reforecast: treat CSV values as cumulative totals by default
        
        # Layer selection (will be set by user)
        self.pipe_layer = None
        self.output_dir = None
        
        # Data storage
        self.df_ltps = None
        self.df_drop = None
        self.pipes_df = None
        self.grouped_df = None
        self.filtered_intersections = None
        
        # Reforecast mode parameters
        self.reforecast_mode = False
        self.forecast_csv_path = None
        self.demand_file_path = None
        
    def run(self):
        """Launch the main conversion dialog."""
        if not self.check_dependencies():
            return
            
        if not self.show_input_dialog():
            return
            
        # Run the conversion process
        self.execute_conversion()
    
    def check_dependencies(self):
        """Check if required dependencies are installed."""
        try:
            from qgis.PyQt.QtWidgets import QMessageBox
            
            if pd is None:
                QMessageBox.critical(
                    self.iface.mainWindow() if self.iface else None,
                    "Missing Dependency",
                    "This plugin requires pandas library.\n\n"
                    "Please install it using:\n"
                    "pip install pandas"
                )
                return False
            return True
            
        except Exception as e:
            print(f"Dependency check error: {str(e)}")
            return False
    
    def show_input_dialog(self):
        """Show dialog to collect user inputs."""
        try:
            from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                           QPushButton, QLineEdit, QFileDialog, QGroupBox,
                                           QFormLayout, QSpinBox, QFrame, QDoubleSpinBox,
                                           QComboBox)
            from qgis.PyQt.QtGui import QFont
            from qgis.PyQt.QtCore import Qt
            from qgis.core import QgsProject, QgsVectorLayer
            
            dialog = QDialog(self.iface.mainWindow() if self.iface else None)
            dialog.setWindowTitle("Pipe-to-Node Load Converter")
            dialog.setMinimumSize(750, 650)
            
            layout = QVBoxLayout()
            
            # Title
            title_label = QLabel(" Pipe-to-Node Load Conversion")
            title_font = QFont()
            title_font.setPointSize(13)
            title_font.setBold(True)
            title_label.setFont(title_font)
            layout.addWidget(title_label)
            
            # Mode selection group
            mode_group = QGroupBox("⚙️ Conversion Mode")
            mode_layout = QVBoxLayout()
            
            self.mode_combo = QComboBox()
            self.mode_combo.addItems([
                "Normal Conversion (Pipe Layer → Node Demands)",
                "Reforecast Mode (Update Existing Demands)"
            ])
            self.mode_combo.setToolTip(
                "Normal Conversion: Full pipe-to-node conversion from QGIS layer\n"
                "Reforecast Mode: Update existing node demands with new forecast values"
            )
            self.mode_combo.currentIndexChanged.connect(lambda: self.update_mode_visibility())
            mode_layout.addWidget(self.mode_combo)
            
            mode_group.setLayout(mode_layout)
            layout.addWidget(mode_group)
            
            # Description (will update based on mode)
            self.desc_label = QLabel()
            self.desc_label.setWordWrap(True)
            self.desc_label.setStyleSheet("color: #666; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
            layout.addWidget(self.desc_label)
            
            # Store dialog and layout for mode updates
            self.dialog = dialog
            self.dialog_layout = layout
            
            # Separator
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            layout.addWidget(line)
            
            # Reforecast mode inputs (will be shown/hidden based on mode)
            self.reforecast_group = QGroupBox(" Reforecast Inputs")
            reforecast_layout = QVBoxLayout()
            
            # Forecast CSV input
            forecast_csv_label = QLabel(" Forecast CSV File:")
            forecast_csv_label.setToolTip(
                "Select the forecast CSV file from forecast plugin\n"
                "Should contain columns: Area, Year, Load (GJ/d)"
            )
            
            forecast_csv_h_layout = QHBoxLayout()
            self.forecast_csv_edit = QLineEdit()
            self.forecast_csv_edit.setPlaceholderText("Select forecast CSV file...")
            forecast_csv_browse_btn = QPushButton(" Browse...")
            forecast_csv_browse_btn.clicked.connect(self.browse_forecast_csv)
            forecast_csv_browse_btn.setToolTip("Browse for forecast CSV")
            
            forecast_csv_h_layout.addWidget(self.forecast_csv_edit)
            forecast_csv_h_layout.addWidget(forecast_csv_browse_btn)
            
            reforecast_layout.addWidget(forecast_csv_label)
            reforecast_layout.addLayout(forecast_csv_h_layout)
            
            # Existing demand file input
            demand_file_label = QLabel(" Existing Node Demand File (node_ex.csv):")
            demand_file_label.setToolTip(
                "Select the existing node_ex.csv demand file\n"
                "This file was created by a previous pipe-to-node conversion"
            )
            
            demand_file_h_layout = QHBoxLayout()
            self.demand_file_edit = QLineEdit()
            self.demand_file_edit.setPlaceholderText("Select existing node_ex.csv...")
            demand_file_browse_btn = QPushButton(" Browse...")
            demand_file_browse_btn.clicked.connect(self.browse_demand_file)
            demand_file_browse_btn.setToolTip("Browse for existing demand file")
            
            demand_file_h_layout.addWidget(self.demand_file_edit)
            demand_file_h_layout.addWidget(demand_file_browse_btn)
            
            reforecast_layout.addWidget(demand_file_label)
            reforecast_layout.addLayout(demand_file_h_layout)

            # Forecast interpretation toggle
            interp_label = QLabel(" Forecast value interpretation:")
            interp_label.setToolTip(
                "Choose how to interpret the 'Load' values in the forecast CSV.\n\n"
                "Cumulative totals (recommended): The value for each period represents the total cumulative\n"
                "load up to that period (e.g., 5-year total, 10-year total). The reforecaster will compute\n"
                "the per-period increment and rebase node loads to those increments.\n\n"
                "Per-period increments: The value for each period already represents the increment for that\n"
                "period only (e.g., the added load between 0→5, 5→10). The reforecaster will use these\n"
                "values directly without differencing."
            )
            self.forecast_interp_checkbox = QComboBox()
            self.forecast_interp_checkbox.addItems([
                "Cumulative totals (compute increments)",
                "Per-period increments (use as-is)"
            ])
            self.forecast_interp_checkbox.setCurrentIndex(0)
            reforecast_layout.addWidget(interp_label)
            reforecast_layout.addWidget(self.forecast_interp_checkbox)
            
            self.reforecast_group.setLayout(reforecast_layout)
            layout.addWidget(self.reforecast_group)
            
            # Input layer group (for normal conversion)
            self.input_group = QGroupBox(" Input Pipe Layer")
            input_layout = QFormLayout()
            
            # Get available vector layers
            project = QgsProject.instance()
            vector_layers = [layer for layer in project.mapLayers().values() 
                           if isinstance(layer, QgsVectorLayer)]
            layer_names = [layer.name() for layer in vector_layers]
            
            # Pipe layer selection
            pipe_layer_label = QLabel(" Pipe Network Layer:")
            pipe_layer_label.setToolTip(
                "Select the pipe layer containing load data\n\n"
                "Required Attributes:\n"
                "  • Name: Pipe identifier (starts with 'LTPS' or 'GA')\n"
                "  • Diameter (mm): Pipe diameter in millimeters\n"
                "  • Load (GJ/d): Load on pipe in GJ/day\n"
                "  • To-Node Name: Destination node identifier\n"
                "  • From-Node Name: Source node identifier\n"
                "  • Description: Pipe description (format: 'Category - Details')\n\n"
                "The Name column will be used to classify pipes:\n"
                "  • Names starting with 'LTPS' = Large Transmission Pipes\n"
                "  • Names starting with 'GA' = Gate Artery pipes"
            )
            
            self.pipe_layer_combo = QComboBox()
            self.pipe_layer_combo.addItems(layer_names)
            self.pipe_layer_combo.setToolTip(
                "Select the layer with pipe data\n"
                "Must have Name, Diameter, Load, To-Node, From-Node, Description attributes"
            )
            input_layout.addRow(pipe_layer_label, self.pipe_layer_combo)
            
            self.input_group.setLayout(input_layout)
            layout.addWidget(self.input_group)
            
            # Output directory group
            self.output_group = QGroupBox(" Output Directory")
            output_layout = QHBoxLayout()
            
            output_label = QLabel(
                "Select folder where all output files will be saved.\n"
                "Four files will be created: node_ex.csv, cat_setup.csv, synergi_demand_script.dsf, connections.csv"
            )
            output_label.setWordWrap(True)
            output_label.setStyleSheet("color: #666; font-size: 10px; margin-bottom: 5px;")
            
            self.output_dir_edit = QLineEdit()
            self.output_dir_edit.setPlaceholderText("Select output directory...")
            output_browse_btn = QPushButton(" Browse...")
            output_browse_btn.clicked.connect(self.browse_output_dir)
            output_browse_btn.setToolTip("Browse for output directory")
            
            output_layout.addWidget(self.output_dir_edit)
            output_layout.addWidget(output_browse_btn)
            
            output_vlayout = QVBoxLayout()
            output_vlayout.addWidget(output_label)
            output_vlayout.addLayout(output_layout)
            self.output_group.setLayout(output_vlayout)
            layout.addWidget(self.output_group)
            
            # Parameters group (for normal conversion)
            self.param_group = QGroupBox("️ Conversion Parameters")
            param_layout = QFormLayout()
            
            # Diameter threshold
            diameter_label = QLabel(" Diameter Threshold (mm):")
            diameter_label.setToolTip(
                "Diameter threshold to distinguish pipe types:\n"
                "  • Pipes > threshold: LTPS (Large Transmission Pipes)\n"
                "  • Pipes ≤ threshold: GA (Gate Artery) pipes\n"
                "Typical value: 900mm\n\n"
                "Note: Pipes are also classified by Name prefix ('LTPS' or 'GA')"
            )
            
            self.diameter_spin = QSpinBox()
            self.diameter_spin.setRange(100, 2000)
            self.diameter_spin.setValue(900)
            self.diameter_spin.setSuffix(" mm")
            self.diameter_spin.setToolTip(
                "Set the diameter threshold for pipe classification\n"
                "Default: 900mm (typical LTPS threshold)"
            )
            param_layout.addRow(diameter_label, self.diameter_spin)
            
            self.param_group.setLayout(param_layout)
            layout.addWidget(self.param_group)
            
            # Processing info (for normal conversion)
            self.info_group = QGroupBox("ℹ️ Processing Steps")
            info_layout = QVBoxLayout()
            
            info_text = QLabel(
                "The conversion will perform the following operations:\n\n"
                "Step 1️⃣: Load and classify pipes by diameter threshold\n"
                "Step 2️⃣: Identify intersection nodes between LTPS and GA pipes\n"
                "Step 3️⃣: Validate connections (2 LTPS + 1 GA per node)\n"
                "Step 4️⃣: Extract flow categories from pipe descriptions\n"
                "Step 5️⃣: Aggregate loads at terminal nodes by category\n"
                "Step 6️⃣: Generate node demand file with categorical data\n"
                "Step 7️⃣: Create Synergi demand script (.dsf) for import\n"
                "Step 8️⃣: Export validated pipe connections for model correction"
            )
            info_text.setWordWrap(True)
            info_text.setStyleSheet("color: #444; font-size: 10px; padding: 5px;")
            info_layout.addWidget(info_text)
            
            self.info_group.setLayout(info_layout)
            layout.addWidget(self.info_group)
            
            # Initialize mode visibility
            self.update_mode_visibility()
            
            # Buttons
            button_layout = QHBoxLayout()
            run_btn = QPushButton(" Run Conversion")
            run_btn.clicked.connect(dialog.accept)
            run_btn.setToolTip("Execute the pipe-to-node conversion process")
            cancel_btn = QPushButton(" Cancel")
            cancel_btn.clicked.connect(dialog.reject)
            
            button_layout.addStretch()
            button_layout.addWidget(run_btn)
            button_layout.addWidget(cancel_btn)
            layout.addLayout(button_layout)
            
            dialog.setLayout(layout)
            
            # Show dialog and collect results
            if dialog.exec_() == dialog.Accepted:
                # Check if reforecast mode
                self.reforecast_mode = self.mode_combo.currentIndex() == 1
                
                if self.reforecast_mode:
                    # Collect reforecast mode inputs
                    self.forecast_csv_path = self.forecast_csv_edit.text()
                    self.demand_file_path = self.demand_file_edit.text()
                    self.output_dir = self.output_dir_edit.text()
                    # Store interpretation flag
                    self.forecast_is_cumulative = (self.forecast_interp_checkbox.currentIndex() == 0)
                else:
                    # Collect normal conversion inputs
                    from qgis.core import QgsProject, QgsVectorLayer
                    project = QgsProject.instance()
                    vector_layers = [layer for layer in project.mapLayers().values() 
                                   if isinstance(layer, QgsVectorLayer)]
                    
                    if self.pipe_layer_combo.currentIndex() >= 0:
                        self.pipe_layer = vector_layers[self.pipe_layer_combo.currentIndex()]
                    
                    self.output_dir = self.output_dir_edit.text()
                    self.diameter_threshold = self.diameter_spin.value()
                
                # Validate inputs
                if not self.validate_inputs():
                    return False
                
                return True
            
            return False
            
        except Exception as e:
            self.show_error("Dialog Error", f"Error showing input dialog: {str(e)}")
            return False
    
    def browse_file(self, line_edit, file_type):
        """Browse for input CSV file."""
        try:
            from qgis.PyQt.QtWidgets import QFileDialog
            
            file_path, _ = QFileDialog.getOpenFileName(
                self.iface.mainWindow() if self.iface else None,
                f"Select {file_type}",
                "",
                "CSV Files (*.csv);;All Files (*.*)"
            )
            
            if file_path:
                line_edit.setText(file_path)
                
        except Exception as e:
            self.show_error("Browse Error", f"Error browsing for file: {str(e)}")
    
    def browse_output_dir(self):
        """Browse for output directory."""
        try:
            from qgis.PyQt.QtWidgets import QFileDialog
            
            dir_path = QFileDialog.getExistingDirectory(
                self.iface.mainWindow() if self.iface else None,
                "Select Output Directory"
            )
            
            if dir_path:
                self.output_dir_edit.setText(dir_path)
                
        except Exception as e:
            self.show_error("Browse Error", f"Error browsing for directory: {str(e)}")
    
    def browse_forecast_csv(self):
        """Browse for forecast CSV file."""
        try:
            from qgis.PyQt.QtWidgets import QFileDialog
            
            file_path, _ = QFileDialog.getOpenFileName(
                self.iface.mainWindow() if self.iface else None,
                "Select Forecast CSV File",
                "",
                "CSV Files (*.csv);;All Files (*.*)"
            )
            
            if file_path:
                self.forecast_csv_edit.setText(file_path)
                
        except Exception as e:
            self.show_error("Browse Error", f"Error browsing for file: {str(e)}")
    
    def browse_demand_file(self):
        """Browse for existing demand file (node_ex.csv)."""
        try:
            from qgis.PyQt.QtWidgets import QFileDialog
            
            file_path, _ = QFileDialog.getOpenFileName(
                self.iface.mainWindow() if self.iface else None,
                "Select Existing Node Demand File",
                "",
                "CSV Files (*.csv);;All Files (*.*)"
            )
            
            if file_path:
                self.demand_file_edit.setText(file_path)
                
        except Exception as e:
            self.show_error("Browse Error", f"Error browsing for file: {str(e)}")
    
    def update_mode_visibility(self):
        """Update dialog UI based on selected mode."""
        is_reforecast = self.mode_combo.currentIndex() == 1
        
        # Update description label
        if is_reforecast:
            self.desc_label.setText(
                "Reforecast mode: Update existing node demands with new forecast values\n\n"
                " How it works:\n"
                "   1. Load forecast CSV (Area, Year, Load columns)\n"
                "   2. Load existing node_ex.csv demand file\n"
                "   3. Parse node descriptions to extract area and year information\n"
                "   4. Calculate load ratios within each area/year group\n"
                "   5. Rebase ratios to new forecast values (maintains distribution)\n"
                "   6. Generate updated node_ex.csv with adjusted loads\n\n"
                " Use Case:\n"
                "   When you have updated forecast values but the pipe network hasn't changed,\n"
                "   use this mode to update node demands without re-running the full conversion.\n\n"
                " Tip: Node descriptions must follow format 'AreaName - Period Year Load'"
            )
        else:
            self.desc_label.setText(
                "Convert pipe-loaded hydraulic models to node-loaded models for Synergi:\n\n"
                " Process Overview:\n"
                "   1. Read pipe data from QGIS layer attribute table\n"
                "   2. Identify LTPS vs GA pipes by Name column pattern and diameter\n"
                "   3. Validate connections between LTPS and GA pipes at intersection nodes\n"
                "   4. Extract flow categories from pipe descriptions\n"
                "   5. Sum loads at terminal nodes for facility load shifting\n"
                "   6. Generate node demand file with proper category assignments\n"
                "   7. Create Synergi demand script (.dsf) for model import\n\n"
                " Outputs Generated:\n"
                "   • node_ex.csv: Node loads aggregated by category\n"
                "   • cat_setup.csv: Flow category configuration\n"
                "   • synergi_demand_script.dsf: Synergi import script\n"
                "   • connections.csv: Validated LTPS-GA pipe connections\n\n"
                " Name Column Pattern:\n"
                "   The 'Name' column is used to distinguish pipe types:\n"
                "   • LTPS pipes: Name starts with 'LTPS' (e.g., 'LTPS_Main_001')\n"
                "   • GA pipes: Name starts with 'GA' (e.g., 'GA_Branch_042')\n\n"
                " Tip: Ensure pipe descriptions follow format 'CategoryName - Details'"
            )
        
        # Show/hide groups based on mode
        self.reforecast_group.setVisible(is_reforecast)
        self.input_group.setVisible(not is_reforecast)
        self.param_group.setVisible(not is_reforecast)
        self.info_group.setVisible(not is_reforecast)
    
    def validate_inputs(self):
        """Validate user inputs before processing."""
        try:
            from qgis.PyQt.QtWidgets import QMessageBox
            
            errors = []
            
            if self.reforecast_mode:
                # Validate reforecast mode inputs
                if not self.forecast_csv_path:
                    errors.append(" Forecast CSV file not specified")
                elif not os.path.exists(self.forecast_csv_path):
                    errors.append(" Forecast CSV file does not exist")
                
                if not self.demand_file_path:
                    errors.append(" Existing demand file (node_ex.csv) not specified")
                elif not os.path.exists(self.demand_file_path):
                    errors.append(" Existing demand file does not exist")
                
                # Check output directory
                if not self.output_dir:
                    errors.append(" Output directory not specified")
                elif not os.path.exists(self.output_dir):
                    try:
                        os.makedirs(self.output_dir)
                    except Exception as e:
                        errors.append(f" Cannot create output directory: {str(e)}")
                
            else:
                # Validate normal conversion inputs
                # Check pipe layer
                if not self.pipe_layer:
                    errors.append(" Pipe layer not selected")
                else:
                    # Check required fields using Synergi column names
                    required_synergi_fields = [
                        'FacNam1005',  # Name
                        'PipeDi2003',  # Diameter
                        'PipeLo2010',  # Load
                        'FacToN1007',  # To-Node
                        'FacFro1006',  # From-Node
                        'FacDes1010'   # Description
                    ]
                    layer_fields = [field.name() for field in self.pipe_layer.fields()]
                    missing_fields = [f for f in required_synergi_fields if f not in layer_fields]
                    
                    if missing_fields:
                        errors.append(f" Layer missing required Synergi fields: {', '.join(missing_fields)}")
                    
                    # Check if layer has features
                    if self.pipe_layer.featureCount() == 0:
                        errors.append(" Selected layer has no features")
                
                # Check output directory
                if not self.output_dir:
                    errors.append(" Output directory not specified")
                elif not os.path.exists(self.output_dir):
                    try:
                        os.makedirs(self.output_dir)
                    except Exception as e:
                        errors.append(f" Cannot create output directory: {str(e)}")
            
            if errors:
                QMessageBox.warning(
                    self.iface.mainWindow() if self.iface else None,
                    "Validation Errors",
                    "Please fix the following issues:\n\n" + "\n".join(errors)
                )
                return False
            
            return True
            
        except Exception as e:
            self.show_error("Validation Error", f"Error validating inputs: {str(e)}")
            return False
    
    def execute_conversion(self):
        """Execute the full conversion process."""
        try:
            # Route to appropriate conversion method
            if self.reforecast_mode:
                return self.execute_reforecast()
            else:
                return self.execute_normal_conversion()
                
        except Exception as e:
            self.show_error("Conversion Error", f"Unexpected error during conversion: {str(e)}")
            return
    
    def execute_normal_conversion(self):
        """Execute normal pipe-to-node conversion."""
        try:
            from qgis.PyQt.QtWidgets import QMessageBox, QProgressDialog
            from qgis.PyQt.QtCore import Qt
            
            # Create progress dialog
            progress = QProgressDialog(
                "Converting pipe loads to node loads...",
                "Cancel",
                0, 8,
                self.iface.mainWindow() if self.iface else None
            )
            progress.setWindowTitle("Pipe-to-Node Conversion")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # Step 1: Load and classify pipes
            progress.setLabelText("Step 1/8: Loading and classifying pipes...")
            progress.setValue(0)
            if not self.load_and_classify_pipes():
                progress.close()
                return
            
            # Step 2: Load all pipes data
            progress.setLabelText("Step 2/8: Loading complete pipe network...")
            progress.setValue(1)
            if not self.load_pipes_data():
                progress.close()
                return
            
            # Step 3: Group loads by terminal nodes
            progress.setLabelText("Step 3/8: Aggregating loads at terminal nodes...")
            progress.setValue(2)
            if not self.group_loads_by_nodes():
                progress.close()
                return
            
            # Step 4: Export node exchange file
            progress.setLabelText("Step 4/8: Exporting node demand file...")
            progress.setValue(3)
            if not self.export_node_exchange():
                progress.close()
                return
            
            # Step 5: Export category setup
            progress.setLabelText("Step 5/8: Creating flow category setup...")
            progress.setValue(4)
            if not self.export_category_setup():
                progress.close()
                return
            
            # Step 6: Create Synergi demand script
            progress.setLabelText("Step 6/8: Generating Synergi demand script...")
            progress.setValue(5)
            if not self.create_synergi_script():
                progress.close()
                return
            
            # Step 7: Find pipe intersections
            progress.setLabelText("Step 7/8: Identifying pipe intersections...")
            progress.setValue(6)
            if not self.find_intersections():
                progress.close()
                return
            
            # Step 8: Validate and export connections
            progress.setLabelText("Step 8/8: Validating and exporting connections...")
            progress.setValue(7)
            if not self.export_connections():
                progress.close()
                return
            
            progress.setValue(8)
            progress.close()
            
            # Show success message with summary
            self.show_success_summary()
            
        except Exception as e:
            self.show_error("Conversion Error", f"Error during conversion: {str(e)}")
    
    def execute_reforecast(self):
        """Execute reforecast mode: update existing demand file with new forecast values."""
        try:
            from qgis.PyQt.QtWidgets import QMessageBox, QProgressDialog
            from qgis.PyQt.QtCore import Qt
            import re
            
            def _normalize_area(name: str):
                """Normalize area names for matching: collapse whitespace and lowercase."""
                if name is None:
                    return None
                return " ".join(str(name).split()).lower()
            
            # Create progress dialog
            progress = QProgressDialog(
                "Reforecasting node demands...",
                "Cancel",
                0, 5,
                self.iface.mainWindow() if self.iface else None
            )
            progress.setWindowTitle("Reforecast Mode")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # Step 1: Load forecast CSV
            progress.setLabelText("Step 1/5: Loading forecast CSV...")
            progress.setValue(0)
            
            try:
                forecast_df = pd.read_csv(self.forecast_csv_path)
                print(f"✓ Loaded forecast CSV: {len(forecast_df)} rows")
                
                # Verify forecast CSV has required columns
                required_cols = ['Area', 'Year', 'Load']
                missing_cols = [c for c in required_cols if c not in forecast_df.columns]
                if missing_cols:
                    self.show_error("Forecast CSV Error", 
                                  f"Forecast CSV missing required columns: {', '.join(missing_cols)}")
                    progress.close()
                    return
                
                # Create area-year lookup dictionary (normalized area names)
                forecast_dict = {}
                forecast_norm_index = {}
                for _, row in forecast_df.iterrows():
                    area_raw = str(row['Area']).strip()
                    year_val = int(row['Year'])
                    load_val = float(row['Load'])
                    key = (area_raw, year_val)
                    forecast_dict[key] = load_val
                    norm_key = (_normalize_area(area_raw), year_val)
                    forecast_norm_index[norm_key] = load_val
                
                print(f"✓ Created forecast lookup with {len(forecast_dict)} area-year combinations")
                
            except Exception as e:
                self.show_error("Forecast CSV Error", f"Error reading forecast CSV: {str(e)}")
                progress.close()
                return
            
            # Step 2: Load existing demand file
            progress.setLabelText("Step 2/5: Loading existing demand file...")
            progress.setValue(1)
            
            try:
                demand_df = pd.read_csv(self.demand_file_path)
                print(f"✓ Loaded existing demand file: {len(demand_df)} nodes")
                
                # Verify demand file has required columns
                if 'Description' not in demand_df.columns or 'Total Load (GJ/d)' not in demand_df.columns:
                    self.show_error("Demand File Error",
                                  "Demand file missing 'Description' or 'Total Load (GJ/d)' columns")
                    progress.close()
                    return
                    
            except Exception as e:
                self.show_error("Demand File Error", f"Error reading demand file: {str(e)}")
                progress.close()
                return
            
            # Step 3: Parse descriptions and extract area/year
            progress.setLabelText("Step 3/5: Parsing node descriptions...")
            progress.setValue(2)
            
            # Pattern to match: "AreaName - Period Year Load" (e.g., "Downtown - 5 Year Load")
            pattern = r'^(.+?)\s*-\s*(\d+)\s+Year\s+Load'
            
            demand_df['Area'] = None
            demand_df['Year'] = None
            demand_df['AreaNorm'] = None
            demand_df['ParsedOK'] = False
            
            for idx, row in demand_df.iterrows():
                desc = str(row['Description']).strip()
                match = re.match(pattern, desc, re.IGNORECASE)
                if match:
                    parsed_area = match.group(1).strip()
                    demand_df.at[idx, 'Area'] = parsed_area
                    demand_df.at[idx, 'AreaNorm'] = _normalize_area(parsed_area)
                    demand_df.at[idx, 'Year'] = int(match.group(2))
                    demand_df.at[idx, 'ParsedOK'] = True
            
            parsed_count = demand_df['ParsedOK'].sum()
            print(f"✓ Parsed {parsed_count}/{len(demand_df)} node descriptions")
            
            if parsed_count == 0:
                self.show_error("Parse Error",
                              "Could not parse any node descriptions. Expected format: 'Area - # Year Load'")
                progress.close()
                return
            
            # Step 4: Calculate load ratios and rebase (using incremental forecast per period or direct increments)
            progress.setLabelText("Step 4/5: Calculating load ratios and rebasing...")
            progress.setValue(3)
            
            # Group by area/year and calculate ratios
            demand_df['New Load (GJ/d)'] = demand_df['Total Load (GJ/d)']  # Default to old value
            
            # Log available forecast years for debugging
            available_years = sorted(set(int(year) for _, year in forecast_dict.keys()))
            available_areas = sorted(set(area for area, _ in forecast_dict.keys()))
            print(f"\nForecast CSV contains:")
            print(f"  Areas: {available_areas}")
            print(f"  Years: {available_years}")
            
            # Log what we're looking for in demand file
            demand_areas = sorted(demand_df[demand_df['ParsedOK']]['Area'].unique())
            demand_years = sorted(demand_df[demand_df['ParsedOK']]['Year'].unique())
            print(f"\nDemand file contains:")
            print(f"  Areas: {demand_areas}")
            print(f"  Years (periods): {demand_years}")
            
            # Check if years look like calendar years vs periods
            if available_years and min(available_years) > 1900:
                print(f"\n⚠ WARNING: Forecast CSV appears to use calendar years ({min(available_years)}-{max(available_years)})")
                print(f"   but demand file uses period years ({demand_years})")
                print(f"   This mismatch will prevent matching!")
                self.show_error("Year Format Mismatch",
                              f"Forecast CSV uses calendar years ({min(available_years)}-{max(available_years)}) "
                              f"but demand file uses period years ({demand_years}).\n\n"
                              f"The forecast CSV 'Year' column should contain period years (5, 10, 15, etc.) "
                              f"not calendar years (2025, 2026, etc.).\n\n"
                              f"Please update your forecast CSV or use the correct export format.")
                progress.close()
                return
            
            # Build per-period increments per area using demand periods actually present
            # Normalize forecast_df areas for robust matching
            forecast_df['Area_str'] = forecast_df['Area'].astype(str).str.strip()
            forecast_df['Area_norm'] = forecast_df['Area_str'].apply(_normalize_area)
            forecast_df['Year_int'] = forecast_df['Year'].astype(int)
            cum_forecast = forecast_df.groupby(['Area_norm', 'Year_int'])['Load'].sum().reset_index()

            # 2) Build increments for each area based on demand periods present for that area
            def build_increments_for_area(area_norm, area_demand_years):
                area_rows = cum_forecast[cum_forecast['Area_norm'] == area_norm]
                area_totals = {int(r['Year_int']): float(r['Load']) for _, r in area_rows.iterrows()}
                if not self.forecast_is_cumulative:
                    # CSV contains per-period increments already; use values directly
                    return {y: float(area_totals.get(y, 0.0)) for y in sorted(area_demand_years)}
                # CSV contains cumulative totals; compute differences
                inc = {}
                prev_total = 0.0
                for y in sorted(area_demand_years):
                    if y not in area_totals:
                        continue
                    total_y = area_totals[y]
                    inc_y = max(total_y - prev_total, 0.0)
                    inc[y] = inc_y
                    prev_total = total_y
                return inc

            # Compute area-specific demand years
            area_to_years = {
                area_norm: sorted(set(int(y) for y in demand_df[(demand_df['ParsedOK']) & (demand_df['AreaNorm'] == area_norm)]['Year'].unique()))
                for area_norm in demand_df[demand_df['ParsedOK']]['AreaNorm'].unique()
            }
            area_to_inc = {area_norm: build_increments_for_area(area_norm, years) for area_norm, years in area_to_years.items()}

            matched_count = 0
            total_groups = 0

            for area_year, group_df in demand_df[demand_df['ParsedOK']].groupby(['AreaNorm', 'Year']):
                area_norm, year = area_year
                total_groups += 1

                # Current total for this area/year group
                current_total = group_df['Total Load (GJ/d)'].sum()
                if current_total == 0:
                    print(f"⚠ Warning: Zero total load for {area_norm} - Year {year}, skipping")
                    continue

                # Incremental forecast for this area/year
                inc_total = area_to_inc.get(area_norm, {}).get(int(year))
                if inc_total is None:
                    print(f"⚠ Warning: No incremental forecast for '{area_norm}' - Year {year}, keeping original loads")
                    continue

                matched_count += 1

                # Scale per-node by ratio
                for idx in group_df.index:
                    old_load = demand_df.at[idx, 'Total Load (GJ/d)']
                    ratio = old_load / current_total if current_total > 0 else 0
                    new_load = ratio * inc_total
                    demand_df.at[idx, 'New Load (GJ/d)'] = new_load

                print(f"✓ Rebased {len(group_df)} nodes for {area_norm} - Year {year} (increment): {current_total:.1f} → {inc_total:.1f} GJ/d")

            print(f"\n✓ Successfully matched {matched_count}/{total_groups} area-year groups (incremental)")
            
            if matched_count == 0:
                self.show_error("No Matches Found",
                              f"Could not match any forecast data to demand file.\n\n"
                              f"Please check that Area names and Year values match between files.")
                progress.close()
                return
            
            # Update the main load column
            demand_df['Total Load (GJ/d)'] = demand_df['New Load (GJ/d)']
            
            # Step 5: Export updated demand file
            progress.setLabelText("Step 5/5: Exporting updated demand file...")
            progress.setValue(4)
            
            # Drop temporary columns
            demand_df = demand_df.drop(columns=['Area', 'Year', 'ParsedOK', 'New Load (GJ/d)'])
            
            # Save to output directory
            output_path = os.path.join(self.output_dir, 'node_ex.csv')
            demand_df.to_csv(output_path, index=False)
            
            print(f"✓ Exported updated demand file: {output_path}")
            
            progress.setValue(5)
            progress.close()
            
            # Show success message
            QMessageBox.information(
                self.iface.mainWindow() if self.iface else None,
                "Reforecast Complete",
                f"Successfully reforecasted node demands!\n\n"
                f"• Processed {parsed_count} nodes\n"
                f"• Applied {len(forecast_dict)} forecast values\n"
                f"• Output: {output_path}\n\n"
                f"The updated demand file is ready for use in Synergi."
            )
            
        except Exception as e:
            self.show_error("Reforecast Error", f"Error during reforecast: {str(e)}")
    
    def load_and_classify_pipes(self):
        """Load pipes data from QGIS layer and classify by Name pattern and diameter."""
        try:
            # Read layer data into pandas DataFrame
            data = []
            for feature in self.pipe_layer.getFeatures():
                attrs = feature.attributes()
                field_names = [field.name() for field in self.pipe_layer.fields()]
                row_dict = dict(zip(field_names, attrs))
                data.append(row_dict)
            
            df = pd.DataFrame(data)
            
            # Rename Synergi columns to internal names
            df = df.rename(columns=COLUMN_MAPPING)
            
            # Validate required columns (after mapping)
            required_columns = ['Name', 'Diameter', 'Load', 'To-Node', 'Description']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                # Show both internal and Synergi names
                synergi_missing = [REVERSE_COLUMN_MAPPING.get(col, col) for col in missing_columns]
                self.show_error(
                    "Missing Columns",
                    f"Pipe layer is missing required Synergi fields:\n{', '.join(synergi_missing)}\n\n"
                    f"(Internal names: {', '.join(missing_columns)})"
                )
                return False
            
            # Classify pipes by Name column pattern (LTPS vs GA)
            # LTPS pipes: Name starts with "LTPS"
            # GA pipes: Name starts with "GA"
            # Also filter by diameter as secondary check
            df_ltps_name = df[df['Name'].str.startswith('LTPS', na=False)]
            df_ga_name = df[df['Name'].str.startswith('GA', na=False)]
            
            # Also apply diameter threshold
            self.df_ltps = df_ltps_name[df_ltps_name['Diameter'] > self.diameter_threshold].sort_values(
                by='Load', ascending=False
            )
            self.df_drop = df_ga_name[df_ga_name['Diameter'] <= self.diameter_threshold].sort_values(
                by='Load', ascending=False
            )
            
            print(f"\n Classified pipes from layer '{self.pipe_layer.name()}':")
            print(f"   LTPS pipes (Name starts with 'LTPS', >{self.diameter_threshold}mm): {len(self.df_ltps)}")
            print(f"   GA pipes (Name starts with 'GA', ≤{self.diameter_threshold}mm): {len(self.df_drop)}")
            print(f"   Total features in layer: {self.pipe_layer.featureCount()}")
            
            if len(self.df_ltps) == 0:
                self.show_error(
                    "No LTPS Pipes Found",
                    "No pipes found with Name starting with 'LTPS' and diameter > threshold.\n\n"
                    "Please verify:\n"
                    "  • Pipe names follow the pattern 'LTPS_xxx' for LTPS pipes\n"
                    f"  • Diameter values are greater than {self.diameter_threshold}mm for LTPS pipes"
                )
                return False
            
            return True
            
        except Exception as e:
            self.show_error("Load Error", f"Error loading and classifying pipes from layer: {str(e)}")
            return False
    
    def load_pipes_data(self):
        """Load complete pipes network data from the same layer."""
        try:
            # Read all features from layer into DataFrame
            # This is the same data as load_and_classify_pipes but includes ALL pipes
            data = []
            for feature in self.pipe_layer.getFeatures():
                attrs = feature.attributes()
                field_names = [field.name() for field in self.pipe_layer.fields()]
                row_dict = dict(zip(field_names, attrs))
                data.append(row_dict)
            
            self.pipes_df = pd.DataFrame(data)
            
            # Rename Synergi columns to internal names
            self.pipes_df = self.pipes_df.rename(columns=COLUMN_MAPPING)
            
            # Validate required columns for connectivity analysis (after mapping)
            required_columns = ['Name', 'To-Node', 'From-Node']
            missing_columns = [col for col in required_columns if col not in self.pipes_df.columns]
            
            if missing_columns:
                synergi_missing = [REVERSE_COLUMN_MAPPING.get(col, col) for col in missing_columns]
                self.show_error(
                    "Missing Columns",
                    f"Pipe layer is missing required Synergi fields:\n{', '.join(synergi_missing)}"
                )
                return False
            
            print(f" Loaded {len(self.pipes_df)} pipes from layer for network analysis")
            return True
            
        except Exception as e:
            self.show_error("Load Error", f"Error loading pipes data from layer: {str(e)}")
            return False
    
    def group_loads_by_nodes(self):
        """Group LTPS loads by terminal nodes and extract flow categories."""
        try:
            # Group by 'To-Node' and aggregate loads
            grouped = self.df_ltps.groupby('To-Node').agg({
                'Load': 'sum',
                'Description': 'first'
            }).reset_index()
            
            # Rename columns (using internal names after mapping)
            grouped.columns = ['To-Node', 'Total Load (GJ/d)', 'Description']
            
            # Extract flow category from description (format: "Category - Details")
            grouped['Flow Category'] = grouped['Description'].str.split('-').str[0].str.strip()
            grouped['Flow Type'] = 'Thermal'
            
            # Pad node names with zeros for consistent formatting
            grouped['Name1'] = grouped['To-Node'].apply(self.pad_node_name)
            
            # Convert to dummy variables for categorical data
            self.grouped_df = pd.get_dummies(grouped, columns=['Flow Category'], dtype=int)
            
            print(f" Grouped loads for {len(self.grouped_df)} terminal nodes")
            print(f"   Flow categories found: {len([col for col in self.grouped_df.columns if col.startswith('Flow Category_')])}")
            
            return True
            
        except Exception as e:
            self.show_error("Grouping Error", f"Error grouping loads: {str(e)}")
            return False
    
    def pad_node_name(self, node_name):
        """Pad numbers in node names to fixed width."""
        if '-' in node_name:
            parts = node_name.split('-')
            if len(parts) >= 2 and parts[-1].isdigit():
                prefix = '-'.join(parts[:-1])
                number = parts[-1]
                return f"{prefix}-{number.zfill(5)}"
        return node_name
    
    def export_node_exchange(self):
        """Export the grouped node loads to CSV."""
        try:
            output_path = os.path.join(self.output_dir, 'node_ex.csv')
            self.grouped_df.to_csv(output_path, index=False)
            print(f" Exported node exchange file: {output_path}")
            return True
            
        except Exception as e:
            self.show_error("Export Error", f"Error exporting node exchange file: {str(e)}")
            return False
    
    def export_category_setup(self):
        """Export flow category setup configuration."""
        try:
            # Get unique flow categories from grouped data
            category_columns = [col for col in self.grouped_df.columns if col.startswith('Flow Category_')]
            
            cat_data = []
            for col in category_columns:
                category_name = col.replace('Flow Category_', '')
                cat_data.append({
                    'Flow Category': category_name,
                    'Flow Type': 'Thermal'
                })
            
            cat_df = pd.DataFrame(cat_data)
            
            output_path = os.path.join(self.output_dir, 'cat_setup.csv')
            cat_df.to_csv(output_path, index=False)
            
            print(f" Exported category setup: {output_path}")
            print(f"   Categories: {', '.join([c['Flow Category'] for c in cat_data])}")
            return True
            
        except Exception as e:
            self.show_error("Export Error", f"Error exporting category setup: {str(e)}")
            return False
    
    def create_synergi_script(self):
        """Create Synergi demand script (.dsf) file."""
        try:
            # Get flow category columns
            category_columns = [col for col in self.grouped_df.columns if col.startswith('Flow Category_')]
            
            # Build script lines
            script_lines = ['# Synergi 4 Demand Script File']
            script_lines.append('')
            
            # Add flow category mappings
            for col in category_columns:
                category_name = col.replace('Flow Category_', '')
                script_lines.append(f'[DATA]NodeFlowByCategory({category_name}),TRUE,[{col}]*[Total Load (GJ/d)]')
            
            script_lines.append('')
            
            # Add configuration settings (using mapped column name)
            config = [
                '[FEATURENAMECOLUMN] To-Node',
                '[ISSINGLEFEATURE] YES',
                '[FEATURETYPE1] Nodes',
                '[DEMANDVAR] 1.070, "X"',
                '[DEMANDVAR] 56.800, "Y"',
                '[DEMANDVAR] 1.000, "Z"',
                '[DELIMITER] ,',
                '[EXPORTREPORT] NO',
                '[SHOWINACTIVEFLOWCATS] NO',
                '[HEADINGSUPPORT] YES',
                '[ZERONODES] NO',
                '[ZEROPIPES] NO',
                '[USEDEMANDFILEFROMMODEL] NO',
                '[USEDSFFROMMODEL] NO',
                '[SAVEDEMANDDATAWITHMODEL] NO',
                '[SELECTEDDEMANDFILEFROMMODEL]',
                '[SELECTEDDSFFROMMODEL]',
                '[SCOPETYPE] 0',
                '[SCOPECRITERIA]'
            ]
            
            script_lines.extend(config)
            
            # Write to file
            output_path = os.path.join(self.output_dir, 'synergi_demand_script.dsf')
            with open(output_path, 'w') as f:
                f.write('\n'.join(script_lines))
            
            print(f" Created Synergi demand script: {output_path}")
            print(f"   Flow categories mapped: {len(category_columns)}")
            return True
            
        except Exception as e:
            self.show_error("Script Error", f"Error creating Synergi script: {str(e)}")
            return False
    
    def find_intersections(self):
        """Find intersection nodes between LTPS and GA pipes."""
        try:
            # Get pipe name sets
            drop_names_set = set(self.df_drop['Name'].tolist())
            ltps_pipe_set = set(self.df_ltps['Name'].tolist())
            
            # Get LTPS and non-LTPS nodes (using mapped column names)
            ltps_mask = self.pipes_df['Name'].str.startswith('LTPS-')
            ltps_nodes = set(self.pipes_df[ltps_mask]['To-Node']).union(
                set(self.pipes_df[ltps_mask]['From-Node'])
            )
            non_ltps_nodes = set(self.pipes_df[~ltps_mask]['To-Node']).union(
                set(self.pipes_df[~ltps_mask]['From-Node'])
            )
            
            # Find intersection nodes
            intersection_nodes = ltps_nodes.intersection(non_ltps_nodes)
            
            # Filter intersections based on criteria
            self.filtered_intersections = {}
            
            for node in intersection_nodes:
                # Get connected pipes
                connected_pipes = self.pipes_df[
                    (self.pipes_df['To-Node'] == node) |
                    (self.pipes_df['From-Node'] == node)
                ]['Name'].tolist()
                
                # Check if any pipe is in drop list
                if not any(pipe in drop_names_set for pipe in connected_pipes):
                    continue
                
                # Count LTPS and GA pipes
                ltps_count = sum(1 for pipe in connected_pipes if pipe.startswith('LTPS-'))
                ga_count = sum(1 for pipe in connected_pipes if pipe.startswith('GA'))
                
                # Filter: 2 LTPS + 1 GA pipe configuration
                if ltps_count == 2 and ga_count == 1:
                    # Remove pipes that are in LTPS main list
                    filtered_pipes = [pipe for pipe in connected_pipes if pipe not in ltps_pipe_set]
                    if filtered_pipes:
                        self.filtered_intersections[node] = filtered_pipes
            
            print(f" Found {len(self.filtered_intersections)} valid intersection nodes")
            return True
            
        except Exception as e:
            self.show_error("Intersection Error", f"Error finding intersections: {str(e)}")
            return False
    
    def export_connections(self):
        """Validate and export pipe connections."""
        try:
            valid_connections = []
            errors = []
            
            for node, pipes in self.filtered_intersections.items():
                # Validate exactly 2 pipes at intersection
                if len(pipes) != 2:
                    errors.append(f"Node {node}: Expected 2 pipes, found {len(pipes)}")
                    continue
                
                # Identify GA and LTPS pipes
                ga_element = None
                ltps_element = None
                
                for pipe in pipes:
                    if pipe.startswith('GA'):
                        ga_element = pipe + '-A'  # Add suffix for GA pipes
                    elif pipe.startswith('LTPS'):
                        ltps_element = pipe
                
                # Validate both elements exist
                if ga_element and ltps_element:
                    valid_connections.append([ga_element, ltps_element])
                else:
                    errors.append(f"Node {node}: Missing GA or LTPS prefix in pipes {pipes}")
            
            # Write valid connections to CSV
            output_path = os.path.join(self.output_dir, 'connections.csv')
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Name0', 'Name1'])  # Headers
                writer.writerows(valid_connections)
            
            print(f" Exported {len(valid_connections)} valid connections: {output_path}")
            
            if errors:
                print(f"️  {len(errors)} validation errors encountered")
                for error in errors[:10]:  # Show first 10 errors
                    print(f"   {error}")
                if len(errors) > 10:
                    print(f"   ... and {len(errors) - 10} more")
            
            return True
            
        except Exception as e:
            self.show_error("Export Error", f"Error exporting connections: {str(e)}")
            return False
    
    def show_success_summary(self):
        """Show success dialog with processing summary."""
        try:
            from qgis.PyQt.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout
            from qgis.PyQt.QtGui import QFont
            from qgis.PyQt.QtCore import Qt
            
            dialog = QDialog(self.iface.mainWindow() if self.iface else None)
            dialog.setWindowTitle("Conversion Complete")
            dialog.setMinimumSize(700, 500)
            
            layout = QVBoxLayout()
            
            # Title
            title_label = QLabel(" Pipe-to-Node Conversion Completed Successfully!")
            title_font = QFont()
            title_font.setPointSize(12)
            title_font.setBold(True)
            title_label.setFont(title_font)
            layout.addWidget(title_label)
            
            # Summary text
            summary = QTextEdit()
            summary.setReadOnly(True)
            
            # Build summary content
            content = []
            content.append("=" * 70)
            content.append("CONVERSION SUMMARY")
            content.append("=" * 70)
            content.append("")
            
            content.append(" INPUT PROCESSING:")
            content.append(f"   • LTPS pipes (>{self.diameter_threshold}mm): {len(self.df_ltps)}")
            content.append(f"   • GA pipes (≤{self.diameter_threshold}mm): {len(self.df_drop)}")
            content.append(f"   • Total network pipes: {len(self.pipes_df)}")
            content.append("")
            
            content.append(" NODE AGGREGATION:")
            content.append(f"   • Terminal nodes with loads: {len(self.grouped_df)}")
            
            # Count flow categories
            category_cols = [col for col in self.grouped_df.columns if col.startswith('Flow Category_')]
            content.append(f"   • Flow categories identified: {len(category_cols)}")
            for col in category_cols:
                cat_name = col.replace('Flow Category_', '')
                count = self.grouped_df[col].sum()
                content.append(f"      - {cat_name}: {count} nodes")
            content.append("")
            
            content.append(" CONNECTION VALIDATION:")
            content.append(f"   • Intersection nodes found: {len(self.filtered_intersections)}")
            
            # Count valid connections
            valid_count = sum(1 for pipes in self.filtered_intersections.values() if len(pipes) == 2)
            content.append(f"   • Valid connections (2 LTPS + 1 GA): {valid_count}")
            content.append("")
            
            content.append(" OUTPUT FILES GENERATED:")
            content.append(f"   • node_ex.csv")
            content.append(f"     Location: {os.path.join(self.output_dir, 'node_ex.csv')}")
            content.append(f"     Contains: Node demands with categorical breakdown")
            content.append("")
            content.append(f"   • cat_setup.csv")
            content.append(f"     Location: {os.path.join(self.output_dir, 'cat_setup.csv')}")
            content.append(f"     Contains: Flow category configuration ({len(category_cols)} categories)")
            content.append("")
            content.append(f"   • synergi_demand_script.dsf")
            content.append(f"     Location: {os.path.join(self.output_dir, 'synergi_demand_script.dsf')}")
            content.append(f"     Contains: Synergi import script with {len(category_cols)} category mappings")
            content.append("")
            content.append(f"   • connections.csv")
            content.append(f"     Location: {os.path.join(self.output_dir, 'connections.csv')}")
            content.append(f"     Contains: {valid_count} validated LTPS-GA pipe connections")
            content.append("")
            
            content.append("=" * 70)
            content.append("NEXT STEPS:")
            content.append("=" * 70)
            content.append("1. Review node_ex.csv for node load assignments")
            content.append("2. Import synergi_demand_script.dsf into Synergi Gas")
            content.append("3. Apply connections.csv to fix split pipe names in model")
            content.append("4. Validate flow categories match your system configuration")
            content.append("")
            
            summary.setPlainText('\n'.join(content))
            summary.setFont(QFont("Courier", 9))
            layout.addWidget(summary)
            
            # Buttons
            button_layout = QHBoxLayout()
            open_folder_btn = QPushButton(" Open Output Folder")
            open_folder_btn.clicked.connect(lambda: self.open_output_folder())
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            
            button_layout.addWidget(open_folder_btn)
            button_layout.addStretch()
            button_layout.addWidget(close_btn)
            layout.addLayout(button_layout)
            
            dialog.setLayout(layout)
            dialog.exec_()
            
        except Exception as e:
            from qgis.PyQt.QtWidgets import QMessageBox
            QMessageBox.information(
                self.iface.mainWindow() if self.iface else None,
                "Conversion Complete",
                f" Conversion completed successfully!\n\nOutput files saved to:\n{self.output_dir}"
            )
    
    def open_output_folder(self):
        """Open the output folder in file explorer."""
        try:
            import subprocess
            import platform
            
            system = platform.system()
            if system == "Windows":
                os.startfile(self.output_dir)
            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", self.output_dir])
            else:  # Linux
                subprocess.Popen(["xdg-open", self.output_dir])
                
        except Exception as e:
            print(f"Could not open folder: {str(e)}")
    
    def show_error(self, title, message):
        """Show error message dialog."""
        try:
            from qgis.PyQt.QtWidgets import QMessageBox
            
            QMessageBox.critical(
                self.iface.mainWindow() if self.iface else None,
                title,
                message
            )
        except:
            print(f"ERROR - {title}: {message}")


# Standalone execution for testing
if __name__ == '__main__':
    import sys
    try:
        from qgis.PyQt.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        converter = PipeToNodeConverter()
        converter.run()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error: {str(e)}")
