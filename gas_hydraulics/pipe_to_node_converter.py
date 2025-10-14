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


class PipeToNodeConverter:
    """Main plugin class for pipe-to-node load conversion."""
    
    def __init__(self, iface=None):
        """Initialize the plugin."""
        self.iface = iface
        
        # Default parameters
        self.diameter_threshold = 900  # mm - threshold for LTPS vs GA pipes
        
        # File paths (will be set by user)
        self.input_node_load_path = None
        self.input_pipes_data_path = None
        self.output_dir = None
        
        # Data storage
        self.df_ltps = None
        self.df_drop = None
        self.pipes_df = None
        self.grouped_df = None
        self.filtered_intersections = None
        
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
                                           QFormLayout, QSpinBox, QFrame, QDoubleSpinBox)
            from qgis.PyQt.QtGui import QFont
            from qgis.PyQt.QtCore import Qt
            
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
            
            # Description
            desc_label = QLabel(
                "Convert pipe-loaded hydraulic models to node-loaded models for Synergi:\n\n"
                " Process Overview:\n"
                "   1. Identify LTPS (Large Transmission Pipes) vs GA (Gate Artery) pipes by diameter\n"
                "   2. Validate connections between LTPS and GA pipes at intersection nodes\n"
                "   3. Extract flow categories from pipe descriptions\n"
                "   4. Sum loads at terminal nodes for facility load shifting\n"
                "   5. Generate node demand file with proper category assignments\n"
                "   6. Create Synergi demand script (.dsf) for model import\n\n"
                " Outputs Generated:\n"
                "   • node_ex.csv: Node loads aggregated by category\n"
                "   • cat_setup.csv: Flow category configuration\n"
                "   • synergi_demand_script.dsf: Synergi import script\n"
                "   • connections.csv: Validated LTPS-GA pipe connections\n\n"
                " Tip: Ensure pipe descriptions follow format 'CategoryName - Details'"
            )
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #666; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
            layout.addWidget(desc_label)
            
            # Separator
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            layout.addWidget(line)
            
            # Input files group
            input_group = QGroupBox(" Input Data Files")
            input_layout = QFormLayout()
            
            # LTPS pipes CSV
            ltps_label = QLabel(" LTPS Pipes CSV:")
            ltps_label.setToolTip(
                "CSV file containing pipe data with loads\n"
                "Required columns:\n"
                "  • Name: Pipe identifier\n"
                "  • Diameter (mm): Pipe diameter in millimeters\n"
                "  • Load (GJ/d): Load on pipe in GJ/day\n"
                "  • To-Node Name: Destination node identifier\n"
                "  • From-Node Name: Source node identifier\n"
                "  • Description: Pipe description (format: 'Category - Details')"
            )
            
            ltps_layout = QHBoxLayout()
            self.ltps_path_edit = QLineEdit()
            self.ltps_path_edit.setPlaceholderText("Select CSV file with pipe loads...")
            ltps_browse_btn = QPushButton(" Browse...")
            ltps_browse_btn.clicked.connect(lambda: self.browse_file(self.ltps_path_edit, "LTPS Pipes CSV"))
            ltps_browse_btn.setToolTip("Browse for LTPS pipes CSV file")
            ltps_layout.addWidget(self.ltps_path_edit)
            ltps_layout.addWidget(ltps_browse_btn)
            input_layout.addRow(ltps_label, ltps_layout)
            
            # All pipes CSV
            pipes_label = QLabel(" All Pipes CSV:")
            pipes_label.setToolTip(
                "CSV file containing complete pipe network data\n"
                "Required columns:\n"
                "  • Name: Pipe identifier\n"
                "  • To-Node Name: Destination node identifier\n"
                "  • From-Node Name: Source node identifier\n"
                "Used for connectivity analysis and intersection detection"
            )
            
            pipes_layout = QHBoxLayout()
            self.pipes_path_edit = QLineEdit()
            self.pipes_path_edit.setPlaceholderText("Select CSV file with all pipes network...")
            pipes_browse_btn = QPushButton(" Browse...")
            pipes_browse_btn.clicked.connect(lambda: self.browse_file(self.pipes_path_edit, "All Pipes CSV"))
            pipes_browse_btn.setToolTip("Browse for complete pipes network CSV file")
            pipes_layout.addWidget(self.pipes_path_edit)
            pipes_layout.addWidget(pipes_browse_btn)
            input_layout.addRow(pipes_label, pipes_layout)
            
            input_group.setLayout(input_layout)
            layout.addWidget(input_group)
            
            # Output directory group
            output_group = QGroupBox(" Output Directory")
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
            output_group.setLayout(output_vlayout)
            layout.addWidget(output_group)
            
            # Parameters group
            param_group = QGroupBox("️ Conversion Parameters")
            param_layout = QFormLayout()
            
            # Diameter threshold
            diameter_label = QLabel(" Diameter Threshold (mm):")
            diameter_label.setToolTip(
                "Diameter threshold to distinguish pipe types:\n"
                "  • Pipes > threshold: LTPS (Large Transmission Pipes)\n"
                "  • Pipes ≤ threshold: GA (Gate Artery) pipes\n"
                "Typical value: 900mm"
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
            
            param_group.setLayout(param_layout)
            layout.addWidget(param_group)
            
            # Processing info
            info_group = QGroupBox("ℹ️ Processing Steps")
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
            
            info_group.setLayout(info_layout)
            layout.addWidget(info_group)
            
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
                self.input_node_load_path = self.ltps_path_edit.text()
                self.input_pipes_data_path = self.pipes_path_edit.text()
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
    
    def validate_inputs(self):
        """Validate user inputs before processing."""
        try:
            from qgis.PyQt.QtWidgets import QMessageBox
            
            errors = []
            
            # Check LTPS pipes file
            if not self.input_node_load_path or not os.path.exists(self.input_node_load_path):
                errors.append(" LTPS Pipes CSV file not found")
            
            # Check all pipes file
            if not self.input_pipes_data_path or not os.path.exists(self.input_pipes_data_path):
                errors.append(" All Pipes CSV file not found")
            
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
    
    def load_and_classify_pipes(self):
        """Load LTPS pipes data and classify by diameter."""
        try:
            # Load the data
            df = pd.read_csv(self.input_node_load_path)
            
            # Validate required columns
            required_columns = ['Name', 'Diameter (mm)', 'Load (GJ/d)', 'To-Node Name', 'Description']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.show_error(
                    "Missing Columns",
                    f"LTPS Pipes CSV is missing required columns:\n{', '.join(missing_columns)}"
                )
                return False
            
            # Filter and sort pipes based on diameter threshold
            self.df_ltps = df[df['Diameter (mm)'] > self.diameter_threshold].sort_values(
                by='Load (GJ/d)', ascending=False
            )
            self.df_drop = df[df['Diameter (mm)'] <= self.diameter_threshold].sort_values(
                by='Load (GJ/d)', ascending=False
            )
            
            print(f"\n Classified pipes:")
            print(f"   LTPS pipes (>{self.diameter_threshold}mm): {len(self.df_ltps)}")
            print(f"   GA pipes (≤{self.diameter_threshold}mm): {len(self.df_drop)}")
            
            return True
            
        except Exception as e:
            self.show_error("Load Error", f"Error loading and classifying pipes: {str(e)}")
            return False
    
    def load_pipes_data(self):
        """Load complete pipes network data."""
        try:
            self.pipes_df = pd.read_csv(self.input_pipes_data_path)
            
            # Validate required columns
            required_columns = ['Name', 'To-Node Name', 'From-Node Name']
            missing_columns = [col for col in required_columns if col not in self.pipes_df.columns]
            
            if missing_columns:
                self.show_error(
                    "Missing Columns",
                    f"All Pipes CSV is missing required columns:\n{', '.join(missing_columns)}"
                )
                return False
            
            print(f" Loaded {len(self.pipes_df)} pipes from network data")
            return True
            
        except Exception as e:
            self.show_error("Load Error", f"Error loading pipes data: {str(e)}")
            return False
    
    def group_loads_by_nodes(self):
        """Group LTPS loads by terminal nodes and extract flow categories."""
        try:
            # Group by 'To-Node Name' and aggregate loads
            grouped = self.df_ltps.groupby('To-Node Name').agg({
                'Load (GJ/d)': 'sum',
                'Description': 'first'
            }).reset_index()
            
            # Rename columns
            grouped.columns = ['To-Node Name', 'Total Load (GJ/d)', 'Description']
            
            # Extract flow category from description (format: "Category - Details")
            grouped['Flow Category'] = grouped['Description'].str.split('-').str[0].str.strip()
            grouped['Flow Type'] = 'Thermal'
            
            # Pad node names with zeros for consistent formatting
            grouped['Name1'] = grouped['To-Node Name'].apply(self.pad_node_name)
            
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
            
            # Add configuration settings
            config = [
                '[FEATURENAMECOLUMN] To-Node Name',
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
            
            # Get LTPS and non-LTPS nodes
            ltps_mask = self.pipes_df['Name'].str.startswith('LTPS-')
            ltps_nodes = set(self.pipes_df[ltps_mask]['To-Node Name']).union(
                set(self.pipes_df[ltps_mask]['From-Node Name'])
            )
            non_ltps_nodes = set(self.pipes_df[~ltps_mask]['To-Node Name']).union(
                set(self.pipes_df[~ltps_mask]['From-Node Name'])
            )
            
            # Find intersection nodes
            intersection_nodes = ltps_nodes.intersection(non_ltps_nodes)
            
            # Filter intersections based on criteria
            self.filtered_intersections = {}
            
            for node in intersection_nodes:
                # Get connected pipes
                connected_pipes = self.pipes_df[
                    (self.pipes_df['To-Node Name'] == node) |
                    (self.pipes_df['From-Node Name'] == node)
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
