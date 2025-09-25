"""Forecast Plugin for QGIS.

This plugin performs 5/10/15/20 year forecasts for gas loads.
Takes current per-area per-class loads, ultimate loads, and growth projections.
Uses forecasting template logic to project future loads by area and class.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Tuple
import numpy as np
from datetime import datetime

LOGGER = logging.getLogger(__name__)

class ForecastPlugin:
    def __init__(self, iface: Any = None):
        """Initialize the Forecast plugin."""
        self.iface = iface
        self.action = None
        
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

    def project_units(self, area_dicts: Dict[str, Dict[int, float]], 
                     growth_dict: Dict[int, float], start_year: int, end_year: int, 
                     areas: List[str], case: str = 'fixed', threshold: float = 0.85) -> Dict[int, List[float]]:
        """Project load units over time using the forecasting template logic.
        
        Args:
            area_dicts: Dictionary mapping area names to {year: load, 'ultimate': ultimate_load}
            growth_dict: Dictionary mapping years to growth values
            start_year: Starting year for projection
            end_year: Ending year for projection
            areas: List of area names
            case: Projection case (fixed, variable, etc.)
            threshold: Development threshold for reduction factor
            
        Returns:
            Dictionary mapping years to list of projected loads by area
        """
        res_dict = {start_year: [area_dicts[area][start_year] for area in areas]}
        
        ultimate_units = {area: area_dicts[area].get('ultimate', area_dicts[area][start_year]) for area in areas}
        initial_supply = [l - u for l, u in zip(ultimate_units.values(), res_dict[start_year])]
        factor_dict = {start_year: [u / s if s != 0 else 1 for u, s in zip(ultimate_units.values(), initial_supply)]}
        
        # Priority areas for enhanced growth in first 5 years
        priority_areas = {
            'Kensington', 'West Industrial', 'Meadow View',
            'Arbour Hills', 'Westgate', 'Southwest', 'South East',
            'Northwest', 'North East'
        }

        years_range = range(start_year, end_year + 1)
        total_units_changes = [growth_dict.get(year, 0) for year in years_range]
        
        for year_idx, year in enumerate(range(start_year + 1, end_year + 1)):
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
            
            # Apply enhanced growth to priority areas for first 5 years
            if year_idx < 5:
                for i, area in enumerate(areas):
                    if area in priority_areas:
                        unit_inv[i] *= 2
                # Re-normalize after applying the boost
                total_modified_inv = sum(unit_inv)
                if total_modified_inv > 0:
                    unit_inv = [u / total_modified_inv for u in unit_inv]
            
            units = np.array(unit_inv) * total_units_changes[year_idx] + res_dict[year - 1]
            
            # Handle excess over ultimate capacity
            diff = np.array(list(ultimate_units.values())) - units
            if any(diff < 0):
                negative_indices = np.where(diff < 0)[0]
                for idx in negative_indices:
                    excess_load = -diff[idx]
                    units[idx] = list(ultimate_units.values())[idx]
                    remaining_indices = [i for i in range(len(units)) if i not in negative_indices]
                    remaining_unit_inv = [unit_inv[i] for i in remaining_indices]
                    total_remaining_inv = sum(remaining_unit_inv)
                    if total_remaining_inv > 0:
                        for i in remaining_indices:
                            units[i] += (unit_inv[i] / total_remaining_inv) * excess_load
                    else:
                        units = list(ultimate_units.values())
                        break

            # Adjustment factor to match total growth
            total_units_added_actual = sum(units) - sum(res_dict[year - 1])
            if total_units_added_actual != 0:
                if total_units_added_actual != total_units_changes[year_idx]:
                    adjustment_factor = total_units_changes[year_idx] / total_units_added_actual
                    units = [u * adjustment_factor for u in units]
            else:
                adjustment_factor = 1
                units = [u * adjustment_factor for u in units]

            res_dict[year] = units
            supply = [l - u for u, l in zip(units, list(ultimate_units.values()))]
            factor_dict[year] = [u / s if s != 0 else 1 for u, s in zip(list(ultimate_units.values()), supply)]
            factor_dict[year] = [max(0, min(f, 5)) for f in factor_dict[year]]
            
            # Ensure no negative or infinite values
            if any(u < 0 or np.isinf(u) for u in res_dict[year]):
                res_dict[year] = list(ultimate_units.values())
            if sum(res_dict[year]) >= sum(list(ultimate_units.values())):
                res_dict[year] = list(ultimate_units.values())
                
        return res_dict

    def create_forecast_scenarios(self, current_loads: Dict[str, Dict[str, float]], 
                                ultimate_loads: Dict[str, Dict[str, float]], 
                                growth_projection: Dict[int, float],
                                forecast_years: List[int] = None) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Create forecast scenarios for residential and commercial/industrial loads.
        
        Args:
            current_loads: Current loads by area and class
            ultimate_loads: Ultimate loads by area and class  
            growth_projection: Annual growth projections
            forecast_years: Years to forecast (default: 5, 10, 15, 20 years from now)
            
        Returns:
            Dictionary with forecast results by class, year, and area
        """
        if forecast_years is None:
            current_year = datetime.now().year
            forecast_years = [current_year + offset for offset in [5, 10, 15, 20]]
        
        # Separate residential and non-residential loads
        area_dicts_res = {}
        area_dicts_non_res = {}
        
        for area in current_loads.keys():
            # Residential loads (including apartments)
            res_current = current_loads[area].get('residential', 0) + current_loads[area].get('apartments', 0)
            res_ultimate = ultimate_loads[area].get('residential', 0) + ultimate_loads[area].get('apartments', 0)
            
            if res_ultimate > 0:
                area_dicts_res[area] = {2025: res_current, 'ultimate': res_ultimate}
            
            # Non-residential loads
            nonres_current = current_loads[area].get('commercial', 0) + current_loads[area].get('industrial', 0)
            nonres_ultimate = ultimate_loads[area].get('commercial', 0) + ultimate_loads[area].get('industrial', 0)
            
            if nonres_ultimate > 0:
                area_dicts_non_res[area] = {2025: nonres_current, 'ultimate': nonres_ultimate}
        
        # Project loads
        max_forecast_year = max(forecast_years)
        
        results = {
            'residential': {},
            'non_residential': {}
        }
        
        if area_dicts_res:
            res_projection = self.project_units(
                area_dicts_res, growth_projection, 2025, max_forecast_year, 
                list(area_dicts_res.keys())
            )
            # Extract forecast years
            for year in forecast_years:
                if year in res_projection:
                    results['residential'][year] = {
                        area: load for area, load in zip(list(area_dicts_res.keys()), res_projection[year])
                    }
        
        if area_dicts_non_res:
            nonres_projection = self.project_units(
                area_dicts_non_res, growth_projection, 2025, max_forecast_year,
                list(area_dicts_non_res.keys())
            )
            # Extract forecast years
            for year in forecast_years:
                if year in nonres_projection:
                    results['non_residential'][year] = {
                        area: load for area, load in zip(list(area_dicts_non_res.keys()), nonres_projection[year])
                    }
        
        return results

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
            LOGGER.info("Running Load Forecast")
            
            if self.iface:
                # Show GUI dialog to get user inputs
                success, inputs = self.show_input_dialog()
                if not success:
                    return
                
                # Create forecasts with user inputs
                forecast_results = self.create_forecast_scenarios(
                    inputs['current_loads'], 
                    inputs['ultimate_loads'], 
                    inputs['growth_projection'],
                    inputs['forecast_years']
                )
                
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
                
                # Create forecasts
                forecast_results = self.create_forecast_scenarios(
                    current_loads, ultimate_loads, growth_projection
                )
                
                # Create output
                self.create_forecast_output(forecast_results)
                
        except Exception as e:
            LOGGER.error(f"Error running Load Forecast: {e}")
            if self.iface:
                from qgis.PyQt.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Error",
                    f"Error running forecast: {str(e)}"
                )

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
                                           QVBoxLayout, QLabel)
            
            widget = QWidget()
            layout = QVBoxLayout()
            
            form_layout = QFormLayout()
            
            # Forecast years selection
            years_label = QLabel("Select forecast years:")
            layout.addWidget(years_label)
            
            # Checkboxes for forecast years
            self.year_checkboxes = {}
            for offset in [5, 10, 15, 20]:
                year = 2025 + offset
                checkbox = QCheckBox(f"{year} ({offset} years)")
                checkbox.setChecked(True)
                self.year_checkboxes[year] = checkbox
                form_layout.addWidget(checkbox)
            
            layout.addLayout(form_layout)
            widget.setLayout(layout)
            
            return widget
            
        except Exception as e:
            LOGGER.error(f"Error creating parameters tab: {e}")
            return None

    def create_loads_input_tab(self, load_type):
        """Create a tab for current or ultimate loads input."""
        try:
            from qgis.PyQt.QtWidgets import (QWidget, QVBoxLayout, QTableWidget, 
                                           QTableWidgetItem, QPushButton, QHBoxLayout)
            
            widget = QWidget()
            layout = QVBoxLayout()
            
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
            
            # Buttons for adding/removing rows
            button_layout = QHBoxLayout()
            add_button = QPushButton("Add Area")
            remove_button = QPushButton("Remove Area")
            
            add_button.clicked.connect(lambda: self.add_area_row(table))
            remove_button.clicked.connect(lambda: self.remove_area_row(table))
            
            button_layout.addWidget(add_button)
            button_layout.addWidget(remove_button)
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
                                           QHBoxLayout, QDoubleSpinBox)
            
            widget = QWidget()
            layout = QVBoxLayout()
            
            # Instructions
            instructions = QLabel("Enter annual growth values (GJ/d per year):")
            layout.addWidget(instructions)
            
            # Simple growth parameters
            params_layout = QHBoxLayout()
            
            # Base growth
            base_label = QLabel("Base Annual Growth:")
            self.base_growth_spin = QDoubleSpinBox()
            self.base_growth_spin.setRange(0, 1000)
            self.base_growth_spin.setValue(100)
            self.base_growth_spin.setSuffix(" GJ/d")
            
            # Growth increment
            increment_label = QLabel("Annual Increment:")
            self.growth_increment_spin = QDoubleSpinBox()
            self.growth_increment_spin.setRange(0, 100)
            self.growth_increment_spin.setValue(2.0)
            self.growth_increment_spin.setSuffix(" GJ/d/year")
            
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
                'growth_projection': growth_projection
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
            
            # Close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)
            
            dialog.setLayout(layout)
            dialog.exec_()
            
        except Exception as e:
            LOGGER.error(f"Error showing forecast results dialog: {e}")