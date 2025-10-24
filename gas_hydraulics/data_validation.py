"""Data Validation Module for Gas Hydraulics Plugin.

This module provides comprehensive data validation and preview
functionality that can be used across all plugin components.
"""

import logging
from collections import defaultdict

LOGGER = logging.getLogger(__name__)


def safe_int(value, default=None):
    """Safely convert value to integer."""
    if value is None:
        return default
    try:
        if isinstance(value, str) and value.strip() == "":
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default


def safe_float(value, default=0.0):
    """Safely convert value to float."""
    if value is None:
        return default
    try:
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_str(value, default=""):
    """Safely convert value to string."""
    if value is None:
        return default
    return str(value)


class DataValidator:
    """Validates input data and provides detailed feedback."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
    
    def add_error(self, message):
        """Add a critical error."""
        self.errors.append(message)
        LOGGER.error(f" VALIDATION ERROR: {message}")
    
    def add_warning(self, message):
        """Add a warning."""
        self.warnings.append(message)
        LOGGER.warning(f"️  VALIDATION WARNING: {message}")
    
    def add_info(self, message):
        """Add informational message."""
        self.info.append(message)
        LOGGER.info(f"ℹ️  VALIDATION INFO: {message}")
    
    def has_errors(self):
        """Check if there are any critical errors."""
        return len(self.errors) > 0
    
    def has_warnings(self):
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    def is_valid(self):
        """Check if validation passed (no errors)."""
        return not self.has_errors()
    
    def get_summary(self):
        """Get validation summary."""
        summary = []
        
        if self.errors:
            summary.append(f" {len(self.errors)} Error(s)")
        if self.warnings:
            summary.append(f"️  {len(self.warnings)} Warning(s)")
        if not self.errors and not self.warnings:
            summary.append(" All checks passed")
        
        return ", ".join(summary)
    
    def get_detailed_report(self):
        """Get detailed validation report."""
        lines = []
        lines.append("="*80)
        lines.append("DATA VALIDATION REPORT")
        lines.append("="*80)
        lines.append("")
        
        if self.errors:
            lines.append(" ERRORS (Must be fixed):")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"  {i}. {error}")
            lines.append("")
        
        if self.warnings:
            lines.append("️  WARNINGS (Review recommended):")
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warning}")
            lines.append("")
        
        if self.info:
            lines.append("ℹ️  INFORMATION:")
            for i, info in enumerate(self.info, 1):
                lines.append(f"  {i}. {info}")
            lines.append("")
        
        lines.append("="*80)
        lines.append(f"RESULT: {self.get_summary()}")
        lines.append("="*80)
        
        return "\n".join(lines)


def validate_forecast_inputs(inputs):
    """Validate forecast input parameters.
    
    Args:
        inputs: Dictionary of input parameters
        
    Returns:
        DataValidator: Validator with validation results
    """
    validator = DataValidator()
    
    LOGGER.info("Validating forecast inputs...")
    
    # Check forecast years
    forecast_years = inputs.get('forecast_years', [])
    if not forecast_years:
        validator.add_error("No forecast years specified")
    else:
        validator.add_info(f"Forecast years: {len(forecast_years)} years ({min(forecast_years)} - {max(forecast_years)})")
        
        # Check for reasonable year range
        year_range = max(forecast_years) - min(forecast_years)
        if year_range > 50:
            validator.add_warning(f"Long forecast period: {year_range} years - results may be less accurate")
        elif year_range < 1:
            validator.add_error("Forecast period must be at least 1 year")
    
    # Check current loads
    current_loads = inputs.get('current_loads', {})
    if not current_loads:
        validator.add_error("No current loads specified")
    else:
        validator.add_info(f"Current loads: {len(current_loads)} zones")
        
        # Check for zero or negative loads
        zero_count = sum(1 for load in current_loads.values() if safe_float(load) <= 0)
        if zero_count > 0:
            validator.add_warning(f"{zero_count} zones have zero or negative current loads")
        
        # Check for reasonable load values
        for zone, load in current_loads.items():
            load_val = safe_float(load)
            if load_val > 100000:  # Arbitrary large value
                validator.add_warning(f"Zone '{zone}' has unusually high load: {load_val}")
    
    # Check ultimate loads
    ultimate_loads = inputs.get('ultimate_loads', {})
    if not ultimate_loads:
        validator.add_error("No ultimate loads specified")
    else:
        validator.add_info(f"Ultimate loads: {len(ultimate_loads)} zones")
        
        # Check ultimate > current
        for zone in current_loads:
            if zone in ultimate_loads:
                current = safe_float(current_loads[zone])
                ultimate = safe_float(ultimate_loads[zone])
                if ultimate < current:
                    validator.add_warning(f"Zone '{zone}': Ultimate load ({ultimate}) < Current load ({current})")
            else:
                validator.add_warning(f"Zone '{zone}' in current loads but not in ultimate loads")
    
    # Check growth rates (only relevant for enhanced mode with regression)
    # In basic mode, priority multipliers are used instead of growth rates
    growth_rates = inputs.get('growth_rates', {})
    
    # Only check growth rates if we have indicators this is enhanced/regression mode
    # Enhanced mode would have population_layer, population_file, or aggregate_polygons
    is_enhanced_mode = any([
        inputs.get('population_layer'),
        inputs.get('population_file'),
        inputs.get('aggregate_polygons') is not None
    ])
    
    if is_enhanced_mode:
        # Enhanced mode: growth rates are relevant
        if not growth_rates:
            validator.add_info("No custom growth rates - will use regression-based forecasting")
        else:
            validator.add_info(f"Custom growth rates: {len(growth_rates)} zones")
            
            for zone, rate in growth_rates.items():
                rate_val = safe_float(rate)
                if rate_val < 0:
                    validator.add_warning(f"Zone '{zone}' has negative growth rate: {rate_val}%")
                elif rate_val > 20:
                    validator.add_warning(f"Zone '{zone}' has very high growth rate: {rate_val}%")
    else:
        # Basic mode: growth rates not used (priority multipliers used instead)
        if growth_rates:
            validator.add_info(f"Note: Custom growth rates ignored in basic mode (uses priority multipliers)")
        # No warning if no growth rates in basic mode - this is expected
    
    # Check zone name consistency
    current_zones = set(current_loads.keys())
    ultimate_zones = set(ultimate_loads.keys())
    growth_zones = set(growth_rates.keys())
    
    if current_zones != ultimate_zones:
        only_current = current_zones - ultimate_zones
        only_ultimate = ultimate_zones - current_zones
        
        if only_current:
            validator.add_warning(f"Zones in current but not ultimate: {', '.join(sorted(only_current))}")
        if only_ultimate:
            validator.add_warning(f"Zones in ultimate but not current: {', '.join(sorted(only_ultimate))}")
    
    return validator


def validate_layer_data(layer, layer_name, required_fields=None):
    """Validate QGIS layer data.
    
    Args:
        layer: QGIS vector layer
        layer_name: Name for reporting
        required_fields: List of required field names (case-insensitive)
        
    Returns:
        DataValidator: Validator with validation results
    """
    validator = DataValidator()
    
    LOGGER.info(f"Validating layer: {layer_name}")
    
    if not layer:
        validator.add_error(f"{layer_name}: Layer is None")
        return validator
    
    if not layer.isValid():
        validator.add_error(f"{layer_name}: Layer is not valid")
        return validator
    
    # Check feature count
    feature_count = layer.featureCount()
    if feature_count == 0:
        validator.add_error(f"{layer_name}: Layer has no features")
    else:
        validator.add_info(f"{layer_name}: {feature_count} features")
    
    # Check geometry type
    geom_type = layer.geometryType()
    geom_type_names = {0: "Point", 1: "Line", 2: "Polygon"}
    validator.add_info(f"{layer_name}: Geometry type = {geom_type_names.get(geom_type, 'Unknown')}")
    
    # Check fields
    field_names = [f.name() for f in layer.fields()]
    validator.add_info(f"{layer_name}: {len(field_names)} fields - {', '.join(field_names)}")
    
    # Check required fields
    if required_fields:
        field_names_upper = [f.upper() for f in field_names]
        for req_field in required_fields:
            if req_field.upper() not in field_names_upper:
                validator.add_error(f"{layer_name}: Missing required field '{req_field}' (case-insensitive)")
            else:
                validator.add_info(f"{layer_name}: Found required field '{req_field}'")
    
    # Check for null geometries
    null_geom_count = sum(1 for feat in layer.getFeatures() if feat.geometry().isNull())
    if null_geom_count > 0:
        validator.add_warning(f"{layer_name}: {null_geom_count} features have null geometry")
    
    # Check for invalid geometries
    invalid_geom_count = sum(1 for feat in layer.getFeatures() if not feat.geometry().isGeosValid())
    if invalid_geom_count > 0:
        validator.add_warning(f"{layer_name}: {invalid_geom_count} features have invalid geometry")
    
    return validator


def show_validation_dialog(validator, title="Data Validation", parent=None):
    """Show validation results in a dialog.
    
    Args:
        validator: DataValidator with validation results
        title: Dialog title
        parent: Parent widget
        
    Returns:
        bool: True if user accepts (or no errors), False if cancelled
    """
    try:
        from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QTextEdit, 
                                         QPushButton, QHBoxLayout, QLabel)
        from qgis.PyQt.QtGui import QFont
        from qgis.PyQt.QtCore import Qt
        
        dialog = QDialog(parent)
        dialog.setWindowTitle(title)
        dialog.setMinimumSize(700, 500)
        
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel(f" {title}")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Summary
        summary_label = QLabel(validator.get_summary())
        summary_label.setFont(QFont("Arial", 12))
        
        if validator.has_errors():
            summary_label.setStyleSheet("color: red; font-weight: bold;")
        elif validator.has_warnings():
            summary_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            summary_label.setStyleSheet("color: green; font-weight: bold;")
        
        layout.addWidget(summary_label)
        
        # Detailed report
        report_text = QTextEdit()
        report_text.setReadOnly(True)
        report_text.setFont(QFont("Courier New", 10))
        report_text.setText(validator.get_detailed_report())
        layout.addWidget(report_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        if validator.has_errors():
            info_label = QLabel(" Errors must be fixed before proceeding")
            info_label.setStyleSheet("color: red;")
            button_layout.addWidget(info_label)
            button_layout.addStretch()
            
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.reject)
            button_layout.addWidget(close_button)
        else:
            if validator.has_warnings():
                info_label = QLabel("️  Warnings detected - Review recommended")
                info_label.setStyleSheet("color: orange;")
                button_layout.addWidget(info_label)
            else:
                info_label = QLabel(" All checks passed")
                info_label.setStyleSheet("color: green;")
                button_layout.addWidget(info_label)
            
            button_layout.addStretch()
            
            cancel_button = QPushButton("Cancel")
            proceed_button = QPushButton(" Proceed")
            
            cancel_button.clicked.connect(dialog.reject)
            proceed_button.clicked.connect(dialog.accept)
            
            button_layout.addWidget(cancel_button)
            button_layout.addWidget(proceed_button)
        
        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        
        # Log to console
        LOGGER.info(validator.get_detailed_report())
        
        result = dialog.exec_()
        
        if validator.has_errors():
            return False
        
        return result == QDialog.Accepted
        
    except Exception as e:
        LOGGER.error(f"Error showing validation dialog: {e}")
        return not validator.has_errors()


def validate_csv_forecast_data(csv_path):
    """Validate forecast CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataValidator: Validator with validation results
    """
    validator = DataValidator()
    
    LOGGER.info(f"Validating CSV: {csv_path}")
    
    import csv
    import os
    
    # Check file exists
    if not os.path.exists(csv_path):
        validator.add_error(f"File does not exist: {csv_path}")
        return validator
    
    validator.add_info(f"File found: {csv_path}")
    
    # Check file size
    file_size = os.path.getsize(csv_path)
    if file_size == 0:
        validator.add_error("File is empty")
        return validator
    
    validator.add_info(f"File size: {file_size:,} bytes")
    
    # Parse CSV
    try:
        areas_found = set()
        years_found = set()
        row_count = 0
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            for row in reader:
                if not row or not row[0]:
                    continue
                
                row_count += 1
                
                # Check for area headers
                if 'SUBZONE' in row[0] or 'AREA' in row[0]:
                    parts = row[0].split(':')
                    if len(parts) > 1:
                        areas_found.add(parts[1].strip())
                
                # Check for year data
                year_val = safe_int(row[0])
                if year_val:
                    years_found.add(year_val)
        
        validator.add_info(f"Total rows: {row_count}")
        
        if not areas_found:
            validator.add_error("No area/subzone sections found in CSV")
        else:
            validator.add_info(f"Areas found: {len(areas_found)} - {', '.join(sorted(areas_found))}")
        
        if not years_found:
            validator.add_error("No year data found in CSV")
        else:
            validator.add_info(f"Years found: {len(years_found)} years ({min(years_found)} - {max(years_found)})")
        
    except Exception as e:
        validator.add_error(f"Error parsing CSV: {str(e)}")
    
    return validator
