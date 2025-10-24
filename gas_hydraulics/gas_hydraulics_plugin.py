"""Main plugin module for GasHydraulics plugin.

This module provides a unified interface to four specialized gas hydraulics analysis tools:
1. Demand File Analysis - Current year load analysis
2. Historical Analysis - Historical load trends
3. Load Forecast - Future load projections
4. Pipe to Node Converter - Convert pipe-loaded models to node-loaded format

The plugin avoids importing QGIS at top-level so it can be imported in a normal
Python environment (for testing).
"""
from __future__ import annotations
import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


class GasHydraulicsPlugin:
    def __init__(self, iface: Any = None):
        """Initialize plugin.

        iface: the QGIS interface object. When None, plugin still works for tests.
        """
        self.iface = iface
        self.actions = []
        
        # Initialize sub-plugins
        self.demand_analysis = None
        self.historical_analysis = None
        self.forecast_plugin = None
        self.pipe_to_node_converter = None
        
        # Check optional dependencies
        self.has_geopandas = self._check_geopandas()
        
    def _check_geopandas(self) -> bool:
        """Check if GeoPandas is available.
        
        Returns:
            bool: True if GeoPandas is installed, False otherwise
        """
        try:
            import geopandas
            import pandas
            LOGGER.info("✓ GeoPandas detected - Synergi export will work")
            return True
        except ImportError:
            LOGGER.warning("⚠ GeoPandas not detected - Synergi export will require manual fix")
            LOGGER.warning("  Install with: pip install geopandas (from OSGeo4W Shell)")
            return False

    def initGui(self):
        """Create GUI elements (only when running inside QGIS)."""
        try:
            # Lazy import of QGIS classes and sub-plugins
            from qgis.PyQt.QtWidgets import QAction, QMenu
            from .demand_file_analysis import DemandFileAnalysisPlugin
            from .historical_analysis import HistoricalAnalysisPlugin
            from .forecast_plugin import ForecastPlugin
            from .pipe_to_node_converter import PipeToNodeConverter

            # Initialize sub-plugins
            self.demand_analysis = DemandFileAnalysisPlugin(self.iface)
            self.historical_analysis = HistoricalAnalysisPlugin(self.iface)
            self.forecast_plugin = ForecastPlugin(self.iface)
            self.pipe_to_node_converter = PipeToNodeConverter(self.iface)

            # Create main menu
            if self.iface:
                # Create actions for each sub-plugin
                demand_action = QAction("Demand File Analysis", self.iface.mainWindow())
                demand_action.triggered.connect(self.demand_analysis.run)
                self.actions.append(demand_action)

                historical_action = QAction("Historical Analysis", self.iface.mainWindow())
                historical_action.triggered.connect(self.historical_analysis.run)
                self.actions.append(historical_action)

                forecast_action = QAction("Load Forecast", self.iface.mainWindow())
                forecast_action.triggered.connect(self.forecast_plugin.run)
                self.actions.append(forecast_action)

                converter_action = QAction("Pipe to Node Converter", self.iface.mainWindow())
                converter_action.triggered.connect(self.pipe_to_node_converter.run)
                self.actions.append(converter_action)

                # Add main selection action
                main_action = QAction("Gas Hydraulics Analysis", self.iface.mainWindow())
                main_action.triggered.connect(self.run)
                self.actions.append(main_action)

                # Add all actions to the Plugins menu
                for action in self.actions:
                    self.iface.addPluginToMenu("&Gas Hydraulics", action)
                    
                # Also add to toolbar for easy access
                self.iface.addToolBarIcon(main_action)
                
                # Show warning if GeoPandas is not available (only once at startup)
                if not self.has_geopandas:
                    from qgis.PyQt.QtWidgets import QMessageBox
                    self.iface.messageBar().pushWarning(
                        "Gas Hydraulics Plugin",
                        "GeoPandas not detected. Synergi export in Load Assignment will require manual fix. "
                        "See plugin log for installation instructions."
                    )

        except Exception as e:
            LOGGER.debug(f"QGIS not available; skipping GUI setup: {e}")

    def unload(self):
        """Cleanup GUI items. Guarded for non-QGIS execution."""
        try:
            if self.actions and self.iface:
                # Remove actions from menu/toolbar
                for action in self.actions:
                    self.iface.removePluginMenu("&Gas Hydraulics", action)
                    self.iface.removeToolBarIcon(action)
                self.actions.clear()
        except Exception as e:
            LOGGER.debug(f"QGIS not available; skipping unload: {e}")

    def run(self):
        """Run main plugin action - show selection dialog."""
        try:
            if self.iface:
                # Show a dialog to select which analysis to run
                from qgis.PyQt.QtWidgets import QMessageBox, QInputDialog
                
                items = [
                    "Demand File Analysis - Current year load breakdown",
                    "Historical Analysis - Historical load trends", 
                    "Load Forecast - Future load projections",
                    "Pipe to Node Converter - Convert pipe-loaded models to node-loaded"
                ]
                
                item, ok = QInputDialog.getItem(
                    self.iface.mainWindow(),
                    "Gas Hydraulics Analysis",
                    "Select analysis type:",
                    items, 0, False
                )
                
                if ok and item:
                    if "Demand File" in item:
                        self.demand_analysis.run()
                    elif "Historical" in item:
                        self.historical_analysis.run()
                    elif "Forecast" in item:
                        self.forecast_plugin.run()
                    elif "Pipe to Node" in item:
                        self.pipe_to_node_converter.run()
            else:
                # Non-QGIS execution - run all analyses
                print("Gas Hydraulics Plugin - Running all analyses:")
                if self.demand_analysis:
                    self.demand_analysis.run()
                if self.historical_analysis:
                    self.historical_analysis.run()
                if self.forecast_plugin:
                    self.forecast_plugin.run()
                
        except Exception as e:
            LOGGER.error(f"Error running Gas Hydraulics plugin: {e}")
