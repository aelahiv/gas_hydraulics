"""gas_hydraulics QGIS plugin package

This package contains four specialized plugins for gas hydraulics analysis:
1. Demand File Analysis - Current year load analysis by area and class
2. Historical Analysis - Historical load trends over 5-year periods  
3. Load Forecast - 5/10/15/20 year load projections
4. Pipe to Node Converter - Convert pipe-loaded models to node-loaded format

The package is designed so it can be imported outside QGIS for testing.
"""

def classFactory(iface):
    """QGIS calls this to instantiate the main plugin."""
    # Import here to avoid QGIS dependency at module import time
    from .gas_hydraulics_plugin import GasHydraulicsPlugin
    return GasHydraulicsPlugin(iface)
