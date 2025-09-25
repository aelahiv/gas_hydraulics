"""Tests for forecast plugin."""
import unittest
import numpy as np
from pathlib import Path
import sys
import os

# Add the gas_hydraulics module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gas_hydraulics'))

from forecast_plugin import ForecastPlugin


class TestForecastPlugin(unittest.TestCase):
    """Test cases for ForecastPlugin."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.plugin = ForecastPlugin()
        
        # Sample area data for forecasting
        self.sample_areas = ['Area1', 'Area2', 'Area3']
        self.sample_area_dicts = {
            'Area1': {2025: 100.0, 'ultimate': 200.0},
            'Area2': {2025: 150.0, 'ultimate': 300.0},
            'Area3': {2025: 80.0, 'ultimate': 180.0}
        }
        self.sample_growth = {2025: 10, 2026: 12, 2027: 8, 2028: 15, 2029: 5, 2030: 7}
    
    def test_project_units_basic(self):
        """Test basic unit projection functionality."""
        result = self.plugin.project_units(
            self.sample_area_dicts,
            self.sample_growth,
            2025,
            2027,
            self.sample_areas
        )
        
        # Check that we get results for each year
        self.assertIn(2025, result)
        self.assertIn(2026, result)
        self.assertIn(2027, result)
        
        # Check initial year matches input
        expected_initial = [100.0, 150.0, 80.0]
        self.assertEqual(result[2025], expected_initial)
        
        # Verify result structure
        for year, values in result.items():
            self.assertEqual(len(values), len(self.sample_areas))
            self.assertTrue(all(isinstance(v, (int, float)) for v in values))
    
    def test_project_units_ultimate_constraints(self):
        """Test that projections don't exceed ultimate loads."""
        # Create scenario where current loads are near ultimate
        high_area_dicts = {
            'Area1': {2025: 190.0, 'ultimate': 200.0},  # 95% of ultimate
            'Area2': {2025: 280.0, 'ultimate': 300.0},  # 93% of ultimate
            'Area3': {2025: 170.0, 'ultimate': 180.0}   # 94% of ultimate
        }
        
        result = self.plugin.project_units(
            high_area_dicts,
            self.sample_growth,
            2025,
            2030,
            self.sample_areas
        )
        
        # Check that no area exceeds its ultimate load
        ultimate_values = [200.0, 300.0, 180.0]
        for year, values in result.items():
            for i, value in enumerate(values):
                self.assertLessEqual(value, ultimate_values[i] + 0.01)  # Small tolerance for floating point
    
    def test_create_forecast_scenarios_structure(self):
        """Test forecast scenario creation structure."""
        current_loads = {
            'Zone1': {'residential': 100.0, 'commercial': 50.0, 'industrial': 25.0},
            'Zone2': {'residential': 150.0, 'commercial': 75.0, 'industrial': 40.0}
        }
        
        ultimate_loads = {
            'Zone1': {'residential': 200.0, 'commercial': 100.0, 'industrial': 50.0},
            'Zone2': {'residential': 300.0, 'commercial': 150.0, 'industrial': 80.0}
        }
        
        growth_projections = {
            2025: {'residential': 5, 'commercial': 3, 'industrial': 2},
            2026: {'residential': 4, 'commercial': 3, 'industrial': 2},
            2027: {'residential': 6, 'commercial': 4, 'industrial': 3}
        }
        
        result = self.plugin.create_forecast_scenarios(
            current_loads,
            ultimate_loads,
            growth_projections,
            2025,
            2030
        )
        
        # Check result structure
        self.assertIn('residential', result)
        self.assertIn('commercial', result)
        self.assertIn('industrial', result)
        
        # Check that each class has year-based projections
        for class_name, class_data in result.items():
            self.assertIsInstance(class_data, dict)
            for year, year_data in class_data.items():
                self.assertIsInstance(year_data, dict)
                # Should have data for each zone
                for zone_name in current_loads.keys():
                    self.assertIn(zone_name, year_data)
    
    def test_priority_areas_logic(self):
        """Test that priority areas receive enhanced growth treatment."""
        priority_areas = ['Kensington', 'West Industrial', 'Meadow View']
        non_priority = ['Regular Area']
        
        all_areas = priority_areas + non_priority
        area_dicts = {area: {2025: 50.0, 'ultimate': 150.0} for area in all_areas}
        growth = {year: 10 for year in range(2025, 2031)}
        
        result = self.plugin.project_units(
            area_dicts,
            growth,
            2025,
            2030,  # 5-year period where priority areas get enhanced growth
            all_areas
        )
        
        # This is a structural test - ensuring the method handles priority areas without error
        # The actual business logic for priority area enhancement would require more detailed verification
        self.assertIsInstance(result, dict)
        self.assertTrue(len(result) > 1)  # Multiple years of results


if __name__ == '__main__':
    unittest.main()