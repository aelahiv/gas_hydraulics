"""Tests for historical analysis plugin."""
import unittest
import pandas as pd
from pathlib import Path
import sys
import os

# Add the gas_hydraulics module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gas_hydraulics'))

from historical_analysis import HistoricalAnalysisPlugin


class TestHistoricalAnalysis(unittest.TestCase):
    """Test cases for HistoricalAnalysisPlugin."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.plugin = HistoricalAnalysisPlugin()
        
        # Create sample historical data
        self.sample_data = pd.DataFrame({
            'Factor Code': ['0', '1', '20', '70', '15'],
            'Base Factor': [0.4, 0.2, 0.1, 0.5, 0.3],
            'Heat Factor': [0.09, 0.05, 0.02, 0.1, 0.06],
            'HUC 3-Year Peak Demand': [1.5, 2.0, 0.5, 1.0, 1.2],
            'Install Date': ['2018-01-15', '2019-06-20', '2021-03-10', '2022-12-05', '2023-02-28'],
            'Use Class': ['RES-001', 'COM-002', 'IND-003', 'RES-004', 'COM-005'],
            'Distribution Pipe': ['PIPE001', 'PIPE002', 'PIPE003', 'PIPE004', 'PIPE005']
        })
    
    def test_calculate_load_factor_codes(self):
        """Test load calculation with factor code logic."""
        result = self.plugin.calculate_load(self.sample_data)
        
        # Factor codes '0', '1', '20' should keep their factors
        mask_keep = result['Factor Code'].astype(str).isin(['0', '1', '20'])
        self.assertTrue(all(result.loc[mask_keep, 'Base Factor'] > 0))
        
        # Other factor codes should have factors zeroed
        mask_zero = ~result['Factor Code'].astype(str).isin(['0', '1', '20'])
        self.assertTrue(all(result.loc[mask_zero, 'Base Factor'] == 0))
        self.assertTrue(all(result.loc[mask_zero, 'Heat Factor'] == 0))
    
    def test_get_5year_periods(self):
        """Test 5-year period generation."""
        periods = self.plugin.get_5year_periods(2000, 2025)
        
        expected_periods = [2000, 2005, 2010, 2015, 2020, 2025]
        self.assertEqual(periods, expected_periods)
        
        # Test edge case
        periods_short = self.plugin.get_5year_periods(2020, 2023)
        expected_short = [2020, 2023]
        self.assertEqual(periods_short, expected_short)
    
    def test_filter_by_use_class(self):
        """Test use class filtering functionality."""
        result = self.plugin.filter_by_use_class(self.sample_data)
        
        self.assertIn('residential', result)
        self.assertIn('commercial', result)
        self.assertIn('industrial', result)
        
        # Verify filtering logic
        self.assertEqual(len(result['residential']), 2)  # RES-001, RES-004
        self.assertEqual(len(result['commercial']), 2)   # COM-002, COM-005
        self.assertEqual(len(result['industrial']), 1)   # IND-003
    
    def test_load_multiplier_settings(self):
        """Test load multiplier configuration."""
        self.plugin.load_multiplier = 1.10
        self.plugin.heat_factor_multiplier = 55.0
        
        # Verify multipliers are applied
        self.assertEqual(self.plugin.load_multiplier, 1.10)
        self.assertEqual(self.plugin.heat_factor_multiplier, 55.0)


if __name__ == '__main__':
    unittest.main()