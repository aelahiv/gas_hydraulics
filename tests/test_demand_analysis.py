"""Tests for demand file analysis plugin."""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the gas_hydraulics module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gas_hydraulics'))

from demand_file_analysis import DemandFileAnalysisPlugin


class TestDemandFileAnalysis(unittest.TestCase):
    """Test cases for DemandFileAnalysisPlugin."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.plugin = DemandFileAnalysisPlugin()
        
        # Create sample service point data
        self.sample_data = pd.DataFrame({
            'Factor Code': ['0', '1', '20', '70', '15'],
            'Base Factor': [0.3987306, 0.2, 0.1, 0.5, 0.3],
            'Heat Factor': [0.0893029, 0.05, 0.02, 0.1, 0.06],
            'HUC 3-Year Peak Demand': [1.5, 2.0, 0.5, 1.0, 1.2],
            'Install Date': ['2020-01-15', '2019-06-20', '2021-03-10', '2018-12-05', '2022-02-28'],
            'Use Class': ['RES-001', 'COM-002', 'IND-003', 'RES-004', 'COM-005'],
            'Distribution Pipe': ['GA11137322', 'GA11137323', 'GA11137324', 'GA11137325', 'GA11137326']
        })
    
    def test_calculate_load_default_factor_codes(self):
        """Test load calculation with default factor codes to keep."""
        result = self.plugin.calculate_load(self.sample_data)
        
        # Factor codes '0', '1', '20' should keep their factors
        # Factor codes '70', '15' should have factors zeroed
        
        # Check that factor codes 0, 1, 20 have non-zero base/heat factors
        mask_keep = result['Factor Code'].astype(str).isin(['0', '1', '20'])
        self.assertTrue(all(result.loc[mask_keep, 'Base Factor'] > 0))
        
        # Check that other factor codes have zeroed factors
        mask_zero = ~result['Factor Code'].astype(str).isin(['0', '1', '20'])
        self.assertTrue(all(result.loc[mask_zero, 'Base Factor'] == 0))
        self.assertTrue(all(result.loc[mask_zero, 'Heat Factor'] == 0))
        
        # Verify load calculation
        expected_load_0 = (1.07 * (0.3987306 + 56.8 * 0.0893029) + 1.5)
        self.assertAlmostEqual(result.iloc[0]['Load'], expected_load_0, places=5)
        
        # Factor code '70' should only have HUC demand
        expected_load_70 = 1.0  # Only HUC 3-Year Peak Demand
        self.assertAlmostEqual(result.iloc[3]['Load'], expected_load_70, places=5)
    
    def test_calculate_load_custom_factor_codes(self):
        """Test load calculation with custom factor codes to keep."""
        custom_codes = ['0', '70']
        result = self.plugin.calculate_load(self.sample_data, custom_codes)
        
        # Only codes '0' and '70' should keep their factors
        mask_keep = result['Factor Code'].astype(str).isin(custom_codes)
        self.assertTrue(all(result.loc[mask_keep, 'Base Factor'] > 0))
        
        # Other codes should be zeroed
        mask_zero = ~result['Factor Code'].astype(str).isin(custom_codes)
        self.assertTrue(all(result.loc[mask_zero, 'Base Factor'] == 0))
    
    def test_filter_by_use_class(self):
        """Test filtering by use class."""
        result = self.plugin.filter_by_use_class(self.sample_data)
        
        self.assertIn('residential', result)
        self.assertIn('commercial', result)
        self.assertIn('industrial', result)
        
        # Check residential filter
        res_df = result['residential']
        self.assertTrue(all(res_df['Use Class'].str.contains('RES', na=False)))
        
        # Check commercial filter
        com_df = result['commercial']
        self.assertTrue(all(com_df['Use Class'].str.contains('COM', na=False)))
        
        # Check industrial filter
        ind_df = result['industrial']
        self.assertTrue(all(ind_df['Use Class'].str.contains('IND', na=False)))
    
    def test_load_multiplier_configuration(self):
        """Test that load multiplier can be configured."""
        self.plugin.load_multiplier = 1.15
        self.plugin.heat_factor_multiplier = 60.0
        
        result = self.plugin.calculate_load(self.sample_data)
        
        # Test first row with factor code '0'
        expected_load = (1.15 * (0.3987306 + 60.0 * 0.0893029) + 1.5)
        self.assertAlmostEqual(result.iloc[0]['Load'], expected_load, places=5)
    
    def test_huc_demand_nan_handling(self):
        """Test handling of NaN values in HUC 3-Year Peak Demand."""
        test_data = self.sample_data.copy()
        test_data.loc[0, 'HUC 3-Year Peak Demand'] = np.nan
        
        result = self.plugin.calculate_load(test_data)
        
        # NaN should be filled with 0
        self.assertEqual(result.iloc[0]['HUC 3-Year Peak Demand'], 0)
        
        # Load calculation should still work
        expected_load = (1.07 * (0.3987306 + 56.8 * 0.0893029) + 0)
        self.assertAlmostEqual(result.iloc[0]['Load'], expected_load, places=5)


if __name__ == '__main__':
    unittest.main()