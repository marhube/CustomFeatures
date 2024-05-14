#Importerer først noen "built-in"- moduler
import os
import sys
import unittest
#**********
#**********
import pandas as pd
import numpy as np
#********

#********* start import testrelaterte moduler
#********* Start import moduler for testing


#**************** Start import selve koden som skal testes
from CustomFeatures.CumulativeMeans import CumulativeMeanCalculator as CMC
#********* Slutt import moduler for testing

# Sample data for summation
data_summation = {
    'time': [1, 1, 2, 2, 3, 3, 4, 4],
    'grouping_var_1': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'grouping_var_2': ['X', 'X', 'Y', 'Y', 'X', 'Y', 'X', 'Y'],
    'my_numeric_col': [10, 20, 30, 40, 50, 60, 70, 80]
}
my_summation_table = pd.DataFrame(data_summation)

# Sample data for prediction
data_prediction = {
    'time': [2, 3, 3, 4, 5],
    'grouping_var_1': ['A', 'A', 'B', 'C', 'D'],
    'grouping_var_2': ['X', 'Z', 'Y', 'Z', 'W']  # Note 'Z' and 'W' do not exist in summation
}
my_prediction_data = pd.DataFrame(data_prediction)



class TestCumulativeCounts(unittest.TestCase):
    def test_init_CumulativeMeanCalculator(self):
        print('Er nå inne i TestCumulativeCounts.test_init_CumulativeMeanCalculator')
        calculator = CMC(['grouping_var_1', 'grouping_var_2'], 'my_numeric_col', 'time')
        self.assertIsInstance(calculator,CMC)
    #   

