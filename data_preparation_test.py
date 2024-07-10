import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

import data_preparation


class DataPreparationTestCase(unittest.TestCase):

    def test_append_last_position_take_last_5(self):

        train_df = pd.DataFrame(pd.DataFrame({'position': [1, 2], 'datetime': [3, 4]}))
        result = data_preparation.append_last_positions(train_df=train_df)
        expected_result = pd.DataFrame()

        assert_frame_equal(result, expected_result)

    def test_append_last_position_take_last_3(self):

        train_df = pd.DataFrame(pd.DataFrame({'position': [1, 2], 'datetime': [3, 4]}))
        result = data_preparation.append_last_positions(train_df=train_df, take_last=3)
        expected_result = pd.DataFrame()

        assert_frame_equal(result, expected_result)

    def test_append_last_circuit_position(self):

        train_df = pd.DataFrame(pd.DataFrame({'position': [1, 2], 'datetime': [3, 4]}))
        result = data_preparation.append_last_circuit_position(train_df=train_df)
        expected_result = pd.DataFrame()

        assert_frame_equal(result, expected_result)


if __name__ == '__main__':
    unittest.main()
