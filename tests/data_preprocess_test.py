import os
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data_preprocess import read_dataframe

def test_read_dataframe():
    # Sample data to return from the mocked pd.read_csv
    sample_data = {
        "column1": [1, 2, 3, 4],
        "column2": ["A", "B", "C", "D"]
    }
    df = pd.DataFrame(sample_data)
    
    # Mocked dataframe that .sample() would return
    sampled_df = df.sample(frac=0.50, random_state=1)

    with patch("pandas.read_csv", return_value=df) as mock_read_csv:
        with patch.object(pd.DataFrame, 'sample', return_value=sampled_df) as mock_sample:
            # Call the function with a fake path
            result = read_dataframe("fakepath")

            # Check that pd.read_csv was called correctly
            mock_read_csv.assert_called_once_with(os.path.join("fakepath", 'historical_data.csv'))

            # Check that .sample was called correctly
            mock_sample.assert_called_once_with(frac=0.50, random_state=1)

            # Assert that the result matches what we expect
            pd.testing.assert_frame_equal(result, sampled_df)