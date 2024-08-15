import pytest
from unittest.mock import patch, mock_open
import pickle

# Assuming these functions are in a file named 'utils.py'
from src.utils import load_pickle, dump_pickle

def test_load_pickle():
    # Simulates the content in the pickle file
    expected_data = {"key": "value"}
    
    # Temporarily replaces built in open function with a mock version that simulates opening a file and reading binary data
    with patch("builtins.open", mock_open(read_data=b"data")) as mock_file:
        # Patches pickle load with a mock object that will return expected data when called
        with patch("pickle.load", return_value=expected_data) as mock_pickle_load:
            # Mock object returns expected data
            result = load_pickle("fakefile.pkl")
            
            # Checks that the mock 'open' function was called exactly once with the specified args
            mock_file.assert_called_once_with("fakefile.pkl", "rb")

            # Checks that the mock pickle.load function was called exactly once during the rest
            mock_pickle_load.assert_called_once()

            assert result == expected_data

def test_dump_pickle():
    # The data that we expect to be saved into the pickle file
    data_to_save = {"key": "value"}
    
    # Temporarily replaces the built-in open function with a mock version that simulates opening a file for writing
    with patch("builtins.open", mock_open()) as mock_file:
        # Patches pickle.dump with a mock object to check that it's called correctly
        with patch("pickle.dump") as mock_pickle_dump:
            # Calls the dump_pickle function to save the data using the mocked objects
            dump_pickle(data_to_save, "fakefile.pkl")
            
            # Checks that the mock 'open' function was called exactly once with the specified args for writing
            mock_file.assert_called_once_with("fakefile.pkl", "wb")

            # Checks that the mock pickle.dump function was called exactly once with the data and the mock file object
            mock_pickle_dump.assert_called_once_with(data_to_save, mock_file())
            
            # Assert that the file handle provided by mock_open was used to write data
            assert mock_pickle_dump.called