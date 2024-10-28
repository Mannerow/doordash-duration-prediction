"""Unit Tests"""

from unittest.mock import mock_open, patch

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer

from src.utils import decode_dataframe, dump_pickle, load_pickle


def test_load_pickle():
    """Tests load_pickle function using a mock."""
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
    """Tests dump_pickle function"""
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


def test_decode_dataframe_dense():
    """Tests decode dataframe"""
    # Sample dense DataFrame
    data = {
        "market_id": [1, 2],
        "created_at": ["2023-01-01", "2023-01-02"],
        "store_id": [111, 222],
        "store_primary_category=fast_food": [1, 0],
        "store_primary_category=grocery": [0, 1],
        "total_items": [10, 20],
        "subtotal": [100, 200],
    }
    df = pd.DataFrame(data)

    # Mock DictVectorizer
    dv = DictVectorizer(sparse=False)
    dv.fit(
        [{"store_primary_category": "fast_food"}, {"store_primary_category": "grocery"}]
    )

    # Decode DataFrame
    result = decode_dataframe(dv, df)

    # Expected output with all columns
    expected_data = {
        "market_id": [1, 2],
        "created_at": ["2023-01-01", "2023-01-02"],
        "actual_delivery_time": [0, 0],  # filled with default value
        "store_id": [111, 222],
        "store_primary_category": ["fast_food", "grocery"],
        "order_protocol": [0, 0],  # filled with default value
        "total_items": [10, 20],
        "subtotal": [100, 200],
        "num_distinct_items": [0, 0],  # filled with default value
        "min_item_price": [0, 0],  # filled with default value
        "max_item_price": [0, 0],  # filled with default value
        "total_onshift_dashers": [0, 0],  # filled with default value
        "total_busy_dashers": [0, 0],  # filled with default value
        "total_outstanding_orders": [0, 0],  # filled with default value
        "estimated_order_place_duration": [0, 0],  # filled with default value
        "estimated_store_to_consumer_driving_duration": [
            0,
            0,
        ],  # filled with default value
    }
    expected_df = pd.DataFrame(expected_data)

    pd.testing.assert_frame_equal(result, expected_df)


def test_decode_dataframe_sparse():
    """Tests sparse matrix"""
    # Sample sparse matrix and feature names
    sparse_data = csr_matrix([[1, 0, 3, 1200], [0, 1, 2, 2400]])
    feature_names = [
        "store_primary_category=fast_food",
        "store_primary_category=grocery",
        "total_items",
        "subtotal",
    ]

    # Mock DictVectorizer, converts feature dictionaries to dense or sparse matrices
    dv = DictVectorizer(sparse=True)
    # Manually setting feature names to similate behaviour of DictVectorizer that has been fit on these features
    dv.feature_names_ = feature_names

    # Decode DataFrame
    result = decode_dataframe(dv, sparse_data)

    # Expected output with all columns
    expected_data = {
        "market_id": [0, 0],
        "created_at": [0, 0],
        "actual_delivery_time": [0, 0],
        "store_id": [0, 0],
        "store_primary_category": ["fast_food", "grocery"],
        "order_protocol": [0, 0],
        "total_items": [3, 2],
        "subtotal": [1200, 2400],
        "num_distinct_items": [0, 0],
        "min_item_price": [0, 0],
        "max_item_price": [0, 0],
        "total_onshift_dashers": [0, 0],
        "total_busy_dashers": [0, 0],
        "total_outstanding_orders": [0, 0],
        "estimated_order_place_duration": [0, 0],
        "estimated_store_to_consumer_driving_duration": [0, 0],
    }
    expected_df = pd.DataFrame(expected_data)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print("Expected")
        print(expected_df)
        print("Result")
        print(result)

    pd.testing.assert_frame_equal(result, expected_df)


def test_decode_dataframe_missing_column():
    """Tests that any missing columns from DF are added to output and filled with default values"""
    # Sample DataFrame missing 'total_items' and other columns
    data = {
        "market_id": [1],
        "created_at": ["2023-01-01"],
        "store_id": [111],
        "store_primary_category=fast_food": [1],
        "subtotal": [100],
    }
    df = pd.DataFrame(data)

    # Mock DictVectorizer
    dv = DictVectorizer(sparse=False)
    dv.fit([{"store_primary_category": "fast_food"}])

    # Decode DataFrame
    result = decode_dataframe(dv, df)

    # Expected output with all columns, missing ones filled with default values
    expected_data = {
        "market_id": [1],
        "created_at": ["2023-01-01"],
        "actual_delivery_time": [0],  # filled with default value
        "store_id": [111],
        "store_primary_category": ["fast_food"],
        "order_protocol": [0],  # filled with default value
        "total_items": [0],  # filled with 0 because it was missing
        "subtotal": [100],
        "num_distinct_items": [0],  # filled with default value
        "min_item_price": [0],  # filled with default value
        "max_item_price": [0],  # filled with default value
        "total_onshift_dashers": [0],  # filled with default value
        "total_busy_dashers": [0],  # filled with default value
        "total_outstanding_orders": [0],  # filled with default value
        "estimated_order_place_duration": [0],  # filled with default value
        "estimated_store_to_consumer_driving_duration": [
            0
        ],  # filled with default value
    }
    expected_df = pd.DataFrame(expected_data)

    pd.testing.assert_frame_equal(result, expected_df)
