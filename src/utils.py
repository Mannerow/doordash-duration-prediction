"""Contains utility functions."""

import pickle

import pandas as pd
from scipy.sparse import csr_matrix


def load_pickle(filename: str):
    """Load pickle from file."""
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def dump_pickle(obj, filename: str):
    """Dump pickle to file."""
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def decode_dataframe(dv, df):
    """Decode a DF from Sparse --> original"""
    # Ensure df is a DataFrame, not a csr_matrix
    if isinstance(df, csr_matrix):
        # Convert sparse matrix to dense matrix
        X_dense = df.toarray()
        # Retrieve feature names from DictVectorizer
        feature_names = dv.get_feature_names_out()
        # Create DataFrame from dense matrix
        df_dense = pd.DataFrame(X_dense, columns=feature_names)
    else:
        df_dense = df

    original_columns = [
        "market_id",
        "created_at",
        "actual_delivery_time",
        "store_id",
        "store_primary_category",
        "order_protocol",
        "total_items",
        "subtotal",
        "num_distinct_items",
        "min_item_price",
        "max_item_price",
        "total_onshift_dashers",
        "total_busy_dashers",
        "total_outstanding_orders",
        "estimated_order_place_duration",
        "estimated_store_to_consumer_driving_duration",
    ]

    # Create a new DataFrame with the same index as the dense DataFrame
    original_df = pd.DataFrame(index=df_dense.index)
    for col in original_columns:
        # A Dict Vectorizer one hot encodes categorical columns like: original_column_name=value
        # We need to find those columns and convert back
        relevant_columns = [
            feature for feature in df_dense.columns if feature.startswith(col + "=")
        ]
        if relevant_columns:
            # Combine the one-hot encoded columns back into a single categorical column
            original_df[col] = (
                df_dense[relevant_columns]
                .idxmax(axis=1)
                .apply(lambda x: x.split("=")[-1])
            )
        elif col in df_dense.columns:
            # If it's a numerical column or directly available
            original_df[col] = df_dense[col]
        else:
            # Handle missing columns as needed (e.g., fill with zeros or NaNs)
            original_df[col] = 0  # or pd.NA, np.nan, etc.

    return original_df
