import numpy as np
import pandas as pd
import ast
from utils.HMD_helper import HMD_yaw

HMD_class = HMD_yaw()


class Tools():
    """
    A utility class providing helper functions for data processing in time-series and participant tracking experiments.

    This class offers methods to:
        - Parse and process DataFrame columns containing string-encoded lists or quaternions.
        - Flatten nested data structures for analysis.
        - Extract and compute statistical features (such as means or yaws)
        from experimental data stored in CSV or DataFrame formats.

    Methods in this class expect data with a specific structure, such as DataFrames with a 'Timestamp'
    column and participant columns containing lists encoded as strings, and may utilise external
    tools for quaternion-to-euler conversions.

    Usage example:
        tools = Tools()
        avg_df = tools.average_dataframe_vectors_with_timestamp(df, "average")
        yaws = tools.all_yaws_per_bin("data.csv")
        flat = tools.flatten_trial_matrix(nested_list)
    """

    def __init__(self) -> None:
        pass

    def average_dataframe_vectors_with_timestamp(self, df, column_name):
        """
        Calculates the mean of all vector-type columns for each row in a DataFrame, excluding the 'Timestamp' column.
        The function assumes that each non-'Timestamp' column contains either a list of numbers or a string
        representation of such a list. It computes the mean of each vector per row, then averages these means
        for each row, associating the result with its timestamp.

        Args:
            df (pd.DataFrame): Input DataFrame, expected to have a 'Timestamp' column and other columns containing
                                vectors (as lists or string representations of lists).
            column_name (str): The name to use for the resulting averaged vector column in the output DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with columns 'Timestamp' and the specified column_name, where each row
                          contains the timestamp and the average of the means of the vectors from that row.
        """
        result = []

        # Iterate over each row in the DataFrame
        for idx, row in df.iterrows():
            timestamp = row["Timestamp"]  # Extract the timestamp for the current row
            row_means = []  # To store means of all vectors in the current row

            # Iterate through all columns in the row except 'Timestamp'
            for col in df.columns:
                if col == "Timestamp":
                    continue  # Skip the timestamp column

                val = row[col]  # Get the value from the current column
                try:
                    # Attempt to parse the value as a vector if it's a string
                    vec = ast.literal_eval(val) if isinstance(val, str) else val

                    # Check if it's a list of numbers (vector)
                    if isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec):
                        row_means.append(np.mean(vec))  # Add the mean of the vector
                except:  # noqa:E722
                    continue

            # Calculate the average of all vector means for the current row
            row_avg = np.mean(row_means) if row_means else np.nan

            # Store the timestamp and calculated average in the result list
            result.append({"Timestamp": timestamp, column_name: row_avg})

        # Return the results as a new DataFrame
        return pd.DataFrame(result)

    def extract_time_series_values(self, df):
        """
        Extracts a list of lists from a DataFrame where each inner list contains
        all the values for a specific timestamp, flattened from the list-encoded columns.

        Parameters:
            df (pd.DataFrame): The input DataFrame, expected to have a 'Timestamp' column
                               and other columns containing string-encoded lists.

        Returns:
            list of list: A list where each inner list contains all values (flattened)
                          for one timestamp (row) in the DataFrame.
        """
        all_values_by_timestep = []

        # Iterate through each row in the DataFrame
        for _, row in df.iterrows():
            row_values = []  # Collect all values for the current timestamp

            # Loop over all columns except 'Timestamp'
            for col in df.columns:
                if col == 'Timestamp':
                    continue  # Skip the 'Timestamp' column

                value = row[col]
                if pd.isna(value):
                    # If value is NaN, treat as an empty list (alternatively, fill with [None] or [0])
                    parsed_list = []
                else:
                    # Convert the string representation of a list to an actual list
                    parsed_list = ast.literal_eval(value)

                # Flatten the list values into the row_values list
                row_values.extend(parsed_list)

            # Append the flattened values for this row/timestamp to the result list
            all_values_by_timestep.append(row_values)

        return all_values_by_timestep

    def all_yaws_per_bin(self, input_csv):
        """
        Reads a CSV file and extracts the yaw values from quaternion lists for each timestamp/bin.

        Parameters:
            input_csv (str): Path to the input CSV file. The file must have a 'Timestamp' column,
                             and all other columns should contain string-encoded lists of quaternions
                             (each quaternion as a list of four numbers).

        Returns:
            yaws_per_bin (list of list):
                - Outer list: one entry per timestamp/bin (row in CSV).
                - Inner list: all yaw values (float) from all participants for that bin.
        """
        df = pd.read_csv(input_csv)

        # All participant columns (exclude the timestamp)
        participant_cols = [col for col in df.columns if col != "Timestamp"]

        yaws_per_bin = []

        # Iterate through each row (timestamp/bin)
        for idx, row in df.iterrows():
            all_yaws = []

            # For each participant's quaternion data
            for col in participant_cols:
                try:
                    # Parse the string-encoded quaternion list
                    quats = ast.literal_eval(row[col])
                    # Ensure the entry is a non-empty list
                    if isinstance(quats, list) and len(quats) > 0:
                        for q in quats:
                            # Convert quaternion to yaw using external method
                            _, _, yaw = HMD_class.quaternion_to_euler(*q)
                            all_yaws.append(yaw)
                except Exception:
                    # Skip if parsing or conversion fails
                    continue

            # Store all yaws found for this timestamp/bin
            yaws_per_bin.append(all_yaws)

        return yaws_per_bin

    def flatten_trial_matrix(self, trial_matrix):
        """
        Flattens a list of lists (matrix) of floats into a single 1D NumPy array.

        Parameters:
            trial_matrix (list of list of float): A nested list where each sublist contains float values.

        Returns:
            np.ndarray: A 1D NumPy array of type float64 containing all the values from the input matrix, flattened.
        """
        flat = []  # Will hold the flattened values

        # Iterate through each sublist in the matrix
        for sublist in trial_matrix:
            flat.extend(sublist)  # Extend the flat list with values from the current sublist

        # Convert the flat list to a NumPy array of type float64
        return np.array(flat, dtype=np.float64)
