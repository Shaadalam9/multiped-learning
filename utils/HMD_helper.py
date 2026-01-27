from custom_logger import CustomLogger
import math
import numpy as np
import pandas as pd
import os
import ast
from collections import defaultdict

logger = CustomLogger(__name__)  # use custom logger


class HMD_yaw():
    """
        A utility class for processing and analysing quaternion-based head-mounted display (HMD) orientation data.

        This class provides methods to:
            - Convert quaternions to Euler angles (roll, pitch, yaw) for easy interpretation and downstream analysis.
            - Average multiple quaternions using Markley's method via eigen decomposition,
                yielding robust mean orientations.
            - Compute average yaw angles per timestamp from participant matrix CSV files.
            - Group files by video_id from a data directory tree, supporting experiments
                organised by video segments or trials.

        Typical use cases include:
            - Preprocessing HMD orientation data for VR/AR experiments.
            - Aggregating orientation data across multiple users or time windows.
            - Batch operations over experiment directories.

        Methods:
            - quaternion_to_euler(w, x, y, z): Converts a single quaternion to Euler angles (roll, pitch, yaw).
            - average_quaternions_eigen(quaternions): Computes the average quaternion from a list.
            - compute_avg_yaw_from_matrix_csv(input_csv, output_csv=None): Computes average yaw per timestamp
                and saves (optionally) as CSV.
            - group_files_by_video_id(data_folder, video_data): Groups CSV files in a directory tree by their video_id.

        Notes:
            - Assumes quaternions use scalar-first format [w, x, y, z].
            - Relies on pandas, numpy, ast, os, math, and collections.defaultdict.
            - The CSV parsing logic assumes columns contain string representations of quaternion lists.

        Example:
            >>> hmd = HMD_yaw()
            >>> roll, pitch, yaw = hmd.quaternion_to_euler(w, x, y, z)
            >>> avg_quat = hmd.average_quaternions_eigen(list_of_quats)
            >>> avg_yaw_df = hmd.compute_avg_yaw_from_matrix_csv('input.csv', 'output.csv')
            >>> grouped_files = hmd.group_files_by_video_id('data/', video_data_df)
        """

    def __init__(self) -> None:
        pass

    def quaternion_to_euler(self, w, x, y, z):
        """
        Converts a quaternion (w, x, y, z) into Euler angles (roll, pitch, yaw).

        The resulting angles are in radians:
            - Roll: rotation around the x-axis
            - Pitch: rotation around the y-axis
            - Yaw: rotation around the z-axis

        Parameters:
            w (float): The scalar component of the quaternion.
            x (float): The x-component of the quaternion.
            y (float): The y-component of the quaternion.
            z (float): The z-component of the quaternion.

        Returns:
            tuple: (roll, pitch, yaw) in radians.
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            # Use 90 degrees if out of range (to handle numerical imprecision)
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # returns in radians
        return roll, pitch, yaw

    def average_quaternions_eigen(self, quaternions):
        """
        Averages a list of quaternions using Markley's method via eigen decomposition.

        Markley's method (see: https://doi.org/10.2514/1.28949) computes the quaternion mean
        that minimises the sum of squared distances on the unit hypersphere, ensuring a robust
        and well-defined average for rotations.

        Args:
            quaternions (List[List[float]]): List of quaternions, each as [w, x, y, z].

        Returns:
            np.ndarray: The average quaternion as a numpy array [w, x, y, z].

        Raises:
            ValueError: If the input list is empty.

        Notes:
            - All input quaternions should be unit quaternions (normalised). This function will
            normalise them if they are not.
            - The sign of the output is chosen so the scalar component (w) is non-negative.
        """
        if len(quaternions) == 0:
            raise ValueError("No quaternions to average.")
        elif len(quaternions) == 1:
            return np.array(quaternions[0])

        # Convert to numpy array and ensure shape (N, 4)
        q_arr = np.array(quaternions)

        # Normalise each quaternion to unit length
        q_arr = np.array([q / np.linalg.norm(q) for q in q_arr])

        # Ensure quaternions are all in the same hemisphere
        # Flip quaternions with negative dot product to the first
        reference = q_arr[0]
        for i in range(1, len(q_arr)):
            if np.dot(reference, q_arr[i]) < 0:
                q_arr[i] = -q_arr[i]

        # Form the symmetric accumulator matrix
        A = np.zeros((4, 4))
        for q in q_arr:
            q = q.reshape(4, 1)  # Make column vector
            A += q @ q.T         # Outer product

        # Normalise by number of quaternions (optional)
        A /= len(q_arr)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        avg_q = eigenvectors[:, np.argmax(eigenvalues)]  # Pick eigenvector with largest eigenvalue

        # Ensure scalar-first order: [w, x, y, z]
        return avg_q if avg_q[0] >= 0 else -avg_q  # Normalise sign

    def compute_avg_yaw_from_matrix_csv(self, input_csv, output_csv=None, force=False):
        """
        Computes the average yaw angle for each timestamp in a CSV containing
        participant quaternion matrices.

        If output_csv is given and force=False and the file already exists,
        it is loaded and returned instead of recomputing.
        """

        # If we already have output & we're not forcing, just load and return
        if output_csv is not None and not force and os.path.isfile(output_csv):
            return pd.read_csv(output_csv)

        df = pd.read_csv(input_csv)
        participant_cols = [col for col in df.columns if col != "Timestamp"]

        results = []

        for _, row in df.iterrows():
            all_quats = []

            for col in participant_cols:
                try:
                    quats = ast.literal_eval(row[col])
                    if isinstance(quats, list) and len(quats) > 0:
                        all_quats.extend(quats)
                except Exception:
                    continue

            if all_quats:
                avg_quat = self.average_quaternions_eigen(all_quats)
                roll, pitch, yaw = self.quaternion_to_euler(*avg_quat)
                results.append({"Timestamp": row["Timestamp"], "AvgYaw": yaw})
            else:
                results.append({"Timestamp": row["Timestamp"], "AvgYaw": None})

        out_df = pd.DataFrame(results)

        if output_csv is not None:
            out_df.to_csv(output_csv, index=False)

        return out_df

    def group_files_by_video_id(self, data_folder, video_data):
        """
        Groups CSV file paths from a directory tree by their associated video_id.

        This method traverses a data folder (including all subdirectories), searching for .csv files.
        Each file is expected to be named in a format containing an underscore and a video_id
        (e.g., "prefix_something_videoid.csv"). Only files whose video_id matches one of the IDs in
        the provided video_data DataFrame will be grouped.

        Args:
            data_folder (str): Path to the root data folder containing CSV files.
            video_data (pd.DataFrame): DataFrame with a 'video_id' column listing valid video IDs.

        Returns:
            defaultdict: Dictionary mapping each video_id to a list of CSV file paths containing that ID.

        Notes:
            - Assumes filenames are structured with underscores, with the video_id
                after the last underscore (before '.csv').
            - Ignores files whose video_id is not present in video_data['video_id'].
            - Recurses into all subfolders of data_folder.
        """

        # Extract unique video IDs from the DataFrame
        video_ids = video_data['video_id'].unique()

        grouped_data = defaultdict(list)  # Dictionary to group file paths by video_id

        # Traverse through the data folder and its subfolders
        for root, _, files in os.walk(data_folder):
            for file in files:
                if file.endswith('.csv'):
                    # Split filename by underscores; expect at least three parts for a video_id
                    file_parts = file.split('_', maxsplit=2)
                    if len(file_parts) > 2:
                        # Extract video_id from the last segment, before '.csv'
                        file_video_id = file_parts[-1].split('.')[0]  # Extract video_id
                        if file_video_id in video_ids:
                            full_path = os.path.join(root, file)
                            grouped_data[file_video_id].append(full_path)

        return grouped_data
