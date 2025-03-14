"""
This script visualizes a time series dataset using Plotly.

Download the imports using:
pip install numpy matplotlib plotly pandas

Run using:
python .\visualize.py

You just have to change the file_prefix variable to visualize a different dataset.
"""


import os
import numpy as np
import pandas as pd
import plotly.express as px
import glob 

# folder containing the files
folder_path = r".\AnomalyDatasets_2021\UCR_TimeSeriesAnomalyDatasets2021\FilesAreInHere\UCR_Anomaly_FullData"

# Define the file prefix [001 - 250]
file_prefix = "001"

# Find the file that starts with the given prefix
matching_files = glob.glob(os.path.join(folder_path, f"{file_prefix}_*.txt"))

if not matching_files:
    raise FileNotFoundError(f"No file found with prefix {file_prefix} in {folder_path}")

# Select the first matching file
file_path = matching_files[0]

# Load data
data = np.loadtxt(file_path)

# Create a DataFrame with an index as time steps
df = pd.DataFrame({"Value": data, "Index": np.arange(len(data))})

# Plot using Plotly for interactive scrolling
fig = px.line(df, x="Index", y="Value", title=file_path)

# Enable horizontal zooming and sliding
fig.update_xaxes(rangeslider_visible=True)
fig.update_layout(height=600, width=1000)

# Show interactive plot
fig.show()
