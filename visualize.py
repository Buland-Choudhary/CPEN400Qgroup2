"""
This script visualizes a time series dataset using Plotly.

Download the imports using:
pip install numpy matplotlib plotly pandas

Run using:
python .\visualize.py

You just have to change the file_prefix variable to visualize a different dataset.
"""


import os
import re
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

# Extract the three numbers from the filename
filename = os.path.basename(file_path)
numbers = re.findall(r'_(\d+)_(\d+)_(\d+)\.txt$', filename)

if numbers:
    training_end, anomaly_start, anomaly_end = map(int, numbers[0])
else:
    # Default values if extraction fails
    training_end, anomaly_start, anomaly_end = None, None, None

# Load data
data = np.loadtxt(file_path)

# Create a DataFrame with an index as time steps
df = pd.DataFrame({"Value": data, "Index": np.arange(len(data))})

# Plot using Plotly for interactive scrolling
fig = px.line(df, x="Index", y="Value", title=file_path.split("\\")[-1])

# Add vertical lines for the three markers if they were successfully extracted
if training_end is not None:
    fig.add_vline(x=training_end, line_width=2, line_dash="dash", line_color="green",
                  annotation_text="Training End", annotation_position="top")
    
if anomaly_start is not None:
    fig.add_vline(x=anomaly_start, line_width=2, line_dash="dash", line_color="red",
                  annotation_text="Anomaly Start", annotation_position="top")
    
if anomaly_end is not None:
    fig.add_vline(x=anomaly_end, line_width=2, line_dash="dash", line_color="blue",
                  annotation_text="Anomaly End", annotation_position="top")


# Enable horizontal zooming and sliding
fig.update_xaxes(rangeslider_visible=True)
fig.update_layout(height=600, width=1000)

if anomaly_start is not None and anomaly_end is not None:
    fig.add_shape(
        type="rect",
        x0=anomaly_start,
        x1=anomaly_end,
        y0=df["Value"].min(),
        y1=df["Value"].max(),
        fillcolor="red",
        opacity=0.2,
        layer="below",
        line_width=0,
    )

# Show interactive plot
fig.show()
