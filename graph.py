import numpy as np

def create_sliding_windows(data, window_size, stride=1):
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i:i + window_size])
    return np.array(windows)

time_series_data = np.random.rand(1000)
window_size = 128
windows = create_sliding_windows(time_series_data, window_size)

print(f"Number of windows: {len(windows)}")