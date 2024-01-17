# import pandas as pd
#
# # Load the sensor data from a CSV file into a pandas DataFrame
# data = pd.read_csv("sensor_data.csv")  # Replace "sensor_data.csv" with your actual file name
#
# # Convert the timestamp column to datetime format
# data["timestamp"] = pd.to_datetime(data["timestamp"])
#
# # Sort the DataFrame by the timestamp column
# data = data.sort_values("timestamp")
#
# # Define the time interval (in seconds) between each sample
# interval = 3600*24  # 1 minute interval
#
# # Create a new DataFrame to store the sampled sensor data
# sampled_data = pd.DataFrame(columns=["timestamp", "reading"])
#
# # Get the minimum and maximum timestamps from the original data
# min_timestamp = data["timestamp"].min()
# max_timestamp = data["timestamp"].max()
#
# # Generate a range of timestamps at the specified interval
# sampled_timestamps = pd.date_range(start=min_timestamp, end=max_timestamp, freq=f"{interval}s")
#
# # Iterate over the sampled timestamps and find the closest sensor reading for each timestamp
# for sampled_timestamp in sampled_timestamps:
#     # Find the closest timestamp in the original data
#     closest_timestamp = data["timestamp"].iloc[(data["timestamp"] - sampled_timestamp).abs().argsort()[0]]
#
#     # Get the corresponding sensor reading
#     sensor_reading = data.loc[data["timestamp"] == closest_timestamp, "reading"].values[0]
#
#     # Append the sampled data to the DataFrame
#     sampled_data = sampled_data.append({"timestamp": closest_timestamp, "reading": sensor_reading}, ignore_index=True)
#
# sampled_data = sampled_data[sampled_data["reading"] >= 5.0]
# sampled_data = sampled_data[sampled_data["reading"] <= 31.0]
#
# # Sort the new DataFrame by the timestamp column
# sampled_data = sampled_data.sort_values("timestamp")
#
# # Save the sampled data to a new CSV file
# sampled_data.to_csv("sampled_sensor_data.csv", index=False)


import pandas as pd
import numpy as np

Raw_folder = "Raw/"
Sampled_folder = "SampledDaily/"
filenames = ["HT_GB", "HT_Vibration", "Generator_GB_nonDrive", "Generator_GB_nonDrive_shaft"]

for i in range(len(filenames)):
    # Load the sensor data from a CSV file into a pandas DataFrame
    data = pd.read_csv(Raw_folder + filenames[i] + ".csv")  # Replace "sensor_data.csv" with your actual file name

    # Convert the timestamp column to datetime format
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Sort the DataFrame by the timestamp column
    data = data.sort_values("timestamp")

    # Define the time interval (in seconds) between each sample
    interval = 3600 * 24    # 1 minute interval

    # Create a new DataFrame to store the sampled sensor data
    sampled_data = pd.DataFrame(columns=["timestamp"] + data.columns[1:].tolist())

    # Get the minimum and maximum timestamps from the original data
    min_timestamp = data["timestamp"].min()
    max_timestamp = data["timestamp"].max()

    # Generate a range of timestamps at the specified interval
    sampled_timestamps = pd.date_range(start=min_timestamp, end=max_timestamp, freq=f"{interval}s")

    # Iterate over the sampled timestamps and find the closest sensor reading for each timestamp
    for sampled_timestamp in sampled_timestamps:
        # Find the closest timestamp in the original data
        closest_timestamp = data["timestamp"].iloc[(data["timestamp"] - sampled_timestamp).abs().argsort()[0]]

        # Get the corresponding sensor readings for all columns
        sensor_readings = data.loc[data["timestamp"] == closest_timestamp, :].values.flatten()[1:]

        # Append the sampled data to the DataFrame
        sampled_data = sampled_data.append(pd.Series([closest_timestamp] + list(sensor_readings), index=sampled_data.columns),
                                           ignore_index=True)

    # Go through all columns of sampled_data and remove values less than 1
    for column in sampled_data.columns[1:]:
        sampled_data[column] = sampled_data[column].map(lambda x: x if x > 2 and x < 30 else pd.NA)

    # Drop the first column (timestamps)
    sampled_data = sampled_data.iloc[:, 1:]

    # Remove empty cells from the dataframe
    sampled_data = sampled_data.apply(lambda x: pd.Series(x.dropna().values))

    # Save the sampled data to a new CSV file
    sampled_data.to_csv(Sampled_folder + filenames[i] + ".csv", index=False)


