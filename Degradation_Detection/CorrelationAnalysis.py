import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def pearson_correlation(power_file, vibration_file):
    df_power = pd.read_csv(power_file)
    df_vibration = pd.read_csv(vibration_file)

    # Get the column names from df_power
    powerhouses = df_power.columns[1:].tolist()

    # Create an empty dictionary to store the correlation results for each column
    correlation_results = {}

    for column in powerhouses:
        print("Computing for column:", column)
        # Skip the first 2 rows and extract data for the column from both dataframes
        data_power = df_power[column][2:]
        data_vibration = df_vibration[column][2:]

        # Convert data_vibration to numeric, skipping non-convertible columns
        try:
            data_power = pd.to_numeric(data_power)
        except ValueError:
            print(f"Skipping column '{column}' as it contains non-numeric data.")
            continue


        # Convert data_vibration to numeric, skipping non-convertible columns
        try:
            data_vibration = pd.to_numeric(data_vibration)
        except ValueError:
            print(f"Skipping column '{column}' as it contains non-numeric data.")
            continue

        # Calculate the difference series of consecutive rows in data_vibration
        diff_vibration = data_vibration.diff()

        # Transform data_power to exponential form
        data_power = data_power
        data_power = np.log(data_power)
        # data_power = data_power

        correlation_series = data_power.rolling(window=30, min_periods=1).corr(diff_vibration)

        # Store the correlation series for the column
        correlation_results[column] = correlation_series

    # Create a figure for plotting
    plt.figure(figsize=(10, 6))

    # Define the rolling window size
    window_size = 15

    # Define a function to calculate the rolling median for a list of values
    def rolling_median(data, window_size):
        return pd.Series(data).rolling(window=window_size, center=True).median()

    # Calculate and plot the rolling median of the correlation series for each column independently
    for column, correlation_series in correlation_results.items():
        if column in ["LI-67", "TT-63", "JC-43"]:
            plt.figure()
            rolling_corr = rolling_median(correlation_series, window_size)
            rolling_corr.plot(label=column)
            plt.xlabel("Time")
            plt.ylabel("Pearson Correlation (Median)")
            plt.title("Degraded bearing at powerhouse " + column)
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    power_file = "Data/Selected/Power.csv"
    vibration_file = "Data/Selected/Vibration_GB.csv"

    pearson_correlation(power_file, vibration_file)
