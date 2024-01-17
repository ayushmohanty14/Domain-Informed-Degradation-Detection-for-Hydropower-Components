import pandas as pd
import re

def convert_kw_to_mw(input_file, output_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Fill NaN values with zero for all columns
    df.fillna(0, inplace=True)

    # Convert all columns starting from the 2nd column to float (1st column is timestamp)
    df.iloc[4:, 1:] = df.iloc[4:, 1:].astype(float)

    # Find all columns with the name "Kilowatts"
    kw_columns = [col for col in df.columns if re.search(r'Kilowatts', col, re.IGNORECASE)]

    # Process each "Kilowatts" column individually
    for column in kw_columns:
        # Convert the numeric data starting from row 5 by dividing by 1000
        df.loc[df.index >= 4, column] *= 0.001

    # Store the output DataFrame as a new CSV file
    df.to_csv(output_file, index=False, header=False)

def preprocess_Vibration(input_file, output_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Fill NaN values with zero for all columns
    df.fillna(0, inplace=True)

    # Store the output DataFrame as a new CSV file
    df.to_csv(output_file, index=False)

def find_common_powerhouses(file1, file2):
    # Read the two CSV files into DataFrames
    df_vibration = pd.read_csv(file1)
    df_power = pd.read_csv(file2)

    # Get the powerhouse names from the 2nd row of both DataFrames
    powerhouses_vibration = df_vibration.iloc[0, 1:].tolist()
    powerhouses_power = df_power.iloc[0, 1:].tolist()

    print(powerhouses_vibration)
    print(powerhouses_power)

    # Find the common powerhouse names in both DataFrames
    common_powerhouses = set(powerhouses_vibration).intersection(powerhouses_power)

    print(common_powerhouses)

    # Find the timestamp range in vibration_file
    vibration_df = pd.read_csv(file1, header=2)  # Set header=2 to skip two rows and start reading from the 3rd row
    vibration_timestamps = vibration_df.iloc[:, 0].tolist()
    vibration_start_time = vibration_timestamps[0]
    vibration_end_time = vibration_timestamps[-1]

    return common_powerhouses, vibration_start_time, vibration_end_time

def create_clean_csv(input_file, output_file, common_powerhouses):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file, header=1)  # Set header=1 to use the 2nd row as column names

    # Filter the DataFrame to include only the columns with common powerhouse names
    columns_to_keep = [df.columns[0]] + [col for col in df.columns[1:] if col in common_powerhouses]
    df = df[columns_to_keep]

    # Store the clean DataFrame as a new CSV file
    df.to_csv(output_file, index=False)

def trim_power_file(power_file, output_file, start_time, end_time):
    # Read the power_file into a DataFrame
    power_df = pd.read_csv(power_file)  # Read the entire file

    # Filter the power_df to include only rows within the specified timestamp range
    power_df = power_df[power_df.iloc[:, 0].between(start_time, end_time)]

    # Store the trimmed DataFrame as a new CSV file
    power_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_power_file = "Data/Raw/RealPower_MeteringAndControl.csv"
    output_power_file = "Data/Pre-Processed/Power.csv"
    selected_power_file = "Data/Selected/Power.csv"
    cleaned_power_file = "Data/Cleaned/Power.csv"

    input_vibration_file = "Data/Raw/HT_GB_Vibration_Daily.csv"
    output_vibration_file = "Data/Pre-Processed/Vibration_GB.csv"
    selected_vibration_file = "Data/Selected/Vibration_GB.csv"
    cleaned_vibration_file = "Data/Cleaned/Vibration_GB.csv"

    # convert_kw_to_mw(input_power_file, output_power_file)
    # preprocess_Vibration(input_vibration_file, output_vibration_file)

    # # Find common powerhouse names in both files
    # common_powerhouses, vibration_start_time, vibration_end_time = find_common_powerhouses(output_vibration_file, output_power_file)
    #
    # # Create clean files with common powerhouse names
    # create_clean_csv(output_vibration_file, selected_vibration_file, common_powerhouses)
    # create_clean_csv(output_power_file, selected_power_file, common_powerhouses)
    #
    # # Trim power_file to the range of timestamps in vibration_file and save as trimmed_power.csv
    # trim_power_file(selected_power_file, cleaned_power_file, vibration_start_time, vibration_end_time)