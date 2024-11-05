#%% Set Up
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

file_name = "CLUSTERED/Clustered_canberra_202309.csv"

# Read the CSV file
df = pd.read_csv(file_name)

# Ensure the 'OPERATIONS_DATE' column is in datetime format
df['OPERATIONS_DATE'] = pd.to_datetime(df['OPERATIONS_DATE'], dayfirst=True)

#%% Split trips into daily trips
# Extract unique dates from the 'OPERATIONS_DATE' column
unique_dates = df['OPERATIONS_DATE'].dt.date.unique()

# Create a directory for daily matrices if it doesn't exist
output_directory = 'DAILY_MATRIX'
os.makedirs(output_directory, exist_ok=True)

#%% Create Daily OD Matrices
for day in unique_dates:
    # Filter the dataframe for the specific day
    daily_df = df[df['OPERATIONS_DATE'].dt.date == day]
    
    # Group by origin and destination clusters to count the number of trips between them for that day
    od_counts = daily_df.groupby(['start_cluster', 'dest_cluster']).size().reset_index(name='trip_count')
    
    # Create the OD matrix using a pivot table
    od_matrix = od_counts.pivot(index='start_cluster', columns='dest_cluster', values='trip_count')
    
    # Fill missing values (for cluster pairs with no trips) with 0
    od_matrix = od_matrix.fillna(0)

    # Export the OD matrix to a CSV file named after the date
    daily_file_name = f"OD_Matrix_{day}.csv"
    daily_file_path = os.path.join(output_directory, daily_file_name)
    
    od_matrix.to_csv(daily_file_path, index=False)

    #%% Plot Heatmap for each day (Optional)
    plt.figure(figsize=(10, 8))
    sns.heatmap(od_matrix, cmap="Oranges", annot=True, fmt='g')

    # Add titles and labels
    plt.tick_params(axis='both', which='both', length=0) #turn off tick marks
    plt.gca().xaxis.tick_top()  # Move the ticks to the top
    plt.gca().xaxis.set_label_position('top')  # Move the X-axis label to the top
    plt.xlabel('Destination Cluster', fontname="Times New Roman", fontsize=12)
    plt.ylabel('Start Cluster', fontname="Times New Roman", fontsize=12)
    plt.title(f"OD Matrix for {day}")

    # Display the heatmap
    plt.show()

    # Confirm export
    print(f"OD Matrix for {day} has been exported to: {daily_file_path}")

#%% Checks (Optional: for one day example)
total_trips = od_matrix.sum().sum()
print(f"Total number of trips in the OD matrix for {day}: {total_trips}")

# Calculate the proportion of trips that start and end in the same cluster (intra-cluster trips)
diagonal_trips = od_matrix.values.diagonal()
same_cluster_trips = diagonal_trips.sum()
proportion_same_cluster = same_cluster_trips / total_trips
print(f"Proportion of trips that start and end in the same cluster for {day}: {proportion_same_cluster:.2%}")
