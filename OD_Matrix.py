#%% Set Up
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

file_name = "CLUSTERED/Clustered_canberra_202309.csv"

df = pd.read_csv(file_name)

#%% Create OD Matrix
# Group by origin and destination clusters to count the number of trips between them
od_counts = df.groupby(['start_cluster', 'dest_cluster']).size().reset_index(name='trip_count')

# Create the OD matrix using a pivot table
od_matrix = od_counts.pivot(index='start_cluster', columns='dest_cluster', values='trip_count')

# Fill missing values (for cluster pairs with no trips) with 0
od_matrix = od_matrix.fillna(0)

#%% Plot on Heatmap and Display OD Matrix

# Set the figure size
plt.figure(figsize=(10, 8))

# Plot the heatmap using seaborn
sns.heatmap(od_matrix, cmap="Oranges", annot=True, fmt='g')

# Add titles and labels
plt.tick_params(axis='both', which='both', length=0) #turn off tick marks
plt.gca().xaxis.tick_top()  # Move the ticks to the top
plt.gca().xaxis.set_label_position('top')  # Move the X-axis label to the top
plt.xlabel('Destination Cluster', fontname = "Times New Roman", fontsize = 12)
plt.ylabel('Start Cluster', fontname = "Times New Roman", fontsize = 12)


# Display the plot
plt.show()
# Step 6: Display the resulting OD matrix
print(od_matrix)

#%% Checks

#Add no. of trips on OD matrix to see if still correct
total_trips = od_matrix.sum().sum()
print(f"Total number of trips in the OD matrix: {total_trips}")

#Determine what proportion of trips are intercluster
# Step 1: Extract the diagonal values of the OD matrix (trips that start and end in the same cluster)
diagonal_trips = od_matrix.values.diagonal()

# Step 2: Sum the diagonal values (total trips starting and ending in the same cluster)
same_cluster_trips = diagonal_trips.sum()

# Step 3: Calculate the total number of trips in the OD matrix
total_trips = od_matrix.sum().sum()

# Step 4: Calculate the proportion of trips that start and end in the same cluster
proportion_same_cluster = same_cluster_trips / total_trips

# Display the results
print(f"Total trips that start and end in the same cluster: {same_cluster_trips}")
print(f"Total trips: {total_trips}")
print(f"Proportion of trips that start and end in the same cluster: {proportion_same_cluster:.2%}")


#%% Export OD Matrix

# Extract the base file name without the directory
base_file_name = os.path.basename(file_name)

# Add the 'OD_' prefix to the base file name
new_file_name = f"OD_{base_file_name}"

# Create the new path in the 'OD_MATRIX' directory
new_file_path = os.path.join('OD_MATRIX', new_file_name)

# Export the OD matrix to the new path
od_matrix.to_csv(new_file_path, index=False)

# Confirm export
print(f"OD Matrix has been exported to: {new_file_path}")

