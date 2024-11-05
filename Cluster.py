#%% Setup
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
import contextily as ctx


# Load your cleaned data

#input variable to change csv file
file_name='CLEAN/canberra_202309.csv'

df1_final = pd.read_csv(file_name)
# df2_final = pd.read_csv('CLEAN/canberra_202310.csv')
# df3_final = pd.read_csv('CLEAN/canberra_202311.csv')
# df4_final = pd.read_csv('CLEAN/canberra_202312.csv')

# Combine all data into one DataFrame
df_combined = pd.concat([df1_final], ignore_index=True)

#%% K-Means Clustering

#extract lat and long, append back to data
df_combined[['start_latitude', 'start_longitude']] = df_combined['ORIGIN_STOP_XY'].str.strip().str.split(expand=True).astype(float)
df_combined[['dest_latitude', 'dest_longitude']] = df_combined['DESTINATION_STOP_XY'].str.strip().str.split(expand=True).astype(float)

coordinates = pd.DataFrame({
    'cluster_latitude': pd.concat([df_combined['start_latitude'], df_combined['dest_latitude']], ignore_index=True),
    'cluster_longitude': pd.concat([df_combined['start_longitude'], df_combined['dest_longitude']], ignore_index=True)
    })

#Drop remaining NaN values and duplicates
coordinates = coordinates.drop_duplicates().dropna()

# Function to compute Sum of Squared Errors (SSE)
def compute_sse(k, coordinates):
    kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42).fit(coordinates)
    #cluster data
    return kmeans.inertia_
    #ntertia function computes the sum of squared distances between each data point and the centroid of each cluster (euclid distance)

# Range of K to try, 31 based on paper
range_k = range(2, 50)
sse = [compute_sse(k, coordinates[['cluster_latitude', 'cluster_longitude']]) for k in range_k]

# Plot SSE for each K
plt.figure(figsize=(10, 6))
plt.plot(range_k, sse, marker='o', color='black')
plt.xticks([0, 10, 20, 30, 40, 49])
plt.xlabel('Number of clusters', fontname='Times New Roman', fontsize=14)
plt.ylabel('SSE', fontname='Times New Roman', fontsize=14)
plt.grid(False)
plt.title("SSE for K values", fontname='Times New Roman', fontsize=14)
plt.xticks(fontname='Times New Roman', fontsize=12)
plt.yticks(fontname='Times New Roman', fontsize=12)
plt.show()


#%% Compute Distance Based Metric

def compute_distance_based_metric(kmeans, coordinates):
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Intra-cluster distance
    intra_distances = []
    for i in range(len(cluster_centers)):
        cluster_points = coordinates[labels == i]
        if len(cluster_points) > 0:
            intra_distances.append(np.mean(cdist(cluster_points, [cluster_centers[i]], 'euclidean')))
    Dintra = np.mean(intra_distances)
    
    # Inter-cluster distance
    inter_distances = cdist(cluster_centers, cluster_centers, 'euclidean')
    np.fill_diagonal(inter_distances, np.inf)  # Ignore self-distance
    Dinter = np.min(inter_distances)
    
    # Combined distance-based metric
    return Dintra / Dinter

#geopandas - change to geodisc distances - WGSA4

distance_metrics = []
for k in range_k:
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(coordinates[['cluster_latitude', 'cluster_longitude']])
    distance_metric = compute_distance_based_metric(kmeans, coordinates[['cluster_latitude', 'cluster_longitude']])
    distance_metrics.append(distance_metric)

min_distance_metric = min(distance_metrics)

# Plot distance_metric for each K
# Plot distance_metric for each K with the specified adjustments
plt.figure(figsize=(10, 6))
plt.plot(range_k, distance_metrics, marker='o', color='black')
plt.xticks([0, 10, 20, 30, 40, 49])
plt.xlabel('Number of clusters', fontname='Times New Roman', fontsize=14)
plt.ylabel('Distance Metric', fontname='Times New Roman', fontsize=14)
plt.grid(False)
plt.title("Distance Metric for K values")
plt.xticks(fontname='Times New Roman', fontsize=12)
plt.yticks(fontname='Times New Roman', fontsize=12)
plt.show()


#%% Compute and Plot Combined Metric

# Scale SSE and distance metrics. Both are being minimized and are positive, 
# thus subtracting them gives combined minimum metric.
def determine_optimal_k(sse, distance_metrics, range_k):
    scaled_sse = (sse - np.min(sse)) / (np.max(sse) - np.min(sse))
    scaled_distance_metric = (distance_metrics - np.min(distance_metrics)) / (np.max(distance_metrics) - np.min(distance_metrics))
    combined_metric = scaled_sse - scaled_distance_metric
    optimal_k = np.argmin(combined_metric) + range_k[0]  # Adjust based on the starting point of range_k
    return combined_metric, optimal_k

# Compute combined metric and find the optimal number of clusters
combined_metric, optimal_k = determine_optimal_k(sse, distance_metrics, list(range_k))
print(f"Optimal number of clusters: {optimal_k}")

# Plot combined_metric for each K
plt.figure(figsize=(10, 6))
plt.plot(range_k, combined_metric, marker='o', color='black')
plt.xticks(range_k)
plt.xlabel('Number of clusters', fontname = "Times New Roman", fontsize = 12 )
plt.ylabel('Combined Metric', fontname = "Times New Roman", fontsize = 12)
plt.title('Combined Metric for different values of K', fontname = "Times New Roman", fontsize = 14)
plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal K = {optimal_k}')
plt.legend()
plt.show()
#graph combined and what comes out

#plot of labels, colour on map

# Recompute metrics for the optimal number of clusters
#n_init = auto means automatitically determines best number of cluster initialisations based on data size and returns result with lowest interta (sum of squared distances from each point to cluster centre)
kmeans = KMeans(n_clusters=optimal_k,n_init='auto', random_state=42).fit(coordinates[['cluster_latitude', 'cluster_longitude']])
distance_based_metric = compute_distance_based_metric(kmeans, coordinates[['cluster_latitude', 'cluster_longitude']].values)

#add cluster label to coordinates, +1 as cluster labels start at 0
coordinates['cluster']= kmeans.fit_predict(coordinates[['cluster_latitude', 'cluster_longitude']])+1

#determine highest and lowest cluster label
max_cluster=max(coordinates['cluster'])
min_cluster=min(coordinates['cluster'])

print(f"Optimal k={optimal_k}:")
print(f"Distance-based metric: {distance_based_metric}")
print(f"Maximum cluster label ={max_cluster}")
print(f"Minimum cluster label ={min_cluster}")

#%% Generate unique values based on latitude and longitude

def generate_unique_values(latitudes, longitudes):
    """
    This function generates unique integers for each (latitude, longitude) pair using Szudzik's pairing function,
    while handling negative values directly within the function.

    Args:
    latitudes: xarray.DataArray or numpy array of latitudes (can be negative)
    longitudes: xarray.DataArray or numpy array of longitudes (can be negative)

    Returns:
    numpy array: A numpy array containing unique values for each (latitude, longitude) pair.
    """
    
    # Initialize an empty array to store the unique values
    unique_values = np.zeros_like(latitudes)
    
    # Iterate over each latitude and longitude pair (1D arrays)
    for i in range(len(latitudes)):
        # Convert latitude to a non-negative integer
        lat = latitudes.iloc[i]
        lat_mapped = 2 * abs(int(lat * 10000)) if lat >= 0 else 2 * abs(int(lat * 10000)) - 1

        # Convert longitude to a non-negative integer
        lon = longitudes.iloc[i]
        lon_mapped = 2 * abs(int(lon * 10000)) if lon >= 0 else 2 * abs(int(lon * 10000)) - 1

        # Apply Szudzik's pairing function directly
        if lat_mapped >= lon_mapped:
            unique_values[i] = lat_mapped * lat_mapped + lon_mapped
        else:
            unique_values[i] = lat_mapped + lon_mapped * lon_mapped
    
    return unique_values

#apply to start coordinates, assign back to DataArray
unique_start_values = generate_unique_values(df_combined["start_latitude"], df_combined["start_longitude"])
df_combined = df_combined.assign(unique_start_value=unique_start_values)

#apply to en coordinates, assign back to DataArray
unique_dest_values = generate_unique_values(df_combined["dest_latitude"], df_combined["dest_longitude"])
df_combined = df_combined.assign(unique_dest_value=unique_dest_values)

#apply to cluster coordinates, assign back to DataArray
unique_cluster_values = generate_unique_values(coordinates["cluster_latitude"], coordinates["cluster_longitude"])
coordinates = coordinates.assign(unique_cluster_value=unique_cluster_values)

#%% Map Cluster Number to Start Lat,Long based on Unique Value

# Check if 'unique_start_values' exists, then rename it
if 'unique_start_value' in df_combined.columns:
    df_combined = df_combined.rename(columns={'unique_start_value': 'unique_value'})
else:
    raise KeyError("Error: 'unique_start_values' not found in df_combined.")

# Check if 'unique_cluster_value' exists, then rename it
if 'unique_cluster_value' in coordinates.columns:
    coordinates = coordinates.rename(columns={'unique_cluster_value': 'unique_value'})
else:
    raise KeyError("Error: 'unique_cluster_value' not found in coordinates.")

# Drop duplicates in coordinates to avoid row multiplication
coordinates = coordinates.drop_duplicates(subset=['unique_value'])

# Perform the left merge to match rows based on 'unique_value'
df_combined = df_combined.merge(coordinates[['unique_value', 'cluster']], on='unique_value', how='left')

# Replace NaN values in 'cluster' with 0 for rows that didn't have a match
df_combined['cluster'].fillna(0, inplace=True)

# Convert 'cluster' to integer if required
df_combined['cluster'] = df_combined['cluster'].astype(int)

df_combined=df_combined.rename(columns={'cluster': 'start_cluster'})

#%% Map Cluster Number to Dest Lat,Long based on Unique Value

# Perform the left merge to match rows based on 'unique_value'
# df_combined = df_combined.merge(coordinates[['unique_dest_value', 'cluster']], on='unique_value', how='left')

df_combined = df_combined.merge(coordinates[['unique_value', 'cluster']], 
                                left_on='unique_dest_value', 
                                right_on='unique_value', 
                                how='left')

# Replace NaN values in 'cluster' with 0 for rows that didn't have a match
df_combined['cluster'].fillna(0, inplace=True)

# Convert 'cluster' to integer if required
df_combined['cluster'] = df_combined['cluster'].astype(int)

df_combined=df_combined.rename(columns={'cluster': 'dest_cluster'})


#%% Map Clusters

# Replace 'cluster_latitude' and 'cluster_longitude' with the actual column names
gdf = gpd.GeoDataFrame(coordinates, geometry=gpd.points_from_xy(coordinates['cluster_longitude'], coordinates['cluster_latitude']))

# Set the coordinate reference system (CRS) to WGS84 (latitude and longitude)
gdf.crs = "EPSG:4326"

# Convert to web mercator (EPSG:3857) for plotting on a tile map
gdf = gdf.to_crs(epsg=3857)

# Plot the data with a more distinguishable colormap (e.g., 'tab20')
fig, ax = plt.subplots(figsize=(12, 12))
gdf.plot(ax=ax, column='cluster', categorical=True, legend=True, cmap='tab20', markersize=100)

# Add a basemap using OpenStreetMap as the source
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Add cluster number labels
# Calculate the centroid for each cluster
centroids = gdf.dissolve(by='cluster').centroid

# Add the cluster number to the plot at the centroid of each cluster
for x, y, label in zip(centroids.x, centroids.y, centroids.index):
    ax.text(x, y, str(label), fontsize=12, color='black', ha='center', va='center', weight='bold')

# Show the plot
plt.show()


#%% Checks

# Check if all rows have valid start_cluster and dest_cluster values > 0 and < 22
valid_rows = (df_combined['start_cluster'] > 0) & (df_combined['start_cluster'] < 22) & \
             (df_combined['dest_cluster'] > 0) & (df_combined['dest_cluster'] < 22)

# Count how many rows meet the condition
valid_count = valid_rows.sum()
total_count = len(df_combined)

# Print the result
if valid_count == total_count:
    print(f"All rows have valid start_cluster and dest_cluster values between 0 and 22.")
else:
    invalid_count = total_count - valid_count
    print(f"{invalid_count} rows have invalid start_cluster or dest_cluster values.")
    
# Display the last few rows to verify the output
print(df_combined.tail())

#check number of trips in each cluster

# Count how many trips start at each cluster
start_cluster_counts = df_combined['start_cluster'].value_counts().sort_index()

# Count how many trips end at each cluster
dest_cluster_counts = df_combined['dest_cluster'].value_counts().sort_index()

# Combine the counts into a DataFrame for easier comparison
cluster_trip_counts = pd.DataFrame({
    'start_trips': start_cluster_counts,
    'end_trips': dest_cluster_counts
}).fillna(0)  # Fill NaN with 0 if a cluster has no trips starting or ending

# Display the number of trips starting and ending at each cluster
print("Trips starting and ending at each cluster:")
print(cluster_trip_counts)

# Check if the total number of start trips equals the total number of end trips
total_start_trips = cluster_trip_counts['start_trips'].sum()
total_end_trips = cluster_trip_counts['end_trips'].sum()

if total_start_trips == total_end_trips:
    print(f"\nThe total number of start trips ({total_start_trips}) is equal to the total number of end trips ({total_end_trips}).")
else:
    print(f"\nMismatch: {total_start_trips} start trips and {total_end_trips} end trips.")


#%% Export

# Extract the base file name without the directory
base_file_name = os.path.basename(file_name)
# Add the 'OD_' prefix to the base file name
new_file_name = f"Clustered_{base_file_name}"
# Create the new path in the 'OD_MATRIX' directory
new_file_path = os.path.join('CLUSTERED', new_file_name)
# Export the data to the new path
df_combined.to_csv(new_file_path, index=False)

# Export cluster trip information
clusterdata_file_name=f"CLUSTERDATA_{base_file_name}"
clusterdata_file_path = os.path.join('CLUSTERED', clusterdata_file_name)
cluster_trip_counts.to_csv(clusterdata_file_path, index=True)

#%% export centroids and start_cluster_counts for eRPCA
# Example: Create a pandas Series

# Export the Series to a CSV file
centroids.to_csv('CLUSTERED/centroids.csv')
start_cluster_counts.to_csv('CLUSTERED/start_cluster_counts.csv')


# Confirm export
print(f"Trip data has been exported to: {new_file_path}")
print(f"Cluster data has been exported to: {clusterdata_file_path}")
