# %% Setup and Imports
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from shapely.geometry import Point 
import geopandas as gpd
import contextily as ctx
from matplotlib.ticker import ScalarFormatter
import shutil
from scipy import stats
import inspect
import math

#%% Clear Output files
# List of folder paths to clear
folder_paths = ['MULTI_GROUP_OUTPUT/DIST_PLOTS', 'MULTI_GROUP_OUTPUT/LS_MATRIX', 'MULTI_GROUP_OUTPUT/RESIDUALS']  # Add as many folders as needed

# Function to clear a folder
def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Delete the file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Delete the directory
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

# Iterate over all folders and clear them
for folder_path in folder_paths:
    clear_folder(folder_path)

#%%
# Matrix dimensions
p = 21
q = 21

# Import Centroids
# Load the CSV file into a pandas DataFrame
df = pd.read_csv('CLUSTERED/centroids.csv')

# Extract x and y coordinates from the 'POINT (x y)' format
df[['x', 'y']] = df['0'].str.extract(r'POINT \(([^ ]+) ([^ ]+)\)')

# Convert x and y to float
df['x'] = df['x'].astype(float)
df['y'] = df['y'].astype(float)

# Create a GeoSeries of Point geometries
geometry = [Point(xy) for xy in zip(df['x'], df['y'])]
centroids = gpd.GeoSeries(geometry)

# %% Helper function to load all daily OD matrices into a list
def load_daily_matrices(directory):
    matrices = []
    filenames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".csv"):
            daily_matrix = pd.read_csv(os.path.join(directory, filename)).to_numpy()
            matrices.append(daily_matrix)
            filenames.append(filename)
    return matrices, filenames

# Load daily OD matrices (multi-group data)
daily_matrices, filenames = load_daily_matrices("DAILY_MATRIX")

# Get mean value of all matrices for initialization
od_mean = np.mean([matrix.mean() for matrix in daily_matrices])

# %% Singular Value Thresholding (SVT) Function with Non-Negative Constraint
def singular_value_thresholding(X, tau):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_thresholded = np.maximum(s - tau, 0)
    L_temp = U @ np.diag(s_thresholded) @ Vt
    L_nonnegative = np.maximum(L_temp, 0)
    return L_nonnegative

# %% Soft Thresholding for Sparse Matrix with Non-Negative Constraint
def soft_thresholding(X, tau):
    S_temp = np.sign(X) * np.maximum(np.abs(X) - tau, 0)
    S_nonnegative = np.maximum(S_temp, 0)
    return S_nonnegative

# %% Multi-Group eRPCA Function (Runs Until Max Iterations)
def erpca_multi_group(daily_matrices, r, s, eta_alpha, eta_beta, alpha, beta, mu, max_iter):

    num_matrices = len(daily_matrices)
    
    # Initialize shared low-rank matrix L
    L = np.mean(daily_matrices, axis=0) - od_mean

    # Initialize Sparse (S), Theta, and Y matrices for each group
    S_list = [np.zeros_like(L) for _ in range(num_matrices)]
    Theta_list = [L + S for S in S_list]
    Y_list = [np.zeros_like(L) for _ in range(num_matrices)]
    
    # Initialize residual lists for tracking convergence
    primal_residuals_list = [[] for _ in range(num_matrices)]
    dual_residuals_list = [[] for _ in range(num_matrices)]
    
    # Stage 1: Optimize L
    for t in range(max_iter):
        L_prev = L.copy()
        L_update_sum = sum([Theta - S + (1 / mu) * Y for Theta, S, Y in zip(Theta_list, S_list, Y_list)])
        L = singular_value_thresholding(L_update_sum / num_matrices, alpha / mu)

    # Stage 2: Fix L, optimize S_g and Theta_g for each group
    for g in range(num_matrices):
        print(f"Stage 2 - Optimizing group {g + 1}/{num_matrices}")
        
        for t in range(max_iter):
            S_prev = S_list[g].copy()
            
            # Update Sparse Matrix S_g
            S_list[g] = soft_thresholding(Theta_list[g] - L + (1 / mu) * Y_list[g], beta / mu)
            
            # Update Natural Parameter Theta_g
            Z_g = L + S_list[g] - (1 / mu) * Y_list[g]
            Theta_list[g] = (daily_matrices[g] + mu * Z_g) / (1 + mu)
            
            # Update Lagrange Multiplier Y_g
            Y_list[g] = Y_list[g] + mu * (Theta_list[g] - L - S_list[g])
            
            # Compute residuals
            primal_residual = np.linalg.norm(Theta_list[g] - L - S_list[g], 'fro')
            dual_residual = mu * np.linalg.norm(S_list[g] - S_prev, 'fro')
            primal_residuals_list[g].append(primal_residual)
            dual_residuals_list[g].append(dual_residual)

            
            # Optional logging
            if t % 1000 == 0:
                print(f"Group {g+1}, Iter {t}")

    return L, S_list, primal_residuals_list, dual_residuals_list

#%% Plot Residuals

def plot_residuals(primal_residuals_list, dual_residuals_list, group_names, plot_start=0):

    plt.figure(figsize=(12, 6))
    
    for g, group_name in enumerate(group_names):
        iterations = range(plot_start, len(primal_residuals_list[g]))
        # Plot Primal Residuals (starting from plot_start)
        plt.subplot(1, 2, 1)
        plt.plot(iterations, primal_residuals_list[g][plot_start:], 
                 marker='o', linestyle='-', label=f'Group {group_name}')
        plt.xlabel('Iteration')
        plt.ylabel('Primal Residual')
        plt.title('Primal Residual Over Iterations')
        plt.legend()
        
        # Plot Dual Residuals (starting from plot_start)
        plt.subplot(1, 2, 2)
        plt.plot(iterations, dual_residuals_list[g][plot_start:], 
                 marker='o', linestyle='-', label=f'Group {group_name}')
        plt.xlabel('Iteration')
        plt.ylabel('Dual Residual')
        plt.title('Dual Residual Over Iterations')
        plt.legend()
    
    plt.tight_layout()
    plt.show()


# %% Heatmap plotting function for L and S matrices
def plot_heatmaps(L, S_list, group_names, output_folder="MULTI_GROUP_OUTPUT/LS_MATRIX"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Plot common low-rank matrix L
    plt.figure(figsize=(8, 6))
    sns.heatmap(L, cmap='viridis', cbar=True, 
                xticklabels=range(1, L.shape[1]+1), yticklabels=range(1, L.shape[0]+1))
    plt.title("Common Low-Rank Matrix L")
    plt.xlabel('Destination Cluster')
    plt.ylabel('Origin Cluster')
    plt.savefig(os.path.join(output_folder, "common_L_matrix.png"))
    plt.close()
    print(f"Common L matrix heatmap saved to {output_folder}/common_L_matrix.png")

    # Plot S matrices for each group in a single figure
    num_groups = len(S_list)
    n_cols = 4  # Adjust as needed
    n_rows = (num_groups + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten()

    for i, (S, group_name) in enumerate(zip(S_list, group_names)):
        ax = axes[i]
        sns.heatmap(S, cmap='coolwarm', ax=ax, cbar=False,
                    xticklabels=range(1, S.shape[1]+1), yticklabels=range(1, S.shape[0]+1))
        ax.set_title(f"S Matrix - {group_name}")
        ax.set_xlabel('Destination Cluster')
        ax.set_ylabel('Origin Cluster')
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "combined_S_matrices.png"))
    plt.close()
    print(f"Combined S matrices heatmap saved to {output_folder}/combined_S_matrices.png")
    
    # Plot residuals (Original - (L+S)) for each group
    residuals_folder = os.path.join(output_folder, "residuals")
    if not os.path.exists(residuals_folder):
        os.makedirs(residuals_folder)

    for i, (S, group_name) in enumerate(zip(S_list, group_names)):
        plt.figure(figsize=(8, 6))
        residual = daily_matrices[i] - (L + S)
        sns.heatmap(residual, cmap='coolwarm', cbar=True,
                    xticklabels=range(1, residual.shape[1]+1), yticklabels=range(1, residual.shape[0]+1))
        plt.title(f"Residual (Original - (L+S)) - {group_name}")
        plt.xlabel('Destination Cluster')
        plt.ylabel('Origin Cluster')
        plt.savefig(os.path.join(residuals_folder, f"Residual_{group_name}.png"))
        plt.close()
    print(f"Residual heatmaps saved to {residuals_folder}")

# %% Initialize CSV for AIC results
def initialize_csv(output_csv="MULTI_GROUP_OUTPUT/aic_results_multi.csv"):
    if os.path.exists(output_csv):
        os.remove(output_csv)  # Delete the existing file
    print(f"{output_csv} has been reset and will start fresh.")

initialize_csv("MULTI_GROUP_OUTPUT/aic_results_multi.csv")

# %% Function to fit distributions and output AIC values
def calculate_best_fit_distribution(trips, start_stop, group_name, matrix_name, output_csv="MULTI_GROUP_OUTPUT/aic_results_multi.csv"):
    distributions = ['norm', 'lognorm', 'expon', 'gamma', 'beta', 'weibull_min', 'poisson']
    best_fit = None
    best_aic = np.inf
    best_params = None
    results = []

    for dist_name in distributions:
        try:
            if dist_name == 'poisson':
                trips_int = np.rint(trips).astype(int)
                lambda_param = np.mean(trips_int)
                loglikelihood = stats.poisson.logpmf(trips_int, lambda_param).sum()
                aic = 2 * 1 - 2 * loglikelihood
                params = (lambda_param,)
            elif dist_name == 'bernoulli':
                trips_bin = (trips > 0).astype(int)
                p_param = np.mean(trips_bin)
                loglikelihood = stats.bernoulli.logpmf(trips_bin, p_param).sum()
                aic = 2 * 1 - 2 * loglikelihood
                params = (p_param,)
            else:
                dist = getattr(stats, dist_name)
                params = dist.fit(trips)
                loglikelihood = dist.logpdf(trips, *params).sum()
                k = len(params)
                aic = 2 * k - 2 * loglikelihood

            results.append((dist_name, aic, params))

            if aic < best_aic:
                best_aic = aic
                best_fit = dist_name
                best_params = params

        except Exception as e:
            print(f"Error fitting {dist_name}: {e}")
            continue

    results.sort(key=lambda x: x[1])

    # Create a DataFrame for the results, including the matrix name and group name
    results_df = pd.DataFrame(results, columns=["Distribution", "AIC", "Parameters"])
    results_df.insert(0, "Matrix", matrix_name)
    results_df.insert(1, "Group", group_name)

    # Add a row for the originating stop and a spacer row
    spacer_row = pd.DataFrame([["", "", "", "", ""]], columns=["Matrix", "Group", "Distribution", "AIC", "Parameters"])
    label_row = pd.DataFrame([[matrix_name, group_name, f"Results for Stop {start_stop}", "", ""]], columns=["Matrix", "Group", "Distribution", "AIC", "Parameters"])

    # Combine label, results, and spacer into one DataFrame
    final_df = pd.concat([label_row, results_df, spacer_row], ignore_index=True)

    # Append to the CSV file if it exists, else create a new one
    if os.path.exists(output_csv):
        final_df.to_csv(output_csv, mode='a', header=False, index=False)
    else:
        final_df.to_csv(output_csv, index=False)
    
    print(f"AIC results for Stop {start_stop} in matrix {matrix_name}, group {group_name} appended to {output_csv}")

    return best_fit, best_aic, best_params

# %% Function to plot histograms for trips from each stop
def plot_histograms_by_stop(S_list, L, distances, group_names, num_stops=21, output_folder="MULTI_GROUP_OUTPUT/DIST_PLOTS"):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Plot histograms for common L matrix
    plot_all_histograms_from_stops(L, distances, num_stops=num_stops, matrix_name="L", group_name="Common_L", output_folder=output_folder)
    
    # Plot histograms for each S matrix
    for S, group_name in zip(S_list, group_names):
        plot_all_histograms_from_stops(S, distances, num_stops=num_stops, matrix_name="S", group_name=group_name, output_folder=output_folder)

# Function to plot all histograms from stops
def plot_all_histograms_from_stops(od_matrix, distances, num_stops=21, matrix_name="L", group_name="", output_folder="MULTI_GROUP_OUTPUT/DIST_PLOTS"):
    
    # Calculate total number of trips from all clusters
    total_trips = od_matrix.sum()

    # Sum trips from each origin
    start_cluster_counts = od_matrix.sum(axis=1)

    # Sort clusters by the number of starting trips in descending order
    sorted_clusters = np.argsort(-start_cluster_counts)[:num_stops] + 1  # Add 1 to convert to cluster numbers

    # Set up subplots grid
    n_cols = 4  # Number of columns for subplots
    n_rows = (num_stops + n_cols - 1) // n_cols

    # Adjust the figure size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 22), constrained_layout=True)
    axes = axes.ravel()

    for i, start_stop in enumerate(sorted_clusters):
        start_stop_idx = start_stop -1

        # Extract trips and distances from the specified stop (start_stop_idx)
        trips_from_stop = od_matrix[start_stop_idx, :]
        distances_from_stop = distances[start_stop_idx, :]

        # Sort trips and distances by closest distance
        sorted_indices = np.argsort(distances_from_stop)
        sorted_trips = trips_from_stop[sorted_indices]

        # Use cluster number for the X-axis labels
        sorted_clusters_dest = sorted_indices + 1  # Adding 1 to make cluster labels 1-based

        # Calculate percentage of trips from this start stop
        cluster_trips = start_cluster_counts[start_stop_idx]  # Number of trips from this cluster
        percentage = (cluster_trips / total_trips) * 100

        # Find the best-fitting distribution if there are non-zero trips
        sorted_trips_nonzero = sorted_trips[sorted_trips > 0]
        if len(sorted_trips_nonzero) > 0:
            # Debug statement to confirm the data and stop number
            print(f"Calling AIC calculation for Stop {start_stop} with data: {sorted_trips_nonzero}")

            best_fit, best_aic, best_params = calculate_best_fit_distribution(
                sorted_trips_nonzero, start_stop, group_name, matrix_name, output_csv="MULTI_GROUP_OUTPUT/aic_results_multi.csv"
            )
        else:
            best_fit = "N/A"
            best_aic = "N/A"
            best_params = None

        # Plot histogram in the current subplot
        axes[i].bar(range(1, len(sorted_trips)+1), sorted_trips, color='blue', edgecolor='black')  # Adjust x to start from 1
        axes[i].set_title(f'Stop {start_stop} ({percentage:.2f}%)\n(Best Fit: {best_fit}, AIC: {best_aic:.2f})', fontsize=10)
        axes[i].set_xticks(range(1, len(sorted_clusters_dest)+1))  # Adjust x-ticks to start from 1
        axes[i].set_xticklabels(sorted_clusters_dest, rotation=45, ha='right', fontsize=8)  # Show only cluster numbers
        axes[i].tick_params(axis='both', labelsize=8)  # Adjust label size for readability
        axes[i].yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure integer labels on Y-axis

    # Hide any extra subplots if num_stops < total axes
    for i in range(num_stops, len(axes)):
        axes[i].axis('off')

    # Center the last figure by setting a main title
    fig.suptitle(f'{matrix_name} Matrix Histograms - {group_name}', fontsize=16, ha='center')
    plt.savefig(os.path.join(output_folder, f"{matrix_name}_histograms_{group_name}.png"))
    plt.close()
    print(f"Histogram plots saved to {output_folder}/{matrix_name}_histograms_{group_name}.png")

# %% Run the multi-group eRPCA
mu = 20
alpha = 1
beta = 1 / math.sqrt(max(p, q))  # Start with a small value for the sparse component
max_iter = 20000  # Adjust as needed
group_names = [f"Day_{i+1}" for i in range(len(daily_matrices))]

# Run the multi-group eRPCA analysis
L, S_list, _, _ = erpca_multi_group(
    daily_matrices, r=20, s=0.1, eta_alpha=0.01, eta_beta=0.01, alpha=alpha, beta=beta, mu=mu, max_iter=max_iter
)

# %% Plot heatmaps for L and S matrices and residuals
plot_heatmaps(L, S_list, group_names)

# Run the multi-group eRPCA analysis
L, S_list, primal_residuals_list, dual_residuals_list = erpca_multi_group(
    daily_matrices, r=20, s=0.1, eta_alpha=0.01, eta_beta=0.01, alpha=alpha, beta=beta, mu=mu, max_iter=max_iter
)

# Plot residuals for each group
plot_residuals(primal_residuals_list, dual_residuals_list, group_names)


# %% Calculate distances
def calculate_euclidean_distances(geoseries):
    coords = np.array([(point.x, point.y) for point in geoseries])
    distances = cdist(coords, coords, metric='euclidean')
    return distances

distances = calculate_euclidean_distances(centroids)

# %% Plot histograms and output AIC values
plot_histograms_by_stop(S_list, L, distances, group_names)

# %% Map areas with most consistent anomalies
# Extract the x (longitude) and y (latitude) from the centroids
centroids_df = pd.DataFrame(centroids.geometry.apply(lambda point: (point.x, point.y)).tolist(), columns=['longitude', 'latitude'])

# Sum the anomalies from S matrices across the month
anomaly_sums = np.sum([S.sum(axis=1) for S in S_list], axis=0)

# Create the GeoDataFrame with the summed anomalies and centroids
gdf = gpd.GeoDataFrame({'anomalies': anomaly_sums, 'cluster': centroids.index + 1}, 
                       geometry=gpd.points_from_xy(centroids_df['longitude'], centroids_df['latitude']), crs='EPSG:4326')

# Create the figure and axis for plotting
fig, ax = plt.subplots(figsize=(12, 12))

# Plot the anomalies on the map
gdf.plot(column='anomalies', cmap='coolwarm', legend=True, markersize=100, ax=ax)

# Add cluster numbers to the center of the circles
for x, y, label in zip(centroids_df['longitude'], centroids_df['latitude'], gdf['cluster']):
    ax.text(x, y, str(label), color='black', fontsize=12, ha='center', va='center')

# Add OpenStreetMap basemap using Mapnik as the source
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Format the colorbar to show whole numbers instead of scientific notation
cbar = ax.get_figure().get_axes()[1]  # Get the colorbar axis
cbar.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))  # Set to whole numbers
cbar.yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure integer tick labels

# Add title and show the plot
plt.title('Spatial Distribution of Anomalies over Canberra')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# %% Plot Anomaly Sums
plt.bar(range(1, len(anomaly_sums)+1), anomaly_sums)  # Adjust x to start from 1
plt.xlabel('Cluster')
plt.ylabel('Anomaly Magnitude')
plt.title('Anomalies Sum by Cluster')
plt.xticks(ticks=range(1, len(anomaly_sums) + 1), labels=range(1, len(anomaly_sums) + 1), rotation=45)  # Label clusters properly
plt.show()

#%% S matrix Colourbar
# After plotting heatmaps
plot_heatmaps(L, S_list, group_names)

# Generate and save the colorbar for the combined S matrices
# Calculate the common vmin and vmax across all S matrices
vmin = min([S.min() for S in S_list])
vmax = max([S.max() for S in S_list])

# Create a figure and axis for the colorbar
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

# Create a colorbar with the same colormap and normalization as the combined S matrices
cmap = plt.cm.coolwarm  # Use the same colormap as in the heatmaps

# Create a ScalarMappable and initialize a data array
norm = plt.Normalize(vmin=vmin, vmax=vmax)
cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                   cax=ax, orientation='horizontal')
cb1.set_label('Value')

# Save the colorbar as a separate image
plt.savefig('MULTI_GROUP_OUTPUT/LS_MATRIX/combined_S_matrices_colorbar.png')
plt.close()
print('Colorbar for combined S matrices saved to MULTI_GROUP_OUTPUT/LS_MATRIX/combined_S_matrices_colorbar.png')

#%% Plot Input Matrices

# Add this code after your plot_heatmaps function, or at the end of your script after the plots have been generated.

def plot_original_matrices_heatmaps(daily_matrices, group_names, output_folder="MULTI_GROUP_OUTPUT/OD_MATRICES"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Calculate the common vmin and vmax across all daily matrices
    vmin = min([matrix.min() for matrix in daily_matrices])
    vmax = max([matrix.max() for matrix in daily_matrices])
    
    num_groups = len(daily_matrices)
    n_cols = 4  # Adjust as needed
    n_rows = (num_groups + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten()
    
    for i, (matrix, group_name) in enumerate(zip(daily_matrices, group_names)):
        ax = axes[i]
        sns.heatmap(matrix, cmap='viridis', ax=ax, cbar=False, vmin=vmin, vmax=vmax,
                    xticklabels=range(1, matrix.shape[1]+1), yticklabels=range(1, matrix.shape[0]+1))
        ax.set_title(f"OD Matrix - {group_name}")
        ax.set_xlabel('Destination Cluster')
        ax.set_ylabel('Origin Cluster')
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "combined_OD_matrices.png"))
    plt.close()
    print(f"Combined OD matrices heatmap saved to {output_folder}/combined_OD_matrices.png")
    
    # Generate and save the colorbar for the combined OD matrices
    # Create a figure and axis for the colorbar
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    
    # Create a colorbar with the same colormap and normalization as the OD matrices heatmaps
    cmap = plt.cm.viridis  # Use the same colormap as in the heatmaps
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                       cax=ax, orientation='horizontal')
    cb1.set_label('Number of Trips')
    
    # Save the colorbar as a separate image
    plt.savefig(os.path.join(output_folder, "combined_OD_matrices_colorbar.png"))
    plt.close()
    print(f"Colorbar for combined OD matrices saved to {output_folder}/combined_OD_matrices_colorbar.png")

plot_original_matrices_heatmaps(daily_matrices, group_names)

#%% Combine residual plots

def plot_combined_residuals(daily_matrices, L, S_list, group_names, output_folder="MULTI_GROUP_OUTPUT/RESIDUALS"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Calculate residuals for each day
    residuals_list = [daily_matrices[i] - (L + S_list[i]) for i in range(len(daily_matrices))]
    
    # Calculate the common vmin and vmax across all residuals
    vmin = min([residual.min() for residual in residuals_list])
    vmax = max([residual.max() for residual in residuals_list])
    
    num_groups = len(residuals_list)
    n_cols = 4  # Adjust as needed
    n_rows = (num_groups + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten()
    
    for i, (residual, group_name) in enumerate(zip(residuals_list, group_names)):
        ax = axes[i]
        sns.heatmap(residual, cmap='coolwarm', ax=ax, cbar=False, vmin=vmin, vmax=vmax,
                    xticklabels=range(1, residual.shape[1]+1), yticklabels=range(1, residual.shape[0]+1))
        ax.set_title(f"Residual - {group_name}")
        ax.set_xlabel('Destination Cluster')
        ax.set_ylabel('Origin Cluster')
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "combined_residuals.png"))
    plt.close()
    print(f"Combined residuals heatmap saved to {output_folder}/combined_residuals.png")
    
    # Generate and save the colorbar for the combined residuals
    # Create a figure and axis for the colorbar
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    
    # Create a colorbar with the same colormap and normalization as the residuals heatmaps
    cmap = plt.cm.coolwarm  # Use the same colormap as in the heatmaps
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                       cax=ax, orientation='horizontal')
    cb1.set_label('Residual Value')
    
    # Save the colorbar as a separate image
    plt.savefig(os.path.join(output_folder, "combined_residuals_colorbar.png"))
    plt.close()
    print(f"Colorbar for combined residuals saved to {output_folder}/combined_residuals_colorbar.png")
    
plot_combined_residuals(daily_matrices, L, S_list, group_names)


#%% Plot AIC Values
# Add this function after your existing plotting functions

def plot_aic_bar_charts(aic_csv_file="MULTI_GROUP_OUTPUT/aic_results_multi.csv", output_folder="MULTI_GROUP_OUTPUT/AIC_PLOTS"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from collections import Counter

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the CSV file
    aic_df = pd.read_csv(aic_csv_file, dtype=str)

    # Remove label rows and spacer rows
    aic_df_clean = aic_df.dropna(subset=['AIC'])
    aic_df_clean = aic_df_clean[aic_df_clean['AIC'] != '']
    aic_df_clean = aic_df_clean[~aic_df_clean['Distribution'].str.contains('Results for Stop')]

    # Exclude Poisson distribution
    aic_df_clean = aic_df_clean[aic_df_clean['Distribution'] != 'poisson']

    # Convert AIC values to float
    aic_df_clean['AIC'] = aic_df_clean['AIC'].astype(float)

    # Initialize a list to collect the best distributions
    best_distributions = []

    # Get the list of groups
    groups = aic_df_clean['Group'].unique()

    # Determine the number of subplots
    num_groups = len(groups)
    n_cols = 4  # Adjust as needed
    n_rows = (num_groups + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for i, group in enumerate(groups):
        ax = axes[i]

        # Get data for this group
        group_df = aic_df_clean[aic_df_clean['Group'] == group]

        # Group by 'Distribution' and average the AIC values across all stops
        dist_aic = group_df.groupby('Distribution')['AIC'].mean().reset_index()

        # Sort distributions by AIC ascending (lowest AIC first)
        dist_aic_sorted = dist_aic.sort_values('AIC')

        # Record the best distribution for this group
        best_distribution = dist_aic_sorted.iloc[0]['Distribution']
        best_distributions.append(best_distribution)

        # Plot the bar chart
        ax.bar(dist_aic_sorted['Distribution'], dist_aic_sorted['AIC'])
        ax.set_title(f'{group} (Best: {best_distribution})')
        ax.set_xlabel('Distribution')
        ax.set_ylabel('Average AIC')
        ax.set_xticklabels(dist_aic_sorted['Distribution'], rotation=45, ha='right')

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    # Determine the most common distribution across all days
    distribution_counts = Counter(best_distributions)
    most_common_distribution = distribution_counts.most_common(1)[0][0]

    # Set the overall title
    fig.suptitle(f'Most Common Best-Fit Distribution: {most_common_distribution}', fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to make room for the suptitle

    # Save the figure
    output_file = os.path.join(output_folder, 'combined_aic_bar_charts.png')
    plt.savefig(output_file)
    plt.close()
    print(f'Combined AIC bar charts saved to {output_file}')

plot_aic_bar_charts()