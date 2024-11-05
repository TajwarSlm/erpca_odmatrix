#%% Setup and Imports
import numpy as np
import pandas as pd
import scipy.sparse.linalg as slinalg
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import svds
import imageio.v2 as imageio
import os
from scipy.stats import mode
import inspect

#%% Load OD Matrix
file_path = "OD_MATRIX/OD_Clustered_canberra_202309.csv"
df_od_matrix = pd.read_csv(file_path)

od_matrix = df_od_matrix.to_numpy()
od_mean = (1/np.sum(od_matrix))*od_matrix

od_original = od_matrix.copy()

#tau optimisation
non_zero_values = od_matrix[od_matrix != 0]  # Exclude zeros for standard deviation calculation
sigma = np.std(non_zero_values)  # Standard deviation of non-zero elements
tau = sigma / np.sqrt(max(od_matrix.shape))  # tau based on standard deviation and matrix size
print(tau)

#%% Singular Value Thresholding (SVT) Function
def singular_value_thresholding(X, tau):
    
    U, s, Vt = np.linalg.svd(X)
    s_thresholded = np.maximum(s - tau, 0)
    L_temp = U @ np.diag(s_thresholded) @ Vt
    L_nonnegative = np.maximum(L_temp, 0)
    return L_nonnegative

#%% Soft Thresholding for Sparse Matrix
def soft_thresholding(X, tau):
    
    S_temp = np.sign(X) * np.maximum(np.abs(X) - tau, 0)
    S_nonnegative = np.maximum(S_temp, 0)
    
    return S_nonnegative

#%% Hyperparameter Tuning Function 
def hyperparameter_tuning(L, S, r, s, eta_alpha, eta_beta, alpha, beta, t):

    # Calculate the current rank of L and sparsity of S
    _, s_values, _ = np.linalg.svd(L, full_matrices=False)
    rank_L = np.sum(s_values > 1e-6)
    sparsity_S = np.sum(S == 0) / S.size

    # Adjust alpha if rank(L) > r
    if rank_L > r:
        alpha = alpha + eta_alpha * np.sqrt(t)
    else:
        alpha = max(alpha - eta_alpha * np.sqrt(t), 1e-6)  # Ensure alpha is non-negative

    # Adjust beta if sparsity of S is less than target sparsity s
    if sparsity_S < s:
        beta = beta + eta_beta * np.sqrt(t)
    else:
        beta = max(beta - eta_beta * np.sqrt(t), 1e-6)  # Ensure beta is non-negative

    return alpha, beta

#%% ADMM Optimization Function for eRPCA
def erpca(od_matrix, r, s, eta_alpha, eta_beta, alpha, beta, mu, max_iter, save_frequency, hyperparameter_tuning_switch, video_output='erpca_evolution.mp4', create_video=True):
    
    # Create a directory to save heatmaps and clear old images
    output_dir = 'erpca_heatmaps'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        clear_image_directory(output_dir)

    # Step 1: Initialize Low-rank (L), Sparse (S), Natural Parameter (Theta), and Lagrange Multiplier (Y) Matrices
        
    # Initialize L as mean valuue
    L = od_mean
    
    # Initialize S
    S = np.zeros_like(od_matrix)  # Initialize S with zeros
    Theta = L + S  # Natural parameter matrix
    Y = np.zeros_like(od_matrix)  # Initalise Y with zeros

    # Lists to store the residuals for plotting
    prime_residuals = []
    dual_residuals = []
    
    # Stage 2: Main ADMM Optimization Loop
    for t in range(max_iter):
        
        print(t)
        
        if hyperparameter_tuning_switch:
            alpha, beta = hyperparameter_tuning(L, S, r, s, eta_alpha, eta_beta, alpha, beta, t)
        
        S_prev = S.copy()
               
        # Low-Rank Update
        L = singular_value_thresholding(Theta - S + (1 / mu) * Y, alpha / mu)
        print(L.mean())
        
        # S Update
        S = soft_thresholding(Theta - L + (1 / mu) * Y, beta / mu)
        print(S.mean())
        
        #Update theta
        Z = L + S - (1 / mu) * Y
        Theta = (od_matrix + mu * Z) / (1 + mu)
        
        #Y Update
        Y = Y + mu*(Theta - L - S)
        
        #check residuals
        
        prime_residual = np.linalg.norm(Theta - L - S, ord='fro')
        print(prime_residual)
        dual_residual = mu*np.linalg.norm(S-S_prev, ord='fro')
        
        prime_residuals.append(prime_residual)
        dual_residuals.append(dual_residual)
        
        # Save heatmaps for visualization at specified intervals
        if t % save_frequency == 0:
            plot_and_save_combined_heatmaps(od_matrix, L, S, t, output_dir)
            
            


    # Create video from saved heatmaps using imageio if enabled
    if create_video:
        create_video_with_imageio(output_dir, video_output, fps=2)

    # Plot the relative residuals and log-likelihood residuals
    plt.figure(figsize=(12, 6))
    
    # Plotting Relative Residuals
    plt.subplot(1, 2, 1)
    plt.plot(range(10, len(prime_residuals)), prime_residuals[10:], marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Prime Residual')
    plt.title('Prime Residual Over Iterations')
    
    # Plotting Log-Likelihood Residuals
    plt.subplot(1, 2, 2)
    plt.plot(range(10,len(dual_residuals)), dual_residuals[10:], marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Dual Residual')
    plt.title('Dual Residual Over Iterations')
    
    plt.tight_layout()
    plt.show()
    
    return L, S, alpha, beta

def clear_image_directory(directory):

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".png"):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

def plot_and_save_combined_heatmaps(od_matrix, L, S, iteration, output_dir):

    # Set consistent vmin and vmax for OD, L, and S matrices
    vmin_common = min(od_matrix.min(), L.min(), S.min())
    vmax_common = max(od_matrix.max(), L.max(), S.max())
    
    # Create a figure with subplots
    plt.figure(figsize=(18, 6))
    
    # Define custom tick labels from 1 to 21
    tick_labels = range(1, 22)
    
    # 1. Heatmap of the Original OD Matrix
    plt.subplot(1, 4, 1)
    sns.heatmap(od_matrix, cmap='viridis', vmin=vmin_common, vmax=vmax_common, 
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title(f"Original OD Matrix (Iteration {iteration})")
    
    # 2. Heatmap of Low-Rank Matrix (L)
    plt.subplot(1, 4, 2)
    sns.heatmap(L, cmap='viridis', vmin=vmin_common, vmax=vmax_common, 
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title(f"Low-Rank Matrix (L) - Iteration {iteration}")
    
    # 3. Heatmap of Sparse Matrix (S)
    plt.subplot(1, 4, 3)
    sns.heatmap(S, cmap='viridis', vmin=vmin_common, vmax=vmax_common, 
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title(f"Sparse Matrix (S) - Iteration {iteration}")
    
    # 4. Residual Heatmap (OD Matrix - (L + S)) with its own color scale
    plt.subplot(1, 4, 4)
    residual = od_matrix - (L + S)
    sns.heatmap(residual, cmap='coolwarm', center=0, 
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title(f"Residual (OD Matrix - (L+S)) - Iteration {iteration}")
    
    # Save the combined heatmap image
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_combined_{iteration:03d}.png")
    plt.close()


def create_video_with_imageio(image_dir='erpca_heatmaps', output_video='erpca_evolution.mp4', fps=2):
    
    # Get the list of image files
    images = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".png")])

    # Read and compile images into a video using imageio
    imageio.mimsave(output_video, [imageio.imread(img) for img in images], fps=fps)
    print(f"Video saved as {output_video}")

#%% Apply eRPCA to OD Matrix

p, q =  np.shape(od_matrix)
# mu = (p * q) / (4 * np.linalg.norm(od_matrix, ord=1))
mu = 15
print(mu)

alpha = 1 # Weight for low-rank component
beta = 1/max(p,q)  # Start with a smaller beta value for sparse component

#%%
create_video = True
save_frequency = 10000
max_iter = 200000
hyperparamater_tuning_switch = False

L, S, final_alpha, final_beta = erpca(od_matrix, r=20, s=0.2, eta_alpha=0.01, eta_beta=0.01,hyperparameter_tuning_switch=hyperparamater_tuning_switch, alpha=alpha, beta=beta, mu=mu, max_iter=max_iter,save_frequency=save_frequency, create_video = create_video)

#round up to nearest integer
L = np.ceil(L).astype(int)
S = np.ceil(S).astype(int)

#%% Convert the resulting L and S matrices back to DataFrames
df_L = pd.DataFrame(L, columns=df_od_matrix.columns, index=df_od_matrix.index)
df_S = pd.DataFrame(S, columns=df_od_matrix.columns, index=df_od_matrix.index)

# Export to Excel files
df_L.to_excel("L_matrix.xlsx", index=False, header=False)  # Exports L matrix to L_matrix.xlsx
df_S.to_excel("S_matrix.xlsx", index=False, header=False)  # Exports S matrix to S_matrix.xlsx

print("Matrices L and S have been exported to L_matrix.xlsx and S_matrix.xlsx respectively.")

#%% Quality Checks

# 1. Reconstruction Error (Frobenius norm)
reconstruction_error = np.linalg.norm(od_matrix - (L + S), ord='fro') / np.linalg.norm(od_matrix, ord='fro')
print(f"Reconstruction Error (Frobenius norm): {reconstruction_error:.5f}")

# 2. Rank of L
rank_L = np.linalg.matrix_rank(L)
print(f"Rank of L: {rank_L}")

rank_S = np.linalg.matrix_rank(S)
print(f"Rank of S: {rank_S}")

# 3. Sparsity of S
sparsity_S = np.sum(S != 0) / S.size  # Fraction of non-zero elements in S
print(f"Sparsity of S: {sparsity_S:.5f}")

#L as % of all trips
L_percent = np.sum(L)/np.sum(L+S)
S_percent = np.sum(S)/np.sum(L+S)

print(f"L is {L_percent: .5f}% of trips")
print(f"S is {S_percent: .5f}% of trips")

#%% Residual Heatmap (Original Matrix - (L + S))
tick_labels = range(1, 22)

plt.figure(figsize=(6, 6))
sns.heatmap(od_matrix - (L + S), cmap='coolwarm', xticklabels=tick_labels, yticklabels=tick_labels)
plt.title("Residual Heatmap (Original Matrix - (L+S)")

#%% Print Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(L, cmap='coolwarm', xticklabels=tick_labels, yticklabels=tick_labels)
plt.title("L")

plt.figure(figsize=(6, 6))
sns.heatmap(S, cmap='coolwarm', xticklabels=tick_labels, yticklabels=tick_labels)
plt.title("S")

# Flatten the matrices for distribution fitting
L_flat = L.flatten()
S_flat = S.flatten()

#%% % of Trips in L and S

# Calculate the sum of all elements in L and S
L_sum = df_L.sum().sum()  # .sum() twice to get the total sum of all cells
S_sum = df_S.sum().sum()

# Decomposed trips (sum of L and S)
decomposed_trips = L_sum + S_sum

# Original trips in the OD matrix
original_trips = df_od_matrix.sum().sum()

# Calculate the percentage of trips in S
S_percent = S_sum / decomposed_trips

# Calculate the difference between original and decomposed trips
trip_difference = original_trips - decomposed_trips
trip_difference_percent = trip_difference/original_trips

# Print the results with formatted strings
print(f'S as a % of Decomposed Trips = {S_percent:.2%}, original trips = {original_trips}')  # Format as percentage

print(f"Trip Difference = {trip_difference} or {trip_difference_percent}%")

#%% Find most common sparse trips
# Find the indices of non-zero elements in the sparse matrix S
non_zero_indices = np.argwhere(S != 0)

# Get the values of the non-zero elements in S
non_zero_values = S[S != 0]

# Create a DataFrame of the non-zero values along with their corresponding origin and destination indices
trip_data = pd.DataFrame(non_zero_indices, columns=["Origin", "Destination"])
trip_data["Flow"] = non_zero_values

# Sort the DataFrame by the flow values (most significant trips first)
trip_data_sorted = trip_data.sort_values(by="Flow", ascending=False)

# Display the top 10 most common trips
print("Top 10 Most Common Trips Based on Sparse Matrix:")
print(trip_data_sorted.head(10))

#%% Find most common L trips
# Find the indices of non-zero elements in the sparse matrix S
non_zero_indices = np.argwhere(L != 0)

# Get the values of the non-zero elements in S
non_zero_values_L = L[L != 0]

# Create a DataFrame of the non-zero values along with their corresponding origin and destination indices
trip_data_L = pd.DataFrame(non_zero_indices, columns=["Origin", "Destination"])
trip_data_L["Flow"] = non_zero_values_L

# Sort the DataFrame by the flow values (most significant trips first)
trip_data_sorted_L = trip_data_L.sort_values(by="Flow", ascending=False)

# Display the top 10 most common trips
print("Top 10 Most Common Trips Based on L Matrix:")
print(trip_data_sorted_L.head(20))

#%% Import
from Cluster import start_cluster_counts
from Cluster import centroids

#%% Fit Distributions to L and S

# Updated function to fit distributions and export AIC results to CSV
def fit_distribution(data, dist_names=['norm', 'expon', 'gamma', 'weibull_min', 'poisson'], matrix_name="L"):
    # Fit multiple distributions and calculate AIC for each
    best_fit = None
    best_aic = np.inf
    results = []

    for dist_name in dist_names:
        dist = getattr(stats, dist_name)
        try:
            if dist_name == 'poisson':
                # Poisson requires integer data
                data_int = np.rint(data).astype(int)  # Round and convert to integer
                lambda_param = np.mean(data_int)  # Poisson distribution has mean as its single parameter
                # Manually calculate log-likelihood for Poisson
                log_likelihood = np.sum(dist.logpmf(data_int, lambda_param))
                aic = 2 * 1 - 2 * log_likelihood  # AIC calculation for Poisson
                params = (lambda_param,)
            else:
                # Fit the distribution using the standard fit method
                params = dist.fit(data)
                log_likelihood = np.sum(dist.logpdf(data, *params))
                k = len(params)  # Number of parameters
                aic = 2 * k - 2 * log_likelihood

            # Append results for each distribution
            results.append({"Distribution": dist_name, "AIC": aic, "Parameters": params})

            # Track the best-fitting distribution
            if aic < best_aic:
                best_aic = aic
                best_fit = (dist_name, params)

        except Exception as e:
            print(f"Error fitting {dist_name}: {e}")
            continue

    # Create a DataFrame from the results and save it as a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{matrix_name}_distribution_fit_AIC.csv", index=False)

    return best_fit, results


#%% Histogram Investigation -  by trip and distance
#%% Calculate cluster-to-cluster distance

def calculate_euclidean_distances(geoseries):

    # Extract x (longitude) and y (latitude) coordinates from the Point objects
    coords = np.array([(point.x, point.y) for point in geoseries])
    
    # Compute pairwise Euclidean distances between coordinates
    distances = cdist(coords, coords, metric='euclidean')
    
    return distances

#%% Combined Histrogram Plot

def initialize_csv(output_csv="aic_results.csv"):
    if os.path.exists(output_csv):
        os.remove(output_csv)  # Delete the existing file
    print(f"{output_csv} has been reset and will start fresh.")

initialize_csv("aic_results.csv")

# Modify the plot_all_histograms_from_stops function
def plot_all_histograms_from_stops(od_matrix, distances, num_stops=21, exclude_stop=13, exclude_stop_flag=False):
    
    # Access the frame of the caller
    frame = inspect.currentframe().f_back
    
    # Get all local variables in the caller's frame
    variables = frame.f_locals
    
    # Find the name of `od_matrix` by checking for reference equality
    od_matrix_name = [name for name, val in variables.items() if val is od_matrix]
    od_matrix_name = od_matrix_name[0] if od_matrix_name else None
    
    # Calculate total number of trips from all clusters
    total_trips = start_cluster_counts.sum()
    
    # Sort clusters by the number of starting trips in descending order
    sorted_clusters = start_cluster_counts.sort_values(ascending=False).index[:num_stops]
    
    # Exclude the specified stop if exclude_stop_flag is True
    if exclude_stop_flag:
        sorted_clusters = [cluster for cluster in sorted_clusters if cluster != exclude_stop]
    
    # Adjust num_stops to reflect the exclusion
    num_stops = min(num_stops, len(sorted_clusters))
    
    # Set up subplots grid: Adjust based on the number of stops
    n_cols = 4  # Number of columns for subplots
    n_rows = (num_stops + n_cols - 1) // n_cols  # Calculate number of rows (compact formula for rounding up)
    
    # Adjust the figure size for an A4 page with normal margins
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 22), constrained_layout=True)
    
    # Flatten axes for easy iteration
    axes = axes.ravel()
    
    # Loop through sorted clusters and plot histograms
    for i, start_stop in enumerate(sorted_clusters):
        start_stop_idx = start_stop - 1
        
        # Extract trips and distances from the specified stop (start_stop_idx)
        trips_from_stop = od_matrix[start_stop_idx, :]
        distances_from_stop = distances[start_stop_idx, :]
        
        # Sort trips and distances by closest distance
        sorted_indices = np.argsort(distances_from_stop)
        sorted_trips = trips_from_stop[sorted_indices]
        
        # Use cluster number for the X-axis labels
        sorted_clusters_dest = sorted_indices + 1  # Adding 1 to make cluster labels 1-based
        
        # Calculate percentage of trips from this start stop
        cluster_trips = start_cluster_counts[start_stop]  # Number of trips from this cluster
        percentage = (cluster_trips / total_trips) * 100

        # Find the best-fitting distribution if there are non-zero trips
        sorted_trips_nonzero = sorted_trips[sorted_trips > 0]
        if len(sorted_trips_nonzero) > 0:
            # Debug statement to confirm the data and stop number
            # print(f"Calling AIC calculation for Stop {start_stop} with data: {sorted_trips_nonzero}")
            
            best_fit, best_aic, best_params = calculate_best_fit_distribution(
                sorted_trips_nonzero, start_stop, output_csv="aic_results.csv", matrix_name=od_matrix_name
            )
        else:
            best_fit = "N/A"
            best_aic = "N/A"
            best_params = None

        
        # Plot histogram in the current subplot
        axes[i].bar(range(len(sorted_trips)), sorted_trips, color='blue', edgecolor='black')
        axes[i].set_title(f'Stop {start_stop} ({percentage:.2f}%)\n(Best Fit: {best_fit}, AIC: {best_aic:.2f})', fontsize=10)
        axes[i].set_xticks(range(len(sorted_clusters_dest)))
        axes[i].set_xticklabels(sorted_clusters_dest, rotation=45, ha='right', fontsize=8)  # Show only cluster numbers
        axes[i].tick_params(axis='both', labelsize=8)  # Adjust label size for readability
        axes[i].yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure integer labels on Y-axis

     # Hide any extra subplots if num_stops < total axes
    for i in range(num_stops, len(axes)):
        axes[i].axis('off')
    
    # Center the last figure by setting a main title
    fig.suptitle(f'Trips from Stops Sorted by Number of Trips (Top {num_stops}, excluding Stop {exclude_stop})', fontsize=16, ha='center')
    plt.show()

# Modify calculate_best_fit_distribution to return parameters
def calculate_best_fit_distribution(trips, start_stop,  output_csv="aic_results.csv", matrix_name="L"):
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
                aic = 2 * len(params) - 2 * loglikelihood

            results.append((dist_name, aic, params))

            if aic < best_aic:
                best_aic = aic
                best_fit = dist_name
                best_params = params

        except Exception as e:
            # print(f"Error fitting {dist_name}: {e}")
            continue

    results.sort(key=lambda x: x[1])
    
        
    # Create a DataFrame for the results, including the matrix name
    results_df = pd.DataFrame(results, columns=["Distribution", "AIC", "Parameters"])
    results_df.insert(0, "Matrix", matrix_name)  # Add matrix name as the first column

    # Add a row for the originating stop and a spacer row
    spacer_row = pd.DataFrame([["", "", "", ""]], columns=["Matrix", "Distribution", "AIC", "Parameters"])
    label_row = pd.DataFrame([[matrix_name, f"Results for Stop {start_stop}", "", ""]], columns=["Matrix", "Distribution", "AIC", "Parameters"])

    # Combine label, results, and spacer into one DataFrame
    final_df = pd.concat([label_row, results_df, spacer_row], ignore_index=True)

    # Append to the CSV file if it exists, else create a new one
    if os.path.exists(output_csv):
        final_df.to_csv(output_csv, mode='a', header=False, index=False)
    else:
        final_df.to_csv(output_csv, index=False)
    
    print(f"AIC results for Stop {start_stop} in matrix {matrix_name} appended to {output_csv}")
    
    return best_fit, best_aic, best_params



#%% Run

distances = calculate_euclidean_distances(centroids)


plot_all_histograms_from_stops(L, distances, num_stops=21)
plot_all_histograms_from_stops(S, distances, num_stops=21)


