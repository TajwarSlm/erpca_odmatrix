#%% Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
from Cluster import start_cluster_counts
from Cluster import centroids


#%% Import OD Matrix
file_path="OD_MATRIX/OD_Clustered_canberra_202309.csv"
df = pd.read_csv(file_path)

#convert to NumPy Array
od_matrix = df.to_numpy()

#%% Flatten Method
#%% Calculate Estimated Rank
def estimate_intrinsic_rank(od_matrix, energy_threshold=0.9, plot=False):
    """
    Estimate the intrinsic rank of a matrix using the Energy Threshold method.
    
    Args:
    od_matrix: 2D numpy array representing the observed matrix.
    energy_threshold: The percentage of energy to be captured by the top `r` singular values (default is 90%).
    plot: Boolean flag to plot singular values and cumulative energy (default is False).
    
    Returns:
    estimated_rank: Estimated rank that captures the specified energy threshold.
    """
    # Step 1: Compute Singular Value Decomposition
    _, singular_values, _ = np.linalg.svd(od_matrix, full_matrices=False)
    
    # Step 2: Calculate Cumulative Energy
    cumulative_energy = np.cumsum(singular_values) / np.sum(singular_values)
    
    # Step 3: Find the minimum rank that captures the desired energy threshold
    estimated_rank = np.searchsorted(cumulative_energy, energy_threshold) + 1  # +1 to account for zero-indexing
    
    # Optional: Plot singular values and cumulative energy
    if plot:
        plt.figure(figsize=(12, 6))

        # Plot Singular Values
        plt.subplot(1, 2, 1)
        plt.plot(singular_values, marker='o')
        plt.title("Singular Values of OD Matrix")
        plt.xlabel("Index")
        plt.ylabel("Singular Value (log scale)")
        plt.yscale('log')
        plt.grid(True)

        # Plot Cumulative Energy
        plt.subplot(1, 2, 2)
        plt.plot(cumulative_energy, marker='o', label="Cumulative Energy")
        plt.axhline(y=energy_threshold, color='r', linestyle='--', label=f"{energy_threshold * 100}% Energy Threshold")
        plt.xlabel("Number of Singular Values")
        plt.ylabel("Cumulative Energy")
        plt.title("Cumulative Energy vs. Number of Singular Values")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    print(f"Estimated Intrinsic Rank: {estimated_rank}")
    return estimated_rank

# Example Usage:
# Assuming `od_matrix` is your numpy array
estimated_rank = estimate_intrinsic_rank(od_matrix, energy_threshold=0.9, plot=True)

#%%
# Function to test multiple distributions on the OD matrix data
def check_distribution(od_matrix):
    # Flatten OD matrix and remove zero values
    od_values = od_matrix.flatten()
    od_values = od_values[od_values > 0]

    # Plot Histogram and Q-Q plot
    plt.figure(figsize=(10, 6))
    sns.histplot(od_values, bins=30, kde=True, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of OD Matrix')
    plt.show()

    stats.probplot(od_values, dist="norm", plot=plt)
    plt.title('Q-Q Plot of OD Matrix')
    plt.show()

    # Shapiro-Wilk Test (keep only 1 normality test)
    shapiro_stat, shapiro_p = stats.shapiro(od_values)
    print(f'Shapiro-Wilk Test: p-value = {shapiro_p:.2f}')

    # Test multiple distributions and calculate AIC
    distributions = ['norm', 'lognorm', 'expon', 'gamma', 'beta', 'weibull_min', 'poisson']
    results = []

    for dist_name in distributions:
        try:
            if dist_name == 'poisson':
                od_values_int = np.rint(od_values).astype(int)
                lambda_param = np.mean(od_values_int)
                loglikelihood = stats.poisson.logpmf(od_values_int, lambda_param).sum()
                aic = 2 * 1 - 2 * loglikelihood  # 1 parameter (lambda)
                params = (lambda_param,)
            elif dist_name == 'bernoulli':
                od_values_bin = (od_values > 0).astype(int)
                p_param = np.mean(od_values_bin)
                loglikelihood = stats.bernoulli.logpmf(od_values_bin, p_param).sum()
                aic = 2 * 1 - 2 * loglikelihood  # 1 parameter (p)
                params = (p_param,)
            else:
                dist = getattr(stats, dist_name)
                params = dist.fit(od_values)
                loglikelihood = dist.logpdf(od_values, *params).sum()
                aic = 2 * len(params) - 2 * loglikelihood

            results.append((dist_name, aic, params))
        except Exception as e:
            print(f"Error fitting {dist_name}: {e}")

    # Sort results by AIC
    results.sort(key=lambda x: x[1])

    # Print sorted AIC results
    print("\nDistribution fitting results (lower AIC is better):")
    print(f"{'Distribution':<15} {'AIC':<10} {'Parameters':<30}")
    for dist_name, aic, params in results:
        print(f"{dist_name:<15} {aic:.2f} {' '.join(f'{p:.2f}' for p in params)}")

    # Display best-fitting distribution
    best_fit = results[0]
    print(f"\nBest fitting distribution: {best_fit[0]} with AIC = {best_fit[1]:.2f}")

    # Plot best-fitting distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(od_values, bins=30, kde=False, color='blue', edgecolor='black', alpha=0.7)

    x = np.linspace(min(od_values), max(od_values), 100)
    if best_fit[0] in ['poisson', 'bernoulli']:
        pmf_fitted = stats.poisson.pmf(x.astype(int), *best_fit[2]) if best_fit[0] == 'poisson' else stats.bernoulli.pmf(x.astype(int), *best_fit[2])
        plt.plot(x, pmf_fitted, 'r-', label=f'Fitted {best_fit[0]}')
    else:
        pdf_fitted = getattr(stats, best_fit[0]).pdf(x, *best_fit[2])
        plt.plot(x, pdf_fitted, 'r-', label=f'Fitted {best_fit[0]}')

    plt.title(f'Best Fitting Distribution: {best_fit[0]}')
    plt.legend()
    plt.show()

    # Bar Chart for AIC values (log scale to handle large differences)
    dist_names, aic_values = zip(*[(r[0], r[1]) for r in results])
    plt.figure(figsize=(10, 6))
    plt.bar(dist_names, aic_values, color='maroon', width=0.4)
    plt.yscale('log')  # Use log scale for y-axis
    plt.xlabel("Distributions")
    plt.ylabel("AIC (Log Scale)")
    plt.title("AIC by Distribution (Log Scale)")
    plt.show()

#%% Check Total Distribution

total_dist = check_distribution(od_matrix)
    
#%% Calculate cluster-to-cluster distance

def calculate_euclidean_distances(geoseries):

    # Extract x (longitude) and y (latitude) coordinates from the Point objects
    coords = np.array([(point.x, point.y) for point in geoseries])
    
    # Compute pairwise Euclidean distances between coordinates
    distances = cdist(coords, coords, metric='euclidean')
    
    return distances


#%% Histrogram from Dest 1

def calculate_best_fit_distribution(trips):

    # Include all relevant distributions: both discrete and continuous
    distributions = ['norm', 'lognorm', 'expon', 'gamma', 'beta', 'weibull_min', 'poisson']
    best_fit = None
    best_aic = np.inf
    results = []

    for dist_name in distributions:
        try:
            if dist_name == 'poisson':
                # Poisson requires integer data
                trips_int = np.rint(trips).astype(int)  # Round and convert to integer
                lambda_param = np.mean(trips_int)
                loglikelihood = stats.poisson.logpmf(trips_int, lambda_param).sum()
                aic = 2 * 1 - 2 * loglikelihood  # Poisson has 1 parameter (lambda)
                params = (lambda_param,)
            elif dist_name == 'bernoulli':
                # Bernoulli requires binary data (0 or 1)
                trips_bin = (trips > 0).astype(int)  # Convert to binary
                p_param = np.mean(trips_bin)
                loglikelihood = stats.bernoulli.logpmf(trips_bin, p_param).sum()
                aic = 2 * 1 - 2 * loglikelihood  # Bernoulli has 1 parameter (p)
                params = (p_param,)
            else:
                # Continuous distributions
                dist = getattr(stats, dist_name)
                params = dist.fit(trips)
                loglikelihood = dist.logpdf(trips, *params).sum()
                aic = 2 * len(params) - 2 * loglikelihood

            results.append((dist_name, aic, params))

            if aic < best_aic:
                best_aic = aic
                best_fit = dist_name

        except Exception as e:
            print(f"Error fitting {dist_name}: {e}")
            continue  # Skip this distribution if there's an error

    # Sort results by AIC
    results.sort(key=lambda x: x[1])

    # Print sorted AIC results
    print("\nDistribution fitting results (lower AIC is better):")
    print(f"{'Distribution':<15} {'AIC':<10} {'Parameters':<30}")
    for dist_name, aic, params in results:
        print(f"{dist_name:<15} {aic:.2f} {' '.join(f'{p:.2f}' for p in params)}")

    # Return the best-fitting distribution name and its AIC
    return best_fit, best_aic

#%% Combined Histrogram Plot
def plot_histogram_from_stop(od_matrix, distances, start_stop=1):
    """
    Plot a histogram for trips from the specified stop to all other stops,
    including the best-fitting distribution based on AIC.
    
    Args:
    od_matrix: 2D numpy array where each element (i, j) is the number of trips from stop i to stop j.
    distances: 2D numpy array of distances between all stops.
    start_stop: The stop index to plot from (1-based index).
    """
    
    # Convert 1-based index to 0-based index for accessing arrays
    start_stop_idx = start_stop - 1
    
    # Extract trips and distances from the specified stop (start_stop_idx)
    trips_from_stop = od_matrix[start_stop_idx, :]
    distances_from_stop = distances[start_stop_idx, :]
    
    # Define destination clusters (1-based indexing for cluster labeling)
    destination_clusters = np.arange(1, len(trips_from_stop) + 1)
    
    # Sort the destination clusters, trips, and distances by distance (closest to farthest)
    sorted_indices = np.argsort(distances_from_stop)
    sorted_clusters = destination_clusters[sorted_indices]
    sorted_trips = trips_from_stop[sorted_indices]
    sorted_distances = distances_from_stop[sorted_indices]

    # Remove zero-trip destinations for better distribution fitting
    sorted_trips_nonzero = sorted_trips[sorted_trips > 0]

    # Find the best-fitting distribution if there are non-zero trips
    if len(sorted_trips_nonzero) > 0:
        best_fit, best_aic = calculate_best_fit_distribution(sorted_trips_nonzero)
    else:
        best_fit = "N/A"
        best_aic = "N/A"

    # Plot the histogram for trips from the specified stop to all other stops
    fig, ax = plt.subplots(figsize=(12, 6))  # Increase figure size for better fit
    
    # Create a bar plot with destination clusters on X-axis
    bars = ax.bar(np.arange(len(sorted_clusters)), sorted_trips, color='blue', edgecolor='black')
    
    # Annotate each bar with the corresponding distance
    for bar, distance in zip(bars, sorted_distances):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{distance:.2g} km', ha='center', va='bottom', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Destination Cluster (sorted by distance)')
    ax.set_ylabel('Number of Trips')
    ax.set_title(f'Trips from Stop {start_stop} to Other Stops (Best Fit: {best_fit}, AIC: {best_aic:.2f})')
    
    # Set X-axis tick labels as cluster numbers only (no distance)
    ax.set_xticks(np.arange(len(sorted_clusters)))
    ax.set_xticklabels(sorted_clusters, rotation=45, ha='right', fontsize=10)
    
    # Ensure that the Y-axis displays integer labels only
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Show the plot
    plt.tight_layout()
    plt.show()

#%% Apply to all Stops with AIC included

def plot_all_histograms_from_stops(od_matrix, distances, num_stops=21):
    """
    Plot histograms for trips from all stops, sorted by the number of trips, 
    and show the best-fitting distribution based on AIC for each.
    
    Args:
    od_matrix: 2D numpy array (OD matrix)
    distances: 2D numpy array (Distance matrix)
    num_stops: Number of stops to plot (default is 21)
    """

    # Calculate total number of trips from all clusters
    total_trips = start_cluster_counts.sum()
    
    # Sort clusters by the number of starting trips in descending order
    sorted_clusters = start_cluster_counts.sort_values(ascending=False).index[:num_stops]
    
    # Set up subplots grid: Adjust based on the number of stops
    n_cols = 7  # Number of columns for subplots
    n_rows = (num_stops + n_cols - 1) // n_cols  # Calculate number of rows (compact formula for rounding up)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12), constrained_layout=True)
    
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
            best_fit, best_aic = calculate_best_fit_distribution(sorted_trips_nonzero)
        else:
            best_fit = "N/A"
            best_aic = "N/A"
        
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
 
    # Set a main title for the figure
    fig.suptitle(f'Trips from Stops Sorted by Number of Trips (Top {num_stops})', fontsize=16)
    plt.show()

#%% Test

distances = calculate_euclidean_distances(centroids)

plot_histogram_from_stop(od_matrix, distances, start_stop=1)
plot_all_histograms_from_stops(od_matrix, distances, num_stops=21)


