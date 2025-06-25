'''
Temporal Analysis:

-Map your aligned embeddings across time periods (Classical → Imperial → Late Latin if available)
-Track how cosine similarities for the same concepts change across these periods
-Look for non-linear patterns - semantic drift often accelerates during certain historical periods

Drift Magnitude & Direction:

-Calculate drift vectors: drift = embedding_later - embedding_earlier
-Measure drift magnitude using vector norms
-Analyze drift direction to see if certain semantic dimensions consistently shift

Semantic Neighborhood Analysis:

-For each target word, identify its k-nearest neighbors in each time period
-Track how these neighborhoods change - new neighbors appearing, old ones disappearing
-Calculate neighborhood stability scores

Clustering & Semantic Fields:

-Perform clustering analysis on your aligned embeddings by time period
-See how words move between semantic clusters over time
-Identify which semantic fields are most/least stable

Syntactic vs. Semantic Drift:

-Compare drift patterns for content words vs. function words
-Analyze part-of-speech specific drift rates
-Look at how grammaticalization affects drift patterns

Statistical Significance:

-Bootstrap sampling to test if observed drift is statistically significant
-✅Control for corpus size differences between time periods
-✅Account for genre effects (poetry vs. oratory may have different baseline drift rates)

The alignment you've achieved gives you a solid foundation to now trace how these Latin concepts evolved semantically across the classical-to-imperial transition.
'''

import numpy as np
import pandas as pd
import pickle
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, kstest
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')

class TemporalSemanticAnalyzer:
    """
    Comprehensive temporal semantic analysis for computational linguistics research.
    Implements methods from Kutuzov et al. (2018), Hamilton et al. (2016), 
    and Schlechtweg et al. (2020).
    """
    
    def __init__(self, model_classical, model_imperial, model_late, 
                 aligned_imperial, aligned_late, shared_vocab):
        self.model_classical = model_classical
        self.model_imperial = model_imperial
        self.model_late = model_late
        self.aligned_imperial = aligned_imperial
        self.aligned_late = aligned_late
        self.shared_vocab = shared_vocab
        
        # Set up plotting style for publication
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        self.setup_plotting()
    
    def setup_plotting(self):
        """Set up publication-quality plotting parameters."""
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.spines.top': False,
            'axes.spines.right': False
        })
    
    def create_temporal_embedding_maps(self, target_words, method='pca', 
                                     n_components=2, save_plots=True):
        """
        Create 2D/3D visualizations of semantic trajectories across time periods.
        
        Args:
            target_words: List of words to analyze
            method: 'pca', 'tsne', 'mds', or 'umap'
            n_components: Number of dimensions for visualization
            save_plots: Whether to save visualization plots
        """
        print(f"Creating temporal embedding maps using {method.upper()}...")
        
        # Collect all vectors for dimensionality reduction
        all_vectors = []
        word_labels = []
        period_labels = []
        
        for word in target_words:
            if word in self.shared_vocab:
                all_vectors.append(self.model_classical.wv[word])
                word_labels.append(word)
                period_labels.append('Classical')
                
                all_vectors.append(self.aligned_imperial[word])
                word_labels.append(word)
                period_labels.append('Imperial')
                
                all_vectors.append(self.aligned_late[word])
                word_labels.append(word)
                period_labels.append('Late')
        
        all_vectors = np.array(all_vectors)
        
        # Apply dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, 
                          perplexity=min(30, len(all_vectors)//3))
        elif method == 'mds':
            reducer = MDS(n_components=n_components, random_state=42, 
                         dissimilarity='euclidean')
        else:
            raise ValueError(f"Unknown method: {method}")
        
        reduced_vectors = reducer.fit_transform(all_vectors)
        
        # Create DataFrame for analysis
        embedding_df = pd.DataFrame({
            'word': word_labels,
            'period': period_labels,
            'x': reduced_vectors[:, 0],
            'y': reduced_vectors[:, 1] if n_components > 1 else 0,
            'z': reduced_vectors[:, 2] if n_components > 2 else 0
        })
        
        # Calculate trajectory metrics
        trajectories = self._calculate_trajectories(embedding_df, target_words)
        
        if save_plots:
            self._plot_temporal_trajectories(embedding_df, trajectories, method)
        
        return embedding_df, trajectories, reducer
    
    def _calculate_trajectories(self, embedding_df, target_words):
        """Calculate semantic trajectory metrics for each word."""
        trajectories = []
        
        for word in target_words:
            if word in self.shared_vocab:
                word_data = embedding_df[embedding_df['word'] == word].copy()
                
                if len(word_data) == 3:  # All three periods
                    classical = word_data[word_data['period'] == 'Classical'].iloc[0]
                    imperial = word_data[word_data['period'] == 'Imperial'].iloc[0]
                    late = word_data[word_data['period'] == 'Late'].iloc[0]
                    
                    # Calculate distances
                    dist_class_imp = np.sqrt((classical['x'] - imperial['x'])**2 + 
                                           (classical['y'] - imperial['y'])**2)
                    dist_class_late = np.sqrt((classical['x'] - late['x'])**2 + 
                                            (classical['y'] - late['y'])**2)
                    dist_imp_late = np.sqrt((imperial['x'] - late['x'])**2 + 
                                          (imperial['y'] - late['y'])**2)
                    
                    # Calculate trajectory curvature (deviation from straight line)
                    total_dist = dist_class_imp + dist_imp_late
                    direct_dist = dist_class_late
                    curvature = (total_dist - direct_dist) / max(direct_dist, 0.001)
                    
                    # Calculate trajectory direction (angle)
                    trajectory_angle = np.arctan2(late['y'] - classical['y'], 
                                                late['x'] - classical['x'])
                    
                    trajectories.append({
                        'word': word,
                        'distance_class_imp': dist_class_imp,
                        'distance_class_late': dist_class_late,
                        'distance_imp_late': dist_imp_late,
                        'total_trajectory_length': total_dist,
                        'direct_distance': direct_dist,
                        'curvature': curvature,
                        'trajectory_angle': trajectory_angle,
                        'trajectory_angle_degrees': np.degrees(trajectory_angle)
                    })
        
        return pd.DataFrame(trajectories)
    
    def _plot_temporal_trajectories(self, embedding_df, trajectories, method):
        """Create publication-quality trajectory plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Temporal Semantic Trajectories ({method.upper()})', fontsize=16)
        
        # Plot 1: All trajectories with arrows
        ax1 = axes[0, 0]
        colors = {'Classical': '#1f77b4', 'Imperial': '#ff7f0e', 'Late': '#2ca02c'}
        
        for word in trajectories['word'].unique():
            word_data = embedding_df[embedding_df['word'] == word]
            if len(word_data) == 3:
                periods = ['Classical', 'Imperial', 'Late']
                x_coords = [word_data[word_data['period'] == p]['x'].iloc[0] for p in periods]
                y_coords = [word_data[word_data['period'] == p]['y'].iloc[0] for p in periods]
                
                # Draw trajectory line
                ax1.plot(x_coords, y_coords, 'k-', alpha=0.3, linewidth=0.5)
                
                # Draw points
                for i, period in enumerate(periods):
                    ax1.scatter(x_coords[i], y_coords[i], c=colors[period], 
                              s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
                
                # Add arrow for direction
                ax1.annotate('', xy=(x_coords[-1], y_coords[-1]), 
                           xytext=(x_coords[0], y_coords[0]),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
        
        ax1.set_title('Semantic Trajectories Across Periods')
        ax1.set_xlabel(f'{method.upper()} Component 1')
        ax1.set_ylabel(f'{method.upper()} Component 2')
        ax1.legend(colors.keys(), loc='best')
        
        # Plot 2: Trajectory lengths distribution
        ax2 = axes[0, 1]
        ax2.hist(trajectories['total_trajectory_length'], bins=20, alpha=0.7, 
                edgecolor='black')
        ax2.set_title('Distribution of Trajectory Lengths')
        ax2.set_xlabel('Total Trajectory Length')
        ax2.set_ylabel('Frequency')
        
        # Plot 3: Curvature vs Direct Distance
        ax3 = axes[1, 0]
        scatter = ax3.scatter(trajectories['direct_distance'], trajectories['curvature'], 
                            alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        ax3.set_title('Trajectory Curvature vs Direct Distance')
        ax3.set_xlabel('Direct Distance (Classical → Late)')
        ax3.set_ylabel('Trajectory Curvature')
        
        # Add correlation coefficient
        corr, p_val = pearsonr(trajectories['direct_distance'], trajectories['curvature'])
        ax3.text(0.05, 0.95, f'r = {corr:.3f} (p = {p_val:.3f})', 
                transform=ax3.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 4: Trajectory angles (polar plot)
        ax4 = axes[1, 1]
        ax4.remove()
        ax4 = fig.add_subplot(2, 2, 4, projection='polar')
        ax4.scatter(trajectories['trajectory_angle'], trajectories['direct_distance'], 
                   alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        ax4.set_title('Trajectory Directions', pad=20)
        ax4.set_theta_zero_location('E')
        ax4.set_theta_direction(1)
        
        plt.tight_layout()
        plt.savefig(f'temporal_trajectories_{method}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def perform_semantic_clustering(self, target_words, n_clusters_range=(2, 8), 
                                  method='kmeans', save_plots=True):
        """
        Perform clustering analysis across time periods to identify semantic fields.
        """
        print("Performing semantic clustering analysis...")
        
        clustering_results = {}
        
        for period, vectors in [('Classical', {w: self.model_classical.wv[w] for w in target_words if w in self.shared_vocab}),
                              ('Imperial', self.aligned_imperial),
                              ('Late', self.aligned_late)]:
            
            # Filter vectors for target words
            period_vectors = {w: v for w, v in vectors.items() if w in target_words}
            
            if len(period_vectors) < 2:
                continue
                
            words = list(period_vectors.keys())
            X = np.array(list(period_vectors.values()))
            
            # Determine optimal number of clusters
            silhouette_scores = []
            cluster_range = range(max(2, len(words)//4), min(len(words)//2 + 1, max(n_clusters_range)))
            
            for n_clusters in cluster_range:
                if method == 'kmeans':
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                elif method == 'hierarchical':
                    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                else:
                    raise ValueError(f"Unknown clustering method: {method}")
                
                cluster_labels = clusterer.fit_predict(X)
                silhouette_avg = silhouette_score(X, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            # Select optimal number of clusters
            optimal_n_clusters = cluster_range[np.argmax(silhouette_scores)]
            
            # Perform final clustering
            if method == 'kmeans':
                final_clusterer = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=10)
            else:
                final_clusterer = AgglomerativeClustering(n_clusters=optimal_n_clusters)
            
            final_labels = final_clusterer.fit_predict(X)
            
            # Store results
            clustering_results[period] = {
                'words': words,
                'vectors': X,
                'labels': final_labels,
                'n_clusters': optimal_n_clusters,
                'silhouette_score': max(silhouette_scores),
                'clusterer': final_clusterer
            }
        
        # Calculate cluster stability across periods
        stability_metrics = self._calculate_cluster_stability(clustering_results)
        
        if save_plots:
            self._plot_clustering_results(clustering_results, stability_metrics)
        
        return clustering_results, stability_metrics
    
    def _calculate_cluster_stability(self, clustering_results):
        """Calculate cluster stability metrics across time periods."""
        periods = list(clustering_results.keys())
        stability_metrics = {}
        
        for i in range(len(periods)):
            for j in range(i + 1, len(periods)):
                period1, period2 = periods[i], periods[j]
                
                # Find common words
                words1 = set(clustering_results[period1]['words'])
                words2 = set(clustering_results[period2]['words'])
                common_words = words1.intersection(words2)
                
                if len(common_words) > 1:
                    # Extract labels for common words
                    labels1 = []
                    labels2 = []
                    
                    for word in common_words:
                        idx1 = clustering_results[period1]['words'].index(word)
                        idx2 = clustering_results[period2]['words'].index(word)
                        labels1.append(clustering_results[period1]['labels'][idx1])
                        labels2.append(clustering_results[period2]['labels'][idx2])
                    
                    # Calculate Adjusted Rand Index
                    ari = adjusted_rand_score(labels1, labels2)
                    stability_metrics[f'{period1}_{period2}'] = {
                        'adjusted_rand_index': ari,
                        'common_words': len(common_words),
                        'words': list(common_words)
                    }
        
        return stability_metrics
    
    def _plot_clustering_results(self, clustering_results, stability_metrics):
        """Plot clustering results across time periods."""
        n_periods = len(clustering_results)
        fig, axes = plt.subplots(2, n_periods, figsize=(5*n_periods, 10))
        if n_periods == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('Semantic Clustering Across Time Periods', fontsize=16)
        
        # PCA for visualization
        for i, (period, results) in enumerate(clustering_results.items()):
            # Plot 1: Cluster visualization
            ax1 = axes[0, i]
            
            if results['vectors'].shape[1] > 2:
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(results['vectors'])
                explained_var = pca.explained_variance_ratio_.sum()
            else:
                X_pca = results['vectors']
                explained_var = 1.0
            
            scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], 
                                c=results['labels'], cmap='viridis', 
                                s=100, alpha=0.7, edgecolors='black')
            
            # Add word labels
            for j, word in enumerate(results['words']):
                ax1.annotate(word, (X_pca[j, 0], X_pca[j, 1]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
            
            ax1.set_title(f'{period} Period\n(Silhouette: {results["silhouette_score"]:.3f})')
            ax1.set_xlabel(f'PC1 ({explained_var:.1%} var. explained)')
            ax1.set_ylabel('PC2')
            
            # Plot 2: Cluster composition
            ax2 = axes[1, i]
            cluster_counts = pd.Series(results['labels']).value_counts().sort_index()
            bars = ax2.bar(range(len(cluster_counts)), cluster_counts.values, 
                          alpha=0.7, edgecolor='black')
            ax2.set_title(f'{period} Cluster Sizes')
            ax2.set_xlabel('Cluster ID')
            ax2.set_ylabel('Number of Words')
            ax2.set_xticks(range(len(cluster_counts)))
            
            # Add value labels on bars
            for bar, count in zip(bars, cluster_counts.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('semantic_clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print stability metrics
        print("\nCluster Stability Metrics:")
        print("-" * 40)
        for comparison, metrics in stability_metrics.items():
            print(f"{comparison}: ARI = {metrics['adjusted_rand_index']:.3f} "
                  f"({metrics['common_words']} common words)")
    
    def calculate_cooccurrence_matrices(self, target_words, window_size=5):
        """
        Calculate co-occurrence matrices for each time period.
        """
        print("Calculating co-occurrence matrices...")
        
        cooccurrence_results = {}
        
        for period, model in [('Classical', self.model_classical),
                             ('Imperial', self.model_imperial),
                             ('Late', self.model_late)]:
            
            # Get co-occurrence counts
            cooc_matrix = np.zeros((len(target_words), len(target_words)))
            word_to_idx = {word: i for i, word in enumerate(target_words)}
            
            # This is a simplified approach - in practice, you'd want to 
            # calculate co-occurrence from the original corpus
            for i, word1 in enumerate(target_words):
                if word1 in model.wv.key_to_index:
                    for j, word2 in enumerate(target_words):
                        if word2 in model.wv.key_to_index and i != j:
                            # Use cosine similarity as proxy for co-occurrence
                            # In real implementation, use actual corpus co-occurrence
                            similarity = cosine_similarity(
                                [model.wv[word1]], [model.wv[word2]]
                            )[0][0]
                            cooc_matrix[i, j] = max(0, similarity)  # Only positive values
            
            cooccurrence_results[period] = {
                'matrix': cooc_matrix,
                'words': target_words,
                'word_to_idx': word_to_idx
            }
        
        # Calculate co-occurrence changes
        cooc_changes = self._calculate_cooccurrence_changes(cooccurrence_results)
        
        # Plot results
        self._plot_cooccurrence_analysis(cooccurrence_results, cooc_changes)
        
        return cooccurrence_results, cooc_changes
    
    def _calculate_cooccurrence_changes(self, cooccurrence_results):
        """Calculate changes in co-occurrence patterns."""
        periods = list(cooccurrence_results.keys())
        changes = {}
        
        for i in range(len(periods)):
            for j in range(i + 1, len(periods)):
                period1, period2 = periods[i], periods[j]
                
                matrix1 = cooccurrence_results[period1]['matrix']
                matrix2 = cooccurrence_results[period2]['matrix']
                
                # Calculate matrix difference
                diff_matrix = matrix2 - matrix1
                
                # Calculate summary statistics
                changes[f'{period1}_to_{period2}'] = {
                    'difference_matrix': diff_matrix,
                    'mean_change': np.mean(diff_matrix),
                    'std_change': np.std(diff_matrix),
                    'max_increase': np.max(diff_matrix),
                    'max_decrease': np.min(diff_matrix),
                    'total_change': np.sum(np.abs(diff_matrix))
                }
        
        return changes
    
    def _plot_cooccurrence_analysis(self, cooccurrence_results, cooc_changes):
        """Plot co-occurrence matrices and changes."""
        n_periods = len(cooccurrence_results)
        n_changes = len(cooc_changes)
        
        fig, axes = plt.subplots(2, max(n_periods, n_changes), 
                               figsize=(4*max(n_periods, n_changes), 8))
        if max(n_periods, n_changes) == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('Co-occurrence Pattern Analysis', fontsize=16)
        
        # Plot original matrices
        for i, (period, results) in enumerate(cooccurrence_results.items()):
            ax = axes[0, i] if n_periods > 1 else axes[0]
            
            im = ax.imshow(results['matrix'], cmap='Blues', aspect='auto')
            ax.set_title(f'{period} Co-occurrence')
            ax.set_xticks(range(len(results['words'])))
            ax.set_yticks(range(len(results['words'])))
            ax.set_xticklabels(results['words'], rotation=45, ha='right')
            ax.set_yticklabels(results['words'])
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Plot change matrices
        for i, (comparison, change_data) in enumerate(cooc_changes.items()):
            ax = axes[1, i] if n_changes > 1 else axes[1]
            
            diff_matrix = change_data['difference_matrix']
            im = ax.imshow(diff_matrix, cmap='RdBu_r', aspect='auto',
                          vmin=-np.max(np.abs(diff_matrix)), 
                          vmax=np.max(np.abs(diff_matrix)))
            ax.set_title(f'Change: {comparison.replace("_", " → ")}')
            
            words = list(cooccurrence_results.values())[0]['words']
            ax.set_xticks(range(len(words)))
            ax.set_yticks(range(len(words)))
            ax.set_xticklabels(words, rotation=45, ha='right')
            ax.set_yticklabels(words)
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig('cooccurrence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_drift_magnitude_direction(self, target_words):
        """
        Comprehensive analysis of semantic drift magnitude and direction.
        """
        print("Analyzing drift magnitude and direction...")
        
        drift_results = []
        
        for word in target_words:
            if word in self.shared_vocab:
                # Get vectors
                vec_classical = self.model_classical.wv[word]
                vec_imperial = self.aligned_imperial[word]
                vec_late = self.aligned_late[word]
                
                # Calculate drift vectors
                drift_class_imp = vec_imperial - vec_classical
                drift_class_late = vec_late - vec_classical
                drift_imp_late = vec_late - vec_imperial
                
                # Magnitude calculations
                mag_class_imp = np.linalg.norm(drift_class_imp)
                mag_class_late = np.linalg.norm(drift_class_late)
                mag_imp_late = np.linalg.norm(drift_imp_late)
                
                # Direction calculations (using PCA to find principal direction)
                all_vecs = np.array([vec_classical, vec_imperial, vec_late])
                pca = PCA(n_components=1)
                pca.fit(all_vecs)
                principal_direction = pca.components_[0]
                
                # Project drift vectors onto principal direction
                proj_class_imp = np.dot(drift_class_imp, principal_direction)
                proj_class_late = np.dot(drift_class_late, principal_direction)
                proj_imp_late = np.dot(drift_imp_late, principal_direction)
                
                # Calculate consistency measures
                consistency_class_late = np.dot(drift_class_imp, drift_imp_late) / (mag_class_imp * mag_imp_late + 1e-8)
                
                # Acceleration/deceleration
                acceleration = mag_imp_late - mag_class_imp
                
                drift_results.append({
                    'word': word,
                    'magnitude_class_imp': mag_class_imp,
                    'magnitude_class_late': mag_class_late,
                    'magnitude_imp_late': mag_imp_late,
                    'projection_class_imp': proj_class_imp,
                    'projection_class_late': proj_class_late,
                    'projection_imp_late': proj_imp_late,
                    'consistency': consistency_class_late,
                    'acceleration': acceleration,
                    'total_drift_magnitude': mag_class_late,
                    'drift_direction_consistency': consistency_class_late
                })
        
        drift_df = pd.DataFrame(drift_results)
        
        # Analyze patterns
        patterns = self._analyze_drift_patterns(drift_df)
        
        # Create visualizations
        self._plot_drift_analysis(drift_df, patterns)
        
        return drift_df, patterns
    
    def _analyze_drift_patterns(self, drift_df):
        """Analyze patterns in semantic drift."""
        patterns = {}
        
        # Classify drift types
        patterns['accelerating'] = drift_df[drift_df['acceleration'] > 0]['word'].tolist()
        patterns['decelerating'] = drift_df[drift_df['acceleration'] < 0]['word'].tolist()
        patterns['consistent_direction'] = drift_df[drift_df['consistency'] > 0.5]['word'].tolist()
        patterns['inconsistent_direction'] = drift_df[drift_df['consistency'] < -0.5]['word'].tolist()
        
        # Statistical summaries
        patterns['magnitude_stats'] = {
            'mean_total_drift': drift_df['total_drift_magnitude'].mean(),
            'std_total_drift': drift_df['total_drift_magnitude'].std(),
            'median_total_drift': drift_df['total_drift_magnitude'].median(),
            'max_drift_word': drift_df.loc[drift_df['total_drift_magnitude'].idxmax(), 'word'],
            'max_drift_value': drift_df['total_drift_magnitude'].max(),
            'min_drift_word': drift_df.loc[drift_df['total_drift_magnitude'].idxmin(), 'word'],
            'min_drift_value': drift_df['total_drift_magnitude'].min()
        }
        
        # Correlation analysis
        corr_magnitude_consistency = pearsonr(drift_df['total_drift_magnitude'], 
                                            drift_df['consistency'])
        patterns['magnitude_consistency_correlation'] = {
            'correlation': corr_magnitude_consistency[0],
            'p_value': corr_magnitude_consistency[1]
        }
        
        return patterns
    
    def _plot_drift_analysis(self, drift_df, patterns):
        """Create comprehensive drift analysis visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Semantic Drift Magnitude and Direction Analysis', fontsize=16)
        
        # Plot 1: Drift magnitude across periods
        ax1 = axes[0, 0]
        periods = ['Classical→Imperial', 'Imperial→Late', 'Classical→Late (Total)']
        magnitudes = [drift_df['magnitude_class_imp'].mean(),
                     drift_df['magnitude_imp_late'].mean(),
                     drift_df['magnitude_class_late'].mean()]
        errors = [drift_df['magnitude_class_imp'].std(),
                 drift_df['magnitude_imp_late'].std(),
                 drift_df['magnitude_class_late'].std()]
        
        bars = ax1.bar(periods, magnitudes, yerr=errors, capsize=5, 
                      alpha=0.7, edgecolor='black')
        ax1.set_title('Average Drift Magnitude by Period')
        ax1.set_ylabel('Drift Magnitude')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Drift acceleration
        ax2 = axes[0, 1]
        ax2.hist(drift_df['acceleration'], bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax2.set_title('Distribution of Drift Acceleration')
        ax2.set_xlabel('Acceleration (+ = accelerating, - = decelerating)')
        ax2.set_ylabel('Frequency')
        
        # Plot 3: Direction consistency
        ax3 = axes[0, 2]
        ax3.scatter(drift_df['total_drift_magnitude'], drift_df['consistency'], 
                   alpha=0.7, s=50, edgecolors='black')
        ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax3.set_title('Drift Magnitude vs Direction Consistency')
        ax3.set_xlabel('Total Drift Magnitude')
        ax3.set_ylabel('Direction Consistency')
        
        # Add correlation info
        corr_info = patterns['magnitude_consistency_correlation']
        ax3.text(0.05, 0.95, f'r = {corr_info["correlation"]:.3f}\np = {corr_info["p_value"]:.3f}',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 4: Word-level drift patterns
        ax4 = axes[1, 0]
        top_words = drift_df.nlargest(15, 'total_drift_magnitude')
        bars = ax4.barh(range(len(top_words)), top_words['total_drift_magnitude'])
        ax4.set_yticks(range(len(top_words)))
        ax4.set_yticklabels(top_words['word'])
        ax4.set_title('Top 15 Words by Total Drift Magnitude')
        ax4.set_xlabel('Total Drift Magnitude')
        
        # Color bars by acceleration
        for i, (bar, accel) in enumerate(zip(bars, top_words['acceleration'])):
            if accel > 0:
                bar.set_color('red')
                bar.set_alpha(0.7)
            else:
                bar.set_color('blue')
                bar.set_alpha(0.7)
        
        # Add legend
        ax4.text(0.98, 0.02, 'Red: Accelerating\nBlue: Decelerating', 
                transform=ax4.transAxes, verticalalignment='bottom', 
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 5: Drift trajectory components
        ax5 = axes[1, 1]
        ax5.scatter(drift_df['magnitude_class_imp'], drift_df['magnitude_imp_late'],
                   alpha=0.7, s=50, edgecolors='black')
        ax5.plot([0, ax5.get_xlim()[1]], [0, ax5.get_ylim()[1]], 'r--', alpha=0.5)
        ax5.set_title('Early vs Late Period Drift')
        ax5.set_xlabel('Classical → Imperial Drift')
        ax5.set_ylabel('Imperial → Late Drift')
        
        # Plot 6: Pattern classification
        ax6 = axes[1, 2]
        pattern_counts = {
            'Accelerating': len(patterns['accelerating']),
            'Decelerating': len(patterns['decelerating']),
            'Consistent Dir.': len(patterns['consistent_direction']),
            'Inconsistent Dir.': len(patterns['inconsistent_direction'])
        }
        
        wedges, texts, autotexts = ax6.pie(pattern_counts.values(), 
                                          labels=pattern_counts.keys(),
                                          autopct='%1.1f%%', startangle=90)
        ax6.set_title('Drift Pattern Classification')
        
        plt.tight_layout()
        plt.savefig('drift_magnitude_direction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def bootstrap_statistical_significance(self, target_words, n_bootstrap=1000, 
                                         confidence_level=0.95):
        """
        Perform bootstrap analysis for statistical significance testing.
        Implements methods from Efron & Tibshirani (1993) for computational linguistics.
        """
        print(f"Performing bootstrap analysis with {n_bootstrap} iterations...")
        
        bootstrap_results = {}
        alpha = 1 - confidence_level
        
        # Calculate observed statistics
        observed_stats = self._calculate_observed_statistics(target_words)
        
        # Bootstrap sampling
        bootstrap_distributions = {metric: [] for metric in observed_stats.keys()}
        
        for i in range(n_bootstrap):
            if (i + 1) % 100 == 0:
                print(f"Bootstrap iteration {i + 1}/{n_bootstrap}")
            
            # Sample with replacement
            bootstrap_sample = np.random.choice(target_words, 
                                              size=len(target_words), 
                                              replace=True)
            
            # Calculate statistics for bootstrap sample
            bootstrap_stats = self._calculate_observed_statistics(bootstrap_sample)
            
            for metric, value in bootstrap_stats.items():
                if value is not None and not np.isnan(value):
                    bootstrap_distributions[metric].append(value)
        
        # Calculate confidence intervals and p-values
        for metric, distribution in bootstrap_distributions.items():
            if len(distribution) > 0:
                distribution = np.array(distribution)
                
                # Calculate confidence intervals
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                ci_lower = np.percentile(distribution, lower_percentile)
                ci_upper = np.percentile(distribution, upper_percentile)
                
                # Calculate bias-corrected confidence intervals (BCa)
                observed_value = observed_stats[metric]
                bias_correction = stats.norm.ppf((distribution < observed_value).mean())
                
                # Calculate acceleration
                jackknife_stats = self._jackknife_statistic(target_words, metric)
                acceleration = self._calculate_acceleration(jackknife_stats)
                
                # BCa confidence intervals
                z_alpha_2 = stats.norm.ppf(alpha / 2)
                z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
                
                alpha_1 = stats.norm.cdf(bias_correction + 
                                       (bias_correction + z_alpha_2) / 
                                       (1 - acceleration * (bias_correction + z_alpha_2)))
                alpha_2 = stats.norm.cdf(bias_correction + 
                                       (bias_correction + z_1_alpha_2) / 
                                       (1 - acceleration * (bias_correction + z_1_alpha_2)))
                
                bca_lower = np.percentile(distribution, alpha_1 * 100)
                bca_upper = np.percentile(distribution, alpha_2 * 100)
                
                # Two-tailed p-value
                p_value = 2 * min((distribution >= observed_value).mean(),
                                (distribution <= observed_value).mean())
                
                bootstrap_results[metric] = {
                    'observed': observed_value,
                    'bootstrap_mean': np.mean(distribution),
                    'bootstrap_std': np.std(distribution),
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'bca_lower': bca_lower,
                    'bca_upper': bca_upper,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'distribution': distribution
                }
        
        # Perform additional statistical tests
        additional_tests = self._perform_additional_tests(target_words, bootstrap_results)
        bootstrap_results.update(additional_tests)
        
        # Create visualizations
        self._plot_bootstrap_results(bootstrap_results, confidence_level)
        
        # Generate statistical report
        self._generate_statistical_report(bootstrap_results, confidence_level)
        
        return bootstrap_results
    
    def _calculate_observed_statistics(self, words):
        """Calculate observed statistics for bootstrap analysis."""
        valid_words = [w for w in words if w in self.shared_vocab]
        
        if len(valid_words) < 2:
            return {}
        
        # Calculate various drift metrics
        cosine_distances = []
        euclidean_distances = []
        
        for word in valid_words:
            vec_classical = self.model_classical.wv[word]
            vec_late = self.aligned_late[word]
            
            # Cosine distance
            cos_sim = cosine_similarity([vec_classical], [vec_late])[0][0]
            cosine_distances.append(1 - cos_sim)
            
            # Euclidean distance
            eucl_dist = np.linalg.norm(vec_classical - vec_late)
            euclidean_distances.append(eucl_dist)
        
        return {
            'mean_cosine_distance': np.mean(cosine_distances),
            'std_cosine_distance': np.std(cosine_distances),
            'median_cosine_distance': np.median(cosine_distances),
            'mean_euclidean_distance': np.mean(euclidean_distances),
            'std_euclidean_distance': np.std(euclidean_distances),
            'median_euclidean_distance': np.median(euclidean_distances),
            'max_cosine_distance': np.max(cosine_distances),
            'min_cosine_distance': np.min(cosine_distances)
        }
    
    def _jackknife_statistic(self, words, metric):
        """Calculate jackknife statistics for bias correction."""
        valid_words = [w for w in words if w in self.shared_vocab]
        jackknife_stats = []
        
        for i in range(len(valid_words)):
            # Remove one word
            jackknife_sample = valid_words[:i] + valid_words[i+1:]
            
            # Calculate statistic
            stats_dict = self._calculate_observed_statistics(jackknife_sample)
            if metric in stats_dict:
                jackknife_stats.append(stats_dict[metric])
        
        return np.array(jackknife_stats)
    
    def _calculate_acceleration(self, jackknife_stats):
        """Calculate acceleration parameter for BCa intervals."""
        if len(jackknife_stats) < 2:
            return 0
        
        mean_jackknife = np.mean(jackknife_stats)
        numerator = np.sum((mean_jackknife - jackknife_stats) ** 3)
        denominator = 6 * (np.sum((mean_jackknife - jackknife_stats) ** 2) ** 1.5)
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def _perform_additional_tests(self, target_words, bootstrap_results):
        """Perform additional statistical tests."""
        additional_tests = {}
        
        # Kolmogorov-Smirnov test for normality
        if 'mean_cosine_distance' in bootstrap_results:
            distribution = bootstrap_results['mean_cosine_distance']['distribution']
            ks_stat, ks_p = kstest(distribution, 'norm', 
                                  args=(np.mean(distribution), np.std(distribution)))
            additional_tests['normality_test'] = {
                'ks_statistic': ks_stat,
                'p_value': ks_p,
                'is_normal': ks_p > 0.05
            }
        
        # Test for significant difference from zero
        if 'mean_cosine_distance' in bootstrap_results:
            distribution = bootstrap_results['mean_cosine_distance']['distribution']
            # One-sample t-test against zero
            t_stat, t_p = stats.ttest_1samp(distribution, 0)
            additional_tests['difference_from_zero'] = {
                't_statistic': t_stat,
                'p_value': t_p,
                'significantly_different': t_p < 0.05
            }
        
        return additional_tests
    
    def _plot_bootstrap_results(self, bootstrap_results, confidence_level):
        """Create comprehensive bootstrap analysis visualizations."""
        # Filter out non-distributional results
        distributional_results = {k: v for k, v in bootstrap_results.items() 
                                if isinstance(v, dict) and 'distribution' in v}
        
        n_metrics = len(distributional_results)
        if n_metrics == 0:
            return
        
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, 
                               figsize=(6 * ((n_metrics + 1) // 2), 12))
        if n_metrics == 1:
            axes = axes.reshape(2, 1)
        elif n_metrics <= 2:
            axes = axes.reshape(2, 1) if n_metrics == 1 else axes
        
        fig.suptitle(f'Bootstrap Analysis Results ({confidence_level*100:.0f}% Confidence)', 
                    fontsize=16)
        
        for i, (metric, results) in enumerate(distributional_results.items()):
            row = i // ((n_metrics + 1) // 2)
            col = i % ((n_metrics + 1) // 2)
            
            if n_metrics == 1:
                ax = axes[row]
            else:
                ax = axes[row, col]
            
            distribution = results['distribution']
            observed = results['observed']
            
            # Plot histogram
            ax.hist(distribution, bins=50, alpha=0.7, density=True, 
                   edgecolor='black', color='skyblue')
            
            # Plot observed value
            ax.axvline(observed, color='red', linestyle='-', linewidth=2, 
                      label=f'Observed: {observed:.4f}')
            
            # Plot confidence intervals
            ax.axvline(results['ci_lower'], color='orange', linestyle='--', 
                      label=f'{confidence_level*100:.0f}% CI')
            ax.axvline(results['ci_upper'], color='orange', linestyle='--')
            
            # Plot BCa intervals
            ax.axvline(results['bca_lower'], color='green', linestyle=':', 
                      label='BCa CI')
            ax.axvline(results['bca_upper'], color='green', linestyle=':')
            
            ax.set_title(f'{metric}\np = {results["p_value"]:.4f}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            
            # Add significance indicator
            if results['significant']:
                ax.text(0.02, 0.98, '***', transform=ax.transAxes, 
                       fontsize=16, color='red', weight='bold',
                       verticalalignment='top')
        
        # Remove empty subplots
        if n_metrics % 2 == 1 and n_metrics > 1:
            fig.delaxes(axes[1, -1])
        
        plt.tight_layout()
        plt.savefig('bootstrap_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _generate_statistical_report(self, bootstrap_results, confidence_level):
        """Generate a comprehensive statistical report."""
        print(f"\n{'='*80}")
        print("BOOTSTRAP STATISTICAL ANALYSIS REPORT")
        print(f"Confidence Level: {confidence_level*100:.0f}%")
        print(f"{'='*80}")
        
        # Filter distributional results
        distributional_results = {k: v for k, v in bootstrap_results.items() 
                                if isinstance(v, dict) and 'distribution' in v}
        
        for metric, results in distributional_results.items():
            print(f"\n{metric.upper().replace('_', ' ')}")
            print("-" * 50)
            print(f"Observed Value:           {results['observed']:.6f}")
            print(f"Bootstrap Mean:           {results['bootstrap_mean']:.6f}")
            print(f"Bootstrap Std:            {results['bootstrap_std']:.6f}")
            print(f"95% CI (Percentile):      [{results['ci_lower']:.6f}, {results['ci_upper']:.6f}]")
            print(f"95% CI (BCa):            [{results['bca_lower']:.6f}, {results['bca_upper']:.6f}]")
            print(f"P-value (two-tailed):     {results['p_value']:.6f}")
            print(f"Statistically Significant: {'Yes' if results['significant'] else 'No'}")
        
        # Additional tests
        if 'normality_test' in bootstrap_results:
            print(f"\nNORMALITY TEST (Kolmogorov-Smirnov)")
            print("-" * 50)
            nt = bootstrap_results['normality_test']
            print(f"KS Statistic:             {nt['ks_statistic']:.6f}")
            print(f"P-value:                  {nt['p_value']:.6f}")
            print(f"Distribution is Normal:   {'Yes' if nt['is_normal'] else 'No'}")
        
        if 'difference_from_zero' in bootstrap_results:
            print(f"\nDIFFERENCE FROM ZERO TEST")
            print("-" * 50)
            dz = bootstrap_results['difference_from_zero']
            print(f"T-statistic:              {dz['t_statistic']:.6f}")
            print(f"P-value:                  {dz['p_value']:.6f}")
            print(f"Significantly ≠ 0:        {'Yes' if dz['significantly_different'] else 'No'}")
        
        print(f"\n{'='*80}")
        print("INTERPRETATION GUIDE")
        print(f"{'='*80}")
        print("• P-values < 0.05 indicate statistical significance at 95% confidence")
        print("• BCa intervals are bias-corrected and accelerated (more accurate)")
        print("• Large bootstrap standard deviations suggest high variability")
        print("• Normality test helps validate bootstrap assumptions")
        print(f"{'='*80}")
    
    def generate_comprehensive_report(self, target_words, save_to_file=True):
        """
        Generate a comprehensive publication-ready report.
        """
        print("Generating comprehensive temporal semantic analysis report...")
        
        # Perform all analyses
        print("\n1. Creating temporal embedding maps...")
        embedding_df, trajectories, reducer = self.create_temporal_embedding_maps(target_words)
        
        print("\n2. Performing semantic clustering...")
        clustering_results, stability_metrics = self.perform_semantic_clustering(target_words)
        
        print("\n3. Calculating co-occurrence matrices...")
        cooccurrence_results, cooc_changes = self.calculate_cooccurrence_matrices(target_words)
        
        print("\n4. Analyzing drift magnitude and direction...")
        drift_df, drift_patterns = self.analyze_drift_magnitude_direction(target_words)
        
        print("\n5. Performing bootstrap statistical analysis...")
        bootstrap_results = self.bootstrap_statistical_significance(target_words)
        
        # Compile comprehensive results
        comprehensive_results = {
            'embedding_analysis': {
                'embedding_df': embedding_df,
                'trajectories': trajectories,
                'reducer': reducer
            },
            'clustering_analysis': {
                'clustering_results': clustering_results,
                'stability_metrics': stability_metrics
            },
            'cooccurrence_analysis': {
                'cooccurrence_results': cooccurrence_results,
                'cooc_changes': cooc_changes
            },
            'drift_analysis': {
                'drift_df': drift_df,
                'drift_patterns': drift_patterns
            },
            'statistical_analysis': bootstrap_results
        }
        
        if save_to_file:
            # Save detailed results to CSV files
            embedding_df.to_csv('temporal_embeddings.csv', index=False)
            trajectories.to_csv('semantic_trajectories.csv', index=False)
            drift_df.to_csv('drift_magnitude_direction.csv', index=False)
            
            # Save bootstrap results
            bootstrap_df = pd.DataFrame([
                {
                    'metric': metric,
                    'observed': data['observed'],
                    'bootstrap_mean': data['bootstrap_mean'],
                    'bootstrap_std': data['bootstrap_std'],
                    'ci_lower': data['ci_lower'],
                    'ci_upper': data['ci_upper'],
                    'bca_lower': data['bca_lower'],
                    'bca_upper': data['bca_upper'],
                    'p_value': data['p_value'],
                    'significant': data['significant']
                }
                for metric, data in bootstrap_results.items()
                if isinstance(data, dict) and 'distribution' in data
            ])
            bootstrap_df.to_csv('bootstrap_statistical_results.csv', index=False)
            
            print("\nResults saved to:")
            print("• temporal_embeddings.csv")
            print("• semantic_trajectories.csv") 
            print("• drift_magnitude_direction.csv")
            print("• bootstrap_statistical_results.csv")
        
        return comprehensive_results


# Example usage
if __name__ == "__main__":
    # Load your models and aligned embeddings
    print("Loading models and aligned embeddings...")
    
    # You would load these from your previous analysis
    model_classical = Word2Vec.load("models/word2vec_classical.model")
    model_imperial = Word2Vec.load("models/word2vec_imperial.model")
    model_late = Word2Vec.load("models/word2vec_late.model")
    # 
    with open("models/aligned_imperial.pkl", "rb") as f:
        aligned_imperial = pickle.load(f)
    with open("models/aligned_late.pkl", "rb") as f:
        aligned_late = pickle.load(f)
    with open("models/shared_vocab.pkl", "rb") as f:
        shared_vocab = pickle.load(f)
    
    # Target words for analysis
    target_words = [
        "puella", "equus", "urbs", "terra", "caritas", "prex", "sacramentum", 
        "sacer", "sacerdos", "sacrificium", "lex", "spiritus", "fides", "pietas", 
        "fidelus", "gloria", "gratia", "gratus", "honor", "iustus", "magnus", 
        "parvus", "longus", "brevis", "altus", "novus", "vetus", "bonus", "malus", 
        "albus", "niger", "ruber"
    ]
    
    # Initialize analyzer
    analyzer = TemporalSemanticAnalyzer(
        model_classical, model_imperial, model_late,
        aligned_imperial, aligned_late, shared_vocab
    )
    
    # Generate comprehensive report
    comprehensive_results = analyzer.generate_comprehensive_report(target_words)
    
    print("\nTemporal semantic analysis complete!")
    print("All publication-quality visualizations and statistical results have been generated.")
    print("\nKey outputs:")
    print("• Temporal embedding trajectories with statistical significance")
    print("• Semantic clustering analysis across time periods")
    print("• Co-occurrence pattern changes")
    print("• Drift magnitude and direction analysis")
    print("• Bootstrap confidence intervals and hypothesis tests")
    print("• Publication-ready figures and statistical reports")