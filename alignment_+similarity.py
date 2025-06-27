import numpy as np
import pickle
from gensim.models import Word2Vec
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import spearmanr, pearsonr
import pandas as pd
from typing import Dict, List, Set, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os

class VectorSpaceAligner:
    # # Create genre metadata structure
    # genre_metadata = {
    #     'text_id': ['classicus_001', 'imperialis_045', 'tardus_123'],
    #     'period': ['classical', 'imperial', 'late'],
    #     'genre': ['poetry', 'history', 'theology'],
    #     'subgenre': ['epic', 'biography', 'homily'],
    #     'author': ['Virgil', 'Tacitus', 'Augustine'],
    #     'word_count': [9896, 15432, 8765]
    # }

    def __init__(self, model_classical, model_imperial, model_late, shared_vocab):
        self.model_classical = model_classical
        self.model_imperial = model_imperial
        self.model_late = model_late
        self.shared_vocab = shared_vocab
        
        # Transformation matrices
        self.R_imp = None
        self.R_late = None
        self.lr_imp = None
        self.lr_late = None
        
    def get_stable_anchor_words(self, min_freq: int = 50) -> List[str]:
        """
        Get stable, high-frequency content words for alignment.
        These should be concrete nouns, body parts, basic verbs, and numbers
        that are less likely to undergo semantic change.
        """
        # Stable content words - concrete nouns, body parts, basic concepts
        candidates = [
            # Body parts (very stable)
            "caput", "corpus", "manus", "pes", "oculus", "os", "cor", "sanguis",
            "dens", "digitus", "brachium", "genu", "nasus", "auris",
            
            # Basic concrete nouns
            "aqua", "ignis", "terra", "sol", "luna", "stella", "caelum",
            "mons", "silva", "arbor", "flos", "herba", "lapis", "ferrum",
            
            # Family relationships
            "pater", "mater", "filius", "filia", "frater", "soror", "vir", "mulier",
            
            # Numbers (very stable)
            "unus", "duo", "tres", "quattuor", "quinque", "sex", "septem", "octo",
            "novem", "decem", "centum", "mille",
            
            # Basic animals
            "canis", "equus", "bos", "ovis", "porcus", "gallus", "piscis", "avis",
            
            # Basic verbs (in infinitive or common forms)
            "dare", "facere", "habere", "venire", "ire", "videre", "audire",
            "dicere", "scribere", "legere", "currere", "stare", "sedere",
            
            # Time concepts
            "dies", "nox", "annus", "mensis", "hora", "tempus",
            
            # Basic adjectives/descriptors
            "magnus", "parvus", "longus", "brevis", "altus", "novus", "vetus",
            "bonus", "malus", "albus", "niger", "ruber"
        ]
        
        # Filter by availability in all models and frequency
        stable_words = []
        for word in candidates:
            if (word in self.shared_vocab and 
                word in self.model_classical.wv.key_to_index and
                word in self.model_imperial.wv.key_to_index and
                word in self.model_late.wv.key_to_index):
                
                # Check frequency in classical model as baseline
                if self.model_classical.wv.get_vecattr(word, "count") >= min_freq:
                    stable_words.append(word)
        
        print(f"Using {len(stable_words)} stable anchor words: {stable_words[:10]}...")
        return stable_words
    
    def get_matrix(self, model, vocab: List[str]) -> np.ndarray:
        """Extract word vectors for given vocabulary."""
        return np.stack([model.wv[word] for word in vocab])
    
    def orthogonal_alignment(self, anchor_words: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Align vector spaces using Orthogonal Procrustes with stable anchor words.
        """
        if anchor_words is None:
            anchor_words = self.get_stable_anchor_words()
        
        print(f"Performing orthogonal alignment with {len(anchor_words)} anchor words...")
        
        # Get matrices for anchor words only
        X = self.get_matrix(self.model_classical, anchor_words)  # target
        Y_imp = self.get_matrix(self.model_imperial, anchor_words)  # source
        Y_late = self.get_matrix(self.model_late, anchor_words)  # source
        
        # Find optimal orthogonal transformations
        self.R_imp, _ = orthogonal_procrustes(Y_imp, X)
        self.R_late, _ = orthogonal_procrustes(Y_late, X)
        
        # Apply transformations to all shared vocabulary
        aligned_imperial = {}
        aligned_late = {}
        
        for word in self.shared_vocab:
            aligned_imperial[word] = self.model_imperial.wv[word] @ self.R_imp
            aligned_late[word] = self.model_late.wv[word] @ self.R_late
        
        # Validate alignment quality
        self._validate_alignment(anchor_words, aligned_imperial, aligned_late, "orthogonal")
        
        return aligned_imperial, aligned_late
    
    def linear_alignment(self, anchor_words: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Align vector spaces using linear regression (allows non-orthogonal transformations).
        """
        if anchor_words is None:
            anchor_words = self.get_stable_anchor_words()
            
        print(f"Performing linear alignment with {len(anchor_words)} anchor words...")
        
        # Get matrices for anchor words
        X = self.get_matrix(self.model_classical, anchor_words)  # target
        Y_imp = self.get_matrix(self.model_imperial, anchor_words)  # source
        Y_late = self.get_matrix(self.model_late, anchor_words)  # source
        
        # Learn linear transformations
        self.lr_imp = LinearRegression(fit_intercept=True).fit(Y_imp, X)
        self.lr_late = LinearRegression(fit_intercept=True).fit(Y_late, X)
        
        # Apply transformations to all shared vocabulary
        aligned_imperial = {}
        aligned_late = {}
        
        for word in self.shared_vocab:
            aligned_imperial[word] = self.lr_imp.predict([self.model_imperial.wv[word]])[0]
            aligned_late[word] = self.lr_late.predict([self.model_late.wv[word]])[0]
        
        # Validate alignment quality
        self._validate_alignment(anchor_words, aligned_imperial, aligned_late, "linear")
        
        return aligned_imperial, aligned_late
    
    def _validate_alignment(self, anchor_words: List[str], 
                          aligned_imperial: Dict, aligned_late: Dict, 
                          method: str):
        """Validate alignment quality using anchor words."""
        print(f"\n=== Validation of {method} alignment ===")
        
        improvements_imp = []
        improvements_late = []
        
        for word in anchor_words[:5]:  # Check first 5 anchor words
            # Original similarities
            orig_sim_imp = cosine_similarity(
                [self.model_classical.wv[word]], 
                [self.model_imperial.wv[word]]
            )[0][0]
            orig_sim_late = cosine_similarity(
                [self.model_classical.wv[word]], 
                [self.model_late.wv[word]]
            )[0][0]
            
            # Aligned similarities
            aligned_sim_imp = cosine_similarity(
                [self.model_classical.wv[word]], 
                [aligned_imperial[word]]
            )[0][0]
            aligned_sim_late = cosine_similarity(
                [self.model_classical.wv[word]], 
                [aligned_late[word]]
            )[0][0]
            
            improvements_imp.append(aligned_sim_imp - orig_sim_imp)
            improvements_late.append(aligned_sim_late - orig_sim_late)
            
            print(f"{word}: Imp {orig_sim_imp:.3f}→{aligned_sim_imp:.3f} "
                  f"(+{aligned_sim_imp-orig_sim_imp:.3f}), "
                  f"Late {orig_sim_late:.3f}→{aligned_sim_late:.3f} "
                  f"(+{aligned_sim_late-orig_sim_late:.3f})")
        
        avg_imp_imp = np.mean(improvements_imp)
        avg_imp_late = np.mean(improvements_late)
        
        print(f"Average similarity improvement - Imperial: {avg_imp_imp:.3f}, Late: {avg_imp_late:.3f}")
        
        if avg_imp_imp > 0 and avg_imp_late > 0:
            print("✓ Alignment successful - similarities improved")
        else:
            print("⚠ Warning: Alignment may not be optimal - similarities did not improve")
    
    def calculate_neighborhood_overlap(self, word: str, k: int = 10) -> Dict[str, float]:
        """
        Calculate neighborhood overlap (Jaccard similarity) between periods.
        Standard measure in computational linguistics for semantic change.
        """
        try:
            # Get k nearest neighbors for each period
            neighbors_classical = set([w for w, _ in self.model_classical.wv.most_similar(word, topn=k)])
            neighbors_imperial = set([w for w, _ in self.model_imperial.wv.most_similar(word, topn=k)])
            neighbors_late = set([w for w, _ in self.model_late.wv.most_similar(word, topn=k)])
            
            # Calculate Jaccard similarities
            jaccard_class_imp = len(neighbors_classical & neighbors_imperial) / len(neighbors_classical | neighbors_imperial)
            jaccard_class_late = len(neighbors_classical & neighbors_late) / len(neighbors_classical | neighbors_late)
            jaccard_imp_late = len(neighbors_imperial & neighbors_late) / len(neighbors_imperial | neighbors_late)
            
            return {
                'jaccard_classical_imperial': jaccard_class_imp,
                'jaccard_classical_late': jaccard_class_late,
                'jaccard_imperial_late': jaccard_imp_late
            }
        except:
            return {
                'jaccard_classical_imperial': np.nan,
                'jaccard_classical_late': np.nan,
                'jaccard_imperial_late': np.nan
            }
    
    def calculate_vector_distances(self, word: str, aligned_imperial: Dict, aligned_late: Dict) -> Dict[str, float]:
        """
        Calculate multiple distance measures commonly used in computational linguistics.
        """
        vec_classical = self.model_classical.wv[word]
        vec_imperial = aligned_imperial[word]
        vec_late = aligned_late[word]
        
        # Cosine similarity (primary metric)
        cos_sim_class_imp = cosine_similarity([vec_classical], [vec_imperial])[0][0]
        cos_sim_class_late = cosine_similarity([vec_classical], [vec_late])[0][0]
        cos_sim_imp_late = cosine_similarity([vec_imperial], [vec_late])[0][0]
        
        # Cosine distance (1 - cosine similarity)
        cos_dist_class_imp = 1 - cos_sim_class_imp
        cos_dist_class_late = 1 - cos_sim_class_late
        cos_dist_imp_late = 1 - cos_sim_imp_late
        
        # Euclidean distance
        eucl_dist_class_imp = np.linalg.norm(vec_classical - vec_imperial)
        eucl_dist_class_late = np.linalg.norm(vec_classical - vec_late)
        eucl_dist_imp_late = np.linalg.norm(vec_imperial - vec_late)
        
        return {
            'cosine_similarity_class_imp': cos_sim_class_imp,
            'cosine_similarity_class_late': cos_sim_class_late,
            'cosine_similarity_imp_late': cos_sim_imp_late,
            'cosine_distance_class_imp': cos_dist_class_imp,
            'cosine_distance_class_late': cos_dist_class_late,
            'cosine_distance_imp_late': cos_dist_imp_late,
            'euclidean_distance_class_imp': eucl_dist_class_imp,
            'euclidean_distance_class_late': eucl_dist_class_late,
            'euclidean_distance_imp_late': eucl_dist_imp_late
        }
    
    def calculate_frequency_statistics(self, word: str) -> Dict[str, float]:
        """
        Calculate frequency-based statistics commonly reported in linguistics.
        """
        # Get raw frequencies
        freq_classical = self.model_classical.wv.get_vecattr(word, "count") if hasattr(self.model_classical.wv, 'get_vecattr') else 0
        freq_imperial = self.model_imperial.wv.get_vecattr(word, "count") if hasattr(self.model_imperial.wv, 'get_vecattr') else 0
        freq_late = self.model_late.wv.get_vecattr(word, "count") if hasattr(self.model_late.wv, 'get_vecattr') else 0
        
        # Calculate relative frequencies (per million words - estimate)
        # Note: You might want to replace these with actual corpus sizes
        corpus_size_classical = 1000000  # Replace with actual size
        corpus_size_imperial = 1000000   # Replace with actual size
        corpus_size_late = 1000000       # Replace with actual size
        
        rel_freq_classical = (freq_classical / corpus_size_classical) * 1000000
        rel_freq_imperial = (freq_imperial / corpus_size_imperial) * 1000000
        rel_freq_late = (freq_late / corpus_size_late) * 1000000
        
        # Frequency ratios
        freq_ratio_imp_class = freq_imperial / max(freq_classical, 1)
        freq_ratio_late_class = freq_late / max(freq_classical, 1)
        freq_ratio_late_imp = freq_late / max(freq_imperial, 1)
        
        return {
            'freq_classical': freq_classical,
            'freq_imperial': freq_imperial,
            'freq_late': freq_late,
            'rel_freq_classical': rel_freq_classical,
            'rel_freq_imperial': rel_freq_imperial,
            'rel_freq_late': rel_freq_late,
            'freq_ratio_imp_class': freq_ratio_imp_class,
            'freq_ratio_late_class': freq_ratio_late_class,
            'freq_ratio_late_imp': freq_ratio_late_imp
        }
    
    def comprehensive_semantic_analysis(self, aligned_imperial: Dict, aligned_late: Dict, 
                                      target_words: List[str]) -> pd.DataFrame:
        """
        Comprehensive semantic drift analysis with all standard computational linguistics metrics.
        """
        results = []
        
        print("Calculating comprehensive semantic drift metrics...")
        
        for i, word in enumerate(target_words):
            if word in self.shared_vocab:
                print(f"Processing word {i+1}/{len(target_words)}: {word}")
                
                # Basic information
                result = {'word': word}
                
                # Vector distance measures
                distance_metrics = self.calculate_vector_distances(word, aligned_imperial, aligned_late)
                result.update(distance_metrics)
                
                # Neighborhood overlap measures
                neighborhood_metrics = self.calculate_neighborhood_overlap(word)
                result.update(neighborhood_metrics)
                
                # Frequency statistics
                freq_metrics = self.calculate_frequency_statistics(word)
                result.update(freq_metrics)
                
                # Semantic change indicators
                result['semantic_change_class_imp'] = distance_metrics['cosine_distance_class_imp']
                result['semantic_change_class_late'] = distance_metrics['cosine_distance_class_late']
                result['semantic_change_total'] = distance_metrics['cosine_distance_class_late'] - distance_metrics['cosine_distance_class_imp']
                
                # Stability indicators (inverse of change)
                result['semantic_stability_class_imp'] = distance_metrics['cosine_similarity_class_imp']
                result['semantic_stability_class_late'] = distance_metrics['cosine_similarity_class_late']
                
                results.append(result)
        
        df = pd.DataFrame(results)
        
        # Add percentile ranks for key metrics (useful for journal reporting)
        df['cosine_distance_class_late_percentile'] = df['cosine_distance_class_late'].rank(pct=True) * 100
        df['semantic_change_total_percentile'] = df['semantic_change_total'].rank(pct=True) * 100
        df['jaccard_classical_late_percentile'] = df['jaccard_classical_late'].rank(pct=True, ascending=False) * 100
        
        return df
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics commonly reported in computational linguistics papers.
        """
        stats = {}
        
        # Cosine similarity statistics
        stats['cosine_similarity'] = {
            'classical_imperial': {
                'mean': df['cosine_similarity_class_imp'].mean(),
                'std': df['cosine_similarity_class_imp'].std(),
                'median': df['cosine_similarity_class_imp'].median(),
                'min': df['cosine_similarity_class_imp'].min(),
                'max': df['cosine_similarity_class_imp'].max()
            },
            'classical_late': {
                'mean': df['cosine_similarity_class_late'].mean(),
                'std': df['cosine_similarity_class_late'].std(),
                'median': df['cosine_similarity_class_late'].median(),
                'min': df['cosine_similarity_class_late'].min(),
                'max': df['cosine_similarity_class_late'].max()
            }
        }
        
        # Semantic change statistics
        stats['semantic_change'] = {
            'classical_imperial': {
                'mean': df['semantic_change_class_imp'].mean(),
                'std': df['semantic_change_class_imp'].std(),
                'median': df['semantic_change_class_imp'].median()
            },
            'classical_late': {
                'mean': df['semantic_change_class_late'].mean(),
                'std': df['semantic_change_class_late'].std(),
                'median': df['semantic_change_class_late'].median()
            },
            'total_change': {
                'mean': df['semantic_change_total'].mean(),
                'std': df['semantic_change_total'].std(),
                'median': df['semantic_change_total'].median()
            }
        }
        
        # Neighborhood overlap statistics
        stats['neighborhood_overlap'] = {
            'classical_imperial': {
                'mean': df['jaccard_classical_imperial'].mean(),
                'std': df['jaccard_classical_imperial'].std()
            },
            'classical_late': {
                'mean': df['jaccard_classical_late'].mean(),
                'std': df['jaccard_classical_late'].std()
            }
        }
        
        # Correlation between frequency and semantic change
        freq_change_corr = pearsonr(df['freq_classical'], df['semantic_change_class_late'])
        stats['frequency_change_correlation'] = {
            'pearson_r': freq_change_corr[0],
            'p_value': freq_change_corr[1]
        }
        
        return stats
    
    def print_journal_ready_results(self, df: pd.DataFrame, stats: Dict, alignment_method: str):
        """
        Print results in a format suitable for computational linguistics journals.
        """
        print(f"\n{'='*60}")
        print(f"COMPUTATIONAL LINGUISTICS ANALYSIS RESULTS")
        print(f"Alignment Method: {alignment_method.upper()}")
        print(f"{'='*60}")
        
        print(f"\n1. COSINE SIMILARITY ANALYSIS")
        print(f"   Classical → Imperial: μ={stats['cosine_similarity']['classical_imperial']['mean']:.3f} "
              f"(σ={stats['cosine_similarity']['classical_imperial']['std']:.3f})")
        print(f"   Classical → Late:     μ={stats['cosine_similarity']['classical_late']['mean']:.3f} "
              f"(σ={stats['cosine_similarity']['classical_late']['std']:.3f})")
        
        print(f"\n2. SEMANTIC CHANGE ANALYSIS (Cosine Distance)")
        print(f"   Classical → Imperial: μ={stats['semantic_change']['classical_imperial']['mean']:.3f} "
              f"(σ={stats['semantic_change']['classical_imperial']['std']:.3f})")
        print(f"   Classical → Late:     μ={stats['semantic_change']['classical_late']['mean']:.3f} "
              f"(σ={stats['semantic_change']['classical_late']['std']:.3f})")
        print(f"   Total Change:         μ={stats['semantic_change']['total_change']['mean']:.3f} "
              f"(σ={stats['semantic_change']['total_change']['std']:.3f})")
        
        print(f"\n3. NEIGHBORHOOD OVERLAP (Jaccard Similarity)")
        print(f"   Classical ∩ Imperial: μ={stats['neighborhood_overlap']['classical_imperial']['mean']:.3f} "
              f"(σ={stats['neighborhood_overlap']['classical_imperial']['std']:.3f})")
        print(f"   Classical ∩ Late:     μ={stats['neighborhood_overlap']['classical_late']['mean']:.3f} "
              f"(σ={stats['neighborhood_overlap']['classical_late']['std']:.3f})")
        
        print(f"\n4. FREQUENCY-CHANGE CORRELATION")
        print(f"   Pearson r = {stats['frequency_change_correlation']['pearson_r']:.3f} "
              f"(p = {stats['frequency_change_correlation']['p_value']:.3f})")
        
        print(f"\n5. TOP 10 WORDS BY SEMANTIC CHANGE")
        top_changed = df.nlargest(10, 'semantic_change_class_late')[
            ['word', 'cosine_similarity_class_late', 'semantic_change_class_late', 
             'jaccard_classical_late', 'freq_classical']
        ]
        print(top_changed.round(3).to_string(index=False))
        
        print(f"\n6. MOST STABLE WORDS (Highest Cosine Similarity)")
        most_stable = df.nlargest(10, 'cosine_similarity_class_late')[
            ['word', 'cosine_similarity_class_late', 'semantic_change_class_late', 
             'jaccard_classical_late', 'freq_classical']
        ]
        print(most_stable.round(3).to_string(index=False))


# Load your models and data
print("Loading models...")
model_classical = Word2Vec.load("models/word2vec_classical.model")
model_imperial = Word2Vec.load("models/word2vec_imperial.model")
model_late = Word2Vec.load("models/word2vec_late.model")

with open("models/alignment_vocab.pkl", "rb") as f:
    shared_vocab = set(pickle.load(f))

print(f"Shared vocabulary size: {len(shared_vocab)}")

# Initialize aligner
aligner = VectorSpaceAligner(model_classical, model_imperial, model_late, shared_vocab)

# Target words for semantic drift analysis
target_words = [
    "puella", "equus", "urbs", "terra", "caritas", "prex", "sacramentum", 
    "sacer", "sacerdos", "sacrificium", "lex", "spiritus", "fides", "pietas", 
    "fidelus", "gloria", "gratia", "gratus", "honor", "iustus", "magnus", 
    "parvus", "longus", "brevis", "altus", "novus", "vetus", "bonus", "malus", 
    "albus", "niger", "ruber"
]

# Perform alignments and comprehensive analysis
print("\n" + "="*50)
print("ORTHOGONAL ALIGNMENT")
print("="*50)
aligned_imp_orth, aligned_late_orth = aligner.orthogonal_alignment()
results_orth = aligner.comprehensive_semantic_analysis(aligned_imp_orth, aligned_late_orth, target_words)
stats_orth = aligner.generate_summary_statistics(results_orth)
aligner.print_journal_ready_results(results_orth, stats_orth, "orthogonal")

print("\n" + "="*50)
print("LINEAR ALIGNMENT")
print("="*50)
aligned_imp_linear, aligned_late_linear = aligner.linear_alignment()
results_linear = aligner.comprehensive_semantic_analysis(aligned_imp_linear, aligned_late_linear, target_words)
stats_linear = aligner.generate_summary_statistics(results_linear)
aligner.print_journal_ready_results(results_linear, stats_linear, "linear")

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

print("\n" + "="*50)
print("SAVING ALIGNED EMBEDDINGS")
print("="*50)

# Save orthogonal alignment results
print("Saving orthogonal alignment results...")
with open("models/aligned_imperial_orthogonal.pkl", "wb") as f:
    pickle.dump(aligned_imp_orth, f)

with open("models/aligned_late_orthogonal.pkl", "wb") as f:
    pickle.dump(aligned_late_orth, f)
# Save results to CSV for further analysis
#results_orth.to_csv('semantic_drift_orthogonal.csv', index=False)
#results_linear.to_csv('semantic_drift_linear.csv', index=False)

print(f"\n\nResults saved to 'semantic_drift_orthogonal.csv' and 'semantic_drift_linear.csv'")
print("These files contain all metrics commonly used in computational linguistics research.")