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
from itertools import combinations

class GenreVectorSpaceAligner:
    
    def __init__(self, models_dict: Dict[str, Word2Vec], shared_vocab: Set[str]):
        """
        Initialize with a dictionary of genre models and shared vocabulary.
        
        Args:
            models_dict: Dictionary mapping genre names to Word2Vec models
            shared_vocab: Set of words shared across models
        """
        self.models = models_dict
        self.shared_vocab = shared_vocab
        self.genre_names = list(models_dict.keys())
        
        # Transformation matrices for each genre pair
        self.transformation_matrices = {}
        
        print(f"Initialized GenreVectorSpaceAligner with {len(self.models)} genre models")
        print(f"Genres: {', '.join(self.genre_names)}")
        print(f"Shared vocabulary size: {len(shared_vocab)}")

    def get_stable_anchor_words(self, min_freq: int = 20) -> List[str]:
        """
        Get stable, high-frequency content words for alignment.
        Uses the same methodology as the original temporal analysis.
        """
        # Same stable content words from original methodology
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
        
        # Filter by availability in models and frequency
        stable_words = []
        for word in candidates:
            if word in self.shared_vocab:
                # Check frequency across models
                sufficient_freq = 0
                for model in self.models.values():
                    if (word in model.wv.key_to_index and 
                        hasattr(model.wv, 'get_vecattr')):
                        try:
                            if model.wv.get_vecattr(word, "count") >= min_freq:
                                sufficient_freq += 1
                        except:
                            pass
                
                # Require word to have sufficient frequency in at least half the models
                if sufficient_freq >= len(self.models) // 2:
                    stable_words.append(word)
        
        print(f"Using {len(stable_words)} stable anchor words: {stable_words[:10]}...")
        return stable_words
    
    def get_matrix(self, model, vocab: List[str]) -> np.ndarray:
        """Extract word vectors for given vocabulary."""
        return np.stack([model.wv[word] for word in vocab])
    
    def orthogonal_alignment(self, reference_genre: str, anchor_words: List[str] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Align all genre vector spaces to a reference genre using Orthogonal Procrustes.
        
        Args:
            reference_genre: Name of the reference genre model
            anchor_words: List of anchor words for alignment
            
        Returns:
            Dictionary mapping genre names to aligned word vectors
        """
        if reference_genre not in self.models:
            raise ValueError(f"Reference genre '{reference_genre}' not found in models")
            
        if anchor_words is None:
            anchor_words = self.get_stable_anchor_words()
        
        print(f"Performing orthogonal alignment with reference: {reference_genre}")
        print(f"Using {len(anchor_words)} anchor words...")
        
        reference_model = self.models[reference_genre]
        
        # Filter anchor words to those available in reference model
        anchor_words = [w for w in anchor_words if w in reference_model.wv.key_to_index]
        
        aligned_genres = {}
        
        for genre_name, model in self.models.items():
            if genre_name == reference_genre:
                # Reference genre doesn't need alignment
                aligned_genres[genre_name] = {word: model.wv[word] for word in self.shared_vocab 
                                            if word in model.wv.key_to_index}
                continue
            
            # Filter anchor words available in both reference and current model
            common_anchors = [w for w in anchor_words if w in model.wv.key_to_index]
            
            if len(common_anchors) < 10:
                print(f"⚠ Warning: Only {len(common_anchors)} anchor words for {genre_name}")
            
            # Get matrices for anchor words
            X = self.get_matrix(reference_model, common_anchors)  # target
            Y = self.get_matrix(model, common_anchors)  # source
            
            # Find optimal orthogonal transformation
            R, _ = orthogonal_procrustes(Y, X)
            self.transformation_matrices[f"{genre_name}_to_{reference_genre}"] = R
            
            # Apply transformation to all shared vocabulary
            aligned_vectors = {}
            for word in self.shared_vocab:
                if word in model.wv.key_to_index:
                    aligned_vectors[word] = model.wv[word] @ R
            
            aligned_genres[genre_name] = aligned_vectors
            
            # Validate alignment quality
            self._validate_alignment(reference_model, model, aligned_vectors, 
                                   common_anchors[:5], genre_name, "orthogonal")
        
        return aligned_genres
    
    def linear_alignment(self, reference_genre: str, anchor_words: List[str] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Align all genre vector spaces to a reference genre using linear regression.
        
        Args:
            reference_genre: Name of the reference genre model
            anchor_words: List of anchor words for alignment
            
        Returns:
            Dictionary mapping genre names to aligned word vectors
        """
        if reference_genre not in self.models:
            raise ValueError(f"Reference genre '{reference_genre}' not found in models")
            
        if anchor_words is None:
            anchor_words = self.get_stable_anchor_words()
            
        print(f"Performing linear alignment with reference: {reference_genre}")
        print(f"Using {len(anchor_words)} anchor words...")
        
        reference_model = self.models[reference_genre]
        
        # Filter anchor words to those available in reference model
        anchor_words = [w for w in anchor_words if w in reference_model.wv.key_to_index]
        
        aligned_genres = {}
        
        for genre_name, model in self.models.items():
            if genre_name == reference_genre:
                # Reference genre doesn't need alignment
                aligned_genres[genre_name] = {word: model.wv[word] for word in self.shared_vocab 
                                            if word in model.wv.key_to_index}
                continue
            
            # Filter anchor words available in both models
            common_anchors = [w for w in anchor_words if w in model.wv.key_to_index]
            
            if len(common_anchors) < 10:
                print(f"⚠ Warning: Only {len(common_anchors)} anchor words for {genre_name}")
            
            # Get matrices for anchor words
            X = self.get_matrix(reference_model, common_anchors)  # target
            Y = self.get_matrix(model, common_anchors)  # source
            
            # Learn linear transformation
            lr = LinearRegression(fit_intercept=True).fit(Y, X)
            
            # Apply transformation to all shared vocabulary
            aligned_vectors = {}
            for word in self.shared_vocab:
                if word in model.wv.key_to_index:
                    aligned_vectors[word] = lr.predict([model.wv[word]])[0]
            
            aligned_genres[genre_name] = aligned_vectors
            
            # Validate alignment quality
            self._validate_alignment(reference_model, model, aligned_vectors, 
                                   common_anchors[:5], genre_name, "linear")
        
        return aligned_genres
    
    def _validate_alignment(self, reference_model, target_model, aligned_vectors: Dict, 
                          anchor_words: List[str], genre_name: str, method: str):
        """Validate alignment quality using anchor words."""
        improvements = []
        
        for word in anchor_words:
            if word not in aligned_vectors:
                continue
                
            # Original similarity
            orig_sim = cosine_similarity(
                [reference_model.wv[word]], 
                [target_model.wv[word]]
            )[0][0]
            
            # Aligned similarity
            aligned_sim = cosine_similarity(
                [reference_model.wv[word]], 
                [aligned_vectors[word]]
            )[0][0]
            
            improvements.append(aligned_sim - orig_sim)
            
            print(f"{word}: {orig_sim:.3f}→{aligned_sim:.3f} (+{aligned_sim-orig_sim:.3f})")
        
        avg_improvement = np.mean(improvements)
        print(f"Average similarity improvement for {genre_name}: {avg_improvement:.3f}")
        
        if avg_improvement > 0:
            print(f"✓ {method.capitalize()} alignment successful for {genre_name}")
        else:
            print(f"⚠ Warning: {method.capitalize()} alignment may not be optimal for {genre_name}")
    
    def calculate_neighborhood_overlap(self, word: str, k: int = 10) -> Dict[str, float]:
        """
        Calculate neighborhood overlap (Jaccard similarity) between all genre pairs.
        """
        overlaps = {}
        
        try:
            # Get k nearest neighbors for each genre
            neighbors = {}
            for genre_name, model in self.models.items():
                if word in model.wv.key_to_index:
                    neighbors[genre_name] = set([w for w, _ in model.wv.most_similar(word, topn=k)])
            
            # Calculate pairwise Jaccard similarities
            for genre1, genre2 in combinations(neighbors.keys(), 2):
                intersection = len(neighbors[genre1] & neighbors[genre2])
                union = len(neighbors[genre1] | neighbors[genre2])
                jaccard = intersection / union if union > 0 else 0
                overlaps[f'jaccard_{genre1}_{genre2}'] = jaccard
                
        except Exception as e:
            print(f"Error calculating neighborhood overlap for {word}: {e}")
            
        return overlaps
    
    def calculate_vector_distances(self, word: str, aligned_genres: Dict[str, Dict[str, np.ndarray]], 
                                 reference_genre: str) -> Dict[str, float]:
        """
        Calculate multiple distance measures between reference genre and all other genres.
        """
        distances = {}
        
        if reference_genre not in aligned_genres or word not in aligned_genres[reference_genre]:
            return distances
            
        reference_vector = aligned_genres[reference_genre][word]
        
        for genre_name in aligned_genres:
            if genre_name == reference_genre or word not in aligned_genres[genre_name]:
                continue
                
            target_vector = aligned_genres[genre_name][word]
            
            # Cosine similarity and distance
            cos_sim = cosine_similarity([reference_vector], [target_vector])[0][0]
            cos_dist = 1 - cos_sim
            
            # Euclidean distance
            eucl_dist = np.linalg.norm(reference_vector - target_vector)
            
            distances[f'cosine_similarity_{reference_genre}_{genre_name}'] = cos_sim
            distances[f'cosine_distance_{reference_genre}_{genre_name}'] = cos_dist
            distances[f'euclidean_distance_{reference_genre}_{genre_name}'] = eucl_dist
        
        return distances
    
    def calculate_frequency_statistics(self, word: str) -> Dict[str, float]:
        """
        Calculate frequency-based statistics across genres.
        """
        freq_stats = {}
        
        for genre_name, model in self.models.items():
            if word in model.wv.key_to_index and hasattr(model.wv, 'get_vecattr'):
                try:
                    freq = model.wv.get_vecattr(word, "count")
                    freq_stats[f'freq_{genre_name}'] = freq
                except:
                    freq_stats[f'freq_{genre_name}'] = 0
            else:
                freq_stats[f'freq_{genre_name}'] = 0
        
        # Calculate frequency ratios
        genre_names = list(self.models.keys())
        for i, genre1 in enumerate(genre_names):
            for genre2 in genre_names[i+1:]:
                freq1 = freq_stats.get(f'freq_{genre1}', 0)
                freq2 = freq_stats.get(f'freq_{genre2}', 0)
                ratio = freq2 / max(freq1, 1)
                freq_stats[f'freq_ratio_{genre2}_{genre1}'] = ratio
        
        return freq_stats
    
    def comprehensive_semantic_analysis(self, aligned_genres: Dict[str, Dict[str, np.ndarray]], 
                                      target_words: List[str], 
                                      reference_genre: str) -> pd.DataFrame:
        """
        Comprehensive semantic drift analysis across genres using the same methodology 
        as the original temporal analysis.
        """
        results = []
        
        print("Calculating comprehensive genre-based semantic drift metrics...")
        
        for i, word in enumerate(target_words):
            if word in self.shared_vocab:
                print(f"Processing word {i+1}/{len(target_words)}: {word}")
                
                # Basic information
                result = {'word': word, 'reference_genre': reference_genre}
                
                # Vector distance measures
                distance_metrics = self.calculate_vector_distances(word, aligned_genres, reference_genre)
                result.update(distance_metrics)
                
                # Neighborhood overlap measures
                neighborhood_metrics = self.calculate_neighborhood_overlap(word)
                result.update(neighborhood_metrics)
                
                # Frequency statistics
                freq_metrics = self.calculate_frequency_statistics(word)
                result.update(freq_metrics)
                
                # Add semantic change indicators for each genre pair
                for genre_name in aligned_genres:
                    if genre_name != reference_genre:
                        cos_dist_key = f'cosine_distance_{reference_genre}_{genre_name}'
                        if cos_dist_key in distance_metrics:
                            result[f'semantic_change_{reference_genre}_{genre_name}'] = distance_metrics[cos_dist_key]
                
                results.append(result)
        
        df = pd.DataFrame(results)
        
        # Add percentile ranks for key metrics
        if not df.empty:
            # Find the main semantic change columns
            semantic_change_cols = [col for col in df.columns if col.startswith('semantic_change_')]
            for col in semantic_change_cols:
                df[f'{col}_percentile'] = df[col].rank(pct=True) * 100
        
        return df
    
    def generate_summary_statistics(self, df: pd.DataFrame, reference_genre: str) -> Dict:
        """
        Generate summary statistics for genre-based analysis.
        """
        stats = {}
        
        # Get all target genres (exclude reference)
        target_genres = [genre for genre in self.genre_names if genre != reference_genre]
        
        # Cosine similarity statistics for each genre pair
        stats['cosine_similarity'] = {}
        for genre in target_genres:
            col_name = f'cosine_similarity_{reference_genre}_{genre}'
            if col_name in df.columns:
                stats['cosine_similarity'][f'{reference_genre}_{genre}'] = {
                    'mean': df[col_name].mean(),
                    'std': df[col_name].std(),
                    'median': df[col_name].median(),
                    'min': df[col_name].min(),
                    'max': df[col_name].max()
                }
        
        # Semantic change statistics
        stats['semantic_change'] = {}
        for genre in target_genres:
            col_name = f'semantic_change_{reference_genre}_{genre}'
            if col_name in df.columns:
                stats['semantic_change'][f'{reference_genre}_{genre}'] = {
                    'mean': df[col_name].mean(),
                    'std': df[col_name].std(),
                    'median': df[col_name].median(),
                    'min': df[col_name].min(),
                    'max': df[col_name].max()
                }
        
        # Cross-genre correlations
        stats['correlations'] = {}
        semantic_change_cols = [col for col in df.columns if col.startswith('semantic_change_')]
        
        if len(semantic_change_cols) > 1:
            corr_matrix = df[semantic_change_cols].corr()
            stats['correlations']['semantic_change_matrix'] = corr_matrix.to_dict()
        
        return stats
    
    def print_journal_ready_results(self, df: pd.DataFrame, stats: Dict, 
                                   reference_genre: str, alignment_method: str):
        """
        Print results in a format suitable for computational linguistics journals.
        """
        print(f"\n{'='*70}")
        print(f"GENRE-BASED SEMANTIC DRIFT ANALYSIS RESULTS")
        print(f"Reference Genre: {reference_genre.upper()}")
        print(f"Alignment Method: {alignment_method.upper()}")
        print(f"{'='*70}")
        
        target_genres = [genre for genre in self.genre_names if genre != reference_genre]
        
        print(f"\n1. COSINE SIMILARITY ANALYSIS")
        for genre in target_genres:
            key = f'{reference_genre}_{genre}'
            if key in stats['cosine_similarity']:
                sim_stats = stats['cosine_similarity'][key]
                print(f"   {reference_genre} → {genre}: μ={sim_stats['mean']:.3f} "
                      f"(σ={sim_stats['std']:.3f})")
        
        print(f"\n2. SEMANTIC CHANGE ANALYSIS (Cosine Distance)")
        for genre in target_genres:
            key = f'{reference_genre}_{genre}'
            if key in stats['semantic_change']:
                change_stats = stats['semantic_change'][key]
                print(f"   {reference_genre} → {genre}: μ={change_stats['mean']:.3f} "
                      f"(σ={change_stats['std']:.3f})")
        
        print(f"\n3. TOP 10 WORDS BY SEMANTIC CHANGE (vs {reference_genre})")
        # Find the column with highest average semantic change
        semantic_cols = [col for col in df.columns if col.startswith('semantic_change_')]
        if semantic_cols:
            # Use the first semantic change column or the one with highest mean
            main_change_col = max(semantic_cols, key=lambda col: df[col].mean())
            top_changed = df.nlargest(10, main_change_col)[
                ['word'] + [col for col in df.columns if 'cosine_similarity' in col or 'semantic_change' in col][:4]
            ]
            print(top_changed.round(3).to_string(index=False))
        
        print(f"\n4. MOST STABLE WORDS ACROSS GENRES")
        if semantic_cols:
            most_stable = df.nsmallest(10, main_change_col)[
                ['word'] + [col for col in df.columns if 'cosine_similarity' in col or 'semantic_change' in col][:4]
            ]
            print(most_stable.round(3).to_string(index=False))


def load_genre_models(models_dir: str = "models_by_genre") -> Dict[str, Word2Vec]:
    """
    Load all Word2Vec models from the specified directory.
    
    Args:
        models_dir: Directory containing the .model files
        
    Returns:
        Dictionary mapping model names to Word2Vec objects
    """
    models = {}
    
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory '{models_dir}' not found")
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.model')]
    
    print(f"Loading models from {models_dir}...")
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        # Extract model name
        model_name = model_file.replace('word2vec_', '').replace('.model', '')
        
        try:
            models[model_name] = Word2Vec.load(model_path)
            print(f"✓ Loaded {model_name}")
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
    
    return models


# MAIN
if __name__ == "__main__":
    
    # Load your models and data
    print("Loading genre models...")
    genre_models = load_genre_models("models_by_genre")
    
    # Load shared vocabulary 
    vocab_path = "models/alignment_vocab.pkl"
    if os.path.exists(vocab_path):
        with open(vocab_path, "rb") as f:
            shared_vocab = set(pickle.load(f))
    else:
        # Calculate shared vocabulary across all genre models
        print("Calculating shared vocabulary across genre models...")
        word_counts = {}
        for model in genre_models.values():
            for word in model.wv.key_to_index:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Words that appear in at least 3 models
        min_models = min(3, len(genre_models) // 2)
        shared_vocab = {word for word, count in word_counts.items() if count >= min_models}
        
        # Save for future use
        os.makedirs("models", exist_ok=True)
        with open(vocab_path, "wb") as f:
            pickle.dump(list(shared_vocab), f)
    
    print(f"Shared vocabulary size: {len(shared_vocab)}")
    
    # Initialize genre aligner
    aligner = GenreVectorSpaceAligner(genre_models, shared_vocab)
    
    # Target words for semantic drift analysis
    target_words = [
        "puella", "equus", "urbs", "terra", "caritas", "prex", "sacramentum", 
        "sacer", "sacerdos", "sacrificium", "lex", "spiritus", "fides", "pietas", 
        "fidelus", "gloria", "gratia", "gratus", "honor", "iustus", "magnus", 
        "parvus", "longus", "brevis", "altus", "novus", "vetus", "bonus", "malus", 
        "albus", "niger", "ruber"
    ]
    
    # Choose reference genre (e.g., 'history_classical' if available)
    available_genres = list(genre_models.keys())
    reference_genre = None
    
    # Try to find a good reference genre
    preferred_references = ['history_classical', 'poetry_classical', 'history_imperial']
    for pref in preferred_references:
        if pref in available_genres:
            reference_genre = pref
            break
    
    if reference_genre is None:
        reference_genre = available_genres[0]  # Use first available
    
    print(f"Using '{reference_genre}' as reference genre")
    
    # Perform alignments and comprehensive analysis
    print("\n" + "="*50)
    print("ORTHOGONAL ALIGNMENT")
    print("="*50)
    aligned_genres_orth = aligner.orthogonal_alignment(reference_genre)
    results_orth = aligner.comprehensive_semantic_analysis(aligned_genres_orth, target_words, reference_genre)
    stats_orth = aligner.generate_summary_statistics(results_orth, reference_genre)
    aligner.print_journal_ready_results(results_orth, stats_orth, reference_genre, "orthogonal")
    
    print("\n" + "="*50)
    print("LINEAR ALIGNMENT")
    print("="*50)
    aligned_genres_linear = aligner.linear_alignment(reference_genre)
    results_linear = aligner.comprehensive_semantic_analysis(aligned_genres_linear, target_words, reference_genre)
    stats_linear = aligner.generate_summary_statistics(results_linear, reference_genre)
    aligner.print_journal_ready_results(results_linear, stats_linear, reference_genre, "linear")
    
    # Save results to CSV for further analysis
    results_orth.to_csv('genre_semantic_drift_orthogonal.csv', index=False)
    results_linear.to_csv('genre_semantic_drift_linear.csv', index=False)
    
    print(f"\n\nResults saved to 'genre_semantic_drift_orthogonal.csv' and 'genre_semantic_drift_linear.csv'")
