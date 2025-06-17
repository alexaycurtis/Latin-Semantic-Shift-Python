import numpy as np
import pickle
from gensim.models import Word2Vec
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import orthogonal_procrustes
import pandas as pd
from typing import Dict, List, Set, Tuple

class VectorSpaceAligner:
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
    
    def measure_semantic_drift(self, aligned_imperial: Dict, aligned_late: Dict, 
                             target_words: List[str]) -> pd.DataFrame:
        """
        Measure semantic drift using cosine similarity (higher = less drift).
        """
        results = []
        
        for word in target_words:
            if word in self.shared_vocab:
                # Calculate similarities (higher = less drift)
                sim_classical_imperial = cosine_similarity(
                    [self.model_classical.wv[word]], 
                    [aligned_imperial[word]]
                )[0][0]
                
                sim_classical_late = cosine_similarity(
                    [self.model_classical.wv[word]], 
                    [aligned_late[word]]
                )[0][0]
                
                # Calculate drift as 1 - similarity (higher = more drift)
                drift_imperial = 1 - sim_classical_imperial
                drift_late = 1 - sim_classical_late
                
                results.append({
                    'word': word,
                    'similarity_classical_imperial': sim_classical_imperial,
                    'similarity_classical_late': sim_classical_late,
                    'drift_classical_imperial': drift_imperial,
                    'drift_classical_late': drift_late,
                    'drift_change': drift_late - drift_imperial  # positive = more drift in late period
                })
        
        return pd.DataFrame(results)

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

# Try both alignment methods
print("\n" + "="*50)
print("ORTHOGONAL ALIGNMENT")
print("="*50)
aligned_imp_orth, aligned_late_orth = aligner.orthogonal_alignment()

print("\n" + "="*50)
print("LINEAR ALIGNMENT") 
print("="*50)
aligned_imp_linear, aligned_late_linear = aligner.linear_alignment()

# Target words for semantic drift analysis
target_words = [
    "puella", "equus", "urbs", "terra", "caritas", "prex", "sacramentum", 
    "sacer", "sacerdos", "sacrificium", "lex", "spiritus", "fides", "pietas", 
    "fidelus", "gloria", "gratia", "gratus", "honor", "iustus", "magnus", "parvus", "longus", "brevis", "altus", "novus", "vetus",
            "bonus", "malus", "albus", "niger", "ruber"
]

# Measure semantic drift with both alignment methods
print("\n" + "="*50)
print("SEMANTIC DRIFT ANALYSIS")
print("="*50)

print("\n--- Using Orthogonal Alignment ---")
results_orth = aligner.measure_semantic_drift(aligned_imp_orth, aligned_late_orth, target_words)
print(results_orth[['word', 'drift_classical_imperial', 'drift_classical_late', 'drift_change']].round(3))

print("\n--- Using Linear Alignment ---")
results_linear = aligner.measure_semantic_drift(aligned_imp_linear, aligned_late_linear, target_words)
print(results_linear[['word', 'drift_classical_imperial', 'drift_classical_late', 'drift_change']].round(3))

# Summary statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)

print("\nOrthogonal Alignment:")
print(f"Average drift Classical→Imperial: {results_orth['drift_classical_imperial'].mean():.3f}")
print(f"Average drift Classical→Late: {results_orth['drift_classical_late'].mean():.3f}")
print(f"Average drift change (Late - Imperial): {results_orth['drift_change'].mean():.3f}")

print("\nLinear Alignment:")
print(f"Average drift Classical→Imperial: {results_linear['drift_classical_imperial'].mean():.3f}")
print(f"Average drift Classical→Late: {results_linear['drift_classical_late'].mean():.3f}")
print(f"Average drift change (Late - Imperial): {results_linear['drift_change'].mean():.3f}")

# Identify words with highest drift
print(f"\nTop 5 words with highest drift in Late period (Linear Alignment):")
top_drift = results_linear.nlargest(5, 'drift_classical_late')[['word', 'drift_classical_late']]
print(top_drift.round(3))