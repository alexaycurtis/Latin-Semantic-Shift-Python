#!/usr/bin/env python3
"""
Word2Vec Model Training for Latin Semantic Drift Analysis
=========================================================

This script trains Word2Vec models on lemmatized Latin texts from different periods
(Classical, Imperial, Late Roman) to enable semantic drift analysis.

Author: [Your Name]
Date: [Current Date]
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Iterator, Tuple
import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('word2vec_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EpochLogger(CallbackAny2Vec):
    """Callback to log training progress"""
    def __init__(self):
        self.epoch = 0
        
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        logger.info(f'Epoch {self.epoch}: Loss = {loss}')
        self.epoch += 1

class LatinCorpusReader:
    """
    Reads and preprocesses lemmatized Latin texts for Word2Vec training
    """
    
    def __init__(self, data_dir: str, min_token_length: int = 2, exclude_function_words: bool = True):
        self.data_dir = Path(data_dir)
        self.min_token_length = min_token_length
        self.exclude_function_words = exclude_function_words
        self.periods = ['classical', 'imperial', 'late']

        # Common Latin function words to exclude
        self.function_words = {
            'et', 'in', 'ad', 'de', 'ex', 'cum', 'per', 'pro', 'ab', 'a',
            'est', 'sunt', 'esse', 'eram', 'erat', 'erant', 'fuit', 'fuerit',
            'hic', 'haec', 'hoc', 'ille', 'illa', 'illud', 'is', 'ea', 'id',
            'qui', 'quae', 'quod', 'quis', 'quid', 'quem', 'quam',
            'non', 'ne', 'nihil', 'nec', 'neque', 'nulla', 'nullus',
            'sed', 'autem', 'enim', 'igitur', 'ergo', 'itaque',
            'si', 'nisi', 'ut', 'ne', 'cum', 'quando', 'ubi',
            'me', 'te', 'se', 'nos', 'vos', 'mihi', 'tibi', 'sibi',
            'meus', 'tuus', 'suus', 'noster', 'vester'
        }
        
    def read_sentences(self, period: str) -> Iterator[List[str]]:
        """
        Generator that yields sentences as lists of tokens for a given period
        """
        period_dir = self.data_dir / period
        if not period_dir.exists():
            logger.warning(f"Directory {period_dir} does not exist")
            return
            
        for file_path in period_dir.glob('*.txt'):
            logger.info(f"Reading {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Split by sentences (assuming one sentence per line)
                        tokens = line.strip().split()
                        # Filter tokens
                        filtered_tokens = []
                        for token in tokens:
                            token_lower = token.lower()
                            # Basic filtering
                            if (len(token) >= self.min_token_length 
                                and token.isalpha()
                                and len(token) <= 20):  # Remove very long tokens
                                
                                # Apply function word filtering if enabled
                                if self.exclude_function_words:
                                    if token_lower not in self.function_words:
                                        filtered_tokens.append(token_lower)
                                else:
                                    filtered_tokens.append(token_lower)
                        if len(filtered_tokens) >= 3:  # Minimum sentence length
                            yield filtered_tokens
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                
    def get_corpus_stats(self, period: str) -> Dict:
        """
        Get basic statistics about the corpus
        """
        sentences = list(self.read_sentences(period))
        vocab = set()
        total_tokens = 0
        
        for sentence in sentences:
            vocab.update(sentence)
            total_tokens += len(sentence)
            
        return {
            'period': period,
            'num_sentences': len(sentences),
            'vocab_size': len(vocab),
            'total_tokens': total_tokens,
            'avg_sentence_length': total_tokens / len(sentences) if sentences else 0
        }

class Word2VecTrainer:
    """
    Trains Word2Vec models for different periods with optimized parameters
    """
    
    def __init__(self, output_dir: str = 'models'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Optimized parameters for historical texts
        self.model_params = {
            'vector_size': 300,      # Higher dimensionality for better semantic representation
            'window': 10,            # Larger window for historical texts (less rigid syntax)
            'min_count': 5,          # Conservative threshold for rare words
            'workers': 4,            # Parallel processing
            'epochs': 20,            # More epochs for better convergence
            'alpha': 0.025,          # Learning rate
            'min_alpha': 0.0001,     # Minimum learning rate
            'sg': 1,                 # Skip-gram (better for rare words)
            'hs': 0,                 # Use negative sampling
            'negative': 20,          # More negative samples for better learning
            'sample': 1e-4,          # Subsampling threshold
            'seed': 42 ,             # Reproducibility
        }

    def get_period_specific_params(self, period: str) -> dict:
        """
        Adjust parameters based on period characteristics
        """
        params = self.model_params.copy()
        
        if period == 'classical':
            # Classical Latin - more standardized
            params.update({
                'min_count': 4,          # Higher threshold for classical texts
                'window': 6,             # Smaller window for more formal syntax
                'sample': 1e-4,          # Higher subsampling threshold
            })
            logger.info(f"Using classical period parameters: min_count=4, window=6")
            
        elif period == 'imperial':
            # Imperial Latin - balance of formal and colloquial
            params.update({
                'min_count': 3,
                'window': 8,
                'sample': 1e-5,
            })
            logger.info(f"Using imperial period parameters: min_count=3, window=8")
            
        elif period == 'late_roman':
            # Late Latin - more variation, Christian terminology
            params.update({
                'min_count': 2,          # Lower threshold for evolving vocabulary
                'window': 10,            # Larger window for changing syntax
                'sample': 1e-6,          # Even lower subsampling
                'epochs': 35,            # More epochs for complex period
            })
            logger.info(f"Using late Roman parameters: min_count=2, window=10, epochs=35")
            
        return params

        
    def train_model(self, corpus_reader: LatinCorpusReader, period: str) -> Word2Vec:
        """
        Train Word2Vec model for a specific period
        """
        logger.info(f"Training Word2Vec model for {period} period")
        
        
        # ADD THIS LINE TO GET PERIOD-SPECIFIC PARAMETERS:
        model_params = self.get_period_specific_params(period)
        
        # Get corpus statistics
        stats = corpus_reader.get_corpus_stats(period)
        logger.info(f"Corpus stats for {period}: {stats}")
        
        # Prepare sentences
        sentences = list(corpus_reader.read_sentences(period))
        
        if not sentences:
            raise ValueError(f"No sentences found for period {period}")
            
        # Initialize model
        model = Word2Vec(
            sentences=sentences,
            callbacks=[EpochLogger()],
            **model_params
        )
        
        # Save model
        model_path = self.output_dir / f'word2vec_{period}.model'
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Save vocabulary
        vocab_path = self.output_dir / f'vocab_{period}.pkl'
        with open(vocab_path, 'wb') as f:
            pickle.dump(list(model.wv.key_to_index.keys()), f)
            
        return model
        
    def train_all_periods(self, corpus_reader: LatinCorpusReader) -> Dict[str, Word2Vec]:
        """
        Train models for all periods
        """
        models = {}
        
        for period in corpus_reader.periods:
            try:
                models[period] = self.train_model(corpus_reader, period)
            except Exception as e:
                logger.error(f"Failed to train model for {period}: {e}")
                
        return models

def create_alignment_corpus(corpus_reader: LatinCorpusReader, min_frequency: int = 5, exclude_function_words: bool = True) -> List[str]:
    """
    Create a vocabulary of words that appear in all periods for alignment
    This is crucial for semantic drift analysis
    """    
    logger.info("Creating alignment vocabulary")
    
    # ADD THIS FUNCTION WORDS SET:
    function_words = {
        'et', 'in', 'ad', 'de', 'ex', 'cum', 'per', 'pro', 'ab', 'a',
        'est', 'sunt', 'esse', 'eram', 'erat', 'erant', 'fuit', 'fuerit',
        'hic', 'haec', 'hoc', 'ille', 'illa', 'illud', 'is', 'ea', 'id',
        'qui', 'quae', 'quod', 'quis', 'quid', 'quem', 'quam',
        'non', 'ne', 'nihil', 'nec', 'neque', 'nulla', 'nullus',
        'sed', 'autem', 'enim', 'igitur', 'ergo', 'itaque',
        'si', 'nisi', 'ut', 'ne', 'cum', 'quando', 'ubi'
    }

        
    period_vocabs = {}
    
    for period in corpus_reader.periods:
        vocab_count = {}
        for sentence in corpus_reader.read_sentences(period):
            for word in sentence:
                vocab_count[word] = vocab_count.get(word, 0) + 1
        
        # Filter by frequency
        if exclude_function_words:
            period_vocabs[period] = {
                word for word, count in vocab_count.items() 
                if count >= min_frequency and word not in function_words
            }
        else:
            period_vocabs[period] = {
                word for word, count in vocab_count.items() 
                if count >= min_frequency
            }

        logger.info(f"{period}: {len(period_vocabs[period])} words above frequency {min_frequency}")
    
    # Find intersection
    alignment_vocab = set.intersection(*period_vocabs.values())
    logger.info(f"Alignment vocabulary: {len(alignment_vocab)} words")
    
    return sorted(list(alignment_vocab))

def main():
    """
    Main training pipeline
    """
    # Configuration
    DATA_DIR = 'lemmatized'  # Adjust path as needed
    
    # Initialize corpus reader
    corpus_reader = LatinCorpusReader(
        DATA_DIR,
        exclude_function_words=True  # Control function word filtering here
        )
    
    # Create alignment vocabulary
    alignment_vocab = create_alignment_corpus(
    corpus_reader, 
    min_frequency=5,
    exclude_function_words=True  # Set to False to include function words in alignment
    )
    
    # Save alignment vocabulary
    os.makedirs('models', exist_ok=True)
    with open('models/alignment_vocab.pkl', 'wb') as f:
        pickle.dump(alignment_vocab, f)
    
    # Initialize trainer
    trainer = Word2VecTrainer()
    
    # Train models
    models = trainer.train_all_periods(corpus_reader)
    
    # Create summary report
    report = []
    for period, model in models.items():
        vocab_size = len(model.wv.key_to_index)
        aligned_words = len([w for w in alignment_vocab if w in model.wv.key_to_index])
        
        report.append({
            'period': period,
            'vocab_size': vocab_size,
            'aligned_words': aligned_words,
            'coverage': aligned_words / len(alignment_vocab) if alignment_vocab else 0
        })
    
    # Save report
    df_report = pd.DataFrame(report)
    df_report.to_csv('models/training_report.csv', index=False)
    
    logger.info("Training complete!")
    logger.info(f"Summary:\n{df_report}")

if __name__ == "__main__":
    main()