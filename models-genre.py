#!/usr/bin/env python3
"""
Metadata-Driven Word2Vec Model Training for Latin Semantic Drift Analysis
=========================================================================

This script trains Word2Vec models on lemmatized Latin texts organized by genre and period
using YAML metadata configuration for systematic semantic drift analysis.

Author: [Your Name]
Date: [Current Date]
"""

import os
import pickle
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Iterator, Tuple, Optional
import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('word2vec_metadata_training.log'),
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

class LatinTextMetadataLoader:
    """Load metadata from YAML configuration"""
    
    def __init__(self, config_file: str = "metadata_config.yaml"):
        self.config_file = config_file
        self.config = None
        self.metadata = None
        
    def load_config(self) -> Dict:
        """Load YAML configuration file."""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {self.config_file}")
            return self.config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.config_file} not found")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
    
    def find_text_files(self, work_config: Dict) -> List[str]:
        """Find actual text files based on work configuration."""
        base_dir = self.config.get('base_directory', 'texts/')
        folder_path = work_config['folder_path']
        file_pattern = work_config['file_pattern']
        books = work_config.get('books', 1)
        
        full_path = os.path.join(base_dir, folder_path)
        found_files = []
        
        if books == 1:
            # Single file work
            file_path = os.path.join(full_path, file_pattern)
            if os.path.exists(file_path):
                found_files.append(file_path)
        else:
            # Multi-book work
            for book_num in range(1, books + 1):
                file_path = os.path.join(full_path, file_pattern.format(book_num))
                if os.path.exists(file_path):
                    found_files.append(file_path)
        
        return found_files
    
    def process_metadata(self) -> Dict[str, List]:
        """Process configuration into metadata structure."""
        if not self.config:
            self.load_config()
        
        # Initialize metadata dictionary
        metadata = {
            'text_id': [],
            'author': [],
            'work': [],
            'work_short': [],
            'period': [],
            'genre': [],
            'subgenre': [],
            'date_composed': [],
            'file_path': [],
            'file_exists': []
        }
        
        for work in self.config['works']:
            # Find actual files
            found_files = self.find_text_files(work)
            
            if found_files:
                for file_path in found_files:
                    text_id = f"{work['author'].lower().replace(' ', '_')}_{work['work_short']}"
                    
                    # Add to metadata
                    metadata['text_id'].append(text_id)
                    metadata['author'].append(work['author'])
                    metadata['work'].append(work['work'])
                    metadata['work_short'].append(work['work_short'])
                    metadata['period'].append(work['period'])
                    metadata['genre'].append(work['genre'])
                    metadata['subgenre'].append(work['subgenre'])
                    metadata['date_composed'].append(work.get('date_composed', 'Unknown'))
                    metadata['file_path'].append(file_path)
                    metadata['file_exists'].append(True)
            else:
                logger.warning(f"No files found for {work['author']} - {work['work']}")
        
        self.metadata = metadata
        return metadata

class MetadataCorpusReader:
    """
    Reads and preprocesses lemmatized Latin texts based on metadata groupings
    """
    
    def __init__(self, metadata_loader: LatinTextMetadataLoader, 
                 min_token_length: int = 2, exclude_function_words: bool = True):
        self.metadata_loader = metadata_loader
        self.min_token_length = min_token_length
        self.exclude_function_words = exclude_function_words
        
        # Load metadata
        if not metadata_loader.metadata:
            metadata_loader.process_metadata()
        self.metadata_df = pd.DataFrame(metadata_loader.metadata)
        
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
        
        logger.info(f"Loaded metadata for {len(self.metadata_df)} texts")
    
    def get_genre_period_combinations(self) -> List[Tuple[str, str]]:
        """Get all unique genre-period combinations"""
        combinations = []
        for period in self.metadata_df['period'].unique():
            for genre in self.metadata_df['genre'].unique():
                # Check if this combination exists
                subset = self.metadata_df[
                    (self.metadata_df['period'] == period) & 
                    (self.metadata_df['genre'] == genre)
                ]
                if len(subset) > 0:
                    combinations.append((genre, period))
        return combinations
    
    def read_sentences_for_group(self, genre: str, period: str) -> Iterator[List[str]]:
        """
        Generator that yields sentences as lists of tokens for a specific genre-period group
        """
        # Get file paths for this genre-period combination
        subset = self.metadata_df[
            (self.metadata_df['period'] == period) & 
            (self.metadata_df['genre'] == genre) &
            (self.metadata_df['file_exists'] == True)
        ]
        
        if len(subset) == 0:
            logger.warning(f"No texts found for {genre} in {period} period")
            return
        
        logger.info(f"Reading {len(subset)} texts for {genre} ({period})")
        
        for _, row in subset.iterrows():
            file_path = row['file_path']
            logger.info(f"Processing {row['author']} - {row['work']}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
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
    
    def get_corpus_stats(self, genre: str, period: str) -> Dict:
        """
        Get basic statistics about a genre-period corpus
        """
        sentences = list(self.read_sentences_for_group(genre, period))
        vocab = set()
        total_tokens = 0
        
        for sentence in sentences:
            vocab.update(sentence)
            total_tokens += len(sentence)
        
        # Get file count
        subset = self.metadata_df[
            (self.metadata_df['period'] == period) & 
            (self.metadata_df['genre'] == genre) &
            (self.metadata_df['file_exists'] == True)
        ]
        
        return {
            'genre': genre,
            'period': period,
            'num_files': len(subset),
            'num_sentences': len(sentences),
            'vocab_size': len(vocab),
            'total_tokens': total_tokens,
            'avg_sentence_length': total_tokens / len(sentences) if sentences else 0,
            'authors': subset['author'].tolist(),
            'works': subset['work'].tolist()
        }

class GenrePeriodWord2VecTrainer:
    """
    Trains Word2Vec models for specific genre-period combinations
    """
    
    def __init__(self, output_dir: str = 'models_by_genre'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Base parameters for historical texts
        self.base_params = {
            'vector_size': 300,      
            'min_count': 3,          # Lower for specialized corpora
            'workers': 4,            
            'epochs': 25,            # More epochs for smaller corpora
            'alpha': 0.025,          
            'min_alpha': 0.0001,     
            'sg': 1,                 # Skip-gram
            'hs': 0,                 
            'negative': 15,          
            'sample': 1e-5,          
            'seed': 42,              
        }
    
    def get_genre_specific_params(self, genre: str, period: str) -> dict:
        """
        Adjust parameters based on genre and period characteristics
        """
        params = self.base_params.copy()
        
        # Genre-specific adjustments
        if genre == 'poetry':
            params.update({
                'window': 5,             # Smaller window for poetic structure
                'min_count': 2,          # Lower threshold for poetic vocabulary
                'epochs': 30,            # More epochs for complex poetic language
            })
            logger.info(f"Using poetry parameters: window=5, min_count=2, epochs=30")
            
        elif genre == 'philosophy':
            params.update({
                'window': 8,             # Medium window for philosophical discourse
                'min_count': 3,          # Standard threshold
                'epochs': 25,
            })
            logger.info(f"Using philosophy parameters: window=8, min_count=3")
            
        elif genre == 'history':
            params.update({
                'window': 7,             # Medium window for narrative
                'min_count': 3,          # Standard threshold
                'epochs': 25,
            })
            logger.info(f"Using history parameters: window=7, min_count=3")
            
        elif genre == 'oratory':
            params.update({
                'window': 6,             # Smaller window for rhetorical structure
                'min_count': 2,          # Lower threshold for rhetorical variety
                'epochs': 25,
            })
            logger.info(f"Using oratory parameters: window=6, min_count=2")
            
        elif genre == 'theology':
            params.update({
                'window': 9,             # Larger window for theological discourse
                'min_count': 2,          # Lower threshold for specialized vocabulary
                'epochs': 30,            # More epochs for complex concepts
            })
            logger.info(f"Using theology parameters: window=9, min_count=2, epochs=30")
        
        # Period-specific adjustments
        if period == 'classical':
            params['min_count'] = max(params['min_count'], 3)  # Higher threshold for classical
        elif period == 'late':
            params['min_count'] = max(params['min_count'] - 1, 1)  # Lower threshold for late Latin
            params['epochs'] += 5  # More epochs for evolving language
            
        return params
    
    def train_model(self, corpus_reader: MetadataCorpusReader, 
                   genre: str, period: str) -> Optional[Word2Vec]:
        """
        Train Word2Vec model for a specific genre-period combination
        """
        logger.info(f"Training Word2Vec model for {genre} ({period})")
        
        # Get genre-period specific parameters
        model_params = self.get_genre_specific_params(genre, period)
        
        # Get corpus statistics
        stats = corpus_reader.get_corpus_stats(genre, period)
        logger.info(f"Corpus stats for {genre} ({period}): {stats}")
        
        # Check if we have enough data
        # if stats['num_sentences'] < 100:
        #     logger.warning(f"Too few sentences ({stats['num_sentences']}) for {genre} ({period}). Skipping.")
        #     return None
        
        # Prepare sentences
        sentences = list(corpus_reader.read_sentences_for_group(genre, period))
        
        if not sentences:
            logger.error(f"No sentences found for {genre} ({period})")
            return None
        
        try:
            # Initialize model
            model = Word2Vec(
                sentences=sentences,
                callbacks=[EpochLogger()],
                **model_params
            )
            
            # Create model filename
            model_filename = f'word2vec_{genre}_{period}.model'
            model_path = self.output_dir / model_filename
            model.save(str(model_path))
            logger.info(f"Model saved to {model_path}")
            
            # Save vocabulary
            vocab_filename = f'vocab_{genre}_{period}.pkl'
            vocab_path = self.output_dir / vocab_filename
            with open(vocab_path, 'wb') as f:
                pickle.dump(list(model.wv.key_to_index.keys()), f)
            
            # Save corpus statistics
            stats_filename = f'stats_{genre}_{period}.pkl'
            stats_path = self.output_dir / stats_filename
            with open(stats_path, 'wb') as f:
                pickle.dump(stats, f)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to train model for {genre} ({period}): {e}")
            return None
    
    def train_all_combinations(self, corpus_reader: MetadataCorpusReader) -> Dict[Tuple[str, str], Word2Vec]:
        """
        Train models for all available genre-period combinations
        """
        models = {}
        combinations = corpus_reader.get_genre_period_combinations()
        
        logger.info(f"Found {len(combinations)} genre-period combinations: {combinations}")
        
        for genre, period in combinations:
            try:
                model = self.train_model(corpus_reader, genre, period)
                if model is not None:
                    models[(genre, period)] = model
            except Exception as e:
                logger.error(f"Failed to train model for {genre} ({period}): {e}")
        
        return models

def create_cross_genre_alignment_vocab(corpus_reader: MetadataCorpusReader, 
                                     min_frequency: int = 3) -> Dict[str, List[str]]:
    """
    Create alignment vocabularies for semantic drift analysis
    """
    logger.info("Creating cross-genre alignment vocabularies")
    
    # Get all combinations
    combinations = corpus_reader.get_genre_period_combinations()
    
    # Count vocabulary across all genre-period combinations
    vocab_counts = {}
    
    for genre, period in combinations:
        combo_vocab = {}
        for sentence in corpus_reader.read_sentences_for_group(genre, period):
            for word in sentence:
                combo_vocab[word] = combo_vocab.get(word, 0) + 1
        
        # Filter by frequency
        filtered_vocab = {
            word for word, count in combo_vocab.items() 
            if count >= min_frequency
        }
        
        vocab_counts[(genre, period)] = filtered_vocab
        logger.info(f"{genre} ({period}): {len(filtered_vocab)} words above frequency {min_frequency}")
    
    # Create different alignment vocabularies
    alignment_vocabs = {}
    
    # 1. Cross-period alignment for each genre (diachronic analysis)
    genres = set(genre for genre, period in combinations)
    for genre in genres:
        genre_combos = [(g, p) for g, p in combinations if g == genre]
        if len(genre_combos) > 1:
            intersection = set.intersection(*[vocab_counts[combo] for combo in genre_combos])
            alignment_vocabs[f'{genre}_diachronic'] = sorted(list(intersection))
            logger.info(f"{genre} diachronic alignment: {len(intersection)} words")
    
    # 2. Cross-genre alignment for each period (synchronic analysis)  
    periods = set(period for genre, period in combinations)
    for period in periods:
        period_combos = [(g, p) for g, p in combinations if p == period]
        if len(period_combos) > 1:
            intersection = set.intersection(*[vocab_counts[combo] for combo in period_combos])
            alignment_vocabs[f'{period}_synchronic'] = sorted(list(intersection))
            logger.info(f"{period} synchronic alignment: {len(intersection)} words")
    
    # 3. Universal alignment (all combinations)
    if len(combinations) > 1:
        universal_intersection = set.intersection(*[vocab_counts[combo] for combo in combinations])
        alignment_vocabs['universal'] = sorted(list(universal_intersection))
        logger.info(f"Universal alignment: {len(universal_intersection)} words")
    
    return alignment_vocabs

def main():
    """
    Main training pipeline using metadata
    """
    # Configuration
    CONFIG_FILE = "metadata_config.yaml"  # Adjust path as needed
    
    # Initialize metadata loader
    metadata_loader = LatinTextMetadataLoader(CONFIG_FILE)
    
    # Initialize corpus reader
    corpus_reader = MetadataCorpusReader(
        metadata_loader,
        exclude_function_words=True
    )
    
    # Print available combinations
    combinations = corpus_reader.get_genre_period_combinations()
    logger.info(f"Available genre-period combinations: {combinations}")
    
    # Create alignment vocabularies
    alignment_vocabs = create_cross_genre_alignment_vocab(corpus_reader, min_frequency=3)
    
    # Save alignment vocabularies
    os.makedirs('models_by_genre', exist_ok=True)
    with open('models_by_genre/alignment_vocabularies.pkl', 'wb') as f:
        pickle.dump(alignment_vocabs, f)
    
    # Initialize trainer
    trainer = GenrePeriodWord2VecTrainer()
    
    # Train models for all combinations
    models = trainer.train_all_combinations(corpus_reader)
    
    # Create summary report
    report = []
    for (genre, period), model in models.items():
        vocab_size = len(model.wv.key_to_index)
        
        # Check alignment coverage for different types
        alignments = {}
        for align_type, align_vocab in alignment_vocabs.items():
            if genre in align_type or period in align_type or align_type == 'universal':
                aligned_words = len([w for w in align_vocab if w in model.wv.key_to_index])
                alignments[align_type] = {
                    'aligned_words': aligned_words,
                    'coverage': aligned_words / len(align_vocab) if align_vocab else 0
                }
        
        report.append({
            'genre': genre,
            'period': period,
            'vocab_size': vocab_size,
            'alignments': alignments
        })
    
    # Save detailed report
    with open('models_by_genre/training_report.pkl', 'wb') as f:
        pickle.dump(report, f)
    
    # Create CSV summary
    csv_report = []
    for item in report:
        base_data = {
            'genre': item['genre'],
            'period': item['period'],
            'vocab_size': item['vocab_size']
        }
        
        # Add alignment data
        for align_type, align_data in item['alignments'].items():
            csv_report.append({
                **base_data,
                'alignment_type': align_type,
                'aligned_words': align_data['aligned_words'],
                'coverage': align_data['coverage']
            })
    
    df_report = pd.DataFrame(csv_report)
    df_report.to_csv('models_by_genre/training_summary.csv', index=False)
    
    logger.info("Training complete!")
    logger.info(f"Trained {len(models)} models for genre-period combinations")
    logger.info(f"Models saved in: {trainer.output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("WORD2VEC TRAINING SUMMARY")
    print("="*60)
    
    for (genre, period), model in models.items():
        print(f"\n{genre.upper()} ({period}):")
        print(f"  Vocabulary size: {len(model.wv.key_to_index)}")
        
        # Show some example words
        vocab_sample = list(model.wv.key_to_index.keys())[:10]
        print(f"  Sample vocabulary: {vocab_sample}")

if __name__ == "__main__":
    main()