# import yaml
# import os
# from pathlib import Path
# import glob
# from typing import Dict, List, Optional

# class LatinTextMetadataLoader:
# # Import Metadata
#     def __init__(self, config_file: str = "metadata_config.yaml"):
#         """Initialize with configuration file path."""
#         self.config_file = config_file
#         self.config = None
#         self.metadata = None
        
#     def load_config(self) -> Dict:
#         """Load YAML configuration file."""
#         try:
#             with open(self.config_file, 'r', encoding='utf-8') as file:
#                 self.config = yaml.safe_load(file)
#             print(f"✓ Loaded configuration from {self.config_file}")
#             return self.config
#         except FileNotFoundError:
#             raise FileNotFoundError(f"Configuration file {self.config_file} not found")
#         except yaml.YAMLError as e:
#             raise ValueError(f"Error parsing YAML file: {e}")
    
#     def get_word_count(self, file_path: str) -> int:
#         """Count words in a text file."""
#         try:
#             encoding = self.config.get('defaults', {}).get('encoding', 'utf-8')
#             with open(file_path, 'r', encoding=encoding) as f:
#                 content = f.read()
#                 return len(content.split())
#         except FileNotFoundError:
#             print(f"⚠ File not found: {file_path}")
#             return 0
#         except Exception as e:
#             print(f"⚠ Error reading {file_path}: {e}")
#             return 0
    
#     def find_text_files(self, work_config: Dict) -> List[str]:
#         """Find actual text files based on work configuration."""
#         base_dir = self.config.get('base_directory', 'texts/')
#         folder_path = work_config['folder_path']
#         file_pattern = work_config['file_pattern']
#         books = work_config.get('books', 1)
        
#         full_path = os.path.join(base_dir, folder_path)
#         found_files = []
        
#         if books == 1:
#             # Single file work
#             file_path = os.path.join(full_path, file_pattern)
#             if os.path.exists(file_path):
#                 found_files.append(file_path)
#         else:
#             # Multi-book work
#             for book_num in range(1, books + 1):
#                 file_path = os.path.join(full_path, file_pattern.format(book_num))
#                 if os.path.exists(file_path):
#                     found_files.append(file_path)
        
#         return found_files
    
#     def generate_text_id(self, author: str, work_short: str, book_num: Optional[int] = None) -> str:
#         """Generate standardized text ID."""
#         author_short = author.lower().replace(' ', '_')
#         if book_num:
#             return f"{author_short}_{work_short}_{book_num:02d}"
#         else:
#             return f"{author_short}_{work_short}"
    
#     def process_metadata(self) -> Dict[str, List]:
#         """Process configuration into the desired metadata structure."""
#         if not self.config:
#             self.load_config()
        
#         # Initialize metadata dictionary
#         metadata = {
#             'text_id': [],
#             'author': [],
#             'work': [],
#             'work_short': [],
#             'period': [],
#             'genre': [],
#             'subgenre': [],
#             'date_composed': [],
#             'book_number': [],
#             'total_books': [],
#             'file_path': [],
#             'word_count': [],
#             'file_exists': []
#         }
        
#         for work in self.config['works']:
#             # Find actual files
#             found_files = self.find_text_files(work)
#             books = work.get('books', 1)
            
#             if books == 1:
#                 # Single file work
#                 text_id = self.generate_text_id(work['author'], work['work_short'])
#                 file_path = found_files[0] if found_files else "FILE_NOT_FOUND"
#                 word_count = self.get_word_count(file_path) if found_files else 0
                
#                 # Add to metadata
#                 metadata['text_id'].append(text_id)
#                 metadata['author'].append(work['author'])
#                 metadata['work'].append(work['work'])
#                 metadata['work_short'].append(work['work_short'])
#                 metadata['period'].append(work['period'])
#                 metadata['genre'].append(work['genre'])
#                 metadata['subgenre'].append(work['subgenre'])
#                 metadata['date_composed'].append(work.get('date_composed', 'Unknown'))
#                 metadata['book_number'].append(1)
#                 metadata['total_books'].append(1)
#                 metadata['file_path'].append(file_path)
#                 metadata['word_count'].append(word_count)
#                 metadata['file_exists'].append(len(found_files) > 0)
                
#             else:
#                 # Multi-book work
#                 for book_num in range(1, books + 1):
#                     text_id = self.generate_text_id(work['author'], work['work_short'], book_num)
                    
#                     # Find corresponding file
#                     book_file = None
#                     for file_path in found_files:
#                         if f"_{book_num:02d}" in file_path or f"_{book_num}" in file_path:
#                             book_file = file_path
#                             break
                    
#                     if not book_file:
#                         book_file = "FILE_NOT_FOUND"
                    
#                     word_count = self.get_word_count(book_file) if book_file != "FILE_NOT_FOUND" else 0
                    
#                     # Add to metadata
#                     metadata['text_id'].append(text_id)
#                     metadata['author'].append(work['author'])
#                     metadata['work'].append(work['work'])
#                     metadata['work_short'].append(work['work_short'])
#                     metadata['period'].append(work['period'])
#                     metadata['genre'].append(work['genre'])
#                     metadata['subgenre'].append(work['subgenre'])
#                     metadata['date_composed'].append(work.get('date_composed', 'Unknown'))
#                     metadata['book_number'].append(book_num)
#                     metadata['total_books'].append(books)
#                     metadata['file_path'].append(book_file)
#                     metadata['word_count'].append(word_count)
#                     metadata['file_exists'].append(book_file != "FILE_NOT_FOUND")
        
#         self.metadata = metadata
#         return metadata
    
#     def get_summary(self) -> Dict:
#         """Get summary statistics of the collection."""
#         if not self.metadata:
#             self.process_metadata()
        
#         total_texts = len(self.metadata['text_id'])
#         total_words = sum(self.metadata['word_count'])
#         existing_files = sum(self.metadata['file_exists'])
#         missing_files = total_texts - existing_files
        
#         # Count by categories
#         authors = set(self.metadata['author'])
#         periods = set(self.metadata['period'])
#         genres = set(self.metadata['genre'])
        
#         return {
#             'total_texts': total_texts,
#             'existing_files': existing_files,
#             'missing_files': missing_files,
#             'total_words': total_words,
#             'unique_authors': len(authors),
#             'unique_periods': len(periods),
#             'unique_genres': len(genres),
#             'authors': sorted(authors),
#             'periods': sorted(periods),
#             'genres': sorted(genres)
#         }
    
#     def print_summary(self):
#         """Print a formatted summary of the collection."""
#         summary = self.get_summary()
        
#         print("\n" + "="*50)
#         print("LATIN TEXT COLLECTION SUMMARY")
#         print("="*50)
#         print(f"Total texts configured: {summary['total_texts']}")
#         print(f"Files found: {summary['existing_files']}")
#         print(f"Files missing: {summary['missing_files']}")
#         print(f"Total word count: {summary['total_words']:,}")
#         print(f"\nUnique authors: {summary['unique_authors']}")
#         print(f"Periods: {', '.join(summary['periods'])}")
#         print(f"Genres: {', '.join(summary['genres'])}")
        
#         if summary['missing_files'] > 0:
#             print(f"\n⚠ Warning: {summary['missing_files']} files not found")
    
#     def export_to_dict(self) -> Dict[str, List]:
#         """Export processed metadata as dictionary (your original format)."""
#         if not self.metadata:
#             self.process_metadata()
#         return self.metadata
    
#     def export_missing_files(self) -> List[str]:
#         """Get list of missing file paths."""
#         if not self.metadata:
#             self.process_metadata()
        
#         missing = []
#         for i, exists in enumerate(self.metadata['file_exists']):
#             if not exists:
#                 missing.append(self.metadata['file_path'][i])
#         return missing


# #Main
# if __name__ == "__main__":
#     # Initialize loader
#     loader = LatinTextMetadataLoader("metadata_config.yaml")
    
#     # Load and process
#     try:
#         config = loader.load_config()
#         metadata = loader.process_metadata()
        
#         # Print summary
#         loader.print_summary()
        
#         # Show sample of metadata
#         print(f"\nFirst 3 text IDs: {metadata['text_id'][:3]}")
#         print(f"First 3 authors: {metadata['author'][:3]}")
#         print(f"First 3 word counts: {metadata['word_count'][:3]}")
        
#         # Show any missing files
#         missing = loader.export_missing_files()
#         if missing:
#             print(f"\nMissing files:")
#             for file in missing[:5]:  # Show first 5
#                 print(f"  - {file}")
        
#     except Exception as e:
#         print(f"Error: {e}")