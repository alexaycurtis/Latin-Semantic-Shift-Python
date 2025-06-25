# Pure Stanza Approach for Latin Lemmatization (Bypasses CLTK UD Feature Issues)
import stanza
from pathlib import Path
import re

# Download Stanza Latin models
print("Downloading Stanza Latin models...")
try:
    stanza.download('la')
    print("Stanza Latin models downloaded successfully!")
except Exception as e:
    print(f"Error downloading Stanza models: {e}")

# Initialize Stanza pipeline directly (no CLTK wrapper)
print("Initializing Stanza Latin pipeline...")
try:
    stanza_nlp = stanza.Pipeline(
        'la', 
        processors='tokenize,pos,lemma', 
        use_gpu=False, 
        verbose=False,
        download_method=None 
    )
    print("Stanza pipeline initialized successfully!")
except Exception as e:
    print(f"Error initializing Stanza pipeline: {e}")
    exit(1)

# Test the pipeline
test_text = "Gallia est omnis divisa in partes tres"
try:
    test_doc = stanza_nlp(test_text)
    print(f"Test analysis complete - found {len(test_doc.sentences[0].words)} words")
    
    # Show lemmatization results
    lemmas = [word.lemma for word in test_doc.sentences[0].words if word.text.isalpha()]
    print(f"Test lemmas: {lemmas}")
    
    # Show detailed word analysis
    print("\nDetailed analysis:")
    for word in test_doc.sentences[0].words:
        print(f"  {word.text} -> {word.lemma} (POS: {word.pos})")
        
except Exception as e:
    print(f"Pipeline test error: {e}")

# Function to clean and preprocess Latin text
def clean_latin_text(text):
    """
    Clean Latin text for better lemmatization
    Removes numbers, excessive whitespace, and normalizes text
    """
    # Remove numbers and digits
    text = re.sub(r'\d+', '', text)
    
    # Remove excessive punctuation but keep sentence boundaries
    text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove very short lines that might be headers/page numbers
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
    
    return '\n'.join(cleaned_lines)

# Function to lemmatize text using pure Stanza
def lemmatize_text_stanza(text):
    """
    Lemmatize Latin text using Stanza directly
    Returns list of lemmas for alphabetic tokens
    """
    try:
        # Clean the text first
        clean_text = clean_latin_text(text)
        
        if not clean_text.strip():
            print("Warning: Text is empty after cleaning")
            return []
        
        # Process with Stanza
        doc = stanza_nlp(clean_text)
        lemmas = []
        
        for sentence in doc.sentences:
            for word in sentence.words:
                # Only include alphabetic words longer than 1 character
                if (word.lemma and 
                    word.text.isalpha() and 
                    len(word.text) > 1 and
                    word.lemma.isalpha()):
                    lemmas.append(word.lemma.lower())
                elif (word.text.isalpha() and 
                      len(word.text) > 1):
                    # Fallback to original word if no lemma
                    lemmas.append(word.text.lower())
        
        return lemmas
    
    except Exception as e:
        print(f"Error lemmatizing text: {e}")
        return []

# Function to process folders with better error handling
def lemmatize_folder(input_dir, output_dir):
    """
    Process all .txt files in input directory and save lemmatized versions
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist!")
        return
    
    txt_files = list(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    print(f"Processing {len(txt_files)} files from {input_dir}...")
    
    for file in txt_files:
        try:
            print(f"Processing: {file.name}")
            
            # Read file with multiple encoding attempts
            raw_text = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    raw_text = file.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if raw_text is None:
                print(f"  Could not read {file.name} with any encoding - skipping")
                continue
            
            # Skip empty files
            if not raw_text.strip():
                print(f"  {file.name} is empty - skipping")
                continue
            
            # Show file stats
            word_count = len(raw_text.split())
            print(f"  File contains ~{word_count} words")
            
            # Lemmatize the text
            lemmas = lemmatize_text_stanza(raw_text)
            
            if lemmas:
                # Join lemmas into a single string
                lemmatized_text = " ".join(lemmas)
                
                # Write to output file
                output_file = output_dir / file.name
                output_file.write_text(lemmatized_text, encoding="utf-8")
                
                print(f"  ✓ Lemmatized: {file.name} ({len(lemmas)} lemmas)")
                
                # Show reduction ratio
                reduction = round((1 - len(lemmas) / word_count) * 100, 1)
                print(f"    Size reduction: {reduction}% (vocabulary normalization)")
            else:
                print(f"  ✗ No lemmas extracted from {file.name}")
                
        except Exception as e:
            print(f"  Error processing {file.name}: {e}")

# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("STANZA-ONLY LATIN LEMMATIZATION")
    print("(Bypasses CLTK UD feature compatibility issues)")
    print("="*60)
    
    # Process each period
    periods = [
        ("data/extradata", "lemmatized/extradata")
    ]
    
    for input_dir, output_dir in periods:
        print(f"\n--- Processing {input_dir} ---")
        lemmatize_folder(input_dir, output_dir)
    
    print("\n" + "="*60)
    print("LEMMATIZATION COMPLETE!")
    print("="*60)