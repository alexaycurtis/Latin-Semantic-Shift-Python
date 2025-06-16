#Download Latin Models
import cltk
from cltk import NLP
from cltk.data.fetch import FetchCorpus
from pathlib import Path
import os
import stanza

print("Downloading Stanza Latin models...")
try:
    stanza.download('la')  # Download Latin models for Stanza
    print("Stanza Latin models downloaded successfully!")
except Exception as e:
    print(f"Error downloading Stanza models: {e}")
    print("Make sure you have internet connection and sufficient disk space")

# Download CLTK-specific models
print("Downloading CLTK Latin models...")
corpus_downloader = FetchCorpus(language="lat")

# Download essential Latin corpora for additional processing
try:
    corpus_downloader.import_corpus("lat_models_cltk")  # Main Latin models
    corpus_downloader.import_corpus("lat_text_latin_library")  # Additional Latin texts
    print("CLTK models downloaded successfully!")
except Exception as e:
    print(f"Error downloading CLTK models: {e}")
    print("You may need to download manually or check your internet connection")

# Initialize the NLP pipeline for Latin
print("Initializing Latin NLP pipeline...")
cltk_nlp = NLP(language="lat")
print(cltk_nlp.pipeline)

# Test the pipeline to ensure all components are loaded
test_text = "Gallia est omnis divisa in partes tres"
try:
    test_doc = cltk_nlp.analyze(test_text)
    print("Pipeline initialized successfully!")
    print(f"Test analysis complete - found {len(test_doc.tokens)} tokens")
    
    # Check what attributes are available in tokens
    if test_doc.tokens:
        sample_token = test_doc.tokens[0]
        print(f"Available attributes for tokens: {dir(sample_token)}")
        
    # Check if lemmata are available at document level
    if hasattr(test_doc, 'lemmata') and test_doc.lemmata:
        print(f"Document-level lemmata available: {test_doc.lemmata[:5]}")
        
except Exception as e:
    print(f"Pipeline initialization error: {e}")
    print("Make sure Stanza Latin models are properly downloaded")

#Identifying Authors for Each Period
#Classical: Cicero, Caesar, Livy, Ovid
#Imperial: 	Seneca, Apuleius, Tertullian
#Late: 	Augustine, Jerome, Ambrose

#Lemmatize text for each period

#Function to lemmatize a specific text
def lemmatize_text_cltk(text):
    doc = cltk_nlp.analyze(text)
    for token in doc.tokens:
        print(vars(token))  # See what's inside
    return [
        token.lemma
        for token in doc.tokens
        if hasattr(token, "pos") and token.pos and token.lemma and token.string.isalpha()
    ]

#Function that loops to create lemmatized versions of each period folder
def lemmatize_folder(input_dir, output_dir):
    """
    Process all .txt files in input directory and save lemmatized versions
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
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
            
            # Read the file with error handling for encoding
            try:
                raw_text = file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    raw_text = file.read_text(encoding="latin-1")
                    print(f"  Used latin-1 encoding for {file.name}")
                except:
                    print(f"  Could not read {file.name} - skipping")
                    continue
            
            # Skip empty files
            if not raw_text.strip():
                print(f"  {file.name} is empty - skipping")
                continue
            
            # Lemmatize the text
            lemmas = lemmatize_text_cltk(raw_text)
            
            if lemmas:
                # Join lemmas into a single string
                lemmatized_text = " ".join(lemmas)
                
                # Write to output file
                output_file = output_dir / file.name
                output_file.write_text(lemmatized_text, encoding="utf-8")
                
                print(f"  ✓ Lemmatized: {file.name} ({len(lemmas)} lemmas)")
            else:
                print(f"  ✗ No lemmas extracted from {file.name}")
                
        except Exception as e:
            print(f"  Error processing {file.name}: {e}")

# Run the lemmatize function for each time period
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting lemmatization process...")
    print("="*50)
    
    # Process each period
    periods = [
        ("data/classical", "lemmatized/classical"),
        ("data/imperial", "lemmatized/imperial"),
        ("data/late", "lemmatized/late")
    ]
    
    for input_dir, output_dir in periods:
        print(f"\n--- Processing {input_dir} ---")
        lemmatize_folder(input_dir, output_dir)
    
    print("\n" + "="*50)
    print("Lemmatization process complete!")
    print("="*50)
