#Download Latin Models
import cltk
from cltk import NLP
from cltk.data.fetch import FetchCorpus
from pathlib import Path
import os


cltk_nlp = NLP(language="lat")

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
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in input_dir.glob("*.txt"):
        raw = file.read_text(encoding="utf-8")
        lemmas = lemmatize_text_cltk(raw)
        out_text = " ".join(lemmas)
        (output_dir / file.name).write_text(out_text, encoding="utf-8")
        print(f"Lemmatized: {file.name}")

#Run the lemmatize function for each time period


lemmatize_folder("data/classical", "lemmatized/classical")
lemmatize_folder("data/imperial", "lemmatized/imperial")
lemmatize_folder("data/late", "lemmatized/late")

