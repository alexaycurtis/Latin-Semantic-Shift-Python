from genre_metaloader import LatinTextMetadataLoader
import pandas as pd

# Initialize the loader
loader = LatinTextMetadataLoader("metadata_config.yaml")

# Load configuration and process metadata
metadata = loader.process_metadata()

# Print summary statistics
loader.print_summary()

# ==========================================
# ACCESSING YOUR METADATA
# ==========================================

#get metadata 
genre_metadata = loader.export_to_dict()

#check
print("Text IDs:", genre_metadata['text_id'][:5])
print("Authors:", genre_metadata['author'][:5])
print("Periods:", genre_metadata['period'][:5])
print("Word counts:", genre_metadata['word_count'][:5])

#Convert to DataFrame for easier manipulation
df = pd.DataFrame(genre_metadata)


#Arrange by period and genre

#Philosophy (Classical - Imperial)
#History (All 3)
#Poetry (ALl 3)
#Oratory (ALl 3)
#Theology (Imperial - Late)

print("\n" + "="*60)
print("SEMANTIC DRIFT ANALYSIS GROUPINGS")
print("="*60)

# Group by period for diachronic analysis
period_groups = {
    'classical': df[df['period'] == 'classical']['text_id'].tolist(),
    'imperial': df[df['period'] == 'imperial']['text_id'].tolist(),
    'late': df[df['period'] == 'late']['text_id'].tolist()
}

print("\n📊 DIACHRONIC GROUPS (Period-based):")
for period, text_ids in period_groups.items():
    word_count = df[df['text_id'].isin(text_ids)]['word_count'].sum()
    print(f"  {period.upper()}: {len(text_ids)} texts, {word_count:,} words")
    print(f"    Text IDs: {text_ids[:3]}{'...' if len(text_ids) > 3 else ''}")

# Group by genre for synchronic analysis
genre_groups = {}
for genre in df['genre'].unique():
    genre_groups[genre] = df[df['genre'] == genre]['text_id'].tolist()

print("\n📚 SYNCHRONIC GROUPS (Genre-based):")
for genre, text_ids in genre_groups.items():
    word_count = df[df['text_id'].isin(text_ids)]['word_count'].sum()
    print(f"  {genre.upper()}: {len(text_ids)} texts, {word_count:,} words")
    print(f"    Text IDs: {text_ids[:3]}{'...' if len(text_ids) > 3 else ''}")

# Cross-tabulation for period-genre combinations
print("\n🔍 PERIOD × GENRE MATRIX:")
crosstab = pd.crosstab(df['period'], df['genre'], margins=True)
print(crosstab)

# Detailed groupings for semantic drift analysis
semantic_drift_groups = {
    # Diachronic: Same genre across periods
    'poetry_diachronic': {
        'classical_poetry': df[(df['period'] == 'classical') & (df['genre'] == 'poetry')]['text_id'].tolist(),
        'imperial_poetry': df[(df['period'] == 'imperial') & (df['genre'] == 'poetry')]['text_id'].tolist(),
        'late_poetry': df[(df['period'] == 'late') & (df['genre'] == 'poetry')]['text_id'].tolist()
    },
    'history_diachronic': {
        'classical_history': df[(df['period'] == 'classical') & (df['genre'] == 'history')]['text_id'].tolist(),
        'imperial_history': df[(df['period'] == 'imperial') & (df['genre'] == 'history')]['text_id'].tolist(),
        'late_history': df[(df['period'] == 'late') & (df['genre'] == 'history')]['text_id'].tolist()
    },
    
    # Synchronic: Different genres within same period
    'classical_synchronic': {
        'classical_poetry': df[(df['period'] == 'classical') & (df['genre'] == 'poetry')]['text_id'].tolist(),
        'classical_history': df[(df['period'] == 'classical') & (df['genre'] == 'history')]['text_id'].tolist(),
        'classical_philosophy': df[(df['period'] == 'classical') & (df['genre'] == 'philosophy')]['text_id'].tolist()
    },
    'imperial_synchronic': {
        'imperial_poetry': df[(df['period'] == 'imperial') & (df['genre'] == 'poetry')]['text_id'].tolist(),
        'imperial_history': df[(df['period'] == 'imperial') & (df['genre'] == 'history')]['text_id'].tolist(),
        'imperial_philosophy': df[(df['period'] == 'imperial') & (df['genre'] == 'philosophy')]['text_id'].tolist()
    },
    'late_synchronic': {
        'late_theology': df[(df['period'] == 'late') & (df['genre'] == 'theology')]['text_id'].tolist(),
        'late_history': df[(df['period'] == 'late') & (df['genre'] == 'history')]['text_id'].tolist()
    }
}

print("\n🔬 DETAILED SEMANTIC DRIFT ANALYSIS GROUPS:")

# Poetry diachronic analysis
poetry_groups = semantic_drift_groups['poetry_diachronic']
print("\n  POETRY ACROSS TIME:")
for period_genre, text_ids in poetry_groups.items():
    if text_ids:  # Only show if texts exist
        word_count = df[df['text_id'].isin(text_ids)]['word_count'].sum()
        print(f"    {period_genre}: {len(text_ids)} texts, {word_count:,} words")

# History diachronic analysis  
history_groups = semantic_drift_groups['history_diachronic']
print("\n  HISTORY ACROSS TIME:")
for period_genre, text_ids in history_groups.items():
    if text_ids:
        word_count = df[df['text_id'].isin(text_ids)]['word_count'].sum()
        print(f"    {period_genre}: {len(text_ids)} texts, {word_count:,} words")

# ==========================================
# WORD2VEC ANALYSIS PREPARATION
# ==========================================

print("\n" + "="*60)
print("WORD2VEC SEMANTIC DRIFT ANALYSIS SETUP")
print("="*60)

# Create comparison pairs for semantic drift
comparison_pairs = [
    # Diachronic comparisons (same genre, different periods)
    ('classical_poetry', 'imperial_poetry', 'Poetry: Classical → Imperial'),
    ('classical_poetry', 'late_poetry', 'Poetry: Classical → Late'),
    ('imperial_poetry', 'late_poetry', 'Poetry: Imperial → Late'),
    
    ('classical_history', 'imperial_history', 'History: Classical → Imperial'),
    ('classical_history', 'late_history', 'History: Classical → Late'),
    ('imperial_history', 'late_history', 'History: Imperial → Late'),
    
    # Synchronic comparisons (same period, different genres)
    ('classical_poetry', 'classical_history', 'Classical: Poetry vs History'),
    ('classical_poetry', 'classical_philosophy', 'Classical: Poetry vs Philosophy'),
    ('classical_history', 'classical_philosophy', 'Classical: History vs Philosophy'),
    
    ('imperial_poetry', 'imperial_history', 'Imperial: Poetry vs History'),
    ('late_theology', 'late_history', 'Late: Theology vs History')
]

print("\n📋 SUGGESTED COMPARISON PAIRS:")
for group1, group2, description in comparison_pairs:
    # Get text IDs for each group
    texts1 = []
    texts2 = []
    
    # Find texts in the detailed groups
    for analysis_type, groups in semantic_drift_groups.items():
        if group1 in groups:
            texts1 = groups[group1]
        if group2 in groups:
            texts2 = groups[group2]
    
    # If not found in detailed groups, try simple period/genre combinations
    if not texts1:
        period, genre = group1.split('_')
        texts1 = df[(df['period'] == period) & (df['genre'] == genre)]['text_id'].tolist()
    if not texts2:
        period, genre = group2.split('_')
        texts2 = df[(df['period'] == period) & (df['genre'] == genre)]['text_id'].tolist()
    
    if texts1 and texts2:
        words1 = df[df['text_id'].isin(texts1)]['word_count'].sum()
        words2 = df[df['text_id'].isin(texts2)]['word_count'].sum()
        print(f"  {description}:")
        print(f"    Group 1: {len(texts1)} texts ({words1:,} words)")
        print(f"    Group 2: {len(texts2)} texts ({words2:,} words)")
        print(f"    Text IDs: {texts1[:2]} vs {texts2[:2]}")
        print()

# ==========================================
# EXPORT FOR WORD2VEC ANALYSIS
# ==========================================

# # Export groupings for semantic drift analysis
# import json

# # Create exportable structure
# export_groups = {
#     'period_groups': period_groups,
#     'genre_groups': genre_groups,
#     'semantic_drift_groups': semantic_drift_groups,
#     'comparison_pairs': [
#         {
#             'group1': pair[0],
#             'group2': pair[1], 
#             'description': pair[2],
#             'texts1': [],  # Will be filled by your analysis script
#             'texts2': []   # Will be filled by your analysis script
#         }
#         for pair in comparison_pairs
#     ]
# }

# # Save groupings for Word2Vec analysis
# with open('semantic_drift_groups.json', 'w') as f:
#     json.dump(export_groups, f, indent=2)

# print("\n💾 EXPORTED FILES:")
# print("  - semantic_drift_groups.json (for Word2Vec analysis)")

# # Also create a simple Python dictionary for immediate use
# with open('semantic_groups.py', 'w') as f:
#     f.write("# Auto-generated semantic drift analysis groups\n\n")
#     f.write("# Period-based groups for diachronic analysis\n")
#     f.write(f"period_groups = {repr(period_groups)}\n\n")
#     f.write("# Genre-based groups for synchronic analysis\n")
#     f.write(f"genre_groups = {repr(genre_groups)}\n\n")
#     f.write("# Detailed groups for complex comparisons\n")
#     f.write(f"semantic_drift_groups = {repr(semantic_drift_groups)}\n")

# print("  - semantic_groups.py (Python dictionaries)")

# # ==========================================
# # STANDARD EXPORTS
# # ==========================================

# # Save to CSV for external use
# df.to_csv('latin_text_metadata.csv', index=False)
# print("\n💾 Standard exports:")
# print("  - latin_text_metadata.csv (full metadata)")

# # Save just the basic metadata dictionary to a Python file
# with open('genre_metadata.py', 'w') as f:
#     f.write("# Auto-generated metadata\n")
#     f.write(f"genre_metadata = {repr(genre_metadata)}")
# print("  - genre_metadata.py (original format)")