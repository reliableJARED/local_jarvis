import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict

#critical to get the wordnet model
nltk.download('wordnet')

def group_words_by_similarity(word_list, similarity_threshold=0.5):
    """
    Groups words based on semantic similarity using WordNet's wup_similarity.
    This is a more lenient method than grouping by exact synsets.
    """
    grouped_words = []
    
    # A simple clustering approach using a threshold
    for word in word_list:
        synsets_word = wn.synsets(word)
        if not synsets_word:
            continue
            
        added_to_group = False
        for group in grouped_words:
            representative_word = group[0]
            synsets_rep = wn.synsets(representative_word)
            if not synsets_rep:
                continue

            # Compare the most probable senses
            similarity = synsets_word[0].wup_similarity(synsets_rep[0])
            if similarity and similarity >= similarity_threshold:
                group.append(word)
                added_to_group = True
                break
        
        if not added_to_group:
            grouped_words.append([word])
            
    return grouped_words

# Example usage
words_to_group = ["car", "auto", "employer", "married", "marry", "working", "owns", "owner", "posses"]
grouped_by_similarity = group_words_by_similarity(words_to_group, similarity_threshold=0.8)

# Print the results
for i, group in enumerate(grouped_by_similarity):
    print(f"Group {i+1}: {group}")