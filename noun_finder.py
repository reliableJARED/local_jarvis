import spacy
from typing import List, Dict, Tuple, Union, Optional

import spacy
from typing import List, Dict, Tuple, Union, Optional


class SpacyNounExtractor:
    """
    A comprehensive class for extracting and analyzing various parts of speech and named entities using spaCy.
    
    Installation:
        pip install spacy
        python -m spacy download en_core_web_sm
    
    Usage:
        analyzer = SpacyTextAnalyzer()
        nouns = analyzer.find_nouns("The cat sat on the mat.")
        verbs = analyzer.find_verbs("The cat quickly ran to the house.")
        entities = analyzer.find_named_entities_by_type("Apple Inc. was founded by Steve Jobs.")
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the text analyzer with a spaCy model.
        
        Args:
            model_name (str): Name of the spaCy model to load
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise OSError(
                f"spaCy model '{model_name}' not found. "
                f"Install with: python -m spacy download {model_name}"
            )
    
    # ==================== NOUN METHODS ====================
    
    def find_nouns(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Find all nouns (regular and proper) in the text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Tuple[str, str, str]]: List of (word, pos_tag, detailed_tag)
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_, token.tag_) 
                for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    
    def find_regular_nouns(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Find only regular nouns (not proper nouns).
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Tuple[str, str, str]]: List of (word, pos_tag, detailed_tag)
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_, token.tag_) 
                for token in doc if token.pos_ == "NOUN"]
    
    def find_proper_nouns(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Find only proper nouns.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Tuple[str, str, str]]: List of (word, pos_tag, detailed_tag)
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_, token.tag_) 
                for token in doc if token.pos_ == "PROPN"]
    
    # ==================== VERB METHODS ====================
    
    def find_verbs(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Find all verbs in the text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Tuple[str, str, str]]: List of (word, pos_tag, detailed_tag)
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_, token.tag_) 
                for token in doc if token.pos_ == "VERB"]
    
    def find_auxiliary_verbs(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Find auxiliary verbs (helping verbs like 'is', 'have', 'will').
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Tuple[str, str, str]]: List of (word, pos_tag, detailed_tag)
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_, token.tag_) 
                for token in doc if token.pos_ == "AUX"]
    
    def find_all_verbs(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Find all verbs including main verbs and auxiliary verbs.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Tuple[str, str, str]]: List of (word, pos_tag, detailed_tag)
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_, token.tag_) 
                for token in doc if token.pos_ in ["VERB", "AUX"]]
    
    def get_verb_lemmas(self, text: str) -> List[Tuple[str, str]]:
        """
        Get verbs with their lemmatized (base) forms.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Tuple[str, str]]: List of (original_word, lemma)
        """
        doc = self.nlp(text)
        return [(token.text, token.lemma_) 
                for token in doc if token.pos_ in ["VERB", "AUX"]]
    
    # ==================== ADJECTIVE METHODS ====================
    
    def find_adjectives(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Find all adjectives in the text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Tuple[str, str, str]]: List of (word, pos_tag, detailed_tag)
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_, token.tag_) 
                for token in doc if token.pos_ == "ADJ"]
    
    def get_adjective_lemmas(self, text: str) -> List[Tuple[str, str]]:
        """
        Get adjectives with their lemmatized (base) forms.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Tuple[str, str]]: List of (original_word, lemma)
        """
        doc = self.nlp(text)
        return [(token.text, token.lemma_) 
                for token in doc if token.pos_ == "ADJ"]
    
    # ==================== ADVERB METHODS ====================
    
    def find_adverbs(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Find all adverbs in the text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Tuple[str, str, str]]: List of (word, pos_tag, detailed_tag)
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_, token.tag_) 
                for token in doc if token.pos_ == "ADV"]
    
    def get_adverb_lemmas(self, text: str) -> List[Tuple[str, str]]:
        """
        Get adverbs with their lemmatized (base) forms.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Tuple[str, str]]: List of (original_word, lemma)
        """
        doc = self.nlp(text)
        return [(token.text, token.lemma_) 
                for token in doc if token.pos_ == "ADV"]
    
    # ==================== NAMED ENTITY METHODS ====================
    
    def find_named_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Find named entities with their categories.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Dict[str, str]]: List of entity dictionaries
        """
        doc = self.nlp(text)
        return [
            {
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_) or 'Unknown'
            }
            for ent in doc.ents
        ]
    
    def find_named_entities_by_type(self, text: str, entity_types: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Find named entities organized by their types.
        
        Args:
            text (str): Input text to analyze
            entity_types (Optional[List[str]]): Specific entity types to extract. 
                                              If None, extracts all types.
            
        Returns:
            Dict[str, List[str]]: Dictionary with entity types as keys and lists of entities as values
        """
        doc = self.nlp(text)
        
        # Define all possible entity types if none specified
        if entity_types is None:
            entity_types = [
                'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 
                'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 
                'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'
            ]
        
        # Initialize result dictionary
        result = {entity_type: [] for entity_type in entity_types}
        
        # Extract entities
        for ent in doc.ents:
            if ent.label_ in result:
                result[ent.label_].append(ent.text)
        
        return result
    
    def find_entities_of_type(self, text: str, entity_type: str) -> List[Dict[str, str]]:
        """
        Find all entities of a specific type.
        
        Args:
            text (str): Input text to analyze
            entity_type (str): Specific entity type to find (e.g., 'PERSON', 'ORG')
            
        Returns:
            List[Dict[str, str]]: List of entities of the specified type
        """
        doc = self.nlp(text)
        return [
            {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': spacy.explain(ent.label_) or 'Unknown'
            }
            for ent in doc.ents if ent.label_ == entity_type.upper()
        ]
    
    def get_entity_statistics(self, text: str) -> Dict[str, int]:
        """
        Get statistics about named entities in the text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict[str, int]: Dictionary with entity types and their counts
        """
        doc = self.nlp(text)
        entity_counts = {}
        
        for ent in doc.ents:
            entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1
        
        return entity_counts
    
    # ==================== NOUN PHRASE METHODS ====================
    
    def find_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases (multi-word noun expressions).
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[str]: List of noun phrases
        """
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]
    
    def get_noun_lemmas(self, text: str) -> List[Tuple[str, str]]:
        """
        Get nouns with their lemmatized (base) forms.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Tuple[str, str]]: List of (original_word, lemma)
        """
        doc = self.nlp(text)
        return [(token.text, token.lemma_) 
                for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    
    # ==================== COMPREHENSIVE ANALYSIS METHODS ====================
    
    def analyze_text(self, text: str) -> Dict[str, Union[List, Dict]]:
        """
        Perform comprehensive analysis of the text including all parts of speech and entities.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict: Comprehensive analysis results
        """
        doc = self.nlp(text)
        
        analysis = {
            'text': text,
            'nouns': {'regular': [], 'proper': [],'pronoun':[]},
            'verbs': {'main': [], 'auxiliary': []},
            'adjectives': [],
            'adverbs': [],
            'named_entities': [],
            'named_entities_by_type': {},
            'noun_phrases': [],
            'lemmas': {'nouns': {}, 'verbs': {}, 'adjectives': {}, 'adverbs': {}},
            'statistics': {}
        }
        
        # Collect different types of words
        for token in doc:
            if token.pos_ == "NOUN":
                analysis['nouns']['regular'].append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'tag': token.tag_
                })
                analysis['lemmas']['nouns'][token.text] = token.lemma_
            elif token.pos_ == "PROPN":
                analysis['nouns']['proper'].append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'tag': token.tag_
                })
                analysis['lemmas']['nouns'][token.text] = token.lemma_
            elif token.pos_ == "PRON":
                analysis['nouns']['pronoun'].append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'tag': token.tag_
                })
                analysis['lemmas']['nouns'][token.text] = token.lemma_
            elif token.pos_ == "VERB":
                analysis['verbs']['main'].append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'tag': token.tag_
                })
                analysis['lemmas']['verbs'][token.text] = token.lemma_
            elif token.pos_ == "AUX":
                analysis['verbs']['auxiliary'].append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'tag': token.tag_
                })
                analysis['lemmas']['verbs'][token.text] = token.lemma_
            elif token.pos_ == "ADJ":
                analysis['adjectives'].append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'tag': token.tag_
                })
                analysis['lemmas']['adjectives'][token.text] = token.lemma_
            elif token.pos_ == "ADV":
                analysis['adverbs'].append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'tag': token.tag_
                })
                analysis['lemmas']['adverbs'][token.text] = token.lemma_
        
        # Named entities
        analysis['named_entities'] = [
            {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': spacy.explain(ent.label_) or 'Unknown'
            }
            for ent in doc.ents
        ]
        
        # Named entities by type
        analysis['named_entities_by_type'] = self.find_named_entities_by_type(text)
        
        # Noun phrases
        analysis['noun_phrases'] = [chunk.text for chunk in doc.noun_chunks]
        
        # Statistics
        analysis['statistics'] = {
            'total_nouns': len(analysis['nouns']['regular']) + len(analysis['nouns']['proper'])+ len(analysis['nouns']['pronoun']),
            'regular_noun_count': len(analysis['nouns']['regular']),
            'proper_noun_count': len(analysis['nouns']['proper']),
            'pronoun_noun_count': len(analysis['nouns']['pronoun']),
            'total_verbs': len(analysis['verbs']['main']) + len(analysis['verbs']['auxiliary']),
            'main_verb_count': len(analysis['verbs']['main']),
            'auxiliary_verb_count': len(analysis['verbs']['auxiliary']),
            'adjective_count': len(analysis['adjectives']),
            'adverb_count': len(analysis['adverbs']),
            'entity_count': len(analysis['named_entities']),
            'noun_phrase_count': len(analysis['noun_phrases']),
            'entity_type_counts': self.get_entity_statistics(text)
        }
        
        return analysis
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """
        Efficiently analyze multiple texts at once.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List[Dict]: List of analysis results for each text
        """
        docs = list(self.nlp.pipe(texts))
        results = []
        
        for text, doc in zip(texts, docs):
            nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
            verbs = [token.text for token in doc if token.pos_ in ["VERB", "AUX"]]
            adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
            adverbs = [token.text for token in doc if token.pos_ == "ADV"]
            entities = [
                {'text': ent.text, 'label': ent.label_}
                for ent in doc.ents
            ]
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            
            results.append({
                'text': text,
                'nouns': nouns,
                'verbs': verbs,
                'adjectives': adjectives,
                'adverbs': adverbs,
                'entities': entities,
                'noun_phrases': noun_phrases
            })
        
        return results
    
    # ==================== UTILITY METHODS ====================
    
    def get_simple_nouns(self, text: str) -> List[str]:
        """
        Get just the noun words as a simple list.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[str]: Simple list of noun words
        """
        doc = self.nlp(text)
        return [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]

    def get_simple_words_by_pos(self, text: str, pos_tags: List[str]) -> List[str]:
        """
        Get words of specific parts of speech as a simple list.
        
        Args:
            text (str): Input text to analyze
            pos_tags (List[str]): List of POS tags to include
            
        Returns:
            List[str]: Simple list of words matching the POS tags
        """
        doc = self.nlp(text)
        return [token.text for token in doc if token.pos_ in pos_tags]
    
    @staticmethod
    def explain_pos_tags() -> Dict[str, str]:
        """
        Get explanations for POS tags.
        
        Returns:
            Dict[str, str]: Dictionary of POS tags and their descriptions
        """
        return {
            # Nouns
            'NOUN': 'Common noun (dog, car, book, freedom)',
            'PROPN': 'Proper noun (John, London, Apple)',
            # Verbs
            'VERB': 'Main verb (run, eat, think, create)',
            'AUX': 'Auxiliary verb (is, have, will, can)',
            # Adjectives and Adverbs
            'ADJ': 'Adjective (big, red, beautiful, smart)',
            'ADV': 'Adverb (quickly, very, well, often)',
            # Detailed tags
            'NN': 'Noun, singular',
            'NNS': 'Noun, plural',
            'NNP': 'Proper noun, singular',
            'NNPS': 'Proper noun, plural',
            'VB': 'Verb, base form',
            'VBD': 'Verb, past tense',
            'VBG': 'Verb, gerund/present participle',
            'VBN': 'Verb, past participle',
            'VBP': 'Verb, non-3rd person singular present',
            'VBZ': 'Verb, 3rd person singular present',
            'JJ': 'Adjective',
            'JJR': 'Adjective, comparative',
            'JJS': 'Adjective, superlative',
            'RB': 'Adverb',
            'RBR': 'Adverb, comparative',
            'RBS': 'Adverb, superlative'
        }
    
    @staticmethod
    def explain_entity_types() -> Dict[str, str]:
        """
        Get explanations for named entity types.
        
        Returns:
            Dict[str, str]: Dictionary of entity types and their descriptions
        """
        return {
            'PERSON': 'People, including fictional characters',
            'NORP': 'Nationalities or religious or political groups',
            'FAC': 'Buildings, airports, highways, bridges, etc.',
            'ORG': 'Companies, agencies, institutions, etc.',
            'GPE': 'Countries, cities, states',
            'LOC': 'Non-GPE locations, mountain ranges, bodies of water',
            'PRODUCT': 'Objects, vehicles, foods, etc. (Not services)',
            'EVENT': 'Named hurricanes, battles, wars, sports events, etc.',
            'WORK_OF_ART': 'Titles of books, songs, etc.',
            'LAW': 'Named documents made into laws',
            'LANGUAGE': 'Any named language',
            'DATE': 'Absolute or relative dates or periods',
            'TIME': 'Times smaller than a day',
            'PERCENT': 'Percentage, including "%"',
            'MONEY': 'Monetary values, including unit',
            'QUANTITY': 'Measurements, as of weight or distance',
            'ORDINAL': '"first", "second", etc.',
            'CARDINAL': 'Numerals that do not fall under another type'
        }


# Example usage and demo functions
def demo_class():
    """Demonstrate the SpacyNounExtractor class"""
    print("=== SpacyNounExtractor Demo ===\n")
    
    try:
        # Initialize the extractor
        extractor = SpacyNounExtractor()
        
        # Test sentences
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Alice and Bob went to Paris last summer.",
            "Apple Inc. released new products in California yesterday.",
            "I love it when you do that",
            "My favorite movie is agengers endgame",
            "This shirt looks so good on me"
        ]
        
        for i, sentence in enumerate(sentences, 1):
            print(f"Sentence {i}: {sentence}")
            
            # Simple noun extraction
            nouns = extractor.get_simple_nouns(sentence)
            print(f"Simple nouns: {nouns}")
            
            # Comprehensive analysis
            analysis = extractor.analyze_text(sentence)
            print(f"Statistics: {analysis['statistics']}")
            
            if analysis['named_entities']:
                entities = [(ent['text'], ent['label']) for ent in analysis['named_entities']]
                print(f"Named entities: {entities}")
            
            print("-" * 50)
        
        # Batch processing example
        print("\nBatch processing:")
        batch_results = extractor.batch_analyze(sentences)
        for result in batch_results:
            print(f"Text: {result['text'][:30]}...")
            print(f"Nouns: {result['nouns']}")

        # Comprehensive analysis on Pride and Prejudice
        print("\nPride and Prejudice processing:")
        PrideAndPrejudice = """It is a truth universally acknowledged, that a single man in possession
of a good fortune must be in want of a wife."""
        analysis = extractor.analyze_text(PrideAndPrejudice)
        print(f"Statistics: {analysis['statistics']}")
        for result in analysis.values():
            print(result)
        
        import pprint

        print("="*50)
        print("prity print\n")
        pprint.pprint(analysis, indent=4)

        # Sample text for testing
        sample_text = """
        Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. 
        The company quickly became a major player in the technology industry. 
        Today, Apple is worth over $2 trillion and employs more than 150,000 people worldwide.
        Tim Cook, the current CEO, has been leading the company since 2011.
        """
        
        # Test various methods
        print("=== NOUNS ===")
        print("All nouns:", extractor.find_nouns(sample_text))
        
        print("\n=== VERBS ===")
        print("All verbs:", extractor.find_all_verbs(sample_text))
        
        print("\n=== ADJECTIVES ===")
        print("Adjectives:", extractor.find_adjectives(sample_text))
        
        print("\n=== ADVERBS ===")
        print("Adverbs:", extractor.find_adverbs(sample_text))
        
        print("\n=== NAMED ENTITIES BY TYPE ===")
        entities_by_type = extractor.find_named_entities_by_type(sample_text)
        for entity_type, entities in entities_by_type.items():
            if entities:  # Only print non-empty types
                print(f"{entity_type}: {entities}")
        
        print("\n=== COMPREHENSIVE ANALYSIS ===")
        full_analysis = extractor.analyze_text(sample_text)
        print("Statistics:", full_analysis['statistics'])
        
        
    except OSError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    demo_class()