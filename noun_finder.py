import spacy
from typing import List, Dict, Tuple, Union, Optional

class SpacyNounExtractor:
    """
    A class for extracting nouns and analyzing text using spaCy.
    
    Installation:
        pip install spacy
        python -m spacy download en_core_web_sm
    
    Usage:
        extractor = SpacyNounExtractor()
        nouns = extractor.find_nouns("The cat sat on the mat.")
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the noun extractor with a spaCy model.
        
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
    
    def analyze_text(self, text: str) -> Dict[str, Union[List, Dict]]:
        """
        Perform comprehensive noun analysis of the text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict: Comprehensive analysis results
        """
        doc = self.nlp(text)
        
        analysis = {
            'text': text,
            'regular_nouns': [],
            'proper_nouns': [],
            'named_entities': [],
            'noun_phrases': [],
            'noun_lemmas': {},
            'statistics': {}
        }
        
        # Collect different types of nouns
        for token in doc:
            if token.pos_ == "NOUN":
                analysis['regular_nouns'].append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'tag': token.tag_
                })
            elif token.pos_ == "PROPN":
                analysis['proper_nouns'].append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'tag': token.tag_
                })
        
        # Named entities
        analysis['named_entities'] = [
            {
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_) or 'Unknown'
            }
            for ent in doc.ents
        ]
        
        # Noun phrases
        analysis['noun_phrases'] = [chunk.text for chunk in doc.noun_chunks]
        
        # Noun lemmas mapping
        analysis['noun_lemmas'] = {
            token.text: token.lemma_ 
            for token in doc if token.pos_ in ["NOUN", "PROPN"]
        }
        
        # Statistics
        analysis['statistics'] = {
            'total_nouns': len(analysis['regular_nouns']) + len(analysis['proper_nouns']),
            'regular_noun_count': len(analysis['regular_nouns']),
            'proper_noun_count': len(analysis['proper_nouns']),
            'entity_count': len(analysis['named_entities']),
            'noun_phrase_count': len(analysis['noun_phrases'])
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
            entities = [
                {'text': ent.text, 'label': ent.label_}
                for ent in doc.ents
            ]
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            
            results.append({
                'text': text,
                'nouns': nouns,
                'entities': entities,
                'noun_phrases': noun_phrases
            })
        
        return results
    
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
    
    def filter_nouns_by_type(self, text: str, noun_types: List[str]) -> List[str]:
        """
        Filter nouns by specific POS types.
        
        Args:
            text (str): Input text to analyze
            noun_types (List[str]): List of POS types to include (e.g., ['NOUN'], ['PROPN'])
            
        Returns:
            List[str]: Filtered list of nouns
        """
        doc = self.nlp(text)
        return [token.text for token in doc if token.pos_ in noun_types]
    
    @staticmethod
    def explain_pos_tags() -> Dict[str, str]:
        """
        Get explanations for noun POS tags.
        
        Returns:
            Dict[str, str]: Dictionary of POS tags and their descriptions
        """
        return {
            'NOUN': 'Common noun (dog, car, book, freedom)',
            'PROPN': 'Proper noun (John, London, Apple)',
            'NN': 'Noun, singular',
            'NNS': 'Noun, plural',
            'NNP': 'Proper noun, singular',
            'NNPS': 'Proper noun, plural'
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
            "Apple Inc. released new products in California yesterday."
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
        
    except OSError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    demo_class()