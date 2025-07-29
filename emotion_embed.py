import os
import uuid
import pickle
import numpy as np
import torch
import requests
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import hashlib


#https://claude.ai/chat/56e733ff-d4e9-4106-8d5b-f611f19d5eb4


# Plutchik's primary emotions with their characteristics
EMOTION_DATABASE = {
    # Joy family
    'joy': {
        'mood': 'Sense of energy and possibility',
        'thoughts': 'Life is going well',
        'responses': 'Sparks creativity, connection, gives energy'
    },
    'ecstasy': {
        'mood': 'Overwhelming euphoria and elation',
        'thoughts': 'Everything is perfect and amazing',
        'responses': 'Boundless enthusiasm, may act impulsively from excitement'
    },
    'serenity': {
        'mood': 'Calm contentment and peace',
        'thoughts': 'Things are pleasant and stable',
        'responses': 'Gentle actions, seeks to maintain harmony'
    },
    
    # Sadness family
    'sadness': {
        'mood': 'Heavy, low energy, withdrawn',
        'thoughts': 'Things aren\'t going well, feeling loss',
        'responses': 'Seeks comfort, may isolate, moves slowly'
    },
    'grief': {
        'mood': 'Profound sorrow and despair',
        'thoughts': 'Something important is gone forever',
        'responses': 'May be inconsolable, needs support, difficulty functioning'
    },
    'pensiveness': {
        'mood': 'Quiet melancholy and reflection',
        'thoughts': 'Contemplating what could have been',
        'responses': 'Introspective, seeks solitude, gentle sadness'
    },
    
    # Trust family
    'trust': {
        'mood': 'Open and accepting',
        'thoughts': 'Others are reliable and good',
        'responses': 'Cooperative, shares freely, seeks connection'
    },
    'admiration': {
        'mood': 'Deep respect and reverence',
        'thoughts': 'This person/thing is truly worthy',
        'responses': 'Wants to learn, emulate, or serve'
    },
    'acceptance': {
        'mood': 'Calm acknowledgment',
        'thoughts': 'This is how things are',
        'responses': 'Goes with the flow, doesn\'t resist'
    },
    
    # Disgust family
    'disgust': {
        'mood': 'Repulsed and rejecting',
        'thoughts': 'This is wrong, contaminated, or inferior',
        'responses': 'Avoids, criticizes, seeks to remove or cleanse'
    },
    'loathing': {
        'mood': 'Intense revulsion and hatred',
        'thoughts': 'This is absolutely abhorrent',
        'responses': 'Strong rejection, may become aggressive to eliminate'
    },
    'boredom': {
        'mood': 'Mild disinterest and restlessness',
        'thoughts': 'This isn\'t worth my attention',
        'responses': 'Seeks stimulation elsewhere, disengages'
    },
    
    # Fear family
    'fear': {
        'mood': 'Anxious alertness and tension',
        'thoughts': 'Something bad might happen',
        'responses': 'Cautious, seeks safety, may freeze or flee'
    },
    'terror': {
        'mood': 'Paralyzing dread',
        'thoughts': 'Immediate danger, might not survive',
        'responses': 'Fight, flight, or freeze response, acts on instinct'
    },
    'apprehension': {
        'mood': 'Mild worry and uncertainty',
        'thoughts': 'Something doesn\'t feel quite right',
        'responses': 'More cautious than usual, seeks reassurance'
    },
    
    # Anger family
    'anger': {
        'mood': 'Heated and energized',
        'thoughts': 'This is unfair, I\'ve been wronged',
        'responses': 'Confrontational, seeks to correct or punish'
    },
    'rage': {
        'mood': 'Burning fury and aggression',
        'thoughts': 'Must destroy the source of this injustice',
        'responses': 'Potentially violent, loses rational control'
    },
    'annoyance': {
        'mood': 'Mildly irritated and impatient',
        'thoughts': 'This is inconvenient or bothersome',
        'responses': 'Short responses, may express frustration verbally'
    },
    
    # Surprise family
    'surprise': {
        'mood': 'Startled and alert',
        'thoughts': 'That was unexpected',
        'responses': 'Heightened attention, pauses to process'
    },
    'amazement': {
        'mood': 'Awed and wonder-struck',
        'thoughts': 'This is incredible and beyond belief',
        'responses': 'Stares, asks questions, wants to understand'
    },
    'distraction': {
        'mood': 'Mildly surprised and unfocused',
        'thoughts': 'Wait, what was that?',
        'responses': 'Attention shifts, momentarily loses focus'
    },
    
    # Anticipation family
    'anticipation': {
        'mood': 'Eager and forward-looking',
        'thoughts': 'Something good is coming',
        'responses': 'Prepares, plans, may act impatiently'
    },
    'vigilance': {
        'mood': 'Intense focus and readiness',
        'thoughts': 'Must be ready for what\'s coming',
        'responses': 'Hyper-alert, prepared for action'
    },
    'interest': {
        'mood': 'Curious and engaged',
        'thoughts': 'I want to know more about this',
        'responses': 'Asks questions, explores, pays attention'
    }
}

# Complex emotions formed by combining primary emotions
COMPLEX_EMOTIONS = {
    'love': {
        'components': ['joy', 'trust'],
        'mood': 'Warm, connected, and devoted',
        'thoughts': 'This person/thing is wonderful and safe',
        'responses': 'Protective, nurturing, wants to be close'
    },
    'submission': {
        'components': ['trust', 'fear'],
        'mood': 'Deferential and compliant',
        'thoughts': 'I should follow their lead',
        'responses': 'Obeys, seeks approval, avoids conflict'
    },
    'awe': {
        'components': ['fear', 'surprise'],
        'mood': 'Humbled and overwhelmed',
        'thoughts': 'This is beyond my understanding',
        'responses': 'Reverent behavior, may feel small or insignificant'
    },
    'disapproval': {
        'components': ['surprise', 'sadness'],
        'mood': 'Disappointed and let down',
        'thoughts': 'This isn\'t what I expected or hoped for',
        'responses': 'Expresses dissatisfaction, may withdraw support'
    },
    'remorse': {
        'components': ['sadness', 'disgust'],
        'mood': 'Regretful and self-reproaching',
        'thoughts': 'I did something wrong and feel bad about it',
        'responses': 'Apologizes, seeks to make amends, self-punishing'
    },
    'contempt': {
        'components': ['disgust', 'anger'],
        'mood': 'Superior and disdainful',
        'thoughts': 'This is beneath me and doesn\'t deserve respect',
        'responses': 'Dismissive, condescending, may ridicule'
    },
    'aggressiveness': {
        'components': ['anger', 'anticipation'],
        'mood': 'Hostile and ready for conflict',
        'thoughts': 'I need to attack before they do',
        'responses': 'Threatening behavior, seeks confrontation'
    },
    'optimism': {
        'components': ['anticipation', 'joy'],
        'mood': 'Hopeful and positive about the future',
        'thoughts': 'Good things are coming',
        'responses': 'Plans enthusiastically, encourages others'
    }
}

class MxBaiEmbedder:
    def __init__(self, pickle_file: str = "embeddings_store.pkl"):
        self.tokenizer = None
        self.model = None
        self.embeddings_store = {}  # Dict to store embeddings with UUID keys
        self.metadata_store = {}    # Store original text and other metadata
        self.pickle_file = pickle_file
        self.emotions_initialized = False
        
        # Load existing embeddings if pickle file exists
        self._load_from_pickle()
        
    def check_internet_connection(self, timeout=5):
        """Check if internet connection is available"""
        try:
            requests.get("https://www.google.com", timeout=timeout)
            return True
        except (requests.ConnectionError, requests.Timeout):
            return False

    def _load_from_pickle(self):
        """Load embeddings and metadata from pickle file"""
        if os.path.exists(self.pickle_file):
            try:
                with open(self.pickle_file, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings_store = data.get('embeddings_store', {})
                    self.metadata_store = data.get('metadata_store', {})
                    self.emotions_initialized = data.get('emotions_initialized', False)
                print(f"Loaded {len(self.embeddings_store)} embeddings from {self.pickle_file}")
                if self.emotions_initialized:
                    print("Emotion embeddings already initialized")
            except Exception as e:
                print(f"Error loading from pickle file: {str(e)}")
                print("Starting with empty embeddings store.")
                self.embeddings_store = {}
                self.metadata_store = {}
                self.emotions_initialized = False
        else:
            print(f"No existing pickle file found at {self.pickle_file}. Starting with empty store.")

    def _save_to_pickle(self):
        """Save embeddings and metadata to pickle file"""
        try:
            data = {
                'embeddings_store': self.embeddings_store,
                'metadata_store': self.metadata_store,
                'emotions_initialized': self.emotions_initialized
            }
            with open(self.pickle_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved {len(self.embeddings_store)} embeddings to {self.pickle_file}")
        except Exception as e:
            print(f"Error saving to pickle file: {str(e)}")

    def load_model(self):
        """Load mixedbread-ai/mxbai-embed-large-v1 model with offline fallback"""
        model_name = "mixedbread-ai/mxbai-embed-large-v1"
        
        # Check internet connection
        has_internet = self.check_internet_connection()
        
        if not has_internet:
            print("No internet connection detected. Setting offline mode...")
            # Set offline environment variables
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
        else:
            print("Internet connection available. Loading model online...")
            # Ensure offline flags are not set
            os.environ.pop('HF_HUB_OFFLINE', None)
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
            os.environ.pop('HF_DATASETS_OFFLINE', None)
        
        try:
            # Load tokenizer and model
            print(f"Loading tokenizer for {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print(f"Loading model {model_name}...")
            self.model = AutoModel.from_pretrained(model_name)
            
            # Set model to evaluation mode
            self.model.eval()
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            if has_internet:
                print("Failed to load online. Trying offline mode...")
                # Set offline flags and retry
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'
                
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModel.from_pretrained(model_name)
                    self.model.eval()
                    print("Model loaded successfully from cache!")
                    return True
                except Exception as offline_e:
                    print(f"Failed to load from cache: {str(offline_e)}")
                    return False
            else:
                print("No internet connection and cache loading failed.")
                return False

    def embed_text_string(self, text: str) -> np.ndarray:
        """
        Create an embedding for a given text string
        
        Args:
            text (str): The input text to embed
            
        Returns:
            np.ndarray: The embedding vector
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                               padding=True, max_length=512)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Use mean pooling of the last hidden state
            # This is a common approach for sentence embeddings
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Apply attention mask and mean pooling
            masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
            summed_embeddings = torch.sum(masked_embeddings, dim=1)
            summed_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = summed_embeddings / summed_mask.unsqueeze(-1)
            
            # Convert to numpy array
            embedding_vector = mean_pooled.squeeze().numpy()
            
            # Normalize the embedding (optional but often helpful for similarity search)
            embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)
            
            return embedding_vector

    def generate_content_id(self, text: str) -> str:
        """Generate deterministic ID based on content
        THIS WILL AUTO overwrite duplicates. So it's impossible to save enter the same thing twice. con being counting and same text different meta data"""
        return hashlib.md5(text.encode()).hexdigest()

    def initialize_emotion_embeddings(self, force_reinitialize: bool = False):
        """
        Initialize embeddings for all emotions in the database
        
        Args:
            force_reinitialize (bool): If True, will re-embed emotions even if already done
        """
        if self.emotions_initialized and not force_reinitialize:
            print("Emotion embeddings already initialized. Use force_reinitialize=True to recreate them.")
            return
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("Initializing emotion embeddings...")
        
        # Combine all emotions
        all_emotions = {**EMOTION_DATABASE, **COMPLEX_EMOTIONS}
        
        emotion_texts = []
        emotion_ids = []
        
        for emotion_name, emotion_data in all_emotions.items():
            # Create a comprehensive text representation of the emotion
            if 'components' in emotion_data:
                # Complex emotion
                emotion_text = f"emotion: {emotion_name}. components: {', '.join(emotion_data['components'])}. mood: {emotion_data['mood']}. thoughts: {emotion_data['thoughts']}. responses: {emotion_data['responses']}"
            else:
                # Primary emotion
                emotion_text = f"emotion: {emotion_name}. mood: {emotion_data['mood']}. thoughts: {emotion_data['thoughts']}. responses: {emotion_data['responses']}"
            
            emotion_texts.append(emotion_text)
            emotion_ids.append(f"emotion_{emotion_name}")
        
        # Batch embed all emotions
        print(f"Embedding {len(emotion_texts)} emotions...")
        for i, (text, emotion_id) in enumerate(zip(emotion_texts, emotion_ids)):
            embedding = self.embed_text_string(text)
            
            # Store with special metadata to mark as emotion
            self.embeddings_store[emotion_id] = embedding
            self.metadata_store[emotion_id] = {
                'text': text,
                'embedding_shape': embedding.shape,
                'created_at': str(uuid.uuid1().time),
                'type': 'emotion',
                'emotion_name': emotion_id.replace('emotion_', ''),
                'emotion_data': all_emotions[emotion_id.replace('emotion_', '')]
            }
            
            if (i + 1) % 5 == 0:
                print(f"Embedded {i + 1}/{len(emotion_texts)} emotions...")
        
        self.emotions_initialized = True
        self._save_to_pickle()
        print(f"Successfully initialized {len(emotion_texts)} emotion embeddings!")

    def find_most_similar_emotion(self, text: str) -> Tuple[str, float, Dict]:
        """
        Find the most similar emotion to the given text
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Tuple[str, float, Dict]: (emotion_name, similarity_score, emotion_attributes)
        """
        if not self.emotions_initialized:
            raise ValueError("Emotion embeddings not initialized. Call initialize_emotion_embeddings() first.")
        
        # Get embedding for the input text
        query_embedding = self.embed_text_string(text)
        
        # Filter only emotion embeddings
        emotion_ids = [eid for eid in self.embeddings_store.keys() if eid.startswith('emotion_')]
        
        if not emotion_ids:
            raise ValueError("No emotion embeddings found in store.")
        
        # Get emotion embeddings
        emotion_embeddings = np.array([self.embeddings_store[eid] for eid in emotion_ids])
        
        # Ensure query embedding is 2D for sklearn
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, emotion_embeddings)[0]
        
        # Find the most similar emotion
        best_idx = np.argmax(similarities)
        best_emotion_id = emotion_ids[best_idx]
        best_similarity = float(similarities[best_idx])
        
        # Get emotion data
        emotion_name = best_emotion_id.replace('emotion_', '')
        emotion_data = self.metadata_store[best_emotion_id]['emotion_data']
        
        return emotion_name, best_similarity, emotion_data

    def analyze_text_emotion(self, text: str, top_n: int = 3) -> List[Tuple[str, float, Dict]]:
        """
        Analyze text and return top N most similar emotions with their attributes
        
        Args:
            text (str): Input text to analyze
            top_n (int): Number of top emotions to return
            
        Returns:
            List[Tuple[str, float, Dict]]: List of (emotion_name, similarity_score, emotion_attributes)
        """
        if not self.emotions_initialized:
            raise ValueError("Emotion embeddings not initialized. Call initialize_emotion_embeddings() first.")
        
        # Get embedding for the input text
        query_embedding = self.embed_text_string(text)
        
        # Filter only emotion embeddings
        emotion_ids = [eid for eid in self.embeddings_store.keys() if eid.startswith('emotion_')]
        
        if not emotion_ids:
            raise ValueError("No emotion embeddings found in store.")
        
        # Get emotion embeddings
        emotion_embeddings = np.array([self.embeddings_store[eid] for eid in emotion_ids])
        
        # Ensure query embedding is 2D for sklearn
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, emotion_embeddings)[0]
        
        # Get top N emotions
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        results = []
        for idx in top_indices:
            emotion_id = emotion_ids[idx]
            similarity_score = float(similarities[idx])
            emotion_name = emotion_id.replace('emotion_', '')
            emotion_data = self.metadata_store[emotion_id]['emotion_data']
            results.append((emotion_name, similarity_score, emotion_data))
        
        return results

    def save_embedding(self, text: str, embedding: Optional[np.ndarray] = None, 
                      custom_id: Optional[str] = None, auto_save: bool = True,
                      check_duplicates: bool = True) -> str:
        """
        Save an embedding to the in-memory store and optionally persist to disk
        
        Args:
            text (str): Original text
            embedding (np.ndarray, optional): Precomputed embedding. If None, will compute it.
            custom_id (str, optional): Custom ID. If None, will generate UUID.
            auto_save (bool): Whether to automatically save to pickle file
            check_duplicates (bool): make sure we haven't embedded the same exact text before
            
        Returns:
            str: The ID used to store the embedding
        """            
        # Generate ID if not provided
        if custom_id is None:
            #embedding_id = str(uuid.uuid4())
            embedding_id = self.generate_content_id(text)
        else:
            embedding_id = custom_id
            
        # Compute embedding if not provided
        if embedding is None:
            embedding = self.embed_text_string(text)
            
        # Store embedding and metadata
        self.embeddings_store[embedding_id] = embedding
        self.metadata_store[embedding_id] = {
            'text': text,
            'embedding_shape': embedding.shape,
            'created_at': str(uuid.uuid1().time),
            'type': 'general'
        }
        
        # Auto-save to pickle file if requested
        if auto_save:
            self._save_to_pickle()
        
        return embedding_id

    def search_embeddings(self, query_embedding: np.ndarray, n: int = 5, 
                         exclude_emotions: bool = False) -> List[Tuple[str, float, str]]:
        """
        Search for the most similar embeddings
        
        Args:
            query_embedding (np.ndarray): The query embedding vector
            n (int): Number of most similar embeddings to return
            exclude_emotions (bool): Whether to exclude emotion embeddings from search
            
        Returns:
            List[Tuple[str, float, str]]: List of (embedding_id, similarity_score, original_text)
        """
        if exclude_emotions:
            # Filter out emotion embeddings
            available_ids = [eid for eid in self.embeddings_store.keys() 
                           if not eid.startswith('emotion_')]
        else:
            available_ids = list(self.embeddings_store.keys())
        
        if len(available_ids) == 0:
            return []
            
        # Get stored embeddings
        stored_embeddings = np.array([self.embeddings_store[id_] for id_ in available_ids])
        
        # Ensure query embedding is 2D for sklearn
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, stored_embeddings)[0]
        
        # Get top n similar embeddings
        top_indices = np.argsort(similarities)[::-1][:n]
        
        results = []
        for idx in top_indices:
            embedding_id = available_ids[idx]
            similarity_score = similarities[idx]
            original_text = self.metadata_store[embedding_id]['text']
            results.append((embedding_id, float(similarity_score), original_text))
            
        return results

    def search_by_text(self, query_text: str, n: int = 5, 
                      exclude_emotions: bool = False) -> List[Tuple[str, float, str]]:
        """
        Convenience method to search by text (will compute embedding first)
        
        Args:
            query_text (str): The query text
            n (int): Number of most similar embeddings to return
            exclude_emotions (bool): Whether to exclude emotion embeddings from search
            
        Returns:
            List[Tuple[str, float, str]]: List of (embedding_id, similarity_score, original_text)
        """
        query_embedding = self.embed_text_string(query_text)
        return self.search_embeddings(query_embedding, n, exclude_emotions)

    def get_stored_count(self, include_emotions: bool = True) -> int:
        """Get the number of stored embeddings"""
        if include_emotions:
            return len(self.embeddings_store)
        else:
            return len([eid for eid in self.embeddings_store.keys() 
                       if not eid.startswith('emotion_')])

    def get_emotion_count(self) -> int:
        """Get the number of stored emotion embeddings"""
        return len([eid for eid in self.embeddings_store.keys() 
                   if eid.startswith('emotion_')])
        
    def get_embedding_info(self, embedding_id: str) -> Dict:
        """Get metadata about a stored embedding"""
        if embedding_id not in self.metadata_store:
            return {}
        return self.metadata_store[embedding_id]

    def clear_all_embeddings(self, auto_save: bool = True):
        """
        Clear all stored embeddings
        
        Args:
            auto_save (bool): Whether to automatically save to pickle file
        """
        self.embeddings_store.clear()
        self.metadata_store.clear()
        self.emotions_initialized = False
        
        if auto_save:
            self._save_to_pickle()

    def batch_save_embeddings(self, texts: List[str], custom_ids: Optional[List[str]] = None) -> List[str]:
        """
        Save multiple embeddings at once (more efficient for large batches)
        
        Args:
            texts (List[str]): List of texts to embed and save
            custom_ids (Optional[List[str]]): Optional list of custom IDs
            
        Returns:
            List[str]: List of IDs used to store the embeddings
        """
        if custom_ids is not None and len(custom_ids) != len(texts):
            raise ValueError("custom_ids must have the same length as texts")
            
        saved_ids = []
        for i, text in enumerate(texts):
            custom_id = custom_ids[i] if custom_ids else None
            # Don't auto-save for each individual embedding in batch
            embedding_id = self.save_embedding(text, custom_id=custom_id, auto_save=False)
            saved_ids.append(embedding_id)
            
        # Save once at the end for efficiency
        self._save_to_pickle()
        return saved_ids

    def delete_embedding(self, embedding_id: str, auto_save: bool = True) -> bool:
        """
        Delete an embedding from the store
        
        Args:
            embedding_id (str): ID of the embedding to delete
            auto_save (bool): Whether to automatically save to pickle file
            
        Returns:
            bool: True if deleted successfully, False if ID not found
        """
        if embedding_id in self.embeddings_store:
            del self.embeddings_store[embedding_id]
            del self.metadata_store[embedding_id]
            
            if auto_save:
                self._save_to_pickle()
            return True
        return False

    def save_to_disk(self):
        """Manually save embeddings to disk"""
        self._save_to_pickle()

    def get_pickle_file_path(self) -> str:
        """Get the path to the pickle file"""
        return self.pickle_file

    def set_pickle_file_path(self, new_path: str):
        """
        Set a new path for the pickle file
        
        Args:
            new_path (str): New path for the pickle file
        """
        self.pickle_file = new_path

# Example usage and testing
if __name__ == "__main__":
    # Initialize the embedder
    embedder = MxBaiEmbedder("emotion_embeddings_store.pkl")
    
    # Load the model
    if not embedder.load_model():
        print("Failed to load model. Exiting.")
        exit(1)
    
    # Initialize emotion embeddings if not already done
    if not embedder.emotions_initialized:
        print("Initializing emotion embeddings...")
        embedder.initialize_emotion_embeddings()
    
    print(f"Total embeddings stored: {embedder.get_stored_count()}")
    print(f"Emotion embeddings: {embedder.get_emotion_count()}")
    
    # Test emotion recognition
    print("\n--- Testing Emotion Recognition ---")
    
    test_texts = [
        "I'm so excited about my vacation next week!",
        "This situation makes me feel really uncomfortable and I want to avoid it.",
        "I can't believe we are finally having sex",
        "I feel so empty and lost without them.",
        "This person is absolutely amazing and I want to spend more time with them.",
        "I need to know what temperature to cook chicken to so it's done.",
        "That was completely unexpected!",
        "I have such high hopes for the future.",
        "This is absolutely disgusting and I want nothing to do with it.",
        "can you tell me a joke.",
        "I'm going to lick your pussy so good until you cum.",
        "snake",
        "water",
        "sunlight",
        "human",
        "cat",
        "dog",
        "chocolate",
        "poop",
        "superman, movie",
        "joke",
        "Microsoft, Google, cloud, computing",
        "intercourse",
        "sex",
        "blowjob",

    ]
    
    for text in test_texts:
        print(f"\nText: '{text}'")
        
        # Get the most similar emotion
        emotion_name, similarity, emotion_data = embedder.find_most_similar_emotion(text)
        
        print(f"Most similar emotion: {emotion_name} (similarity: {similarity:.4f})")
        print(f"  Mood: {emotion_data['mood']}")
        print(f"  Thoughts: {emotion_data['thoughts']}")
        print(f"  Responses: {emotion_data['responses']}")
        
        # Get top 3 emotions for comparison
        top_emotions = embedder.analyze_text_emotion(text, top_n=3)
        print("  Top 3 emotions:")
        for i, (emo_name, sim_score, emo_data) in enumerate(top_emotions, 1):
            print(f"    {i}. {emo_name} ({sim_score:.4f})")
    
    print(f"\nEmbeddings saved to: {embedder.get_pickle_file_path()}")