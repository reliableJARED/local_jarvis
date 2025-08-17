import logging
import sys, subprocess
from typing import Dict



logging.basicConfig(level=logging.INFO)


class EmotionEngine:

    def __init__(self):
        self.embedder =None #MxBaiEmbedder()
        self.np = None
        self.vector_dimensions = 384
        self._check_and_install_dependencies()
        self._initialize_embedders()
        
        self.default_emotion = 'joy'
        # Plutchik's primary emotions with their characteristics
        self.emotion_database = {
                # Joy family
                'joy': {
                    'mood': 'Sense of energy and possibility',
                    'thoughts': 'Life is going well',
                    'responses': 'Sparks creativity, connection, gives energy',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'ecstasy': {
                    'mood': 'Overwhelming euphoria and elation',
                    'thoughts': 'Everything is perfect and amazing',
                    'responses': 'Boundless enthusiasm, may act impulsively from excitement',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'serenity': {
                    'mood': 'Calm contentment and peace',
                    'thoughts': 'Things are pleasant and stable',
                    'responses': 'Gentle actions, seeks to maintain harmony',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                
                # Sadness family
                'sadness': {
                    'mood': 'Heavy, low energy, withdrawn',
                    'thoughts': 'Things aren\'t going well, feeling loss',
                    'responses': 'Seeks comfort, may isolate, moves slowly',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'grief': {
                    'mood': 'Profound sorrow and despair',
                    'thoughts': 'Something important is gone forever',
                    'responses': 'May be inconsolable, needs support, difficulty functioning',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'pensiveness': {
                    'mood': 'Quiet melancholy and reflection',
                    'thoughts': 'Contemplating what could have been',
                    'responses': 'Introspective, seeks solitude, gentle sadness',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                
                # Trust family
                'trust': {
                    'mood': 'Open and accepting',
                    'thoughts': 'Others are reliable and good',
                    'responses': 'Cooperative, shares freely, seeks connection',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'admiration': {
                    'mood': 'Deep respect and reverence',
                    'thoughts': 'This person/thing is truly worthy',
                    'responses': 'Wants to learn, emulate, or serve',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'acceptance': {
                    'mood': 'Calm acknowledgment',
                    'thoughts': 'This is how things are',
                    'responses': 'Goes with the flow, doesn\'t resist',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                
                # Disgust family
                'disgust': {
                    'mood': 'Repulsed and rejecting',
                    'thoughts': 'This is wrong, contaminated, or inferior',
                    'responses': 'Avoids, criticizes, seeks to remove or cleanse',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'loathing': {
                    'mood': 'Intense revulsion and hatred',
                    'thoughts': 'This is absolutely abhorrent',
                    'responses': 'Strong rejection, may become aggressive to eliminate',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'boredom': {
                    'mood': 'Mild disinterest and restlessness',
                    'thoughts': 'This isn\'t worth my attention',
                    'responses': 'Seeks stimulation elsewhere, disengages',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                
                # Fear family
                'fear': {
                    'mood': 'Anxious alertness and tension',
                    'thoughts': 'Something bad might happen',
                    'responses': 'Cautious, seeks safety, may freeze or flee',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'terror': {
                    'mood': 'Paralyzing dread',
                    'thoughts': 'Immediate danger, might not survive',
                    'responses': 'Fight, flight, or freeze response, acts on instinct',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'apprehension': {
                    'mood': 'Mild worry and uncertainty',
                    'thoughts': 'Something doesn\'t feel quite right',
                    'responses': 'More cautious than usual, seeks reassurance',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                
                # Anger family
                'anger': {
                    'mood': 'Heated and energized',
                    'thoughts': 'This is unfair, I\'ve been wronged',
                    'responses': 'Confrontational, seeks to correct or punish',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'rage': {
                    'mood': 'Burning fury and aggression',
                    'thoughts': 'Must destroy the source of this injustice',
                    'responses': 'Potentially violent, loses rational control',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'annoyance': {
                    'mood': 'Mildly irritated and impatient',
                    'thoughts': 'This is inconvenient or bothersome',
                    'responses': 'Short responses, may express frustration verbally',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                
                # Surprise family
                'surprise': {
                    'mood': 'Startled and alert',
                    'thoughts': 'That was unexpected',
                    'responses': 'Heightened attention, pauses to process',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'amazement': {
                    'mood': 'Awed and wonder-struck',
                    'thoughts': 'This is incredible and beyond belief',
                    'responses': 'Stares, asks questions, wants to understand',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'distraction': {
                    'mood': 'Mildly surprised and unfocused',
                    'thoughts': 'Wait, what was that?',
                    'responses': 'Attention shifts, momentarily loses focus',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                
                # Anticipation family
                'anticipation': {
                    'mood': 'Eager and forward-looking',
                    'thoughts': 'Something good is coming',
                    'responses': 'Prepares, plans, may act impatiently',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'vigilance': {
                    'mood': 'Intense focus and readiness',
                    'thoughts': 'Must be ready for what\'s coming',
                    'responses': 'Hyper-alert, prepared for action',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'interest': {
                    'mood': 'Curious and engaged',
                    'thoughts': 'I want to know more about this',
                    'responses': 'Asks questions, explores, pays attention',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
            
            # Complex emotions formed by combining primary emotions
                'love': {
                    'components': ['joy', 'trust'],
                    'mood': 'Warm, connected, and devoted',
                    'thoughts': 'This person/thing is wonderful and safe',
                    'responses': 'Protective, nurturing, wants to be close',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'submission': {
                    'components': ['trust', 'fear'],
                    'mood': 'Deferential and compliant',
                    'thoughts': 'I should follow their lead',
                    'responses': 'Obeys, seeks approval, avoids conflict',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'awe': {
                    'components': ['fear', 'surprise'],
                    'mood': 'Humbled and overwhelmed',
                    'thoughts': 'This is beyond my understanding',
                    'responses': 'Reverent behavior, may feel small or insignificant',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'disapproval': {
                    'components': ['surprise', 'sadness'],
                    'mood': 'Disappointed and let down',
                    'thoughts': 'This isn\'t what I expected or hoped for',
                    'responses': 'Expresses dissatisfaction, may withdraw support',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'remorse': {
                    'components': ['sadness', 'disgust'],
                    'mood': 'Regretful and self-reproaching',
                    'thoughts': 'I did something wrong and feel bad about it',
                    'responses': 'Apologizes, seeks to make amends, self-punishing',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'contempt': {
                    'components': ['disgust', 'anger'],
                    'mood': 'Superior and disdainful',
                    'thoughts': 'This is beneath me and doesn\'t deserve respect',
                    'responses': 'Dismissive, condescending, may ridicule',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'aggressiveness': {
                    'components': ['anger', 'anticipation'],
                    'mood': 'Hostile and ready for conflict',
                    'thoughts': 'I need to attack before they do',
                    'responses': 'Threatening behavior, seeks confrontation',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                },
                'optimism': {
                    'components': ['anticipation', 'joy'],
                    'mood': 'Hopeful and positive about the future',
                    'thoughts': 'Good things are coming',
                    'responses': 'Plans enthusiastically, encourages others',
                    'embedding': self.np.random.normal(0, 1, self.vector_dimensions)  # Placeholder embedding
                }
            }
        
        #embed and store emotions
        self._initialize_emotion_embeddings()

    def _check_and_install_dependencies(self) -> None:
        """
        Check if required dependencies are installed and install if missing.
        Stores imported modules as instance attributes.
        """
       
        # Check and install NumPy
        try:
            import numpy as np
            self.np = np
        except ImportError:
            logging.debug("NumPy not found. Installing numpy...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
            try:
                import numpy as np
                self.np = np
            except ImportError:
                raise ImportError("WARNING! Failed to install or import NumPy")
        
    def _initialize_embedders(self):
        """Initialize all embedding models."""
        try:
            
            from mxbai_embed import MxBaiEmbedder
            
            logging.debug("Initializing text embedder...")
            
            self.embedder= MxBaiEmbedder()
            
            
            logging.debug("Text embedder initialized successfully")
            
        except ImportError as e:
            logging.error(f"Failed to import embedder class MxBaiEmbedder: {e}")
            raise
    
    def _initialize_emotion_embeddings(self):
        """
        Initialize embeddings for all emotions in the database
        """
        print("Initializing emotion embeddings...")
        
        for emotion in self.emotion_database.keys():
            # Generate embedding using the embedding model
            em = self.emotion_database[emotion]
            
            # Create text to embed from emotion characteristics
            emotion_text = f"{em['mood']} {em['thoughts']}"
            
            # Get embedding - this returns a list
            embedding_result = self.embedder.embed_string(emotion_text)
            
            if embedding_result is None:
                print(f"Warning: Failed to generate embedding for emotion: {emotion}")
                continue
                
            # Convert to numpy array and store
            embedding_array = self.np.array(embedding_result)
            self.emotion_database[emotion]['embedding'] = embedding_array
            
            print(f"Initialized embedding for emotion: {emotion} (shape: {embedding_array.shape})")
        
        print("Finished initializing emotion embeddings.")
    
    def get_emotional_reaction(self, text: str) -> Dict:
        """
        Generate an emotional reaction to an input text.
        
        Args:
            text: Input text to analyze emotion embedding similarity
            
        Returns:
            Dict: {'emotion': 'joy',
                'mood': 'Sense of energy and possibility',
                'thoughts': 'Life is going well',
                'responses': 'Sparks creativity, connection, gives energy',
                'embedding': vector with self.vector_dimensions,
                'similarity': float  # Added similarity score
                }
        """
        # Embed the input text with the same model used for emotions
        input_embedding_result = self.embedder.embed_string(text)
        
        logging.debug(f"Input embedding result type: {type(input_embedding_result)}")
        logging.debug(f"Input embedding result shape/length: {len(input_embedding_result) if input_embedding_result else 'None'}")

        # Check if input_embedding is valid
        if input_embedding_result is None:
            logging.error("Error: Input text embedding is None")
            return {"Error": "No emotional reaction, couldn't embed input"}
        
        # Convert to numpy array for easier manipulation
        input_embedding = self.np.array(input_embedding_result)
        
        if len(input_embedding) != self.vector_dimensions:
            logging.error(f"Error: Input embedding shape is {input_embedding.shape}, expected ({self.vector_dimensions},)")
            return {"Error": "No emotional reaction, wrong embedding shape"}
    
        # Find the closest emotion embedding
        best_emotion = self.default_emotion
        best_similarity = -1.0
        
        for emotion, data in self.emotion_database.items():
            emotion_embedding = data['embedding']
            if emotion_embedding is None:
                logging.debug(f"Warning: {emotion} has no embedding, skipping")
                continue
            
            # Ensure emotion embedding is numpy array
            if not isinstance(emotion_embedding, self.np.ndarray):
                emotion_embedding = self.np.array(emotion_embedding)
            
            # Check if dimensions match
            if len(emotion_embedding) != self.vector_dimensions:
                logging.debug(f"Warning: {emotion} embedding has wrong dimensions ({len(emotion_embedding)} vs {self.vector_dimensions})")
                continue
            
            # Calculate cosine similarity between input and emotion embeddings
            # Cosine similarity = dot product / (norm1 * norm2)
            dot_product = self.np.dot(input_embedding, emotion_embedding)
            norm_input = self.np.linalg.norm(input_embedding)
            norm_emotion = self.np.linalg.norm(emotion_embedding)
            
            # Avoid division by zero
            if norm_input == 0 or norm_emotion == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm_input * norm_emotion)
            
            logging.debug(f"Emotion: {emotion}, Similarity: {similarity:.4f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_emotion = emotion
        
        logging.debug(f"Best matching emotion: {best_emotion} (similarity: {best_similarity:.4f})")
        
        # Create result dictionary
        result = {
            'emotion': best_emotion,
            'similarity': best_similarity
        }
        
        # Add all the emotion data
        result.update(self.emotion_database[best_emotion])
        
        return result


# Example usage and testing
if __name__ == "__main__":

        emotion = EmotionEngine()

        print(emotion.emotion_database)

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
        "chocolate bar",
        "poop",
        "superman movie",
        "joke",
        "Microsoft, Google, cloud, computing",
        "intercourse",
        "sex",
        "please give me a blowjob",
        "I need you to mow the lawn",

    ]
        for string in test_texts:

            result = emotion.get_emotional_reaction(string)
            
            print(string,":",result['emotion'],result['similarity'],result['mood'],result['thoughts'],">>",result['responses'])
