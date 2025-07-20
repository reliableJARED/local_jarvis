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

class MxBaiEmbedder:
    def __init__(self, pickle_file: str = "embeddings_store.pkl"):
        self.tokenizer = None
        self.model = None
        self.embeddings_store = {}  # Dict to store embeddings with UUID keys
        self.metadata_store = {}    # Store original text and other metadata
        self.pickle_file = pickle_file
        
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
                print(f"Loaded {len(self.embeddings_store)} embeddings from {self.pickle_file}")
            except Exception as e:
                print(f"Error loading from pickle file: {str(e)}")
                print("Starting with empty embeddings store.")
                self.embeddings_store = {}
                self.metadata_store = {}
        else:
            print(f"No existing pickle file found at {self.pickle_file}. Starting with empty store.")

    def _save_to_pickle(self):
        """Save embeddings and metadata to pickle file"""
        try:
            data = {
                'embeddings_store': self.embeddings_store,
                'metadata_store': self.metadata_store
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
        """Generate deterministic ID based on conten
        THIS WILL AUTO overwrite duplicates. So it's impossible to save enter the same thing twice. con being counting and same text different meta data"""
        return hashlib.md5(text.encode()).hexdigest()

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
            'created_at': str(uuid.uuid1().time)
        }
        
        # Auto-save to pickle file if requested
        if auto_save:
            self._save_to_pickle()
        
        return embedding_id

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

    def search_embeddings(self, query_embedding: np.ndarray, n: int = 5) -> List[Tuple[str, float, str]]:
        """
        Search for the most similar embeddings
        
        Args:
            query_embedding (np.ndarray): The query embedding vector
            n (int): Number of most similar embeddings to return
            
        Returns:
            List[Tuple[str, float, str]]: List of (embedding_id, similarity_score, original_text)
        """
        if len(self.embeddings_store) == 0:
            return []
            
        # Get all stored embeddings
        stored_ids = list(self.embeddings_store.keys())
        stored_embeddings = np.array([self.embeddings_store[id_] for id_ in stored_ids])
        
        # Ensure query embedding is 2D for sklearn
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, stored_embeddings)[0]
        
        # Get top n similar embeddings
        top_indices = np.argsort(similarities)[::-1][:n]
        
        results = []
        for idx in top_indices:
            embedding_id = stored_ids[idx]
            similarity_score = similarities[idx]
            original_text = self.metadata_store[embedding_id]['text']
            results.append((embedding_id, float(similarity_score), original_text))
            
        return results

    def search_by_text(self, query_text: str, n: int = 5) -> List[Tuple[str, float, str]]:
        """
        Convenience method to search by text (will compute embedding first)
        
        Args:
            query_text (str): The query text
            n (int): Number of most similar embeddings to return
            
        Returns:
            List[Tuple[str, float, str]]: List of (embedding_id, similarity_score, original_text)
        """
        query_embedding = self.embed_text_string(query_text)
        return self.search_embeddings(query_embedding, n)

    def get_stored_count(self) -> int:
        """Get the number of stored embeddings"""
        return len(self.embeddings_store)
        
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
        
        if auto_save:
            self._save_to_pickle()

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
    # Initialize the embedder (will automatically load existing embeddings)
    embedder = MxBaiEmbedder()
    
    # Load the model
    if not embedder.load_model():
        print("Failed to load model. Exiting.")
        exit(1)
    
    print(f"Starting with {embedder.get_stored_count()} existing embeddings")
    
    # Example texts to embed
    sample_texts = [
        "prompt: what temperature to cook chicken. response: cook chicken to an internal temperature of 165 degrees fahrenheit to ensure food safety and eliminate harmful bacteria like salmonella. use a meat thermometer inserted into the thickest part of the breast or thigh, avoiding bone contact. the chicken should rest for 3-5 minutes after cooking to allow juices to redistribute throughout the meat, resulting in a more tender and flavorful final product.",
        
        "prompt: how often should i water my houseplants. response: water houseplants when the top inch of soil feels dry to the touch, which typically occurs every 7-10 days for most indoor plants. factors like humidity, temperature, pot size, and plant type significantly affect watering frequency. succulents need water less frequently, perhaps every 2-3 weeks, while tropical plants may require more frequent watering. always check soil moisture rather than following a rigid schedule, and ensure proper drainage to prevent root rot.",
        
        "prompt: what is the best workout routine for beginners. response: beginners should start with a balanced routine that includes 150 minutes of moderate cardio per week, strength training 2-3 times per week targeting all major muscle groups, and flexibility exercises daily. start with bodyweight exercises like squats, push-ups, and planks, gradually increasing intensity and duration. rest days are crucial for recovery and muscle growth. consider working with a trainer initially to ensure proper form and prevent injury while building a sustainable fitness habit.",
        
        "prompt: how many hours of sleep do adults need each night. response: adults need 7-9 hours of quality sleep per night for optimal health and cognitive function. establishing a consistent sleep schedule helps regulate your circadian rhythm, while creating a cool, dark, and quiet sleep environment promotes deeper rest. avoid caffeine 6 hours before bedtime, limit screen time 1 hour before sleep, and consider relaxation techniques like meditation or gentle stretching to prepare your body and mind for restorative sleep.",
        
        "prompt: how much money should i save for retirement. response: financial experts recommend saving 10-15% of your gross income for retirement, starting as early as possible to take advantage of compound interest. if you begin saving in your 20s, you may need to save less due to longer growth periods, while starting later requires higher contribution rates. utilize employer 401k matching programs, consider roth ira contributions for tax-free growth, and regularly review and adjust your investment portfolio based on your age, risk tolerance, and retirement timeline.",
        
        "prompt: what is the most effective way to learn a new language. response: effective language learning combines multiple approaches including daily practice with native speakers through conversation exchange apps, immersion through media consumption like podcasts and movies, structured grammar study using textbooks or apps, and consistent vocabulary building through spaced repetition systems. dedicate at least 30 minutes daily to practice, focus on practical phrases you'll actually use, and don't be afraid to make mistakes as they're essential for learning progress and building confidence.",
        
        "prompt: how often should i clean my gutters. response: clean gutters at least twice yearly, typically in late spring and early fall, to prevent water damage and ice dams. homes surrounded by trees may require more frequent cleaning, potentially 3-4 times per year. signs that gutters need attention include water overflow during rain, sagging sections, or visible plant growth. regular maintenance prevents costly repairs to your roof, foundation, and landscaping while ensuring proper water drainage away from your home's structure.",
        
        "prompt: when should i change my car's oil. response: change your car's oil every 3,000-7,500 miles depending on your vehicle's age, driving conditions, and oil type used. newer cars with synthetic oil can often go 7,500-10,000 miles between changes, while older vehicles or those driven in severe conditions like stop-and-go traffic, extreme temperatures, or dusty environments may need more frequent changes. check your owner's manual for manufacturer recommendations and monitor oil level and color regularly between scheduled changes.",
        
        "prompt: when is the best time to plant vegetables in my garden. response: plant vegetables based on your local climate zone and the last expected frost date, typically 2-6 weeks after the last frost for warm-season crops like tomatoes, peppers, and squash. cool-season vegetables like lettuce, spinach, and peas can be planted 2-4 weeks before the last frost. use companion planting techniques to maximize space and natural pest control, prepare soil with compost and organic matter, and consider succession planting for continuous harvests throughout the growing season.",
        
        "prompt: how can i protect my computer from malware and viruses. response: protect your computer by installing reputable antivirus software and keeping it updated, enabling automatic system updates for your operating system, avoiding suspicious email attachments and downloads from untrusted sources, and using strong unique passwords for all accounts. regularly backup important data to external drives or cloud storage, be cautious when clicking links or downloading software, use a firewall for additional protection, and consider using a password manager to maintain secure login credentials across all your online accounts."
    ]
    
    print("\n--- Embedding and storing sample texts ---")
    # Embed and store sample texts (only if we don't have many embeddings already)
    if embedder.get_stored_count() < 5:
        stored_ids = embedder.batch_save_embeddings(sample_texts)
        for i, text in enumerate(sample_texts):
            print(f"Stored: {text[:50]}... (ID: {stored_ids[i][:8]}...)")
    else:
        print("Already have embeddings stored, skipping sample text embedding.")
    
    print(f"\nTotal embeddings stored: {embedder.get_stored_count()}")
    
    # Test search functionality
    print("\n--- Testing search functionality ---")
    query_text = "neural network training"
    print(f"Searching for: '{query_text}'")
    
    search_results = embedder.search_by_text(query_text, n=3)
    
    print("\nTop 3 similar embeddings:")
    for i, (emb_id, score, text) in enumerate(search_results, 1):
        print(f"{i}. Similarity: {score:.4f}")
        print(f"   Text: {text}")
        print(f"   ID: {emb_id[:8]}...")
        print()

    # Test search functionality with different query
    print("\n--- Testing search functionality with date query ---")
    query_text = "prompt:tell me a joke. response: Knock knock! Who's there? Interrupting cow. Interrupting cow w-- MOOOOO!"
    print(f"Searching for: '{query_text}'")
    
    search_results = embedder.search_by_text(query_text, n=3)
    
    print("\nTop 3 similar embeddings:")
    for i, (emb_id, score, text) in enumerate(search_results, 1):
        print(f"{i}. Similarity: {score:.4f}")
        print(f"   Text: {text}")
        print(f"   ID: {emb_id[:8]}...")
        print()
        
    print(f"\nEmbeddings saved to: {embedder.get_pickle_file_path()}")