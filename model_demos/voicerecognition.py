import sqlite3
import numpy as np
import torch
import logging
from typing import List, Tuple, Optional, Dict
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['SPEECHBRAIN_CACHE_FOLDER'] = os.path.abspath('pretrained_models')

from speechbrain.inference.speaker import EncoderClassifier


class VoiceRecognitionSystem:
    """
    Simplified voice recognition system using standard SQLite for storage.
    
    Uses SpeechBrain's ECAPA-TDNN model trained on Voxceleb1+Voxceleb2.
    Model performance: 0.80% EER on Voxceleb1-test (cleaned).
    
    Features:
        - Adaptive sample cluster refinement (automatically replaces weak samples)
        - Audio should be 16kHz, single channel
        - Uses cosine similarity for speaker verification
        - Embeddings are 192-dimensional (see speechbrain docs)
    """

    def __init__(self, db_path=":memory:", embedding_dim=192, use_gpu=False, max_samples_per_profile=10):
        """
        Initialize the voice recognition system.
        
        Args:
            db_path: Path to SQLite database (default: in-memory)
            embedding_dim: Dimension of voice embeddings (ECAPA-TDNN uses 192)
            use_gpu: Whether to use GPU for inference (default: False)
            max_samples_per_profile: Maximum voice samples per speaker (default: 10)
        """
        self.embedding_dim = embedding_dim
        self.recognition_threshold = 0.4
        self.new_profile_threshold = 0.7  # Used to determine if a new sample is actually a voice of existing profile
        self.max_samples_per_profile = max_samples_per_profile
        
        # Connect to standard SQLite database
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Initialize SpeechBrain model with optional GPU support
        run_opts = {"device": "cuda"} if use_gpu else {}
        self.speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts=run_opts,
            use_auth_token=False,
        )
        
        # Create tables
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        # Table for speaker profiles
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                human_id TEXT PRIMARY KEY,
                samples_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table for average voice embeddings (for quick matching)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles_vec (
                human_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                FOREIGN KEY (human_id) REFERENCES profiles(human_id) ON DELETE CASCADE
            )
        """)
        
        # Table for individual voice samples (up to max_samples_per_profile per person)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS voice_samples (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                human_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (human_id) REFERENCES profiles(human_id) ON DELETE CASCADE
            )
        """)
        
        # Create index for faster queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_voice_samples_human_id 
            ON voice_samples(human_id)
        """)
        
        self.conn.commit()
    
    # ==================== SQL Helper Functions ====================
    
    @staticmethod
    def _serialize_embedding(embedding: np.ndarray) -> bytes:
        """Convert numpy array to bytes for storage."""
        return embedding.tobytes()
    
    @staticmethod
    def _deserialize_embedding(blob: bytes) -> np.ndarray:
        """Convert bytes back to numpy array."""
        return np.frombuffer(blob, dtype=np.float32)
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two normalized embeddings."""
        # For normalized embeddings, cosine similarity is just the dot product
        return float(np.dot(emb1, emb2))
    
    def _insert_profile_embedding(self, human_id: str, embedding: np.ndarray):
        """Insert or update profile average embedding."""
        blob = self._serialize_embedding(embedding)
        self.conn.execute("""
            INSERT OR REPLACE INTO profiles_vec (human_id, embedding)
            VALUES (?, ?)
        """, (human_id, blob))
    
    def _insert_voice_sample(self, human_id: str, embedding: np.ndarray) -> int:
        """Insert a voice sample and return its ID."""
        blob = self._serialize_embedding(embedding)
        cursor = self.conn.execute("""
            INSERT INTO voice_samples (human_id, embedding)
            VALUES (?, ?)
        """, (human_id, blob))
        return cursor.lastrowid
    
    def _get_all_profile_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """Retrieve all profile embeddings for similarity search."""
        rows = self.conn.execute("""
            SELECT human_id, embedding FROM profiles_vec
        """).fetchall()
        
        return [(row['human_id'], self._deserialize_embedding(row['embedding'])) 
                for row in rows]
    
    def _get_voice_samples(self, human_id: str) -> List[Tuple[int, np.ndarray]]:
        """Get all voice samples for a speaker."""
        rows = self.conn.execute("""
            SELECT sample_id, embedding FROM voice_samples
            WHERE human_id = ?
        """, (human_id,)).fetchall()
        
        return [(row['sample_id'], self._deserialize_embedding(row['embedding'])) 
                for row in rows]
    
    def _find_nearest_profile(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Find the nearest profile to the given embedding."""
        profiles = self._get_all_profile_embeddings()
        
        if not profiles:
            return None, 0.0
        
        best_match = None
        best_similarity = -1.0
        
        for human_id, profile_emb in profiles:
            similarity = self._cosine_similarity(embedding, profile_emb)
            print(f"Internal Search Similarity: {similarity}")
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = human_id
        
        return best_match, best_similarity
    
    # ==================== Core Audio Processing ====================
    
    def extract_audio_embedding(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """
        Extract normalized embedding from audio data.
        
        Args:
            audio_data: Audio as numpy array
            sample_rate: Sample rate of the audio (default: 16000 as required by model)
            
        Note:
            Model expects 16kHz single-channel audio. Audio will be converted to mono if stereo.
        """
        try:
            # Ensure audio is single channel (mono)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=0)
                logging.info("Converted stereo audio to mono")
            
            # Warn if sample rate is not 16kHz
            if sample_rate != 16000:
                logging.warning(
                    f"Audio should be 16kHz for optimal performance, got {sample_rate}Hz. "
                    "Consider resampling your audio."
                )
            
            # Convert to tensor and add batch dimension
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32)).unsqueeze(0)
            
            with torch.no_grad():
                embedding = self.speaker_model.encode_batch(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            # Normalize the embedding for cosine similarity
            norm = np.linalg.norm(embedding)
            normalized = (embedding / norm if norm > 0 else embedding).astype(np.float32)
            
            return normalized
            
        except Exception as e:
            logging.error(f"Error extracting embedding: {e}")
            return None
    
    def _calculate_cohesion_score(self, embedding: np.ndarray, cluster: List[np.ndarray]) -> float:
        """
        Calculate how well an embedding fits with a cluster.
        Returns average cosine similarity to all embeddings in the cluster.
        
        Args:
            embedding: The embedding to score (should be normalized)
            cluster: List of embeddings forming the cluster (should be normalized)
            
        Returns:
            Average similarity score (higher = better fit, range: -1 to 1)
        """
        if not cluster:
            return 0.0
        
        # For normalized embeddings, cosine similarity is just the dot product
        similarities = [self._cosine_similarity(embedding, cluster_emb) for cluster_emb in cluster]
        
        return float(np.mean(similarities))
    
    def _select_representative_samples(self, embeddings: List[np.ndarray], 
                                      max_samples: int = None) -> List[np.ndarray]:
        """
        Select the most representative cluster of samples from a larger set.
        Selects samples closest to the overall centroid.
        
        Args:
            embeddings: List of voice embeddings
            max_samples: Maximum number of samples to return (default: self.max_samples_per_profile)
            
        Returns:
            List of selected embeddings that form the most cohesive cluster
        """
        if max_samples is None:
            max_samples = self.max_samples_per_profile
            
        if len(embeddings) <= max_samples:
            return embeddings
        
        embeddings_array = np.array(embeddings)
        
        # Calculate overall centroid
        centroid = np.mean(embeddings_array, axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # Normalize
        
        # Calculate distances from each embedding to the centroid
        # For normalized vectors: distance = 1 - cosine_similarity
        similarities = [self._cosine_similarity(emb, centroid) for emb in embeddings_array]
        distances = [1 - sim for sim in similarities]
        
        # Select the max_samples closest to the centroid
        closest_indices = np.argsort(distances)[:max_samples]
        
        selected_embeddings = [embeddings[i] for i in closest_indices]
        
        logging.info(f"Selected {max_samples} most representative samples from {len(embeddings)} total")
        return selected_embeddings
    
    # ==================== Public API Methods ====================
    
    def verify_speakers(self, audio1: np.ndarray, audio2: np.ndarray, 
                       sample_rate: int = 16000) -> Tuple[float, bool]:
        """
        Verify if two audio samples are from the same speaker.
        
        Args:
            audio1: First audio sample
            audio2: Second audio sample
            sample_rate: Sample rate of both audio samples
            
        Returns:
            Tuple of (similarity_score, is_same_speaker)
        """
        emb1 = self.extract_audio_embedding(audio1, sample_rate)
        emb2 = self.extract_audio_embedding(audio2, sample_rate)
        
        if emb1 is None or emb2 is None:
            return 0.0, False
        
        # Calculate cosine similarity (embeddings are already normalized)
        similarity = self._cosine_similarity(emb1, emb2)
        is_same = similarity >= self.recognition_threshold
        
        return float(similarity), is_same
    
    def add_speaker_profile(
        self, 
        human_id: str, 
        audio_samples: List[np.ndarray],
        sample_rate: int = 16000,
        **attributes
    ) -> bool:
        """
        Add a new speaker profile with audio samples and attributes.
        
        Args:
            human_id: Unique identifier for the speaker
            audio_samples: List of audio samples (np.ndarray), each should be 16kHz mono
            sample_rate: Sample rate of the audio samples (default: 16000)
            **attributes: Additional profile attributes (name, age, etc.)
            
        Note:
            If more than max_samples_per_profile samples provided, will automatically
            select the most representative cluster.
        """
        try:
            # Extract embeddings from audio samples
            embeddings = []
            for audio_sample in audio_samples:
                embedding = self.extract_audio_embedding(audio_sample, sample_rate)
                if embedding is not None:
                    embeddings.append(embedding)
            
            if not embeddings:
                logging.error(f"Failed to create embeddings for {human_id}")
                return False
            
            # Select most representative samples if we have too many
            if len(embeddings) > self.max_samples_per_profile:
                embeddings = self._select_representative_samples(embeddings)
            
            # Check if voice already exists
            # Use average of all embeddings for checking
            check_embedding = np.mean(embeddings, axis=0)
            check_embedding = check_embedding / np.linalg.norm(check_embedding)
            existing_id, similarity = self._find_nearest_profile(check_embedding)
            
            if existing_id and similarity >= self.new_profile_threshold:
                logging.warning(f"Voice already exists as {existing_id} (similarity: {similarity:.3f})")
                return False
            
            # Calculate average embedding
            avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
            
            # Add any additional columns from attributes
            attr_columns = list(attributes.keys())
            if attr_columns:
                # Check if columns exist, if not add them
                for col in attr_columns:
                    try:
                        self.conn.execute(f"ALTER TABLE profiles ADD COLUMN {col} TEXT")
                    except sqlite3.OperationalError:
                        pass  # Column already exists
            
            # Insert profile attributes
            columns = ['human_id', 'samples_count'] + attr_columns
            placeholders = ', '.join(['?'] * len(columns))
            values = [human_id, len(embeddings)] + [attributes[k] for k in attr_columns]
            
            self.conn.execute(
                f"INSERT INTO profiles ({', '.join(columns)}) VALUES ({placeholders})",
                values
            )
            
            # Insert average embedding
            self._insert_profile_embedding(human_id, avg_embedding)
            
            # Insert individual samples
            for embedding in embeddings:
                self._insert_voice_sample(human_id, embedding)
            
            self.conn.commit()
            logging.info(f"Added speaker profile: {human_id} with {len(embeddings)} samples")
            return True
            
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error adding speaker profile {human_id}: {e}")
            return False
    
    def recognize_speaker(self, audio_data: np.ndarray,
                          threshold: float = 0.0, 
                          sample_rate: int = 16000) -> Tuple[Optional[str], float]:
        """
        Recognize speaker from audio sample.
        
        Args:
            audio_data: Audio sample (should be 16kHz mono)
            threshold: Recognition threshold (default: uses self.recognition_threshold)
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (human_id, similarity) or (None, 0.0) if no match
        """
        if threshold == 0.0:
            threshold = self.recognition_threshold

        embedding = self.extract_audio_embedding(audio_data, sample_rate)
        if embedding is None:
            return None, 0.0
        
        # Find nearest match using helper function
        human_id, similarity = self._find_nearest_profile(embedding)
        
        if human_id and similarity >= threshold:
            return human_id, similarity
        
        return None, 0.0
    
    def add_voice_sample(self, human_id: str, audio_sample: np.ndarray,
                        sample_rate: int = 16000) -> bool:
        """
        Add an additional voice sample to existing profile.
        
        If profile has < max_samples: adds the new sample
        If profile has max_samples: evaluates if new sample fits the cluster better
        than the least cohesive existing sample. If so, replaces it.
        
        Args:
            human_id: Speaker's voice ID
            audio_sample: Audio sample (should be 16kHz mono)
            sample_rate: Sample rate of the audio
            
        Returns:
            True if sample was added/replaced, False if rejected
        """
        # Check if profile exists
        result = self.conn.execute(
            "SELECT samples_count FROM profiles WHERE human_id = ?",
            (human_id,)
        ).fetchone()
        
        if not result:
            logging.error(f"Profile {human_id} not found")
            return False
        
        samples_count = result['samples_count']
        
        # Extract embedding from new sample
        new_embedding = self.extract_audio_embedding(audio_sample, sample_rate)
        if new_embedding is None:
            return False
        
        try:
            # Get all existing samples
            samples = self._get_voice_samples(human_id)
            existing_embeddings = [emb for _, emb in samples]
            sample_ids = [sid for sid, _ in samples]
            
            if samples_count < self.max_samples_per_profile:
                # Still have room, just add the new sample
                self._insert_voice_sample(human_id, new_embedding)
                existing_embeddings.append(new_embedding)
                
                logging.info(f"Added sample to {human_id} ({samples_count + 1}/{self.max_samples_per_profile})")
                
            else:
                # Profile is at capacity - check if new sample should replace an existing one
                
                # Calculate cohesion score for the new sample with existing cluster
                new_sample_score = self._calculate_cohesion_score(
                    new_embedding, existing_embeddings
                )
                
                # Calculate cohesion score for each existing sample with the rest
                cohesion_scores = []
                for i, emb in enumerate(existing_embeddings):
                    # Calculate score with all OTHER samples (excluding itself)
                    other_embeddings = existing_embeddings[:i] + existing_embeddings[i+1:]
                    score = self._calculate_cohesion_score(emb, other_embeddings)
                    cohesion_scores.append(score)
                
                # Find the least cohesive existing sample
                least_cohesive_idx = np.argmin(cohesion_scores)
                least_cohesive_score = cohesion_scores[least_cohesive_idx]
                
                # If new sample fits better than the least cohesive, replace it
                if new_sample_score > least_cohesive_score:
                    sample_id_to_replace = sample_ids[least_cohesive_idx]
                    
                    # Delete the least cohesive sample
                    self.conn.execute(
                        "DELETE FROM voice_samples WHERE sample_id = ?",
                        (sample_id_to_replace,)
                    )
                    
                    # Insert the new sample
                    self._insert_voice_sample(human_id, new_embedding)
                    
                    # Update the embeddings list for recalculation
                    existing_embeddings[least_cohesive_idx] = new_embedding
                    
                    logging.info(
                        f"Replaced least cohesive sample for {human_id} "
                        f"(old score: {least_cohesive_score:.3f}, new score: {new_sample_score:.3f})"
                    )
                else:
                    logging.info(
                        f"New sample not cohesive enough for {human_id} "
                        f"(score: {new_sample_score:.3f} vs worst existing: {least_cohesive_score:.3f})"
                    )
                    return False
            
            # Recalculate average embedding with updated samples
            avg_embedding = np.mean(existing_embeddings, axis=0).astype(np.float32)
            
            # Update average in profiles_vec
            self._insert_profile_embedding(human_id, avg_embedding)
            
            # Update sample count and timestamp
            self.conn.execute(
                "UPDATE profiles SET samples_count = ?, updated_at = CURRENT_TIMESTAMP WHERE human_id = ?",
                (len(existing_embeddings), human_id)
            )
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error adding sample to {human_id}: {e}")
            return False
    
    def get_speaker_profile(self, human_id: str) -> Optional[Dict]:
        """Get complete profile information for a speaker."""
        result = self.conn.execute(
            "SELECT * FROM profiles WHERE human_id = ?",
            (human_id,)
        ).fetchone()
        
        if result:
            return dict(result)
        
        return None
    
    def get_voice_sample_quality_metrics(self, human_id: str) -> Optional[Dict]:
        """
        Get quality metrics about a speaker's voice samples.
        
        Returns:
            Dictionary with cohesion scores and statistics
        """
        # Get all samples
        samples = self._get_voice_samples(human_id)
        
        if not samples:
            return None
        
        embeddings = [emb for _, emb in samples]
        
        # Calculate centroid
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        # Calculate distance of each sample to centroid
        distances = [1 - self._cosine_similarity(emb, centroid) for emb in embeddings]
        
        # Calculate pairwise cohesion scores
        cohesion_scores = []
        for i, emb in enumerate(embeddings):
            other_embeddings = embeddings[:i] + embeddings[i+1:]
            if other_embeddings:
                score = self._calculate_cohesion_score(emb, other_embeddings)
                cohesion_scores.append(score)
        
        return {
            'human_id': human_id,
            'num_samples': len(embeddings),
            'avg_distance_to_centroid': float(np.mean(distances)),
            'max_distance_to_centroid': float(np.max(distances)),
            'min_distance_to_centroid': float(np.min(distances)),
            'avg_cohesion_score': float(np.mean(cohesion_scores)) if cohesion_scores else 0.0,
            'min_cohesion_score': float(np.min(cohesion_scores)) if cohesion_scores else 0.0,
            'cluster_quality': 'good' if np.mean(distances) < 0.1 else 'fair' if np.mean(distances) < 0.2 else 'poor'
        }
    
    def update_profile_attribute(self, human_id: str, **attributes) -> bool:
        """Update profile attributes."""
        if not attributes:
            return False
        
        # Ensure columns exist
        for col in attributes.keys():
            try:
                self.conn.execute(f"ALTER TABLE profiles ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
        
        set_clause = ', '.join([f"{key} = ?" for key in attributes.keys()])
        values = list(attributes.values()) + [human_id]
        
        try:
            self.conn.execute(
                f"UPDATE profiles SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE human_id = ?",
                values
            )
            self.conn.commit()
            return True
        except Exception as e:
            logging.error(f"Error updating profile {human_id}: {e}")
            return False
    
    def list_all_speakers(self) -> List[str]:
        """Get list of all registered speaker IDs."""
        results = self.conn.execute("SELECT human_id FROM profiles").fetchall()
        return [r['human_id'] for r in results]
    
    def delete_profile(self, human_id: str) -> bool:
        """Delete a speaker profile completely."""
        try:
            self.conn.execute("DELETE FROM profiles WHERE human_id = ?", (human_id,))
            self.conn.execute("DELETE FROM profiles_vec WHERE human_id = ?", (human_id,))
            self.conn.execute("DELETE FROM voice_samples WHERE human_id = ?", (human_id,))
            self.conn.commit()
            logging.info(f"Deleted profile: {human_id}")
            return True
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error deleting profile {human_id}: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        self.conn.close()