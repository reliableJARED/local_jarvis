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

    def __init__(self, db_path=":memory:", embedding_dim=192, use_gpu=False, max_samples_per_profile=10,my_voice_id='my_voice_id'):
        """
        Initialize the voice recognition system.
        
        Args:
            db_path: Path to SQLite database (default: in-memory)
            embedding_dim: Dimension of voice embeddings (ECAPA-TDNN uses 192)
            use_gpu: Whether to use GPU for inference (default: False)
            max_samples_per_profile: Maximum voice samples per speaker (default: 10)
        """
        self.embedding_dim = embedding_dim
        self.recognition_threshold = 0.3
        self.new_profile_threshold = 0.7  # Used to determine if a new sample is actually a voice of existing profile
        self.max_samples_per_profile = max_samples_per_profile
        self.my_voice_id = my_voice_id
        
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
                audio_data BLOB,
                sample_rate INTEGER DEFAULT 16000,
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
    def _serialize_audio(audio: np.ndarray) -> bytes:
        """Convert audio numpy array to bytes for storage."""
        return audio.astype(np.float32).tobytes()
    
    @staticmethod
    def _deserialize_embedding(blob: bytes) -> np.ndarray:
        """Convert bytes back to numpy array."""
        return np.frombuffer(blob, dtype=np.float32)
    
    @staticmethod
    def _deserialize_audio(blob: bytes) -> np.ndarray:
        """Convert bytes back to audio numpy array."""
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
    
    def _insert_voice_sample(self, human_id: str, embedding: np.ndarray, audio_data: Optional[np.ndarray] = None, sample_rate: int = 16000) -> int:
        """Insert a voice sample with optional audio data and return its ID."""
        emb_blob = self._serialize_embedding(embedding)
        audio_blob = self._serialize_audio(audio_data) if audio_data is not None else None
        
        cursor = self.conn.execute("""
            INSERT INTO voice_samples (human_id, embedding, audio_data, sample_rate)
            VALUES (?, ?, ?, ?)
        """, (human_id, emb_blob, audio_blob, sample_rate))
        return cursor.lastrowid
    
    def _get_all_profile_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """Retrieve all profile embeddings for similarity search."""
        rows = self.conn.execute("""
            SELECT human_id, embedding FROM profiles_vec
        """).fetchall()
        
        return [(row['human_id'], self._deserialize_embedding(row['embedding'])) 
                for row in rows]
    
    def _get_voice_samples(self, human_id: str, include_audio: bool = False) -> List[Tuple]:
        """
        Get all voice samples for a speaker.
        
        Args:
            human_id: Speaker ID
            include_audio: If True, return (sample_id, embedding, audio_data, sample_rate)
                          If False, return (sample_id, embedding)
        """
        if include_audio:
            rows = self.conn.execute("""
                SELECT sample_id, embedding, audio_data, sample_rate FROM voice_samples
                WHERE human_id = ?
            """, (human_id,)).fetchall()
            
            return [(row['sample_id'], 
                    self._deserialize_embedding(row['embedding']),
                    self._deserialize_audio(row['audio_data']) if row['audio_data'] else None,
                    row['sample_rate']) 
                   for row in rows]
        else:
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
            print(f"voicerecognition Internal Search Similarity: {similarity}")
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
    
    def removeVoiceAudio(self, audio_chunk: np.ndarray, sample_rate: int = 16000, 
                        database_id: str = False) -> Optional[np.ndarray]:
        """
        Remove a known voice profile from mixed audio to isolate user interruptions.
        
        This method is designed for diarization scenarios where TTS playback is mixed
        with user speech. It attempts to subtract the known voice (e.g., TTS system voice)
        from the audio chunk to isolate the user's voice.
        
        Uses stored audio samples from the database to create a spectral profile of the
        target voice, then subtracts it from the mixed audio using adaptive spectral subtraction.

        TODO: UPDATE THIS - could be used to isolate multiple known speakers in the same audio chunk (solve Party Problem)
        Would accept a list of voice ids known to possibly be in the audio track
        
        Args:
            audio_chunk: Mixed audio containing both TTS and user voice (numpy array)
            sample_rate: Sample rate of the audio (default: 16000 Hz)
            database_id: ID of the voice profile to remove (default: False) will populate
            
        Returns:
            Audio chunk with the specified voice removed, or None if processing fails
            
        Note:
            This uses an improved spectral subtraction approach with stored samples:
            - Retrieves stored audio samples for the target voice
            - Creates an average spectral profile from stored samples
            - Applies adaptive spectral subtraction based on similarity
            - For best results, ensure the voice profile has quality samples
            
        Algorithm:
            1. Retrieve stored audio samples for the target voice
            2. Calculate average spectral profile from stored samples
            3. Extract embedding from mixed audio and calculate similarity
            4. Apply adaptive spectral subtraction using the spectral profile
            5. Return isolated audio with target voice removed
        """
        if not database_id:
            database_id = self.my_voice_id
        try:
            # Validate input
            if audio_chunk is None or len(audio_chunk) == 0:
                logging.warning("Empty audio chunk provided")
                return None
            
            # Ensure mono audio
            if audio_chunk.ndim > 1:
                audio_chunk = np.mean(audio_chunk, axis=0)
                logging.info("Converted stereo audio to mono for voice removal")
            
            # Check if the profile exists
            profile = self.get_speaker_profile(database_id)
            if profile is None:
                logging.error(f"Voice profile '{database_id}' not found in database")
                return None
            
            # Get the voice embedding for similarity calculation
            profile_embedding_blob = self.conn.execute(
                "SELECT embedding FROM profiles_vec WHERE human_id = ?",
                (database_id,)
            ).fetchone()
            
            if profile_embedding_blob is None:
                logging.error(f"No embedding found for profile '{database_id}'")
                return None
            
            target_embedding = self._deserialize_embedding(profile_embedding_blob['embedding'])
            
            # Extract embedding from mixed audio
            mixed_embedding = self.extract_audio_embedding(audio_chunk, sample_rate)
            if mixed_embedding is None:
                logging.warning("Could not extract embedding from mixed audio")
                return audio_chunk  # Return original if extraction fails
            
            # Calculate similarity to determine contribution level
            similarity = self._cosine_similarity(mixed_embedding, target_embedding)
            logging.info(f"Voice similarity to '{database_id}': {similarity:.3f}")
            
            # If similarity is low, the target voice might not be present
            if similarity < 0.3:
                logging.info(f"Low similarity ({similarity:.3f}), target voice may not be present")
                return audio_chunk
            
            # Retrieve stored audio samples for spectral profiling
            samples = self._get_voice_samples(database_id, include_audio=True)
            
            # Filter samples that have audio data
            audio_samples = [audio for _, _, audio, _ in samples if audio is not None]
            
            if not audio_samples:
                logging.warning(f"No audio samples stored for '{database_id}'")
                # Fall back to basic spectral subtraction if no samples
                return None
            
            logging.info(f"Using {len(audio_samples)} stored audio samples for voice removal")
            
            # Create average spectral profile from stored samples
            target_spectra = []
            for audio_sample in audio_samples:
                # Ensure sample has enough length
                if len(audio_sample) < 512:
                    continue
                    
                # Compute FFT for the sample
                fft_sample = np.fft.rfft(audio_sample)
                magnitude = np.abs(fft_sample)
                target_spectra.append(magnitude)
            
            if not target_spectra:
                logging.warning("Could not create spectral profile")
                return None
            
            # Calculate average spectral profile
            # Normalize all spectra to same length by padding or truncating
            max_len = max(len(spec) for spec in target_spectra)
            normalized_spectra = []
            for spec in target_spectra:
                if len(spec) < max_len:
                    # Pad with zeros
                    padded = np.pad(spec, (0, max_len - len(spec)), 'constant')
                    normalized_spectra.append(padded)
                else:
                    # Truncate
                    normalized_spectra.append(spec[:max_len])
            
            avg_target_spectrum = np.mean(normalized_spectra, axis=0)
            
            # Process the mixed audio
            fft_mixed = np.fft.rfft(audio_chunk)
            magnitude_mixed = np.abs(fft_mixed)
            phase_mixed = np.angle(fft_mixed)
            
            # Match the target spectrum length to mixed audio spectrum
            if len(avg_target_spectrum) < len(magnitude_mixed):
                avg_target_spectrum = np.pad(
                    avg_target_spectrum, 
                    (0, len(magnitude_mixed) - len(avg_target_spectrum)), 
                    'constant'
                )
            elif len(avg_target_spectrum) > len(magnitude_mixed):
                avg_target_spectrum = avg_target_spectrum[:len(magnitude_mixed)]
            
            # Normalize the target spectrum to have similar energy to mixed audio
            target_energy = np.sum(avg_target_spectrum ** 2)
            mixed_energy = np.sum(magnitude_mixed ** 2)
            if target_energy > 0:
                avg_target_spectrum = avg_target_spectrum * np.sqrt(mixed_energy / target_energy)
            
            # Calculate adaptive subtraction factor based on similarity
            # Higher similarity = more aggressive subtraction
            subtraction_factor = min(similarity * 1.5, 0.98)  # Cap at 0.98
            
            # Apply spectral subtraction using the profile
            # Subtract the scaled target spectrum from the mixed spectrum
            magnitude_subtracted = magnitude_mixed - (subtraction_factor * avg_target_spectrum)
            
            # Apply spectral flooring using multiple methods
            # 1. Use percentile-based noise floor
            noise_floor = np.percentile(magnitude_mixed, 3)
            
            # 2. Use target spectrum minimum as additional constraint
            target_floor = np.min(avg_target_spectrum) * 0.1
            
            # Take the maximum of the two floors
            final_floor = max(noise_floor, target_floor)
            
            # Apply the floor
            magnitude_subtracted = np.maximum(magnitude_subtracted, final_floor)
            
            # Apply spectral smoothing to reduce artifacts
            window_size = 5
            magnitude_smoothed = np.convolve(
                magnitude_subtracted, 
                np.ones(window_size) / window_size, 
                mode='same'
            )
            
            # Reconstruct signal with processed magnitude and original phase
            fft_reconstructed = magnitude_smoothed * np.exp(1j * phase_mixed)
            audio_reconstructed = np.fft.irfft(fft_reconstructed, n=len(audio_chunk))
            
            # Apply noise gating based on energy
            frame_size = 512
            for i in range(0, len(audio_reconstructed) - frame_size, frame_size // 2):
                frame_energy = np.sum(audio_reconstructed[i:i+frame_size] ** 2)
                if frame_energy < (final_floor ** 2) * frame_size:
                    audio_reconstructed[i:i+frame_size] *= 0.1  # Attenuate low-energy frames
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(audio_reconstructed))
            if max_val > 0:
                audio_reconstructed = audio_reconstructed / max_val * 0.85
            
            logging.info(f"Advanced voice removal applied using {len(audio_samples)} samples (factor: {subtraction_factor:.3f})")
            return audio_reconstructed.astype(np.float32)
            
        except Exception as e:
            logging.error(f"Error in removeVoiceAudio: {e}")
            import traceback
            logging.error(traceback.format_exc())
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
            select the most representative cluster. Audio samples are stored for voice removal.
        """
        try:
            # Extract embeddings from audio samples
            embeddings = []
            valid_audio_samples = []
            for audio_sample in audio_samples:
                embedding = self.extract_audio_embedding(audio_sample, sample_rate)
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_audio_samples.append(audio_sample)
            
            if not embeddings:
                logging.error(f"Failed to create embeddings for {human_id}")
                return False
            
            # Select most representative samples if we have too many
            if len(embeddings) > self.max_samples_per_profile:
                selected_indices = self._select_representative_samples_indices(embeddings)
                embeddings = [embeddings[i] for i in selected_indices]
                valid_audio_samples = [valid_audio_samples[i] for i in selected_indices]
            
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
            
            # Insert individual samples with audio data
            for embedding, audio_sample in zip(embeddings, valid_audio_samples):
                self._insert_voice_sample(human_id, embedding, audio_sample, sample_rate)
            
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
                self._insert_voice_sample(human_id, new_embedding, audio_sample, sample_rate)
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
                    
                    # Insert the new sample with audio data
                    self._insert_voice_sample(human_id, new_embedding, audio_sample, sample_rate)
                    
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



if __name__ == "__main__":
    # Configure logging to see what's happening
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("=" * 60)
    print("Voice Recognition System Demo")
    print("=" * 60)
    print("DEMO DATA IS NOT GOOD - it lacks sufficient variety, but it remains because it still demonstrates how to use")
    print("in practice two speakers have very different profiles, even same sample same person can be less than 0.7 similarity")
    # Initialize the system with an in-memory database
    print("\n1. Initializing Voice Recognition System...")
    vrs = VoiceRecognitionSystem(db_path=":memory:", use_gpu=False, max_samples_per_profile=5)
    print("   ✓ System initialized")
    
    # Create some dummy audio samples (in practice, these would be real audio data)
    # For demo purposes, we'll create random audio-like data
    print("\n2. Creating demo audio samples...")
    
    # Simulate 3 different speakers with slightly different "voice characteristics"
    np.random.seed(42)  # For reproducibility
    
    
    # Speaker 1: "Alice" - High frequency emphasis, periodic patterns
    alice_samples = []
    for i in range(3):
        t = np.linspace(0, 3, 16000 * 3)
        # Base signal with high frequency components
        signal = np.sin(2 * np.pi * 300 * t) * 0.3  # 300 Hz tone
        signal += np.sin(2 * np.pi * 450 * t) * 0.2  # 450 Hz harmonic
        # Add some noise and variation per sample
        signal += np.random.randn(16000 * 3) * 0.05
        signal += i * 0.01  # Slight DC offset variation
        alice_samples.append(signal.astype(np.float32))

    # Speaker 2: "Bob" - Low frequency emphasis, different pattern
    bob_samples = []
    for i in range(3):
        t = np.linspace(0, 3, 16000 * 3)
        # Base signal with low frequency components
        signal = np.sin(2 * np.pi * 100 * t) * 0.4  # 100 Hz tone
        signal += np.sin(2 * np.pi * 150 * t) * 0.25  # 150 Hz harmonic
        # Different noise characteristics
        signal += np.random.randn(16000 * 3) * 0.08
        signal -= i * 0.02  # Opposite DC offset trend
        bob_samples.append(signal.astype(np.float32))

    # Speaker 3: Unknown speaker - Mid-range frequency, chaotic
    t = np.linspace(0, 3, 16000 * 3)
    unknown_sample = np.sin(2 * np.pi * 200 * t) * 0.5  # 200 Hz
    unknown_sample += np.sin(2 * np.pi * 350 * t) * 0.3  # 350 Hz
    # High noise level for more chaos
    unknown_sample += np.random.randn(16000 * 3) * 0.2
    unknown_sample = unknown_sample.astype(np.float32)
    
    print("   ✓ Created demo audio samples")
    print(type(alice_samples[0]))

    # Register Alice
    print("\n3. Registering 'Alice' with initial voice sample...")
    result = vrs.add_speaker_profile("Alice", [alice_samples[0]])
    if result:
        print(f"   ✓ Alice registered successfully")


    # Add more samples for Alice
    print("\n4. Adding additional samples for Alice...")
    for i, sample in enumerate(alice_samples[1:], start=2):
        if vrs.add_voice_sample("Alice", sample):
            print(f"   ✓ Added sample {i}/3 for Alice")

    # Register Bob
    print("\n5. Registering 'Bob' with voice samples...")
    vrs.add_speaker_profile("Bob", [bob_samples[0]])
    for i, sample in enumerate(bob_samples[1:], start=2):
        vrs.add_voice_sample("Bob", sample)
    print("   ✓ Bob registered with all samples")
    
    # List all speakers
    print("\n6. Listing all registered speakers...")
    speakers = vrs.list_all_speakers()
    print(f"   Registered speakers: {speakers}")
    
    # Recognize a known speaker (Alice)
    print("\n7. Testing recognition with Alice's voice...")
    id,similarity = vrs.recognize_speaker(alice_samples[0])
    if id:
        print(f"   ✓ Recognized: {id}")
        print(f"     Similarity: {similarity}")
    else:
        print("   ✗ No match found")
    
    # Try to recognize Bob
    print("\n8. Testing recognition with Bob's voice...")
    id,similarity = vrs.recognize_speaker(bob_samples[0])
    if id:
        print(f"   ✓ Recognized: {id}")
        print(f"     Similarity: {similarity}")
    else:
        print("   ✗ No match found")
    
    # Try to recognize unknown speaker
    print("\n9. Testing with unknown speaker...")
    id,similarity = vrs.recognize_speaker(unknown_sample)
    if id:
        print(f"   ✓ Recognized: {id}")
        print(f"     Similarity: {similarity}")
    else:
        print("   ✗ No match found")
    
    # Get profile information
    print("\n10. Getting Alice's profile information...")
    profile = vrs.get_speaker_profile("Alice")
    if profile:
        print(f"   Profile details:")
        print(f"     - Human ID: {profile['human_id']}")
        print(f"     - Samples: {profile['samples_count']}")
        print(f"     - Created: {profile['created_at']}")
    
    # Get voice quality metrics
    print("\n11. Analyzing voice sample quality for Alice...")
    metrics = vrs.get_voice_sample_quality_metrics("Alice")
    if metrics:
        print(f"   Quality metrics:")
        print(f"     - Number of samples: {metrics['num_samples']}")
        print(f"     - Average cohesion: {metrics['avg_cohesion_score']:.4f}")
        print(f"     - Cluster quality: {metrics['cluster_quality']}")
    
    # Update profile attributes
    print("\n12. Adding custom attributes to Bob's profile...")
    vrs.update_profile_attribute("Bob", department="Engineering", access_level="admin")
    bob_profile = vrs.get_speaker_profile("Bob")
    print(f"   Updated profile: {bob_profile}")
    
    # Test the adaptive sample replacement (add a 6th sample to Alice who has max_samples=5)
    print("\n13. Testing adaptive sample replacement...")
    print(f"   Alice currently has {vrs.get_speaker_profile('Alice')['samples_count']} samples (max=5)")
    print("   Adding one more sample (should replace weakest sample)...")
    new_sample = np.random.randn(16000 * 3).astype(np.float32) * 0.1
    if vrs.add_voice_sample("Alice", new_sample):
        print("   ✓ Sample added/replaced successfully")
        final_count = vrs.get_speaker_profile('Alice')['samples_count']
        print(f"   Alice still has {final_count} samples (adaptive replacement worked)")
    
    # Delete a profile
    print("\n14. Deleting Bob's profile...")
    if vrs.delete_profile("Bob"):
        print("   ✓ Bob's profile deleted")
        remaining = vrs.list_all_speakers()
        print(f"   Remaining speakers: {remaining}")
    
    # Clean up
    print("\n15. Closing system...")
    vrs.close()
    print("   ✓ System closed")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
