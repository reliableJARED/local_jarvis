"""
Enhanced modular speech processing system with word boundary detection for dynamic live transcripts
Also uses speaker diarization to recognize voices
"""

import torch
import sounddevice as sd
import numpy as np
import time
from datetime import datetime
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from collections import deque
import threading
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List, Tuple
import pickle
import os
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine, pdist, squareform
import sys
import warnings
warnings.filterwarnings("ignore")

# Try to import speaker embedding models
try:
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False

@dataclass
class TranscriptSegment:
    """Represents a transcribed speech segment"""
    text: str
    speaker_id: str
    start_time: float
    end_time: float
    confidence: float
    is_final: bool = False

@dataclass
class SpeechEvent:
    """Represents a speech detection event"""
    event_type: str  # 'speech_start', 'speech_end', 'silence'
    timestamp: float
    audio_data: Optional[np.ndarray] = None

@dataclass
class SpeakerProfile:
    """Enhanced speaker profile with multiple embeddings and metadata"""
    speaker_id: str
    embeddings: List[np.ndarray]  # Store multiple embeddings for better accuracy
    first_seen: datetime
    last_updated: datetime
    total_samples: int
    confidence_scores: List[float]  # Track confidence for each embedding
    
    def get_average_embedding(self) -> np.ndarray:
        """Get average embedding from all samples"""
        if not self.embeddings:
            return np.array([])
        
        # Weight embeddings by confidence if available
        if len(self.confidence_scores) == len(self.embeddings):
            weights = np.array(self.confidence_scores)
            weights = weights / np.sum(weights)  # Normalize
            
            weighted_embeddings = [emb * weight for emb, weight in zip(self.embeddings, weights)]
            return np.mean(weighted_embeddings, axis=0)
        else:
            return np.mean(self.embeddings, axis=0)
    
    def add_embedding(self, embedding: np.ndarray, confidence: float = 1.0):
        """Add a new embedding sample"""
        self.embeddings.append(embedding)
        self.confidence_scores.append(confidence)
        self.total_samples += 1
        self.last_updated = datetime.now()
        
        # Keep only the best N embeddings to prevent memory bloat
        max_embeddings = 10
        if len(self.embeddings) > max_embeddings:
            # Remove embeddings with lowest confidence
            sorted_indices = np.argsort(self.confidence_scores)
            keep_indices = sorted_indices[-max_embeddings:]
            
            self.embeddings = [self.embeddings[i] for i in keep_indices]
            self.confidence_scores = [self.confidence_scores[i] for i in keep_indices]

@dataclass
class SpeakerCluster:
    """Represents a cluster of embeddings for a speaker"""
    cluster_id: int
    embeddings: List[np.ndarray]
    confidence_scores: List[float]
    centroid: np.ndarray
    cluster_quality: float  # Silhouette score or similar
    sample_count: int
    
    def update_centroid(self):
        """Recalculate centroid with confidence weighting"""
        if not self.embeddings:
            return
        
        try:
            # Ensure embeddings are numpy arrays
            embeddings_array = [np.array(emb) if not isinstance(emb, np.ndarray) else emb 
                            for emb in self.embeddings]
            
            if len(self.confidence_scores) == len(embeddings_array):
                weights = np.array(self.confidence_scores)
                if np.sum(weights) > 0:  # Avoid division by zero
                    weights = weights / np.sum(weights)
                    weighted_embeddings = [emb * weight for emb, weight in zip(embeddings_array, weights)]
                    self.centroid = np.mean(weighted_embeddings, axis=0)
                else:
                    self.centroid = np.mean(embeddings_array, axis=0)
            else:
                self.centroid = np.mean(embeddings_array, axis=0)
        except Exception as e:
            print(f"Error updating centroid: {e}")
            # Fallback: just use simple mean if weighting fails
            try:
                embeddings_array = [np.array(emb) if not isinstance(emb, np.ndarray) else emb 
                                for emb in self.embeddings]
                self.centroid = np.mean(embeddings_array, axis=0)
            except:
                self.centroid = np.array([])

@dataclass
class ClusteredSpeakerProfile:
    """Enhanced speaker profile with clustering support"""
    speaker_id: str
    clusters: List[SpeakerCluster]
    first_seen: datetime
    last_updated: datetime
    total_samples: int
    last_clustering: Optional[datetime] = None
    clustering_version: int = 0
    
    def get_primary_embedding(self) -> np.ndarray:
        """Get the best representative embedding"""
        if not self.clusters:
            return np.array([])
        
        # Find cluster with highest quality and most samples
        best_cluster = max(self.clusters, 
                          key=lambda c: c.cluster_quality * np.log(c.sample_count + 1))
        return best_cluster.centroid
    
    def get_all_centroids(self) -> List[np.ndarray]:
        """Get centroids from all clusters"""
        return [cluster.centroid for cluster in self.clusters if len(cluster.centroid) > 0]
    
    def add_embedding(self, embedding: np.ndarray, confidence: float = 1.0):
        """Add embedding to the most appropriate cluster"""
        if not self.clusters:
            # Create first cluster
            cluster = SpeakerCluster(
                cluster_id=0,
                embeddings=[embedding],
                confidence_scores=[confidence],
                centroid=embedding.copy(),
                cluster_quality=1.0,
                sample_count=1
            )
            self.clusters.append(cluster)
        else:
            # Find best matching cluster
            best_cluster = None
            best_similarity = 0.0
            
            for cluster in self.clusters:
                similarity = 1 - cosine(embedding, cluster.centroid)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster
            
            # Add to best cluster if similarity is good enough, otherwise create new cluster
            if best_similarity > 0.7:  # Threshold for same cluster
                best_cluster.embeddings.append(embedding)
                best_cluster.confidence_scores.append(confidence)
                best_cluster.sample_count += 1
                best_cluster.update_centroid()
            else:
                # Create new cluster
                new_cluster_id = max(c.cluster_id for c in self.clusters) + 1
                cluster = SpeakerCluster(
                    cluster_id=new_cluster_id,
                    embeddings=[embedding],
                    confidence_scores=[confidence],
                    centroid=embedding.copy(),
                    cluster_quality=1.0,
                    sample_count=1
                )
                self.clusters.append(cluster)
        
        self.total_samples += 1
        self.last_updated = datetime.now()

class SpeakerIdentifier:
    """Speaker identifier with automatic clustering capabilities"""
    
    def __init__(self, speaker_db_path: str = "clustered_speaker_profiles.pkl",
                 auto_save_interval: int = 30,
                 clustering_interval: int = 300,  # 5 minutes
                 min_samples_for_clustering: int = 10):
        
        self.speaker_db_path = speaker_db_path
        self.auto_save_interval = auto_save_interval
        self.clustering_interval = clustering_interval
        self.min_samples_for_clustering = min_samples_for_clustering
        
        self.speaker_profiles: Dict[str, ClusteredSpeakerProfile] = {}
        self.current_speaker_id = None
        self.speaker_counter = 0
        self._save_lock = threading.Lock()
        self._cluster_lock = threading.Lock()
        self._last_save_time = 0
        self._last_clustering_time = 0
        
        # Load speaker embedding model
        if SPEECHBRAIN_AVAILABLE:
            self.speaker_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
        else:
            self.speaker_model = None
        
        # Load existing profiles
        self.load_speaker_profiles()
        
        # Start background threads
        self._auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self._auto_save_thread.start()
        
        self._clustering_thread = threading.Thread(target=self._clustering_loop, daemon=True)
        self._clustering_thread.start()
    
    def extract_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from audio (same as before)"""
        if not SPEECHBRAIN_AVAILABLE or self.speaker_model is None:
            # Fallback: simple audio features
            if len(audio_data) == 0:
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            
            energy = np.mean(audio_data ** 2)
            fft = np.fft.fft(audio_data)
            spectral_centroid = np.mean(np.abs(fft))
            zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(audio_data))))
            
            embedding = np.array([energy * 1000, spectral_centroid / 1000, zero_crossing_rate * 100, 0, 0])
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        
        try:
            # Use SpeechBrain model
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                embedding = self.speaker_model.encode_batch(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding
        except Exception:
            return np.random.normal(0, 0.01, 192) / 192
    
    def identify_speaker(self, audio_data: np.ndarray, threshold: float = 0.3,
                        update_embedding: bool = True) -> str:
        """Identify speaker using clustered embeddings"""
        embedding = self.extract_embedding(audio_data)
        
        # Find best matching speaker across all their clusters
        best_match_id = None
        best_similarity = 0.0
        
        for speaker_id, profile in self.speaker_profiles.items():
            # Check against all cluster centroids
            for centroid in profile.get_all_centroids():
                if len(centroid) > 0:
                    similarity = 1 - cosine(embedding, centroid)
                    if similarity > best_similarity and similarity > threshold:
                        best_similarity = similarity
                        best_match_id = speaker_id
        
        if best_match_id is not None:
            if update_embedding:
                self.update_speaker_embedding(best_match_id, embedding, best_similarity)
            return best_match_id
        else:
            # Create new speaker
            new_speaker_id = f"SPEAKER_{self.speaker_counter:02d}"
            self.speaker_counter += 1
            self.create_new_speaker(new_speaker_id, embedding)
            return new_speaker_id
    
    def create_new_speaker(self, speaker_id: str, embedding: np.ndarray, confidence: float = 1.0):
        """Create a new speaker profile with initial cluster"""
        profile = ClusteredSpeakerProfile(
            speaker_id=speaker_id,
            clusters=[],
            first_seen=datetime.now(),
            last_updated=datetime.now(),
            total_samples=0
        )
        profile.add_embedding(embedding, confidence)
        self.speaker_profiles[speaker_id] = profile
        print(f"Created new clustered speaker profile: {speaker_id}")
    
    def update_speaker_embedding(self, speaker_id: str, embedding: np.ndarray, confidence: float = 1.0):
        """Update speaker embedding and trigger clustering if needed"""
        if speaker_id in self.speaker_profiles:
            profile = self.speaker_profiles[speaker_id]
            profile.add_embedding(embedding, confidence)
            
            # Check if this speaker needs re-clustering
            if (profile.total_samples >= self.min_samples_for_clustering and
                profile.total_samples % 20 == 0):  # Re-cluster every 20 new samples
                self._schedule_speaker_clustering(speaker_id)
    
    def _schedule_speaker_clustering(self, speaker_id: str):
        """Schedule clustering for a specific speaker"""
        def cluster_task():
            self._cluster_speaker(speaker_id)
        
        threading.Thread(target=cluster_task, daemon=True).start()
    
    def _cluster_speaker(self, speaker_id: str):
        """Perform clustering analysis for a specific speaker"""
        if speaker_id not in self.speaker_profiles:
            return
        
        with self._cluster_lock:
            profile = self.speaker_profiles[speaker_id]
            
            # Collect all embeddings from all clusters
            all_embeddings = []
            all_confidences = []
            
            for cluster in profile.clusters:
                all_embeddings.extend(cluster.embeddings)
                all_confidences.extend(cluster.confidence_scores)
            
            if len(all_embeddings) < self.min_samples_for_clustering:
                return
            
            print(f"Clustering {len(all_embeddings)} embeddings for {speaker_id}")
            
            # Convert to numpy array
            embeddings_array = np.array(all_embeddings)
            confidences_array = np.array(all_confidences)
            
            # Try different clustering approaches
            best_clusters = self._find_optimal_clusters(embeddings_array, confidences_array)
            
            if best_clusters:
                # Update speaker profile with new clusters
                new_clusters = []
                for i, (cluster_embeddings, cluster_confidences) in enumerate(best_clusters):
                    cluster = SpeakerCluster(
                        cluster_id=i,
                        embeddings=cluster_embeddings,
                        confidence_scores=cluster_confidences,
                        centroid=np.array([]),  # Will be calculated
                        cluster_quality=0.0,
                        sample_count=len(cluster_embeddings)
                    )
                    cluster.update_centroid()
                    
                    # Calculate cluster quality (compactness)
                    if len(cluster_embeddings) > 1:
                        distances = pdist([cluster.centroid] + cluster_embeddings, metric='cosine')
                        cluster.cluster_quality = 1.0 - np.mean(distances)
                    else:
                        cluster.cluster_quality = 1.0
                    
                    new_clusters.append(cluster)
                
                profile.clusters = new_clusters
                profile.last_clustering = datetime.now()
                profile.clustering_version += 1
                
                print(f"Updated {speaker_id} with {len(new_clusters)} clusters")
    
    def _find_optimal_clusters(self, embeddings: np.ndarray, confidences: np.ndarray) -> List[Tuple[List, List]]:
        """Find optimal clustering for embeddings"""
        if len(embeddings) < 3:
            return [(embeddings.tolist(), confidences.tolist())]
        
        best_score = -1
        best_clusters = None
        
        # Try DBSCAN clustering
        try:
            # Use DBSCAN with cosine distance
            distance_matrix = squareform(pdist(embeddings, metric='cosine'))
            dbscan = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
            cluster_labels = dbscan.fit_predict(distance_matrix)
            
            # Check if clustering is good
            if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:
                score = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_clusters = self._group_by_labels(embeddings, confidences, cluster_labels)
        except Exception as e:
            print(f"DBSCAN clustering failed: {e}")
        
        # Try K-means with different k values
        for k in range(2, min(5, len(embeddings) // 2 + 1)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                # Calculate silhouette score
                score = silhouette_score(embeddings, cluster_labels, metric='cosine')
                if score > best_score:
                    best_score = score
                    best_clusters = self._group_by_labels(embeddings, confidences, cluster_labels)
            except Exception as e:
                print(f"K-means clustering (k={k}) failed: {e}")
        
        # If no good clustering found, return single cluster
        if best_clusters is None or best_score < 0.3:
            return [(embeddings.tolist(), confidences.tolist())]
        
        return best_clusters
    
    def _group_by_labels(self, embeddings: np.ndarray, confidences: np.ndarray, 
                        labels: np.ndarray) -> List[Tuple[List, List]]:
        """Group embeddings by cluster labels"""
        clusters = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points in DBSCAN
                continue
            
            mask = labels == label
            cluster_embeddings = embeddings[mask].tolist()
            cluster_confidences = confidences[mask].tolist()
            clusters.append((cluster_embeddings, cluster_confidences))
        
        return clusters
    
    def _clustering_loop(self):
        """Background thread for periodic clustering"""
        while True:
            time.sleep(self.clustering_interval)
            current_time = time.time()
            
            if current_time - self._last_clustering_time > self.clustering_interval:
                self._perform_global_clustering()
                self._last_clustering_time = current_time
    
    def _perform_global_clustering(self):
        """Perform clustering analysis on all speakers that need it"""
        speakers_to_cluster = []
        
        for speaker_id, profile in self.speaker_profiles.items():
            # Check if speaker needs clustering
            needs_clustering = (
                profile.total_samples >= self.min_samples_for_clustering and
                (profile.last_clustering is None or 
                 (datetime.now() - profile.last_clustering).seconds > self.clustering_interval * 2)
            )
            
            if needs_clustering:
                speakers_to_cluster.append(speaker_id)
        
        if speakers_to_cluster:
            print(f"Performing clustering analysis on {len(speakers_to_cluster)} speakers")
            for speaker_id in speakers_to_cluster:
                self._cluster_speaker(speaker_id)
    
    def get_clustering_stats(self) -> Dict[str, Dict]:
        """Get clustering statistics for all speakers"""
        stats = {}
        for speaker_id, profile in self.speaker_profiles.items():
            cluster_info = []
            for cluster in profile.clusters:
                cluster_info.append({
                    'cluster_id': cluster.cluster_id,
                    'sample_count': cluster.sample_count,
                    'quality': cluster.cluster_quality,
                    'avg_confidence': np.mean(cluster.confidence_scores) if cluster.confidence_scores else 0.0
                })
            
            stats[speaker_id] = {
                'total_samples': profile.total_samples,
                'num_clusters': len(profile.clusters),
                'last_clustering': profile.last_clustering,
                'clustering_version': profile.clustering_version,
                'clusters': cluster_info
            }
        
        return stats
    
    def merge_speakers(self, speaker_id1: str, speaker_id2: str, keep_id: str = None) -> str:
        """
        Merge two speaker profiles (useful for correcting misidentifications)
        
        Args:
            speaker_id1, speaker_id2: Speaker IDs to merge
            keep_id: Which ID to keep (if None, keeps speaker_id1)
        
        Returns:
            The retained speaker ID
        """
        if speaker_id1 not in self.speaker_profiles or speaker_id2 not in self.speaker_profiles:
            raise ValueError("One or both speakers not found")
        
        profile1 = self.speaker_profiles[speaker_id1]
        profile2 = self.speaker_profiles[speaker_id2]
        
        # Determine which profile to keep
        if keep_id is None:
            keep_id = speaker_id1
        elif keep_id not in [speaker_id1, speaker_id2]:
            raise ValueError("keep_id must be one of the speakers being merged")
        
        # Merge embeddings
        if keep_id == speaker_id1:
            profile1.embeddings.extend(profile2.embeddings)
            profile1.confidence_scores.extend(profile2.confidence_scores)
            profile1.total_samples += profile2.total_samples
            profile1.last_updated = max(profile1.last_updated, profile2.last_updated)
            # Remove the other profile
            del self.speaker_profiles[speaker_id2]
            return speaker_id1
        else:
            profile2.embeddings.extend(profile1.embeddings)
            profile2.confidence_scores.extend(profile1.confidence_scores)
            profile2.total_samples += profile1.total_samples
            profile2.last_updated = max(profile1.last_updated, profile2.last_updated)
            # Remove the other profile
            del self.speaker_profiles[speaker_id1]
            return speaker_id2
    
    def get_speaker_stats(self) -> Dict[str, Dict]:
        """Get statistics about all known speakers"""
        stats = {}
        for speaker_id, profile in self.speaker_profiles.items():
            stats[speaker_id] = {
                'total_samples': profile.total_samples,
                'first_seen': profile.first_seen,
                'last_updated': profile.last_updated,
                'avg_confidence': np.mean(profile.confidence_scores) if profile.confidence_scores else 0.0
            }
        return stats
    
    def save_speaker_profiles(self, filepath: str = None):
        """Save speaker profiles to pickle file"""
        if filepath is None:
            filepath = self.speaker_db_path
        
        with self._save_lock:
            try:
                # Create backup of existing file
                if os.path.exists(filepath):
                    backup_path = f"{filepath}.backup"
                    os.rename(filepath, backup_path)
                
                with open(filepath, 'wb') as f:
                    pickle.dump({
                        'speaker_profiles': self.speaker_profiles,
                        'speaker_counter': self.speaker_counter,
                        'save_timestamp': datetime.now()
                    }, f)
                
                # Remove backup if save was successful
                backup_path = f"{filepath}.backup"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                
                self._last_save_time = time.time()
                print(f"Saved {len(self.speaker_profiles)} speaker profiles to {filepath}")
                
            except Exception as e:
                print(f"Error saving speaker profiles: {e}")
                # Restore backup if save failed
                backup_path = f"{filepath}.backup"
                if os.path.exists(backup_path):
                    os.rename(backup_path, filepath)
    
    def load_speaker_profiles(self, filepath: str = None):
        """Load speaker profiles from pickle file"""
        if filepath is None:
            filepath = self.speaker_db_path
        
        if not os.path.exists(filepath):
            print(f"No existing speaker database found at {filepath}")
            return
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.speaker_profiles = data.get('speaker_profiles', {})
            self.speaker_counter = data.get('speaker_counter', 0)
            save_timestamp = data.get('save_timestamp', 'unknown')
            
            print(f"Loaded {len(self.speaker_profiles)} speaker profiles from {filepath}")
            print(f"Database last saved: {save_timestamp}")
            
            # Print speaker stats
            for speaker_id, profile in self.speaker_profiles.items():
                print(f"  {speaker_id}: {profile.total_samples} samples, "
                      f"last updated {profile.last_updated}")
            
        except Exception as e:
            print(f"Error loading speaker profiles: {e}")
            self.speaker_profiles = {}
            self.speaker_counter = 0
    
    def _schedule_save(self):
        """Schedule an asynchronous save"""
        def save_task():
            self.save_speaker_profiles()
        
        threading.Thread(target=save_task, daemon=True).start()
    
    def _auto_save_loop(self):
        """Background thread for periodic auto-saves"""
        while True:
            time.sleep(self.auto_save_interval)
            if time.time() - self._last_save_time > self.auto_save_interval:
                if self.speaker_profiles:  # Only save if we have data
                    self._schedule_save()
    
    def cleanup(self):
        """Clean up and save before shutdown"""
        self.save_speaker_profiles()

class SpeechCallback(ABC):
    """Abstract base class for speech processing callbacks"""
    
    @abstractmethod
    def on_speech_start(self, event: SpeechEvent):
        """Called when speech is detected"""
        pass
    
    @abstractmethod
    def on_speech_end(self, event: SpeechEvent):
        """Called when speech ends"""
        pass
    
    @abstractmethod
    def on_transcript_update(self, segment: TranscriptSegment):
        """Called with real-time transcript updates"""
        pass
    
    @abstractmethod
    def on_transcript_final(self, segment: TranscriptSegment):
        """Called with final transcript"""
        pass
    
    @abstractmethod
    def on_speaker_change(self, old_speaker: Optional[str], new_speaker: str):
        """Called when speaker changes"""
        pass

class AudioCapture:
    """Handles real-time audio capture"""
    
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
    
    def audio_callback(self, indata, frames, time, status):
        """Audio callback for sounddevice"""
        audio_data = indata.flatten()
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        self.audio_queue.put(audio_data.copy())
    
    def start(self):
        """Start audio capture"""
        self.is_recording = True
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype=np.float32,
            device=sd.default.device[0]  # Explicitly use input device
        )
        self.stream.start()
    
    def stop(self):
        """Stop audio capture"""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
    
    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """Get next audio chunk (non-blocking)"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

class VoiceActivityDetector:
    """Handles voice activity detection"""
    
    def __init__(self, threshold=0.3, sample_rate=16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.vad_history = deque(maxlen=5)
        
        # Load Silero VAD
        self.vad_model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
    
    def detect_speech(self, audio_chunk: np.ndarray) -> bool:
        """Detect if audio chunk contains speech"""
        audio_tensor = torch.from_numpy(audio_chunk).float()
        
        # Ensure correct length for VAD model
        if len(audio_tensor) != 512:
            if len(audio_tensor) > 512:
                audio_tensor = audio_tensor[:512]
            else:
                padded = torch.zeros(512)
                padded[:len(audio_tensor)] = audio_tensor
                audio_tensor = padded
        
        speech_prob = self.vad_model(audio_tensor, self.sample_rate)
        
        if hasattr(speech_prob, 'item'):
            speech_prob = speech_prob.item()
        elif isinstance(speech_prob, torch.Tensor):
            speech_prob = speech_prob.cpu().numpy()
            if speech_prob.ndim > 0:
                speech_prob = speech_prob[0]
        
        # Smooth VAD decisions
        has_speech_raw = speech_prob > self.threshold
        self.vad_history.append(has_speech_raw)
        
        # Majority vote for smoothing
        if len(self.vad_history) >= 3:
            recent_decisions = list(self.vad_history)[-3:]
            has_speech = sum(recent_decisions) >= 2
        else:
            has_speech = has_speech_raw
        
        return has_speech

class WordBoundaryDetector:
    """Detects word boundaries in audio streams"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.energy_history = deque(maxlen=20)  # Store last 20 energy values
        self.vad_history = deque(maxlen=10)     # Store last 10 VAD decisions
        self.silence_threshold = 0.001          # Energy threshold for silence
        self.min_silence_duration = 0.1        # Minimum silence duration for word boundary (100ms)
        self.energy_drop_threshold = 0.3       # Energy drop threshold for boundary detection
        
    def calculate_energy(self, audio_chunk: np.ndarray) -> float:
        """Calculate RMS energy of audio chunk"""
        if len(audio_chunk) == 0:
            return 0.0
        return np.sqrt(np.mean(audio_chunk ** 2))
    
    def detect_word_boundary(self, audio_chunk: np.ndarray, has_speech: bool) -> bool:
        """
        Detect if we're at a good word boundary based on energy history and VAD
        
        Args:
            audio_chunk: Current audio chunk
            has_speech: Current VAD decision
            
        Returns:
            True if this is a good word boundary
        """
        current_energy = self.calculate_energy(audio_chunk)
        self.energy_history.append(current_energy)
        self.vad_history.append(has_speech)
        
        # Need some history to make decisions
        if len(self.energy_history) < 5:
            return False
        
        # Convert to lists for easier manipulation
        energy_list = list(self.energy_history)
        vad_list = list(self.vad_history)
        
        # Strategy 1: Detect silence gaps (classic word boundary indicator)
        if self._detect_silence_gap(energy_list, vad_list):
            return True
        
        # Strategy 2: Detect significant energy drops
        if self._detect_energy_drop(energy_list):
            return True
        
        # Strategy 3: Detect VAD transitions (speech to non-speech)
        if self._detect_vad_transition(vad_list):
            return True
        
        return False
    
    def _detect_silence_gap(self, energy_list: List[float], vad_list: List[bool]) -> bool:
        """Detect silence gaps that indicate word boundaries"""
        if len(energy_list) < 5 or len(vad_list) < 5:
            return False
        
        # Look for recent silence followed by speech
        recent_energy = energy_list[-3:]
        recent_vad = vad_list[-3:]
        
        # Check if we have low energy AND no speech detection
        has_silence = all(e < self.silence_threshold for e in recent_energy)
        has_no_speech = not any(recent_vad)
        
        if has_silence and has_no_speech:
            # Check if we had speech before this silence
            if len(vad_list) >= 6:
                previous_vad = vad_list[-6:-3]
                had_previous_speech = any(previous_vad)
                return had_previous_speech
        
        return False
    
    def _detect_energy_drop(self, energy_list: List[float]) -> bool:
        """Detect significant energy drops that might indicate word boundaries"""
        if len(energy_list) < 5:
            return False
        
        # Calculate moving averages
        recent_avg = np.mean(energy_list[-3:])
        previous_avg = np.mean(energy_list[-6:-3])
        
        # Detect significant drop
        if previous_avg > 0:
            energy_drop_ratio = (previous_avg - recent_avg) / previous_avg
            return energy_drop_ratio > self.energy_drop_threshold
        
        return False
    
    def _detect_vad_transition(self, vad_list: List[bool]) -> bool:
        """Detect VAD transitions that indicate word boundaries"""
        if len(vad_list) < 4:
            return False
        
        # Look for speech-to-silence transitions
        # Pattern: [speech, speech, no_speech, no_speech] or similar
        recent_pattern = vad_list[-4:]
        
        # Check for transition from speech to silence
        if recent_pattern[:2] == [True, True] and recent_pattern[2:] == [False, False]:
            return True
        
        # Check for transition from silence to speech (end of pause)
        if recent_pattern[:2] == [False, False] and recent_pattern[2:] == [True, True]:
            return True
        
        return False

class SpeechTranscriber:
    """Handles speech transcription using Whisper"""
    
    def __init__(self, model_name="openai/whisper-small"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        self.sample_rate = 16000
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio data to text"""
        if len(audio_data) == 0:
            return ""
        
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        inputs = self.processor(
            audio_data,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones(inputs["input_features"].shape[:-1], dtype=torch.long)
        
        with torch.no_grad():
            predicted_ids = self.model.generate(
                inputs["input_features"],
                attention_mask=inputs.get("attention_mask"),
                language="en",
                task="transcribe",
                max_length=448,
                do_sample=False
            )
        
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription.strip()


class VoiceAssistantSpeechProcessor:
    """Main speech processing system for voice assistants"""
    
    def __init__(self, callback: SpeechCallback):
        self.callback = callback
        
        # Initialize components
        self.audio_capture = AudioCapture()
        self.vad = VoiceActivityDetector()
        self.transcriber = SpeechTranscriber()
        self.speaker_id = SpeakerIdentifier()
        self.word_boundary_detector = WordBoundaryDetector()
        
        # State tracking
        self.is_speaking = False
        self.current_speaker = None
        self.speech_buffer = deque()
        self.silence_start = None
        self.speech_start_time = None
        
        # Real-time transcription
        self.realtime_buffer = deque()
        self.last_realtime_transcript = ""
        self.last_realtime_time = 0
        self.realtime_transcript_thread = None
        
        # Configuration
        self.silence_duration_threshold = 1.5
        self.min_speech_duration = 0.8
        self.max_speech_duration = 10.0
        
        # Enhanced real-time configuration - use min max to transcribe in real time and try dynamic word breaks in between that range
        self.realtime_update_interval = 0.5      # Minimum update interval (500ms)
        self.realtime_max_interval = 2.5         # Maximum update interval (5 * minimum)
        self.realtime_min_duration = 1.0         # Minimum audio for real-time transcription
        
        # Processing thread
        self.processing_thread = None
        self.should_stop = False
    
    def start(self):
        """Start the speech processing system"""
        self.should_stop = False
        self.audio_capture.start()
        self.processing_thread = threading.Thread(target=self._process_audio, daemon=True)
        self.processing_thread.start()
        
        # Start real-time transcription thread
        self.realtime_transcript_thread = threading.Thread(target=self._realtime_transcription_loop, daemon=True)
        self.realtime_transcript_thread.start()
    
    def stop(self, force_kill=False, timeout=1.0):
        """
        Stop the speech processing system
        
        Args:
            force_kill (bool): If True, forcefully terminate threads that don't stop gracefully
            timeout (float): How long to wait for graceful shutdown before force killing
        """
        print("Stopping speech processor...")
        
        # Set stop flag
        self.should_stop = True
        
        # Stop audio capture immediately
        try:
            self.audio_capture.stop()
        except Exception as e:
            print(f"Error stopping audio capture: {e}")
        
        # Clean up speaker identifier (saves profiles)
        try:
            self.speaker_id.cleanup()
        except Exception as e:
            print(f"Error cleaning up speaker identifier: {e}")
        
        # List of threads to manage
        threads_to_stop = []
        if self.processing_thread and self.processing_thread.is_alive():
            threads_to_stop.append(("processing_thread", self.processing_thread))
        if self.realtime_transcript_thread and self.realtime_transcript_thread.is_alive():
            threads_to_stop.append(("realtime_transcript_thread", self.realtime_transcript_thread))
        
        # Also check speaker identifier threads
        if hasattr(self.speaker_id, '_auto_save_thread') and self.speaker_id._auto_save_thread.is_alive():
            threads_to_stop.append(("speaker_auto_save_thread", self.speaker_id._auto_save_thread))
        if hasattr(self.speaker_id, '_clustering_thread') and self.speaker_id._clustering_thread.is_alive():
            threads_to_stop.append(("speaker_clustering_thread", self.speaker_id._clustering_thread))
        
        if not threads_to_stop:
            print("No active threads to stop.")
            return
        
        print(f"Stopping {len(threads_to_stop)} threads...")
        
        # First attempt: Graceful shutdown
        for thread_name, thread in threads_to_stop:
            print(f"Waiting for {thread_name} to stop gracefully...")
            thread.join(timeout=timeout)
        
        # Check which threads are still alive
        still_alive = [(name, thread) for name, thread in threads_to_stop if thread.is_alive()]
        
        if not still_alive:
            print("All threads stopped gracefully.")
            return
        
        # If force_kill is requested or any threads are still alive
        if force_kill or still_alive:
            print(f"Force killing {len(still_alive)} remaining threads...")
            
            for thread_name, thread in still_alive:
                try:
                    self._force_kill_thread(thread, thread_name)
                except Exception as e:
                    print(f"Error force killing {thread_name}: {e}")
        
        print("Speech processor stopped.")
    
    def _force_kill_thread(self, thread, thread_name="unknown"):
        """
        Force kill a thread using ctypes (platform dependent)
        WARNING: This is potentially dangerous and should be used as last resort
        """
        if not thread.is_alive():
            return
        
        print(f"Force killing thread: {thread_name}")
        
        try:
            # Get thread ID
            thread_id = thread.ident
            if thread_id is None:
                print(f"Could not get thread ID for {thread_name}")
                return
            
            # Platform-specific thread termination
            if sys.platform == "win32":
                # Windows
                import ctypes.wintypes
                kernel32 = ctypes.windll.kernel32
                
                # Open thread handle
                THREAD_TERMINATE = 0x0001
                thread_handle = kernel32.OpenThread(THREAD_TERMINATE, False, thread_id)
                
                if thread_handle:
                    # Terminate thread
                    kernel32.TerminateThread(thread_handle, 0)
                    kernel32.CloseHandle(thread_handle)
                    print(f"Thread {thread_name} terminated (Windows)")
                else:
                    print(f"Could not open thread handle for {thread_name}")
            
            elif sys.platform.startswith("linux") or sys.platform == "darwin":
                # Linux/macOS - use pthread_cancel
                import ctypes.util
                
                # Load pthread library
                pthread_lib_name = ctypes.util.find_library("pthread")
                if pthread_lib_name:
                    pthread = ctypes.CDLL(pthread_lib_name)
                    
                    # Cancel thread
                    result = pthread.pthread_cancel(ctypes.c_ulong(thread_id))
                    if result == 0:
                        print(f"Thread {thread_name} cancelled (Unix)")
                    else:
                        print(f"Failed to cancel thread {thread_name}: {result}")
                else:
                    print("Could not find pthread library")
            
            else:
                print(f"Unsupported platform for force killing: {sys.platform}")
        
        except Exception as e:
            print(f"Error in force kill for {thread_name}: {e}")
        
        # Give a moment for cleanup
        time.sleep(0.1)

    def _process_audio(self):
        """Main audio processing loop"""
        audio_buffer = deque(maxlen=int(16000 * 0.5))  # 0.5 second buffer
        
        while not self.should_stop:
            try:
                # Get audio chunk
                chunk = self.audio_capture.get_audio_chunk()
                if chunk is None:
                    time.sleep(0.01)
                    continue
                
                audio_buffer.extend(chunk)
                
                # Check for speech
                if len(audio_buffer) >= 512:
                    recent_audio = np.array(list(audio_buffer)[-512:])
                    has_speech = self.vad.detect_speech(recent_audio)
                    
                    current_time = time.time()
                    
                    if has_speech:
                        if not self.is_speaking:
                            # Speech started
                            self.is_speaking = True
                            self.speech_start_time = current_time
                            self.speech_buffer.clear()
                            self.realtime_buffer.clear()
                            self.last_realtime_transcript = ""
                            
                            # Include some pre-speech audio
                            self.speech_buffer.extend(list(audio_buffer))
                            self.realtime_buffer.extend(list(audio_buffer))
                            
                            self.callback.on_speech_start(SpeechEvent(
                                event_type='speech_start',
                                timestamp=current_time,
                                audio_data=recent_audio
                            ))
                        else:
                            # Continue speech
                            self.speech_buffer.extend(chunk)
                            self.realtime_buffer.extend(chunk)
                        
                        self.silence_start = None
                    
                    else:
                        # No speech detected
                        if self.is_speaking:
                            if self.silence_start is None:
                                self.silence_start = current_time
                            
                            silence_duration = current_time - self.silence_start
                            
                            if silence_duration >= self.silence_duration_threshold:
                                # Speech ended
                                self._finalize_speech_segment()
                        
                        if self.speech_buffer:
                            self.speech_buffer.extend(chunk)
                            self.realtime_buffer.extend(chunk)
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error in audio processing: {e}")
                time.sleep(0.1)
    
    def _realtime_transcription_loop(self):
        """Enhanced background thread for real-time transcription with word boundary detection"""
        while not self.should_stop:
            try:
                current_time = time.time()
                time_since_last_update = current_time - self.last_realtime_time
                
                # Only process if we're currently speaking
                if self.is_speaking and self.realtime_buffer:
                    
                    # Get current audio buffer
                    current_audio = np.array(list(self.realtime_buffer))
                    audio_duration = len(current_audio) / 16000
                    
                    # Check if we have enough audio for processing
                    if audio_duration >= self.realtime_min_duration:
                        
                        # Determine if we should process now
                        should_process = False
                        
                        # Always process if we've hit the maximum interval
                        if time_since_last_update >= self.realtime_max_interval:
                            should_process = True
                            print(f"🕐 Processing due to max interval ({self.realtime_max_interval}s)")
                        
                        # Process if we've hit minimum interval AND found a word boundary
                        elif time_since_last_update >= self.realtime_update_interval:
                            
                            # Check for word boundary using recent audio
                            if len(current_audio) >= 1024:  # Need enough audio for boundary detection
                                recent_chunk = current_audio[-1024:]
                                has_speech = self.vad.detect_speech(recent_chunk[-512:])
                                
                                is_word_boundary = self.word_boundary_detector.detect_word_boundary(
                                    recent_chunk, has_speech
                                )
                                
                                if is_word_boundary:
                                    should_process = True
                                    print(f"🎯 Processing due to word boundary detected")
                        
                        # Process the audio if conditions are met
                        if should_process:
                            try:
                                # Get current speaker (use cached if available)
                                if self.current_speaker is None and len(current_audio) > 8000:  # ~0.5 seconds
                                    speaker_id = self.speaker_id.identify_speaker(current_audio)
                                    if speaker_id != self.current_speaker:
                                        old_speaker = self.current_speaker
                                        self.current_speaker = speaker_id
                                        self.callback.on_speaker_change(old_speaker, speaker_id)
                                else:
                                    speaker_id = self.current_speaker or "UNKNOWN"
                                
                                # Transcribe current buffer
                                transcript = self.transcriber.transcribe(current_audio)
                                
                                # Only send update if transcript changed significantly
                                if (transcript.strip() and 
                                    transcript != self.last_realtime_transcript and
                                    len(transcript.strip()) > 2):
                                    
                                    segment = TranscriptSegment(
                                        text=transcript,
                                        speaker_id=speaker_id,
                                        start_time=self.speech_start_time or current_time,
                                        end_time=current_time,
                                        confidence=0.8,  # Lower confidence for real-time
                                        is_final=False
                                    )
                                    
                                    self.callback.on_transcript_update(segment)
                                    self.last_realtime_transcript = transcript
                                    
                                    print(f"📱 Real-time update: {len(current_audio)/16000:.1f}s audio, "
                                          f"interval: {time_since_last_update:.1f}s")
                                
                                self.last_realtime_time = current_time
                                
                            except Exception as e:
                                print(f"Error in real-time transcription: {e}")
                
                time.sleep(0.05)  # Check every 50ms for more responsive boundary detection
                
            except Exception as e:
                print(f"Error in real-time transcription loop: {e}")
                time.sleep(0.5)
    
    def _finalize_speech_segment(self):
        """Process and finalize a speech segment"""
        if not self.speech_buffer or not self.speech_start_time:
            return
        
        speech_audio = np.array(list(self.speech_buffer))
        speech_duration = len(speech_audio) / 16000
        
        # Check minimum duration
        if speech_duration < self.min_speech_duration:
            self._reset_speech_state()
            return
        
        try:
            # Identify speaker (use full audio for final identification)
            speaker_id = self.speaker_id.identify_speaker(speech_audio)
            
            # Check for speaker change
            if speaker_id != self.current_speaker:
                old_speaker = self.current_speaker
                self.current_speaker = speaker_id
                self.callback.on_speaker_change(old_speaker, speaker_id)
            
            # Final transcription (usually more accurate than real-time)
            transcript = self.transcriber.transcribe(speech_audio)
            
            if transcript.strip():
                # Create final transcript segment
                segment = TranscriptSegment(
                    text=transcript,
                    speaker_id=speaker_id,
                    start_time=self.speech_start_time,
                    end_time=time.time(),
                    confidence=1.0,  # Higher confidence for final transcription
                    is_final=True
                )
                
                self.callback.on_transcript_final(segment)
            
            # Send speech end event
            self.callback.on_speech_end(SpeechEvent(
                event_type='speech_end',
                timestamp=time.time(),
                audio_data=speech_audio
            ))
        
        except Exception as e:
            print(f"Error processing speech segment: {e}")
        
        finally:
            self._reset_speech_state()
    
    def _reset_speech_state(self):
        """Reset speech processing state"""
        self.is_speaking = False
        self.speech_buffer.clear()
        self.realtime_buffer.clear()
        self.silence_start = None
        self.speech_start_time = None
        self.last_realtime_transcript = ""
        self.last_realtime_time = 0
        # Reset word boundary detector state
        self.word_boundary_detector.energy_history.clear()
        self.word_boundary_detector.vad_history.clear()

# Example usage for voice assistant
class VoiceAssistantCallback(SpeechCallback):
    """Example callback implementation for a voice assistant"""
    
    def __init__(self):
        self.current_conversation = []
        self.active_speakers = set()
    
    def on_speech_start(self, event: SpeechEvent):
        print(f"🎤 Speech detected at {event.timestamp:.1f}")
    
    def on_speech_end(self, event: SpeechEvent):
        print(f"🔇 Speech ended at {event.timestamp:.1f}")
    
    def on_transcript_update(self, segment: TranscriptSegment):
        print(f"📝 [{segment.speaker_id}] (live): {segment.text}")
    
    def on_transcript_final(self, segment: TranscriptSegment):
        print(f"✅ [{segment.speaker_id}] (final): {segment.text}")
        self.current_conversation.append(segment)
        self.active_speakers.add(segment.speaker_id)
    
    def on_speaker_change(self, old_speaker: Optional[str], new_speaker: str):
        if old_speaker is not None:
            print(f"👥 Speaker change: {old_speaker} → {new_speaker}")
        else:
            print(f"👤 New speaker: {new_speaker}")
    
    def get_conversation_history(self) -> List[TranscriptSegment]:
        """Get full conversation history"""
        return self.current_conversation.copy()
    
    def get_recent_transcript(self, seconds: float = 30.0) -> str:
        """Get recent transcript as formatted string"""
        cutoff_time = time.time() - seconds
        recent_segments = [s for s in self.current_conversation if s.start_time >= cutoff_time]
        
        result = []
        for segment in recent_segments:
            result.append(f"[{segment.speaker_id}]: {segment.text}")
        
        return "\n".join(result)

if __name__ == "__main__":
    # Example usage
    callback = VoiceAssistantCallback()
    processor = VoiceAssistantSpeechProcessor(callback)
    
    print("Starting enhanced voice assistant speech processor...")
    print("Features: Dynamic word boundary detection, adaptive real-time transcription")
    print("Speak into the microphone. Press Ctrl+C to stop.")
    
    try:
        processor.start()
        
        # Main loop - your voice assistant logic would go here
        while True:
            time.sleep(1)
            
            # Example: Get recent conversation every 10 seconds
            recent = callback.get_recent_transcript(10.0)
            if recent:
                print(f"\n--- Recent conversation ---\n{recent}\n")
    
    except KeyboardInterrupt:
        print("\nStopping...")
        processor.stop()