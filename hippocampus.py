"""
Core Architecture of Hippocampus Memory System - Fixed for Single CUDA Device
Fully modular design with all methods contained within the class.
"""

import os
import logging
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Manager
import queue  # Import for Empty exception
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import uuid
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ModalityType(Enum):
    """Enumeration for different modality types."""
    TEXT = "text"
    IMAGE = "image" 
    AUDIO = "audio"
    SCENE = "scene"
    NOUNS = "nouns"

@dataclass
class MemoryItem:
    """Data class for memory items with metadata."""
    id: str
    content: Any
    modality: ModalityType
    timestamp: float
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class Hippocampus:
    """
    Unified memory consolidation system integrating multimodal embeddings, storage and search.
    full modularity.
    """
    
    def __init__(self, 
                 data_directory: str = "./hippocampus_data",
                 enable_multiprocessing: bool = True,
                 max_workers: int = None,
                 cuda_device: Optional[int] = None,  # None = CPU, 0 or 1 = specific CUDA
                 batch_size: int = 8):
        """
        Initialize the Hippocampus memory system.
        
        Args:
            data_directory: Directory for storing memory data
            enable_multiprocessing: Whether to use multiprocessing
            max_workers: Maximum number of worker processes
            cuda_device: CUDA device to use (None for CPU, 0 or 1 for specific GPU)
            batch_size: Batch size for processing
        """
        self.data_directory = data_directory
        self.enable_multiprocessing = enable_multiprocessing
        self.max_workers = max_workers or min(mp.cpu_count(), 4)
        self.cuda_device = str(cuda_device)
        self.batch_size = batch_size
        
        self._check_and_install_dependencies()
        # Set CUDA device BEFORE any imports or initializations
        self._configure_cuda()
        
        # Initialize components
        self.memory_store = None
        self.embedders = {}
        self.processing_queue = None
        self.result_queue = None
        self.stop_event = None
        self.worker_processes = []
        self.torch = None
        
        # Thread safety
        self.lock = threading.Lock()
        self.manager = Manager() if enable_multiprocessing else None
        
        # Performance tracking
        self.stats = {
            'items_processed': 0,
            'embeddings_generated': 0,
            'searches_performed': 0,
            'avg_embedding_time': 0.0,
            'avg_search_time': 0.0
        }
        
        # Initialize system
        self._setup_system()
        
        device_str = f"CUDA:{cuda_device}" if cuda_device is not None else "CPU"
        logging.info(f"Hippocampus initialized with {self.max_workers} workers on {device_str}")
        logging.info(f"Multiprocessing: {enable_multiprocessing}")

    def _check_and_install_dependencies(self) -> None:
        """
        Check if required dependencies are installed and install if missing.
        Stores imported modules as instance attributes.
        """
        import subprocess
        import sys
        # Check and install PyTorch
        try:
            import torch
            self.torch = torch
           
        except ImportError:
            logging.debug("PyTorch not found. Installing torch...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
            try:
                import torch
                self.torch = torch
                logging.debug("PyTorch installed successfully")
            except ImportError:
                raise ImportError("WARNING! Failed to install or import PyTorch")
            
    def _configure_cuda(self):
        """Configure CUDA device for the entire process."""
        import os
        if self.cuda_device is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cuda_device)
            logging.info(f"Set CUDA_VISIBLE_DEVICES to {self.cuda_device}")
        else:
            # Check if CUDA is available
            if self.torch.cuda.is_available():
                # Use the first available GPU
                os.environ['CUDA_VISIBLE_DEVICES'] = "0"
                logging.info("CUDA is available. Using GPU device 0")
            else:
                # Use CPU
                os.environ['CUDA_VISIBLE_DEVICES'] = ""
                logging.info("CUDA not available. Using CPU")

    def _setup_system(self):
        """Set up the complete memory system."""
        # Create data directory
        os.makedirs(self.data_directory, exist_ok=True)
        
        # Initialize memory store
        database_config = {
            'conversations': 384,
            'images': 1024,
            'audio': 128,
            'scene': 384,
            'nouns': 384
        }
        
        self.memory_store = self._initialize_memory_store(database_config)
        
        # Initialize embedders only for synchronous processing or main process
        if not self.enable_multiprocessing:
            self._initialize_embedders()
        
        # Set up multiprocessing if enabled
        if self.enable_multiprocessing:
            self._setup_multiprocessing()

    def _initialize_memory_store(self, database_config: Dict[str, int]):
        """Initialize the MemRecall storage system."""
        try:
            from faiss_db import MemRecall
            return MemRecall(
                data_directory=self.data_directory,
                databases=database_config
            )
        except ImportError:
            logging.error("MemRecall class not found. Please ensure it's importable.")
            raise

    def _initialize_embedders(self):
        """Initialize all embedding models."""
        try:
            from vggish_embed import VGGishEmbedder
            from beit_embed import BeITEmbedder
            from mxbai_embed import MxBaiEmbedder
            
            logging.debug("Initializing embedders...")
            
            self.embedders[ModalityType.TEXT] = MxBaiEmbedder(cuda_device = self.cuda_device)
            self.embedders[ModalityType.SCENE] = self.embedders[ModalityType.TEXT]
            self.embedders[ModalityType.NOUNS] = self.embedders[ModalityType.TEXT]
            self.embedders[ModalityType.IMAGE] = BeITEmbedder(cuda_device = self.cuda_device)
            self.embedders[ModalityType.AUDIO] = VGGishEmbedder()#will only use CPU at the moment
            
            logging.debug("All embedders initialized successfully")
            
        except ImportError as e:
            logging.error(f"Failed to import embedder classes: {e}")
            raise

    @staticmethod
    def _init_embedders_for_worker(worker_id: int):
        """
        Initialize embedders for a worker process.
        Static method to avoid pickling issues.
        """
        cuda_device = str(os.environ['CUDA_VISIBLE_DEVICES'])
        logging.debug(f"Worker {worker_id} initializing embedders")
        
        worker_embedders = {}
        try:
            from vggish_embed import VGGishEmbedder
            from beit_embed import BeITEmbedder
            from mxbai_embed import MxBaiEmbedder
            
            worker_embedders[ModalityType.TEXT] = MxBaiEmbedder(cuda_device=cuda_device)
            worker_embedders[ModalityType.IMAGE] = BeITEmbedder(cuda_device=cuda_device)
            worker_embedders[ModalityType.AUDIO] = VGGishEmbedder()
            worker_embedders[ModalityType.SCENE] = worker_embedders[ModalityType.TEXT]
            worker_embedders[ModalityType.NOUNS] = worker_embedders[ModalityType.TEXT]
            
            logging.info(f"Worker {worker_id} embedders initialized successfully")
            return worker_embedders
            
        except Exception as e:
            logging.error(f"Worker {worker_id} failed to initialize embedders: {e}")
            return {}

    @staticmethod
    def _process_embedding_task(embedders: Dict, task_data: Dict) -> Dict:
        """Process a single embedding task. Static method for multiprocessing."""
        start_time = time.time()
        
        try:
            memory_item = task_data['memory_item']
            modality = memory_item.modality
            
            if modality not in embedders:
                return {
                    'success': False,
                    'error': f"No embedder for modality {modality}",
                    'memory_item': memory_item
                }
            
            embedder = embedders[modality]
            
            if modality == ModalityType.TEXT or modality == ModalityType.SCENE or modality == ModalityType.NOUNS:
                embedding = embedder.embed_string(memory_item.content)
            elif modality == ModalityType.IMAGE:
                embedding = embedder.embed_image(memory_item.content)
            elif modality == ModalityType.AUDIO:
                if isinstance(memory_item.content, str):
                    embedding_array = embedder.embed_wav_file(memory_item.content)
                    embedding = embedding_array.mean(axis=0).tolist()
                else:
                    embedding_array = embedder.embed_audio_data(memory_item.content)
                    embedding = embedding_array.mean(axis=0).tolist()
            else:
                raise ValueError(f"Unsupported modality: {modality}")
            
            memory_item.embedding = embedding
            
            return {
                'success': True,
                'memory_item': memory_item,
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'memory_item': task_data['memory_item']
            }

    @staticmethod
    def _process_batch_embedding_task(embedders: Dict, task_data: Dict) -> Dict:
        """Process a batch embedding task. Static method for multiprocessing."""
        start_time = time.time()
        
        try:
            memory_items = task_data['memory_items']
            modality = task_data['modality']
            
            if modality not in embedders:
                return {
                    'success': False,
                    'error': f"No embedder for modality {modality}",
                    'memory_items': memory_items
                }
            
            embedder = embedders[modality]
            
            if modality in [ModalityType.TEXT, ModalityType.SCENE, ModalityType.NOUNS]:
                contents = [item.content for item in memory_items]
                embeddings = embedder.embed_batch(contents)
            elif modality == ModalityType.IMAGE:
                contents = [item.content for item in memory_items]
                embeddings = embedder.embed_batch(contents)
            else:  # AUDIO
                embeddings = []
                for item in memory_items:
                    if isinstance(item.content, str):
                        emb_array = embedder.embed_wav_file(item.content)
                        embeddings.append(emb_array.mean(axis=0).tolist())
                    else:
                        emb_array = embedder.embed_audio_data(item.content)
                        embeddings.append(emb_array.mean(axis=0).tolist())
            
            for item, embedding in zip(memory_items, embeddings):
                item.embedding = embedding
            
            return {
                'success': True,
                'memory_items': memory_items,
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'memory_items': task_data['memory_items']
            }

    @classmethod
    def _worker_process_function(cls, worker_id: int, task_queue: Queue, result_queue: Queue, 
                                stop_event: Event):
        """
        Worker process function. Class method to maintain access to static methods.
        """
        # Initialize embedders within this worker process
        embedders = cls._init_embedders_for_worker(worker_id)
        
        if not embedders:
            logging.error(f"Worker {worker_id} failed to initialize, exiting")
            return
        
        # Process tasks until stop event is set
        while not stop_event.is_set():
            try:
                # Get task with timeout
                task = task_queue.get(timeout=1.0)
                
                if task is None:  # Poison pill
                    break
                
                task_type, task_data = task
                
                if task_type == 'embed':
                    result = cls._process_embedding_task(embedders, task_data)
                    result_queue.put(('embed_result', result))
                
                elif task_type == 'batch_embed':
                    result = cls._process_batch_embedding_task(embedders, task_data)
                    result_queue.put(('batch_embed_result', result))
                
            except queue.Empty:
                # This is normal - just timeout waiting for tasks
                continue
            except Exception as e:
                if not stop_event.is_set():
                    logging.error(f"Worker {worker_id} error processing task: {e}")
                    import traceback
                    logging.debug(f"Worker {worker_id} traceback: {traceback.format_exc()}")
                continue
        
        logging.info(f"Worker {worker_id} shutting down")

    def _setup_multiprocessing(self):
        """Set up multiprocessing queues and events."""
        # Ensure we're using spawn method on Windows
        if mp.get_start_method() != 'spawn':
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # Already set
        
        self.processing_queue = Queue()
        self.result_queue = Queue()
        self.stop_event = Event()
        
        # Start worker processes
        for i in range(self.max_workers):
            worker = Process(
                target=self._worker_process_function,
                args=(i, self.processing_queue, self.result_queue, self.stop_event)
            )
            worker.start()
            self.worker_processes.append(worker)
            
        # Start result collector thread
        self.result_thread = threading.Thread(target=self._collect_results)
        self.result_thread.daemon = True
        self.result_thread.start()

    def _collect_results(self):
        """Collect results from worker processes."""
        while not self.stop_event.is_set():
            try:
                result_type, result_data = self.result_queue.get(timeout=1.0)
                
                if result_type in ['embed_result', 'batch_embed_result']:
                    self._handle_embedding_result(result_data)
                
            except queue.Empty:
                # This is normal - just timeout waiting for results
                continue
            except Exception as e:
                if not self.stop_event.is_set():
                    logging.error(f"Result collector error: {e}")
                continue

    def _handle_embedding_result(self, result_data: Dict):
        """Handle embedding results and store in memory."""
        if result_data['success']:
            if 'memory_item' in result_data:
                self._store_memory_item(result_data['memory_item'])
            elif 'memory_items' in result_data:
                for item in result_data['memory_items']:
                    self._store_memory_item(item)
            
            # Update stats
            with self.lock:
                self.stats['embeddings_generated'] += 1
                self.stats['avg_embedding_time'] = (
                    self.stats['avg_embedding_time'] * (self.stats['embeddings_generated'] - 1) +
                    result_data['processing_time']
                ) / self.stats['embeddings_generated']
        else:
            logging.error(f"Embedding failed: {result_data['error']}")

    def _store_memory_item(self, memory_item: MemoryItem):
        """Store a memory item in the appropriate database."""
        try:
            db_mapping = {
                ModalityType.TEXT: 'conversations',
                ModalityType.IMAGE: 'images', 
                ModalityType.AUDIO: 'audio',
                ModalityType.SCENE: 'scene',
                ModalityType.NOUNS: 'nouns'
            }
            
            db_name = db_mapping.get(memory_item.modality)
            if db_name and memory_item.embedding:
                import numpy as np
                vector = np.array(memory_item.embedding)
                
                success = self.memory_store.add_to_database(
                    database_name=db_name,
                    item_id=memory_item.id,
                    vector=vector,
                    metadata={
                        'content': memory_item.content,
                        'timestamp': memory_item.timestamp,
                        'modality': memory_item.modality.value,
                        **memory_item.metadata
                    }
                )
                
                if success:
                    with self.lock:
                        self.stats['items_processed'] += 1
                else:
                    logging.error(f"Failed to store memory item {memory_item.id}")
            
        except Exception as e:
            logging.error(f"Error storing memory item: {e}")

    def add_memory(self, 
                   content: Any,
                   modality: ModalityType,
                   metadata: Optional[Dict[str, Any]] = None,
                   async_process: bool = True) -> str:
        """Add a new memory to the system."""
        memory_id = str(uuid.uuid4())
        memory_item = MemoryItem(
            id=memory_id,
            content=content,
            modality=modality,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        if self.enable_multiprocessing and async_process:
            task = ('embed', {'memory_item': memory_item})
            self.processing_queue.put(task)
        else:
            self._process_memory_sync(memory_item)
        
        logging.debug(f"Added memory {memory_id} ({modality.value})")
        return memory_id

    def add_memories_batch(self,
                          items: List[Tuple[Any, ModalityType, Optional[Dict[str, Any]]]],
                          async_process: bool = True) -> List[str]:
        """Add multiple memories in batch."""
        memory_items = []
        memory_ids = []
        modality_groups = {}
        
        for content, modality, metadata in items:
            memory_id = str(uuid.uuid4())
            memory_item = MemoryItem(
                id=memory_id,
                content=content,
                modality=modality,
                timestamp=time.time(),
                metadata=metadata or {}
            )
            
            memory_items.append(memory_item)
            memory_ids.append(memory_id)
            
            if modality not in modality_groups:
                modality_groups[modality] = []
            modality_groups[modality].append(memory_item)
        
        if self.enable_multiprocessing and async_process:
            for modality, items_group in modality_groups.items():
                task = ('batch_embed', {
                    'memory_items': items_group,
                    'modality': modality
                })
                self.processing_queue.put(task)
        else:
            for memory_item in memory_items:
                self._process_memory_sync(memory_item)
        
        logging.debug(f"Added {len(memory_items)} memories in batch")
        return memory_ids

    def _process_memory_sync(self, memory_item: MemoryItem):
        """Process a memory item synchronously."""
        # Ensure embedders are initialized for sync processing
        if not self.embedders:
            self._initialize_embedders()
            
        start_time = time.time()
        
        try:
            modality = memory_item.modality
            embedder = self.embedders.get(modality)
            
            if not embedder:
                logging.error(f"No embedder available for modality {modality}")
                return
            
            if modality in [ModalityType.TEXT, ModalityType.SCENE, ModalityType.NOUNS]:
                embedding = embedder.embed_string(memory_item.content)
            elif modality == ModalityType.IMAGE:
                embedding = embedder.embed_image(memory_item.content)
            elif modality == ModalityType.AUDIO:
                if isinstance(memory_item.content, str):
                    embedding_array = embedder.embed_wav_file(memory_item.content)
                    embedding = embedding_array.mean(axis=0).tolist()
                else:
                    embedding_array = embedder.embed_audio_data(memory_item.content)
                    embedding = embedding_array.mean(axis=0).tolist()
            else:
                logging.error(f"Unsupported modality: {modality}")
                return
            
            memory_item.embedding = embedding
            self._store_memory_item(memory_item)
            
            processing_time = time.time() - start_time
            with self.lock:
                self.stats['embeddings_generated'] += 1
                self.stats['avg_embedding_time'] = (
                    self.stats['avg_embedding_time'] * (self.stats['embeddings_generated'] - 1) +
                    processing_time
                ) / self.stats['embeddings_generated']
            
        except Exception as e:
            logging.error(f"Error processing memory synchronously: {e}")

    def search_memory(self,
                     query: Any,
                     modality: ModalityType,
                     top_k: int = 5,
                     database_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar memories."""
        start_time = time.time()
        logging.debug(query)
        logging.debug(modality)
        logging.debug(top_k)
        logging.debug(database_name)
        
        try:
            # For searching, we need embedders in the main process
            if not self.embedders:
                self._initialize_embedders()
            
            embedder = self.embedders.get(modality)
            if not embedder:
                logging.error(f"No embedder available for modality {modality}")
                return []
            
            if modality in [ModalityType.TEXT, ModalityType.SCENE, ModalityType.NOUNS]:
                query_embedding = embedder.embed_string(query)
            elif modality == ModalityType.IMAGE:
                query_embedding = embedder.embed_image(query)
            elif modality == ModalityType.AUDIO:
                if isinstance(query, str):
                    embedding_array = embedder.embed_wav_file(query)
                    query_embedding = embedding_array.mean(axis=0).tolist()
                else:
                    embedding_array = embedder.embed_audio_data(query)
                    query_embedding = embedding_array.mean(axis=0).tolist()
            else:
                logging.error(f"Unsupported modality: {modality}")
                return []
            
            if not database_name:
                db_mapping = {
                    ModalityType.TEXT: 'conversations',
                    ModalityType.IMAGE: 'images',
                    ModalityType.AUDIO: 'audio', 
                    ModalityType.SCENE: 'scene',
                    ModalityType.NOUNS: 'nouns'
                }
                database_name = db_mapping.get(modality, 'conversations')
            
            import numpy as np
            query_vector = np.array(query_embedding)
            print("="*50)
            results = self.memory_store.similarity_search(
                query_vector=query_vector,
                database_name=database_name,
                top_k=top_k
            )
            logging.debug(results)
            search_time = time.time() - start_time
            with self.lock:
                self.stats['searches_performed'] += 1
                self.stats['avg_search_time'] = (
                    self.stats['avg_search_time'] * (self.stats['searches_performed'] - 1) +
                    search_time
                ) / self.stats['searches_performed']
            
            logging.info(f"Search completed in {search_time:.3f}s, found {len(results)} results")
            return results
            
        except Exception as e:
            logging.error(f"Search error: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        with self.lock:
            stats = self.stats.copy()
        
        try:
            for db_name in self.memory_store.list_databases():
                db_info = self.memory_store.get_database_info(db_name)
                stats[f'{db_name}_items'] = db_info.get('total_items', 0)
        except Exception as e:
            logging.error(f"Error getting database stats: {e}")
        
        return stats

    def save_all_memories(self) -> bool:
        """Save all memories to persistent storage."""
        try:
            result = self.memory_store.save_all_databases()
            
            # The result from save_all_databases appears to be a list of success messages
            # Check if we got any actual failures
            if result is None:
                logging.warning("Save returned None - assuming success")
                return True
            elif isinstance(result, bool):
                success = result
            elif isinstance(result, dict):
                # Check if all values are truthy
                success = all(result.values()) if result else True
            elif isinstance(result, list):
                # If it's a list of messages, check if any contain "failed"
                success = not any("failed" in str(item).lower() for item in result)
            else:
                # Unknown format, log it and assume success if not explicitly failed
                logging.debug(f"Save result format: {type(result)} - {result}")
                success = True
            
            if success:
                logging.debug("All memories saved successfully")
            else:
                logging.error("Some memories failed to save")
            
            return success
            
        except Exception as e:
            logging.error(f"Error saving memories: {e}")
            return False

    def shutdown(self):
        """Gracefully shutdown the system."""
        logging.info("Shutting down Hippocampus...")
        
        # Save all memories
        self.save_all_memories()
        
        if self.enable_multiprocessing and self.stop_event:
            # Signal workers to stop
            self.stop_event.set()
            
            # Send poison pills to workers
            for _ in range(self.max_workers):
                try:
                    self.processing_queue.put(None, timeout=1.0)
                except:
                    pass
            
            # Wait for workers to finish
            for worker in self.worker_processes:
                worker.join(timeout=5.0)
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=2.0)
            
            # Stop result collector
            if hasattr(self, 'result_thread') and self.result_thread.is_alive():
                self.result_thread.join(timeout=5.0)
        
        logging.info("Hippocampus shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.debug(exc_type)
        logging.debug(exc_val)
        logging.debug(exc_tb)
        """Context manager exit."""
        self.shutdown()


# Example usage and testing
if __name__ == "__main__":
    """
    from hippocampus import Hippocampus, ModalityType

    # Use CUDA device 0
    hippocampus = Hippocampus(cuda_device=0)

    # Use CUDA device 1
    hippocampus = Hippocampus(cuda_device=1)  

    # Use CPU
    hippocampus = Hippocampus(cuda_device=None)

    # Import in another script
    with Hippocampus(cuda_device=0, max_workers=4) as memory_system:
        memory_system.add_memory("Test content", ModalityType.TEXT)
    """
    # Windows multiprocessing guard
    mp.freeze_support()
    
    # Example usage
    try:
        # Initialize with CUDA device 0 (or 1+ if multi GPU system)
        with Hippocampus(
            enable_multiprocessing=True, 
            max_workers=2,
            cuda_device=1  # Use 0 for first GPU, 1 for second, None for CPU or will default to first GPU if cuda detected
        ) as hippocampus:
            
            # Add text memories
            text_id = hippocampus.add_memory(
                "The user mentioned they like coffee in the morning",
                ModalityType.TEXT,
                {"source": "conversation", "importance": "high"}
            )
            
            # Add batch memories
            batch_items = [
                ("Good morning", ModalityType.TEXT, {"time": "09:00"}),
                ("How are you today?", ModalityType.TEXT, {"time": "09:01"}),
                ("I'm doing well, thanks", ModalityType.TEXT, {"time": "09:02"})
            ]
            batch_ids = hippocampus.add_memories_batch(batch_items)
            
            # Wait for processing
            time.sleep(3)
            
            # Search for similar memories
            print("="*50)
            print("search memory")
            results = hippocampus.search_memory(
                "coffee morning routine",
                ModalityType.TEXT,
                top_k=3,
                database_name='conversations'
            )
            
            print(f"Search results: {results}")
            
            # Print statistics
            stats = hippocampus.get_statistics()
            print(f"System statistics: {stats}")
            
    except Exception as e:
        logging.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()