

"""
Cross-Platform Multiprocessing Fixes for Hippocampus Memory System
Addresses Windows, Linux, and macOS compatibility issues.
"""

import os
import sys
import platform
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
    Cross-platform compatible with proper multiprocessing handling and async support.
    """
    
    def __init__(self, 
                 data_directory: str = "./hippocampus_data",
                 enable_multiprocessing: bool = True,
                 max_workers: int = None,
                 cuda_device: Optional[int] = None,
                 batch_size: int = 8,
                 enable_async: bool = True,
                 thread_pool_workers: int = 4):
        """
        Initialize the Hippocampus memory system.
        
        Args:
            data_directory: Directory for storing memory data
            enable_multiprocessing: Whether to use multiprocessing
            max_workers: Maximum number of worker processes
            cuda_device: CUDA device to use (None for CPU, 0 or 1 for specific GPU)
            batch_size: Batch size for processing
            enable_async: Whether to enable async processing (alternative to multiprocessing)
            thread_pool_workers: Number of thread pool workers for async processing
        """
        

        self.data_directory = data_directory
        self.enable_multiprocessing = enable_multiprocessing
        self.max_workers = max_workers or min(mp.cpu_count(), 4)
        self.cuda_device = str(cuda_device) if cuda_device is not None else None
        self.batch_size = batch_size
        self.enable_async = enable_async
        self.thread_pool_workers = thread_pool_workers
        
        # Cross-platform multiprocessing setup
        self._setup_multiprocessing_method()
        
        self._check_and_install_dependencies()
        # Set CUDA device BEFORE any imports or initializations
        self._configure_cuda()
        
        # Initialize components
        self.memory_system = None
        self.embedders = {}
        self.amygdala = None #Emotion engine connection
        self.processing_queue = None
        self.result_queue = None
        self.stop_event = None
        self.worker_processes = []
        self.torch = None
        
        # Thread safety
        self.lock = threading.Lock()
        self.manager = None
        
        # Async components
        self._thread_pool = None
        self._pending_tasks = set()

        # multiprocessing task tracking
        self._pending_mp_count = 0
        self._pending_mp_event = threading.Event()
        self._pending_mp_event.set()  # Start in "complete" state
        
        # Performance tracking
        self.stats = {
            'items_processed': 0,
            'embeddings_generated': 0,
            'searches_performed': 0,
            'avg_embedding_time': 0.0,
            'avg_search_time': 0.0
        }
        
        # Initialize system
        self._convo_db = 'conversations'
        self._images_db = 'images'
        self._audio_db = 'audio'
        self._scene_db = 'scene'
        self._noun_db = 'noun'

        self._txt_embed_dimensions = 384
        self._img_embed_dimensions = 1024
        self._aud_embed_dimensions = 128
        self.database_config = {
                self._convo_db: self._txt_embed_dimensions,
                self._images_db: self._img_embed_dimensions,
                self._audio_db: self._aud_embed_dimensions,
                self._scene_db: self._txt_embed_dimensions,
                self._noun_db: self._txt_embed_dimensions
            }
        
        self._setup_system()#default will use self.database_config
        
        device_str = f"CUDA:{cuda_device}" if cuda_device is not None else "CPU"
        processing_mode = "Multiprocessing" if enable_multiprocessing else ("Async" if enable_async else "Sync")
        logging.info(f"Hippocampus initialized with {self.max_workers} workers on {device_str} using {processing_mode}")
        logging.info(f"Platform: {platform.system()}")

    def _setup_multiprocessing_method(self):
        """
        Set up the appropriate multiprocessing start method for each platform.
        """
        current_platform = platform.system().lower()
        
        try:
            # Get current start method
            current_method = mp.get_start_method(allow_none=True)
            
            if current_platform == "windows":
                # Windows multiprocessing guard
                mp.freeze_support()
                # Windows only supports 'spawn'
                if current_method != 'spawn':
                    mp.set_start_method('spawn', force=True)
                    logging.info("Set multiprocessing method to 'spawn' for Windows")
                    
            elif current_platform == "darwin":  # macOS
                # macOS: prefer 'spawn' for CUDA compatibility, fallback to 'fork'
                if current_method not in ['spawn', 'fork']:
                    try:
                        mp.set_start_method('spawn', force=True)
                        logging.info("Set multiprocessing method to 'spawn' for macOS")
                    except RuntimeError:
                        try:
                            mp.set_start_method('fork', force=True)
                            logging.info("Set multiprocessing method to 'fork' for macOS")
                        except RuntimeError:
                            logging.warning("Could not set multiprocessing method on macOS")
                            
            elif current_platform == "linux":
                # Linux: prefer 'spawn' for CUDA compatibility, fallback to 'fork'
                if current_method not in ['spawn', 'fork']:
                    try:
                        mp.set_start_method('spawn', force=True)
                        logging.info("Set multiprocessing method to 'spawn' for Linux")
                    except RuntimeError:
                        try:
                            mp.set_start_method('fork', force=True)
                            logging.info("Set multiprocessing method to 'fork' for Linux")
                        except RuntimeError:
                            logging.warning("Could not set multiprocessing method on Linux")
            else:
                logging.warning(f"Unknown platform: {current_platform}, using default multiprocessing method")
                
        except Exception as e:
            logging.warning(f"Error setting multiprocessing method: {e}")
            # Continue with default method

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
            if self.torch and self.torch.cuda.is_available():
                # Use the first available GPU
                os.environ['CUDA_VISIBLE_DEVICES'] = "0"
                logging.info("CUDA is available. Using GPU device 0")
            else:
                # Use CPU
                os.environ['CUDA_VISIBLE_DEVICES'] = ""
                logging.info("CUDA not available. Using CPU")

    def _setup_system(self,database_config=None):
        """Set up the complete memory system."""
        # Create data directory
        os.makedirs(self.data_directory, exist_ok=True)
        
        # Initialize memory store
        if database_config == None:
            database_config = self.database_config
        
        self.memory_system = self._initialize_memory_store(database_config)

        #setup the emotion engine
        self._initialize_emotion_engine()
        
        # Always initialize embedders in main process for search functionality
        # Even when using multiprocessing/async, we need embedders for search_memory()
        self._initialize_embedders()
        
        # Set up processing method
        if self.enable_multiprocessing:
            self._setup_multiprocessing()
        elif self.enable_async:
            self._setup_async_processing()

    def _setup_async_processing(self):
        """Set up async processing components."""
        try:
            # Create thread pool for CPU-bound embedding tasks
            self._thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.thread_pool_workers,
                thread_name_prefix="HippocampusAsync"
            )
            logging.info(f"Initialized async thread pool with {self.thread_pool_workers} workers")
        except Exception as e:
            logging.error(f"Failed to setup async processing: {e}")
            self.enable_async = False

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
            #text
            self.embedders[ModalityType.TEXT] = MxBaiEmbedder(cuda_device=self.cuda_device)
            self.embedders[ModalityType.SCENE] = self.embedders[ModalityType.TEXT]
            self.embedders[ModalityType.NOUNS] = self.embedders[ModalityType.TEXT]
            #image
            self.embedders[ModalityType.IMAGE] = BeITEmbedder(cuda_device=self.cuda_device)
            #audio
            self.embedders[ModalityType.AUDIO] = VGGishEmbedder()  # CPU only
            
            logging.debug("All embedders initialized successfully")
            
        except ImportError as e:
            logging.error(f"Failed to import embedder classes: {e}")
            raise
    
    def _initialize_emotion_engine(self):
        """Initialize the emotion engine."""
        try:
            from emotion_embed2 import EmotionEngine
            
            logging.debug("Initializing emotion engine...")
            #text
            self.amygdala = EmotionEngine()

            logging.debug("Emotion Engine initialized successfully")
            
        except ImportError as e:
            logging.error(f"Failed to import EmotionEngine class: {e}")
            raise

    @staticmethod
    def _init_embedders_for_worker(worker_id: int, cuda_device: Optional[str] = None):
        """
        Initialize embedders for a worker process.
        Static method to avoid pickling issues.
        """
        # Get CUDA device from environment or parameter
        if cuda_device is None:
            cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if cuda_device and cuda_device != '':
                cuda_device = cuda_device.split(',')[0]  # Take first device if multiple
            else:
                cuda_device = None
        
        logging.debug(f"Worker {worker_id} initializing embedders with CUDA device: {cuda_device}")
        
        worker_embedders = {}
        try:
            from vggish_embed import VGGishEmbedder
            from beit_embed import BeITEmbedder
            from mxbai_embed import MxBaiEmbedder
            #text embedding
            worker_embedders[ModalityType.TEXT] = MxBaiEmbedder(cuda_device=cuda_device)
            worker_embedders[ModalityType.SCENE] = worker_embedders[ModalityType.TEXT]
            worker_embedders[ModalityType.NOUNS] = worker_embedders[ModalityType.TEXT]
            #image embedding
            worker_embedders[ModalityType.IMAGE] = BeITEmbedder(cuda_device=cuda_device)
            #audio embedding
            worker_embedders[ModalityType.AUDIO] = VGGishEmbedder()  # CPU only
            
            
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
                                stop_event: Event, cuda_device: Optional[str] = None):
        """
        Worker process function. Class method to maintain access to static methods.
        """
        try:
            # Set up signal handling for graceful shutdown on Unix systems
            if platform.system() != "Windows":
                import signal
                def signal_handler(signum, frame):
                    logging.info(f"Worker {worker_id} received signal {signum}, shutting down")
                    stop_event.set()
                
                signal.signal(signal.SIGTERM, signal_handler)
                signal.signal(signal.SIGINT, signal_handler)
        except Exception as e:
            logging.debug(f"Worker {worker_id} could not set up signal handling: {e}")
        
        # Initialize embedders within this worker process
        embedders = cls._init_embedders_for_worker(worker_id, cuda_device)
        
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
                
                # Note: multiprocessing.Queue doesn't have task_done() method
                
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
        """Set up multiprocessing queues and events with cross-platform compatibility."""
        try:
            # Create queues and events
            self.processing_queue = Queue()
            self.result_queue = Queue()
            self.stop_event = Event()
            
            # Start worker processes
            for i in range(self.max_workers):
                worker = Process(
                    target=self._worker_process_function,
                    args=(i, self.processing_queue, self.result_queue, self.stop_event, self.cuda_device),
                    name=f"HippocampusWorker-{i}"
                )
                worker.daemon = False  # Explicitly set daemon to False for proper cleanup
                worker.start()
                self.worker_processes.append(worker)
                
            # Start result collector thread
            self.result_thread = threading.Thread(target=self._collect_results, name="ResultCollector")
            self.result_thread.daemon = True
            self.result_thread.start()
            
            logging.info(f"Started {len(self.worker_processes)} worker processes")
            
        except Exception as e:
            logging.error(f"Failed to setup multiprocessing: {e}")
            self.enable_multiprocessing = False
            raise

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
            items_processed = 0
            if 'memory_item' in result_data:
                self._store_memory_item(result_data['memory_item'])
                items_processed = 1
            elif 'memory_items' in result_data:
                for item in result_data['memory_items']:
                    self._store_memory_item(item)
                    items_processed += 1

            # Decrement pending count using existing lock
            with self.lock:
                self._pending_mp_count = max(0, self._pending_mp_count - items_processed)
                if self._pending_mp_count == 0:
                    self._pending_mp_event.set()  # Mark as complete

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
                
                success = self.memory_system.add_to_database(
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

    def _process_batch_sync(self, memory_items: List[MemoryItem]):
        """
        Process a batch of memory items synchronously.
        This runs in a thread pool to avoid blocking the main thread.
        """
        if not memory_items:
            return
        
        # Ensure embedders are initialized
        if not self.embedders:
            self._initialize_embedders()
        
        # Group by modality for batch embedding
        modality_groups = {}
        for item in memory_items:
            if item.modality not in modality_groups:
                modality_groups[item.modality] = []
            modality_groups[item.modality].append(item)
        
        # Process each modality group
        for modality, items in modality_groups.items():
            try:
                embedder = self.embedders.get(modality)
                if not embedder:
                    logging.error(f"No embedder available for modality {modality}")
                    continue
                
                # Batch embed if possible, otherwise process individually
                if modality in [ModalityType.TEXT, ModalityType.SCENE, ModalityType.NOUNS]:
                    contents = [item.content for item in items]
                    try:
                        embeddings = embedder.embed_batch(contents)
                        for item, embedding in zip(items, embeddings):
                            item.embedding = embedding
                            self._store_memory_item(item)
                    except (AttributeError, NotImplementedError):
                        # Fallback to individual processing
                        for item in items:
                            embedding = embedder.embed_string(item.content)
                            item.embedding = embedding
                            self._store_memory_item(item)
                
                elif modality == ModalityType.IMAGE:
                    contents = [item.content for item in items]
                    try:
                        embeddings = embedder.embed_batch(contents)
                        for item, embedding in zip(items, embeddings):
                            item.embedding = embedding
                            self._store_memory_item(item)
                    except (AttributeError, NotImplementedError):
                        # Fallback to individual processing
                        for item in items:
                            embedding = embedder.embed_image(item.content)
                            item.embedding = embedding
                            self._store_memory_item(item)
                
                elif modality == ModalityType.AUDIO:
                    # Audio typically requires individual processing
                    for item in items:
                        if isinstance(item.content, str):
                            embedding_array = embedder.embed_wav_file(item.content)
                            embedding = embedding_array.mean(axis=0).tolist()
                        else:
                            embedding_array = embedder.embed_audio_data(item.content)
                            embedding = embedding_array.mean(axis=0).tolist()
                        
                        item.embedding = embedding
                        self._store_memory_item(item)
                
            except Exception as e:
                logging.error(f"Error processing batch for modality {modality}: {e}")

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
        
        if self.enable_multiprocessing and async_process and self.processing_queue:
            with self.lock:  # Use existing lock
                task = ('embed', {'memory_item': memory_item})
                self.processing_queue.put(task)
                self._pending_mp_count += 1
                self._pending_mp_event.clear()

        elif self.enable_async and async_process and self._thread_pool:
            # Use async thread pool
            def process_and_store():
                self._process_memory_sync(memory_item)
            
            future = self._thread_pool.submit(process_and_store)
            self._pending_tasks.add(future)
            
            # Remove completed tasks
            future.add_done_callback(lambda f: self._pending_tasks.discard(f))
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
        
        if self.enable_multiprocessing and async_process and self.processing_queue:
            with self.lock:  # Use existing lock
                for modality, items_group in modality_groups.items():
                    task = ('batch_embed', {
                        'memory_items': items_group,
                        'modality': modality
                    })
                    self.processing_queue.put(task)
                    self._pending_mp_count += len(items_group)
                    self._pending_mp_event.clear()  # Mark as incomplete

        elif self.enable_async and async_process and self._thread_pool:
            # Use async thread pool for batch processing
            futures = []
            for modality, items_group in modality_groups.items():
                # Split large groups into smaller batches
                for i in range(0, len(items_group), self.batch_size):
                    batch = items_group[i:i + self.batch_size]
                    future = self._thread_pool.submit(self._process_batch_sync, batch)
                    futures.append(future)
                    self._pending_tasks.add(future)
                    
                    # Remove completed tasks
                    future.add_done_callback(lambda f: self._pending_tasks.discard(f))
        else:
            # Sync processing
            for memory_item in memory_items:
                self._process_memory_sync(memory_item)
        
        logging.debug(f"Added {len(memory_items)} memories in batch")
        return memory_ids

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all pending tasks (both multiprocessing and async) to complete.
        """
        success = True
        
        # Wait for multiprocessing tasks
        if self.enable_multiprocessing:
            if not self._pending_mp_event.wait(timeout=timeout):
                with self.lock:  # Use existing lock to check count
                    remaining = self._pending_mp_count
                logging.warning(f"Timeout waiting for {remaining} multiprocessing tasks")
                success = False
        
        # Wait for async thread pool tasks (existing code)
        if self.enable_async and self._pending_tasks:
            try:
                done, not_done = concurrent.futures.wait(
                    self._pending_tasks, 
                    timeout=timeout,
                    return_when = concurrent.futures.ALL_COMPLETED
                )
                self._pending_tasks.clear()
                if not_done:
                    logging.warning(f"Timeout waiting for {len(not_done)} async tasks")
                    success = False
            except Exception as e:
                logging.error(f"Error waiting for async tasks: {e}")
                success = False
        
        return success

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

    def emotional_reaction(self,query:str) -> Dict:
        """Take an input string and return the emotion most similar to the string
        Arg:
            query: string to get an emotional reaction to
        Returns:
            Dict: {'emotion': 'joy',
                'mood': 'Sense of energy and possibility',
                'thoughts': 'Life is going well',
                'responses': 'Sparks creativity, connection, gives energy',
                'embedding': vector with self.vector_dimensions,
                'similarity': float  # Added similarity score
                }
                'error': if there was an error, will be the error string - key only exists if there was error
        """
        try:

            result = self.amygdala.get_emotional_reaction(query)

            logging.debug(f"Emotional reaction completed, triggered {result}")
            return result
        
        except Exception as e:
            logging.error(f"Emotional reaction error: {e}")
            return {'error':e}

    def search_memory(self,
                     query: Any,
                     modality: ModalityType,
                     top_k: int = 5,
                     database_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in a database."""
        start_time = time.time()
        
        try:
            # Ensure embedders are initialized in the main process for searching
            if not self.embedders:
                logging.debug("Initializing embedders for search operation...")
                self._initialize_embedders()
            
            # Double-check that the specific embedder exists and is properly initialized
            embedder = self.embedders.get(modality)
            if not embedder:
                logging.error(f"No embedder available for modality {modality}")
                logging.debug(f"Available embedders: {list(self.embedders.keys())}")
                return []
            
            # Validate that the embedder is properly loaded
            try:
                # Check if embedder has the required methods and is properly initialized
                if modality in [ModalityType.TEXT, ModalityType.SCENE, ModalityType.NOUNS]:
                    if not hasattr(embedder, 'embed_string'):
                        logging.error(f"Text embedder missing embed_string method")
                        return []
                    # Test with a simple string to ensure it's working
                    if not hasattr(embedder, 'model') or embedder.model is None:
                        logging.error(f"Text embedder model not properly loaded")
                        return []
                elif modality == ModalityType.IMAGE:
                    if not hasattr(embedder, 'embed_image'):
                        logging.error(f"Image embedder missing embed_image method")
                        return []
                    if not hasattr(embedder, 'model') or embedder.model is None:
                        logging.error(f"Image embedder model not properly loaded")
                        return []
                elif modality == ModalityType.AUDIO:
                    if not hasattr(embedder, 'embed_wav_file') and not hasattr(embedder, 'embed_audio_data'):
                        logging.error(f"Audio embedder missing required methods")
                        return []
            except Exception as validation_error:
                logging.error(f"Embedder validation failed for {modality}: {validation_error}")
                # Try to reinitialize the specific embedder
                try:
                    logging.info(f"Attempting to reinitialize {modality} embedder...")
                    if modality in [ModalityType.TEXT, ModalityType.SCENE, ModalityType.NOUNS]:
                        from mxbai_embed import MxBaiEmbedder
                        self.embedders[modality] = MxBaiEmbedder(cuda_device=self.cuda_device)
                        if modality == ModalityType.TEXT:
                            self.embedders[ModalityType.SCENE] = self.embedders[ModalityType.TEXT]
                            self.embedders[ModalityType.NOUNS] = self.embedders[ModalityType.TEXT]
                    elif modality == ModalityType.IMAGE:
                        from beit_embed import BeITEmbedder
                        self.embedders[modality] = BeITEmbedder(cuda_device=self.cuda_device)
                    elif modality == ModalityType.AUDIO:
                        from vggish_embed import VGGishEmbedder
                        self.embedders[modality] = VGGishEmbedder()
                    
                    embedder = self.embedders.get(modality)
                    logging.info(f"Successfully reinitialized {modality} embedder")
                except Exception as reinit_error:
                    logging.error(f"Failed to reinitialize {modality} embedder: {reinit_error}")
                    return []
            
            # Generate query embedding
            try:
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
                
                if not query_embedding:
                    logging.error(f"Failed to generate embedding for query")
                    return []
                    
            except Exception as embed_error:
                logging.error(f"Error generating query embedding: {embed_error}")
                import traceback
                logging.debug(f"Embedding error traceback: {traceback.format_exc()}")
                return []
            
            # Determine database name
            if not database_name:
                db_mapping = {
                    ModalityType.TEXT: self._convo_db,
                    ModalityType.IMAGE: self._images_db,
                    ModalityType.AUDIO: self._audio_db, 
                    ModalityType.SCENE: self._scene_db,
                    ModalityType.NOUNS: self._noun_db
                }
                database_name = db_mapping.get(modality, self._convo_db)
            
            # Perform similarity search
            import numpy as np
            query_vector = np.array(query_embedding)
            
            if not self.memory_system:
                logging.error("Memory store not initialized")
                return []
            
            results = self.memory_system.similarity_search(
                query_vector=query_vector,
                database_name=database_name,
                top_k=top_k
            )
            
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
            import traceback
            logging.debug(f"Search error traceback: {traceback.format_exc()}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        with self.lock:
            stats = self.stats.copy()
        
        try:
            if self.memory_system:
                for db_name in self.memory_system.list_databases():
                    db_info = self.memory_system.get_database_info(db_name)
                    stats[f'{db_name}_items'] = db_info.get('total_items', 0)
        except Exception as e:
            logging.error(f"Error getting database stats: {e}")
        
        return stats

    def save_all_memories(self) -> bool:
        """Save all memories to persistent storage."""
        try:
            if not self.memory_system:
                logging.warning("No memory store available for saving")
                return False
                
            result = self.memory_system.save_all_databases()
            
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

    """
    convenience wrapper methods for embedding
    """
    def embed_string(self, sentence: str) -> List[float]:
        """
        Embed a single string using the text embedder.
        
        Args:
            sentence: Input string to embed
            
        Returns:
            List of embedding values
        """
        try:
            # Ensure embedders are initialized
            if not self.embedders:
                self._initialize_embedders()
            
            text_embedder = self.embedders.get(ModalityType.TEXT)
            if not text_embedder:
                logging.error("Text embedder not available")
                return []
            
            embedding = text_embedder.embed_string(sentence)
            logging.debug(f"Generated embedding for string (length: {len(embedding)})")
            return embedding
            
        except Exception as e:
            logging.error(f"Error embedding string: {e}")
            return []

    def string_embedding_similarity(self, string1: str, string2: str) -> float:
        """
        Calculate cosine similarity between two strings using their embeddings.
        
        Args:
            string1: First string to compare
            string2: Second string to compare
            
        Returns:
            Cosine similarity score between -1 and 1 (higher = more similar)
        """
        try:
            # Get embeddings for both strings
            embedding1 = self.embed_string(string1)
            embedding2 = self.embed_string(string2)
            
            if not embedding1 or not embedding2:
                logging.error("Failed to generate embeddings for similarity calculation")
                return 0.0
            
            # Convert to numpy arrays for calculation
            import numpy as np
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                logging.warning("Zero norm vector detected in similarity calculation")
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            logging.debug(f"Calculated similarity: {similarity:.4f}")
            return float(similarity)
            
        except Exception as e:
            logging.error(f"Error calculating string similarity: {e}")
            return 0.0
    
    def shutdown(self):
        """Gracefully shutdown the system with cross-platform compatibility."""
        logging.info("Shutting down Hippocampus...")
        
        # Wait for async tasks to complete
        if self.enable_async and self._pending_tasks:
            logging.info(f"Waiting for {len(self._pending_tasks)} pending async tasks...")
            self.wait_for_completion(timeout=10.0)
        
        # Save all memories
        self.save_all_memories()
        
        if self.enable_multiprocessing and self.stop_event:
            # Signal workers to stop
            self.stop_event.set()
            
            # Send poison pills to workers
            if self.processing_queue:
                for _ in range(self.max_workers):
                    try:
                        self.processing_queue.put(None, timeout=1.0)
                    except Exception as e:
                        logging.debug(f"Error sending poison pill: {e}")
            
            # Wait for workers to finish gracefully
            for i, worker in enumerate(self.worker_processes):
                if worker.is_alive():
                    logging.debug(f"Waiting for worker {i} to finish...")
                    worker.join(timeout=5.0)
                    
                    if worker.is_alive():
                        logging.warning(f"Worker {i} did not finish gracefully, terminating...")
                        try:
                            worker.terminate()
                            worker.join(timeout=2.0)
                        except Exception as e:
                            logging.error(f"Error terminating worker {i}: {e}")
                        
                        # Force kill on Unix systems if still alive
                        if worker.is_alive() and platform.system() != "Windows":
                            try:
                                import os
                                import signal
                                os.kill(worker.pid, signal.SIGKILL)
                                worker.join(timeout=1.0)
                            except Exception as e:
                                logging.error(f"Error force killing worker {i}: {e}")
            
            # Stop result collector thread
            if hasattr(self, 'result_thread') and self.result_thread.is_alive():
                logging.debug("Stopping result collector thread...")
                self.result_thread.join(timeout=5.0)
                if self.result_thread.is_alive():
                    logging.warning("Result collector thread did not stop gracefully")
        
        # Shutdown thread pool
        if self.enable_async and self._thread_pool:
            logging.info("Shutting down thread pool...")
            self._thread_pool.shutdown(wait=True, timeout=10.0)
        
        # Clean up resources
        try:
            if hasattr(self, 'processing_queue') and self.processing_queue:
                # Close the queue to indicate no more items will be put
                self.processing_queue.close()
                # Note: multiprocessing.Queue doesn't have join_thread() in all Python versions
                try:
                    self.processing_queue.join_thread()
                except AttributeError:
                    # join_thread() not available, this is normal for some Python versions
                    pass
        except Exception as e:
            logging.debug(f"Error closing processing queue: {e}")
            
        try:
            if hasattr(self, 'result_queue') and self.result_queue:
                self.result_queue.close()
                try:
                    self.result_queue.join_thread()
                except AttributeError:
                    # join_thread() not available, this is normal for some Python versions
                    pass
        except Exception as e:
            logging.debug(f"Error closing result queue: {e}")
        
        logging.info("Hippocampus shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            logging.error(f"Exception in context manager: {exc_type.__name__}: {exc_val}")
        self.shutdown()

# Example usage and testing
if __name__ == "__main__":
    """
    from hippocampus import Hippocampus, ModalityType

    # Use CUDA device 0
    hippocampus = Hippocampus(cuda_device=0)

    # Use CUDA device 1
    hippocampus = Hippocampus(cuda_device=1)  

    # Use CUDA if detected, default device 0 else CPU
    hippocampus = Hippocampus(cuda_device=None)

    # Import in another script
    with Hippocampus(cuda_device=0, max_workers=4) as memory_system:
        memory_system.add_memory("Test content", ModalityType.TEXT)
    """
    # Windows multiprocessing guard
    #mp.freeze_support()
    
    # Example usage
    try:
        # Initialize with CUDA device 0 (or 1+ if multi GPU system)
        with Hippocampus(
            enable_multiprocessing=True, 
            max_workers=2,
            cuda_device=1  # Use 0 for first GPU, 1 for second, None for CPU or will default to first GPU if cuda detected
        ) as hippocampus:
            
            # Add text 
            text_id = hippocampus.add_memory(
                "The user mentioned they like coffee in the morning",
                ModalityType.TEXT,
                {"source": "conversation", "importance": "high"}
            )
            
            # Add batch memories
            batch_items = [
                ("Good morning", ModalityType.TEXT, {"time": "09:00"}),
                ("How are you today?", ModalityType.TEXT, {"time": "09:01"}),
                ("I'm doing well, thanks", ModalityType.TEXT, {"time": "09:02"}),
                ("My dog's name is king", ModalityType.TEXT, {"time": "09:02"})
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

            # Get emotional reaction
            print("="*50)
            print("trigger emotion")
            results = hippocampus.emotional_reaction(
                "coffee morning routine"
            )
            
            print(f"Emotion results: {results}")
            
            # Print statistics
            stats = hippocampus.get_statistics()
            print(f"System statistics: {stats}")
            
    except Exception as e:
        logging.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()