"""
BeIT Image Embedder - A class to handle image embedding operations using Microsoft's BEiT model.
Based on: https://huggingface.co/microsoft/beit-large-patch16-224

This class checks for required dependencies, installs them if missing, and provides methods
for embedding single images, calculating similarity, and embedding batches of images.

Supports various input formats:
- PIL Images
- File paths (string)
- NumPy arrays
- Base64 encoded strings

@article  
  author    = {Hangbo Bao and
               Li Dong and
               Furu Wei},
  title     = {BEiT: {BERT} Pre-Training of Image Transformers},
  journal   = {CoRR},
  volume    = {abs/2106.08254},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.08254},
  archivePrefix = {arXiv},
  eprint    = {2106.08254},
  timestamp = {Tue, 29 Jun 2021 16:55:04 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2106-08254.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

"""
import subprocess
import sys
import socket
import logging
import base64
import io
from typing import Dict, List, Union, Any
from pathlib import Path

# BeITEmbedder._check_and_install_dependencies() is called in the constructor
# it will import the required modules and check if they are installed, installing them if necessary.
# numpy, torch, transformers, and PIL are required for this class to function

logging.basicConfig(level=logging.DEBUG)


class BeITEmbedder:
    """
    BEiT image embedding, generates high-dimensional embeddings for input images.
    Extracts features from the last hidden layer before classification.
    """
    def __init__(self, model_name: str = "microsoft/beit-large-patch16-224", cuda_device: str = None):

        self.model_name = model_name
        self.device = None
        self.model = None
        self.processor = None
        self.embedding_dim = None
        self.cuda_device = cuda_device
        self.torch = None
        self._check_and_install_dependencies(model_name)
        
    def _configure_cuda(self):
        """Configure CUDA device for the entire process."""
        import os
        if self.cuda_device is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cuda_device)
            logging.info(f"Set CUDA_VISIBLE_DEVICES to {self.cuda_device}")
        else:
             # Use CPU
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
            logging.info("CUDA not available. Using CPU")
        return True
    
    def _get_best_device(self, torch) -> "torch.device":
        """
        Determine the best available device with priority: CUDA > MPS > CPU
        
        Args:
            torch: PyTorch module
            
        Returns:
            torch.device: Best available device
        """
        if torch.cuda.is_available():
            #set cuda device env
            _ = self._configure_cuda()
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logging.debug(f"CUDA available - BeIT Embedder Using GPU {self.cuda_device}: {gpu_name}")
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.debug("MPS (Apple Silicon) available - Using MPS")
            return device
        else:
            device = torch.device("cpu")
            logging.debug("Using CPU")
            return device
    
    def use_local_files(self, host="8.8.8.8", port=53, timeout=3):
        """
        Check internet connectivity by attempting to connect to DNS server.
        
        Returns:
            bool: True if NO connection, False otherwise
        """
        logging.debug(f"Checking internet connectivity to {host}:{port} with timeout {timeout} seconds")
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            logging.debug("Internet connection is available.")
            return False
        except socket.error:
            logging.debug("No internet connection detected.")
            return True
        
    def _check_and_install_dependencies(self, model_name) -> None:
        """
        Check if required dependencies are installed and install if missing.
        Stores imported modules as instance attributes.
        """
        offline_mode = self.use_local_files()
        
        # Check and install transformers
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            self.AutoImageProcessor = AutoImageProcessor
            self.AutoModelForImageClassification = AutoModelForImageClassification
        except ImportError:
            logging.debug("transformers not found. Installing transformers...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
            try:
                from transformers import AutoImageProcessor, AutoModelForImageClassification
                self.AutoImageProcessor = AutoImageProcessor
                self.AutoModelForImageClassification = AutoModelForImageClassification
            except ImportError:
                raise ImportError("WARNING! Failed to install or import transformers")

        # Check and install PyTorch
        try:
            import torch
            self.torch = torch
            self.device = self._get_best_device(torch)
            logging.debug(f"Using device: {self.device}")
        except ImportError:
            logging.debug("PyTorch not found. Installing torch...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
            try:
                import torch
                self.torch = torch
                self.device = self._get_best_device(torch)
                logging.debug(f"Using device: {self.device}")
            except ImportError:
                raise ImportError("WARNING! Failed to install or import PyTorch")

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

        # Check and install PIL
        try:
            from PIL import Image
            self.Image = Image
        except ImportError:
            logging.debug("PIL not found. Installing Pillow...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
            try:
                from PIL import Image
                self.Image = Image
            except ImportError:
                raise ImportError("WARNING! Failed to install or import PIL/Pillow")
        
        # Load model and processor
        try:
            self.processor = self.AutoImageProcessor.from_pretrained(
                model_name, 
                local_files_only=offline_mode
            )
            self.model = self.AutoModelForImageClassification.from_pretrained(
                model_name, 
                local_files_only=offline_mode,
                output_hidden_states=True  # We need hidden states for embeddings
            ).to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logging.debug(f"Model and processor loaded successfully on {self.device}")
            
            # Log additional device info
            if self.device.type == "cuda":
                logging.debug(f"CUDA device count: {torch.cuda.device_count()}")
                logging.debug(f"Current CUDA device: {torch.cuda.current_device()}")
                logging.debug(f"CUDA memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
            elif self.device.type == "mps":
                logging.debug("Using Apple Silicon GPU acceleration")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model or processor: {str(e)}")
    
    def _preprocess_image(self, image_input: Union[str, "PIL.Image.Image", "np.ndarray", bytes]) -> "PIL.Image.Image":
        """
        Convert various image input formats to PIL Image.
        
        Args:
            image_input: Image in various formats (file path, PIL Image, numpy array, base64 bytes)
            
        Returns:
            PIL Image object
        """
        if isinstance(image_input, str):
            # Handle file path or base64 string
            if image_input.startswith('data:image') or len(image_input) > 500:  # Likely base64
                # Extract base64 data if it's a data URL
                if image_input.startswith('data:image'):
                    image_input = image_input.split(',')[1]
                try:
                    image_bytes = base64.b64decode(image_input)
                    image = self.Image.open(io.BytesIO(image_bytes))
                except Exception as e:
                    raise ValueError(f"Failed to decode base64 image: {str(e)}")
            else:
                # Handle file path
                if not Path(image_input).exists():
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                image = self.Image.open(image_input)
        
        elif isinstance(image_input, bytes):
            # Handle raw bytes
            image = self.Image.open(io.BytesIO(image_input))
        
        elif hasattr(image_input, 'shape') and hasattr(image_input, 'dtype'):
            # Handle numpy array
            if image_input.dtype != self.np.uint8:
                # Normalize if not uint8
                if image_input.max() <= 1.0:
                    image_input = (image_input * 255).astype(self.np.uint8)
                else:
                    image_input = image_input.astype(self.np.uint8)
            
            # Convert from numpy array to PIL
            if len(image_input.shape) == 3:
                image = self.Image.fromarray(image_input)
            elif len(image_input.shape) == 2:
                image = self.Image.fromarray(image_input, mode='L')
            else:
                raise ValueError(f"Unsupported numpy array shape: {image_input.shape}")
        
        elif hasattr(image_input, 'mode') and hasattr(image_input, 'size'):
            # Already a PIL Image
            image = image_input
        
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    
    def _extract_embeddings(self, outputs) -> "np.ndarray":
        """
        Extract embeddings from model outputs using the last hidden state.
        
        Args:
            outputs: Model outputs containing hidden_states
            
        Returns:
            Pooled embeddings as numpy array
        """
        # Get the last hidden state (before classification head)
        last_hidden_state = outputs.hidden_states[-1]  # Shape: [batch_size, num_patches + 1, hidden_size]
        
        # Use the [CLS] token embedding (first token) as the image representation
        cls_embeddings = last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]
        
        return cls_embeddings.detach().cpu().numpy()
    
    def embed_image(self, image: Union[str, "PIL.Image.Image", "np.ndarray", bytes]) -> List[float]:
        """
        Embed a single image.
        
        Args:
            image: Input image in various formats (file path, PIL Image, numpy array, base64)
            
        Returns:
            List of embedding values
        """
        # Preprocess image
        pil_image = self._preprocess_image(image)
        
        # Process image with the processor
        inputs = self.processor(images=pil_image, return_tensors="pt")
        
        # Move inputs to device
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        
        # Get model outputs
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract embeddings
        embeddings = self._extract_embeddings(outputs)
        
        # Store embedding dimension for reference
        if self.embedding_dim is None:
            self.embedding_dim = embeddings.shape[1]
            logging.debug(f"Embedding dimension: {self.embedding_dim}")
        
        # Return as list (single embedding)
        return embeddings[0].tolist()
    
    def embed_batch(self, images: List[Union[str, "PIL.Image.Image", "np.ndarray", bytes]]) -> List[List[float]]:
        """
        Embed a batch of images.
        
        Args:
            images: List of images in various formats
            
        Returns:
            List of embedding lists
        """
        # Preprocess all images
        pil_images = [self._preprocess_image(img) for img in images]
        
        # Process images with the processor
        inputs = self.processor(images=pil_images, return_tensors="pt")
        
        # Move inputs to device
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        
        # Get model outputs
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract embeddings
        embeddings = self._extract_embeddings(outputs)
        
        # Store embedding dimension for reference
        if self.embedding_dim is None:
            self.embedding_dim = embeddings.shape[1]
            logging.debug(f"Embedding dimension: {self.embedding_dim}")
        
        # Return as list of lists
        return embeddings.tolist()
    
    def similarity(self, embeddings1: Union[List[float], List[List[float]]], 
                   embeddings2: Union[List[float], List[List[float]]]) -> Union[float, List[List[float]]]:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity score(s)
        """
        # Convert to numpy arrays
        emb1 = self.np.array(embeddings1)
        emb2 = self.np.array(embeddings2)
        
        # Ensure 2D arrays
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)
        
        # Calculate cosine similarity
        dot_product = self.np.dot(emb1, emb2.T)
        norms1 = self.np.linalg.norm(emb1, axis=1, keepdims=True)
        norms2 = self.np.linalg.norm(emb2, axis=1, keepdims=True)
        
        similarity_matrix = dot_product / (norms1 * norms2.T)
        
        # Return appropriate format
        if similarity_matrix.shape == (1, 1):
            return float(similarity_matrix[0, 0])
        else:
            return similarity_matrix.tolist()
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            Embedding dimension 1024 for BEiT large model
        """
        if self.embedding_dim is None:
            # Create a dummy image to determine embedding dimension
            dummy_image = self.Image.new('RGB', (224, 224), color='white')
            self.embed_image(dummy_image)
        
        return self.embedding_dim


# Example usage
if __name__ == "__main__":
    # Initialize the BeITEmbedder model   
    model = BeITEmbedder()
    
    # Example with dummy images (you can replace with actual image paths)
    print("Creating example images...")
    
    # Create some example images
    from PIL import Image, ImageDraw
    
    # Create a red square
    img1 = Image.new('RGB', (224, 224), color='red')
    draw1 = ImageDraw.Draw(img1)
    draw1.rectangle([50, 50, 174, 174], fill='white')
    
    # Create a blue circle
    img2 = Image.new('RGB', (224, 224), color='blue')
    draw2 = ImageDraw.Draw(img2)
    draw2.ellipse([50, 50, 174, 174], fill='white')
    
    # Create a green triangle (approximated with polygon)
    img3 = Image.new('RGB', (224, 224), color='green')
    draw3 = ImageDraw.Draw(img3)
    draw3.polygon([(112, 50), (50, 174), (174, 174)], fill='white')
    
    images = [img1, img2, img3]
    
    # Single image embedding
    print("Embedding single image...")
    embedding = model.embed_image(images[0])
    print(f"Single embedding shape: {len(embedding)}")
    print(f"Embedding dimension: {model.get_embedding_dimension()}")
    print(f"First 5 embedding values: {embedding[:5]}")
    
    # Batch embedding
    print("\nEmbedding batch of images...")
    embeddings = model.embed_batch(images)
    print(f"Batch embeddings shape: {len(embeddings)} x {len(embeddings[0])}")
    
    # Similarity calculation
    sim_score = model.similarity(embeddings[0], embeddings[1])
    print(f"\nSimilarity between image 1 and 2: {sim_score:.4f}")
    
    # Similarity matrix
    sim_matrix = model.similarity(embeddings, embeddings)
    print(f"Similarity matrix shape: {len(sim_matrix)} x {len(sim_matrix[0])}")
    print("Similarity matrix:")
    for i, row in enumerate(sim_matrix):
        print(f"Image {i+1}: {[f'{val:.3f}' for val in row]}")
    
    # Test with file path (if you have actual images)
    # embedding_from_file = model.embed_image("path/to/your/image.jpg")
    
    print(f"\nModel loaded successfully with embedding dimension: {model.get_embedding_dimension()}")