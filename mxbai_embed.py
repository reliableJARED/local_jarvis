"""
https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1
A class to handle embedding and similarity operations using the MxBai model.
This class checks for required dependencies, installs them if missing, and provides methods
for embedding strings, calculating similarity, and embedding batches of sentences.

@online{xsmall2024mxbai,
  title={Every Byte Matters: Introducing mxbai-embed-xsmall-v1},
  author={Sean Lee and Julius Lipp and Rui Huang and Darius Koenig},
  year={2024},
  url={https://www.mixedbread.ai/blog/mxbai-embed-xsmall-v1},
}
"""
import subprocess
import sys
import socket
import logging
from typing import Dict, List, Union
# MxBaiEmbedder._check_and_install_dependencies() is called in the constructor
# it will import the required modules and check if they are installed, installing them if necessary.
# numpy, torch, and transformers are required for this class to function

logging.basicConfig(level=logging.DEBUG)


class MxBaiEmbedder:
    """
    A class to handle embedding and similarity operations using the MxBai model.
    """
    def __init__(self, model_name: str = "mixedbread-ai/mxbai-embed-xsmall-v1"):
        self.model_name = model_name
        self.device = None
        self.model = None
        self.tokenizer = None
        
        self._check_and_install_dependencies(model_name)
        
       
    def _get_best_device(self, torch) -> "torch.device":
        """
        Determine the best available device with priority: CUDA > MPS > CPU
        
        Args:
            torch: PyTorch module
            
        Returns:
            torch.device: Best available device
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logging.debug(f"CUDA available - Using GPU: {gpu_name}")
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
        # Attempt to connect to a well-known DNS server (Google's public DNS)
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            logging.debug("Internet connection is available.")
            return False
        except socket.error:
            logging.debug("No internet connection detected.")
            # If connection fails, assume we are in offline mode
            return True
        
    def _check_and_install_dependencies(self, model_name) -> None:
        """
        Check if required dependencies are installed and install if missing.
        Stores imported modules as instance attributes.
        """
        offline_mode = self.use_local_files()
        
        # Check and install transformers
        try:
            from transformers import AutoModel, AutoTokenizer
            self.AutoModel = AutoModel
            self.AutoTokenizer = AutoTokenizer
        except ImportError:
            logging.debug("transformers not found. Installing transformers...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
            try:
                from transformers import AutoModel, AutoTokenizer
                self.AutoModel = AutoModel
                self.AutoTokenizer = AutoTokenizer
            except ImportError:
                raise ImportError("WARNING! Failed to install or import transformers")

        # Check and install PyTorch
        try:
            import torch
            self.torch = torch
            # Determine device with priority: CUDA > MPS > CPU
            self.device = self._get_best_device(torch)
            logging.debug(f"Using device: {self.device}")
        except ImportError:
            logging.debug("PyTorch not found. Installing torch...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
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
        
        # Load model and tokenizer
        try:
            self.tokenizer = self.AutoTokenizer.from_pretrained(
                model_name, 
                local_files_only=offline_mode
            )
            self.model = self.AutoModel.from_pretrained(
                model_name, 
                local_files_only=offline_mode
            ).to(self.device)
            logging.debug(f"Model and tokenizer loaded successfully on {self.device}")
            
            # Log additional device info
            if self.device.type == "cuda":
                logging.debug(f"CUDA device count: {self.torch.cuda.device_count()}")
                logging.debug(f"Current CUDA device: {self.torch.cuda.current_device()}")
                logging.debug(f"CUDA memory allocated: {self.torch.cuda.memory_allocated()/1024**2:.1f} MB")
            elif self.device.type == "mps":
                logging.debug("Using Apple Silicon GPU acceleration")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer: {str(e)}")
    
    def _pooling(self, outputs: "torch.Tensor", inputs: Dict) -> "np.ndarray":
        """
        Apply mean pooling to the model outputs using attention mask.
        
        Args:
            outputs: Model outputs (last_hidden_state)
            inputs: Tokenizer inputs containing attention_mask
            
        Returns:
            Pooled embeddings as numpy array
        """
        # Mean pooling with attention mask
        outputs = self.torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1
        ) / self.torch.sum(inputs["attention_mask"], dim=1, keepdim=True)
        
        return outputs.detach().cpu().numpy()
    
    def embed_string(self, sentence: str) -> List[float]:
        """
        Embed a single string.
        
        Args:
            sentence: Input string to embed
            
        Returns:
            List of embedding values
        """
        # Tokenize input
        inputs = self.tokenizer(
            sentence, 
            padding=True, 
            truncation=True,
            return_tensors='pt'
        )
        
        # Move inputs to device
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        
        # Get model outputs
        with self.torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state
        
        # Apply pooling
        embeddings = self._pooling(outputs, inputs)
        
        # Return as list (single embedding)
        return embeddings[0].tolist()
    
    def embed_batch(self, sentences: List[str]) -> List[List[float]]:
        """
        Embed a batch of sentences.
        
        Args:
            sentences: List of input strings to embed
            
        Returns:
            List of embedding lists
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            sentences, 
            padding=True, 
            truncation=True,
            return_tensors='pt'
        )
        
        # Move inputs to device
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        
        # Get model outputs
        with self.torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state
        
        # Apply pooling
        embeddings = self._pooling(outputs, inputs)
        
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


# Example usage
# Note: Ensure you have the transformers library installed and the model downloaded.
if __name__ == "__main__":
    # Initialize the MxBaiEmbedder model   
    model = MxBaiEmbedder()

    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium."
    ]
    
    # Single string embedding
    embedding = model.embed_string(sentences[0])
    print(f"Single embedding shape: {len(embedding)}")
    print(f"First embedding: {embedding[:5]}...")  # Show first 5 values
    
    # Batch embedding
    embeddings = model.embed_batch(sentences)
    print(f"Batch embeddings shape: {len(embeddings)} x {len(embeddings[0])}")
    
    # Similarity calculation
    sim_score = model.similarity(embeddings[0], embeddings[1])
    print(f"Similarity between first two sentences: {sim_score:.4f}")
    
    # Similarity matrix
    sim_matrix = model.similarity(embeddings, embeddings)
    print(f"Similarity matrix shape: {len(sim_matrix)} x {len(sim_matrix[0])}")