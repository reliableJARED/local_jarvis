# Example usage and testing
"""
FAISS Similarity Search Module

A modular class for implementing Facebook AI Similarity Search (FAISS) with multiple
database support for different data types including text, image vectors, audio vectors,
scene descriptions, and noun descriptions. Handles loading, saving, and searching
across multiple databases with FAISS indices.

Dependencies:
    - faiss-cpu: pip install faiss-cpu
    - numpy: pip install numpy
    
Note: GPU versions of FAISS exist (faiss-gpu) but this implementation uses CPU version
for broader compatibility. For GPU acceleration, replace faiss-cpu with faiss-gpu.

https://claude.ai/chat/759383e6-5de6-4aef-be70-1f4e35b8a1e3
"""

import os
import pickle
import subprocess
import sys
from typing import Dict, List, Tuple, Optional, Any
import logging

# Non standard library Dependencies will be imported and checked during initialization


class MemRecall:
    """
    A comprehensive FAISS-based python dict storage and similarity search system supporting multiple databases
    for different data types.
    
    This class provides functionality to:
    - Load and manage multiple FAISS indices
    - Save and load databases as pickle files
    - Perform similarity searches across different data types
    - Handle text, image vectors, audio vectors, scene descriptions, and noun data
    
    Attributes:
        conversations_db (Dict): Database for conversation text data
        images_db (Dict): Database for image vector data
        audio_db (Dict): Database for audio clip vector data
        scene_db (Dict): Database for scene descriptions (text)
        nouns_db (Dict): Database for known people, places, things (text)
        emotions_db (Dict): Database for embedded emotion descriptions (text)
    """
    
    def __init__(self, data_directory: str = "./faiss_data", 
                 databases: Optional[Dict[str, int]] = None):
        """
        Initialize the FAISS Similarity Search system.
        
        Args:
            data_directory (str): Directory path to store database files
            databases (Optional[Dict[str, int]]): Dictionary of database names and their vector dimensions.
                                                 If None, uses default databases.
        """
        self.data_directory = data_directory
        
        # Default database configuration
        self._default_databases = {
            'conversations': 384,  # Text descriptions
            'images': 512,         # Common image embedding size
            'audio': 128,          # Common audio embedding size
            'scene': 384,          # Text descriptions
            'nouns': 384,          # Text descriptions
            'emotions': 384        # Text descriptions
        }
        
        # Set database configuration
        self.databases = databases if databases is not None else self._default_databases.copy()
        if databases is None:
            print("Initialize using default database configuration.")
        
        # Dynamically create database attributes
        for db_name in self.databases.keys():
            setattr(self, f"{db_name}_db", {})
        
        # FAISS indices for each database type (dynamically created) Optional[Any] = Optional[faiss.IndexFlatIP]
        self._faiss_indices: Dict[str, Optional[Any]] = {
            db_name: None for db_name in self.databases.keys()
        }
        
        # Check and install dependencies first
        self._check_and_install_dependencies()
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_directory, exist_ok=True)
        
        # Load existing databases
        self._load_all_databases()

        # Show all loaded databases
        print(f"Loaded databases: {self.list_databases()}")

        # Show database information
        for db_name in self.databases.keys():
            info = self.get_database_info(db_name)
            print(f"Database '{db_name}' info: {info}")
    
    def _check_and_install_dependencies(self) -> None:
        """
        Check if required dependencies are installed and install if missing.
        Stores imported modules as instance attributes.
        """
        # Check and install FAISS
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            print("FAISS not found. Installing faiss-cpu...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
            try:
                import faiss
                self.faiss = faiss
            except ImportError:
                raise ImportError("WARNING! Failed to install or import FAISS")

        # Check and install NumPy
        try:
            import numpy as np
            self.np = np
        except ImportError:
            print("NumPy not found. Installing numpy...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
            try:
                import numpy as np
                self.np = np
            except ImportError:
                raise ImportError("WARNING! Failed to install or import NumPy")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """
        Check if required dependencies are installed.
        
        Returns:
            Dict[str, bool]: Dictionary with dependency names and installation status
        """
        dependencies = {
            'faiss': hasattr(self, 'faiss') and self.faiss is not None,
            'numpy': hasattr(self, 'np') and self.np is not None
        }
        
        return dependencies
    
    def save_database(self, database_name: str, data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save a database dictionary as a pickle file.
        
        Args:
            database_name (str): Name of the database to save
            data (Optional[Dict[str, Any]]): Dictionary data to save. If None, saves the current database state.
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if database exists in configuration
            if database_name not in self.databases:
                print(f"Database '{database_name}' not found in configured databases")
                return False
            
            # Get data to save - either provided data or current database state
            if data is None:
                if hasattr(self, f"{database_name}_db"):
                    data = getattr(self, f"{database_name}_db")
                else:
                    print(f"Database attribute '{database_name}_db' not found")
                    return False
            
            filename = os.path.join(self.data_directory, f"{database_name}_db.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update the corresponding database attribute
            setattr(self, f"{database_name}_db", data)
            
            print(f"Database '{database_name}' saved successfully to {filename}")
            return True

        except Exception as e:
            print(f"Error saving database '{database_name}': {str(e)}")
            return False

    def save_all_databases(self) -> Dict[str, bool]:
        """
        Save all databases to their respective pickle files.
        
        Returns:
            Dict[str, bool]: Dictionary mapping database names to save success status
        """
        results = {}
        
        try:
            for db_name in self.databases.keys():
                try:
                    # Save each database individually and capture the result
                    results[db_name] = self.save_database(db_name)
                except Exception as e:
                    print(f"Error saving database '{db_name}': {str(e)}")
                    results[db_name] = False
            
            return results
            
        except Exception as e:
            print(f"Error when saving all databases: {str(e)}")
            # Return partial results if we got some, otherwise return empty dict
            return results if results else {db_name: False for db_name in self.databases.keys()}
    
    def load_database(self, database_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a database from a pickle file and rebuild FAISS index.
        
        Args:
            database_name (str): Name of the database to load
            
        Returns:
            Optional[Dict[str, Any]]: Loaded database dictionary or None if failed
        """
        try:
            filename = os.path.join(self.data_directory, f"{database_name}_db.pkl")
            
            if not os.path.exists(filename):
                print(f"Database file '{filename}' not found. Creating empty database.")
                return {}
            
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            # Update the corresponding database attribute
            setattr(self, f"{database_name}_db", data)
            
            # Rebuild FAISS index from loaded data
            if data:  # Only if there's data to rebuild from
                self._rebuild_faiss_index(database_name, data)
            
            print(f"Database '{database_name}' loaded successfully from {filename}")
            return data
            
        except Exception as e:
            print(f"Error loading database '{database_name}': {str(e)}")
            return None
    
    def _load_all_databases(self) -> None:
        """
        Load all databases on initialization.
        """
        for db_name in self.databases.keys():
            loaded_db = self.load_database(db_name)
            if loaded_db is not None:
                setattr(self, f"{db_name}_db", loaded_db)
    
    def _rebuild_faiss_index(self, database_name: str, data: Dict[str, Any]) -> None:
        """
        Rebuild FAISS index from loaded database data.
        
        Args:
            database_name (str): Name of the database
            data (Dict[str, Any]): Database data with vectors
        """
        try:
            if not data:
                return
                
            # Get vector dimension from database config
            dimension = self.databases.get(database_name)
            if dimension is None:
                print(f"No dimension found for database '{database_name}'")
                return
            
            # Initialize FAISS index
            self._initialize_faiss_index(database_name, dimension)
            
            # Collect all vectors and update index positions
            vectors = []
            items_by_position = {}
            
            for item_id, item_data in data.items():
                if 'vector' in item_data:
                    vector = item_data['vector']
                    # Ensure vector is normalized
                    if hasattr(vector, 'shape'):
                        vector_norm = vector / self.np.linalg.norm(vector)
                    else:
                        vector_array = self.np.array(vector)
                        vector_norm = vector_array / self.np.linalg.norm(vector_array)
                    
                    vectors.append(vector_norm.astype(self.np.float32))
                    # Store the mapping for this position
                    position = len(vectors) - 1
                    items_by_position[position] = item_id
                    # Update the item's index position
                    item_data['index_position'] = position
            
            if vectors:
                # Add all vectors to FAISS index at once
                vectors_array = self.np.vstack(vectors)
                self._faiss_indices[database_name].add(vectors_array)
                print(f"Rebuilt FAISS index for '{database_name}' with {len(vectors)} vectors")
            else:
                print(f"No vectors found in database '{database_name}' to rebuild index")
                
        except Exception as e:
            print(f"Error rebuilding FAISS index for '{database_name}': {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

    def _initialize_faiss_index(self, database_name: str, dimension: int) -> Any:
        """
        Initialize a FAISS index for a specific database.
        
        Args:
            database_name (str): Name of the database
            dimension (int): Vector dimension for the index
            
        Returns:
            faiss.IndexFlatIP: Initialized FAISS index
        """
        # Using IndexFlatIP for inner product similarity (cosine similarity with normalized vectors)
        index = self.faiss.IndexFlatIP(dimension)
        self._faiss_indices[database_name] = index
        return index
    
    def add_to_database(self, database_name: str, item_id: str, vector: List[float], 
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add an item to a specific database.
        
        Args:
            database_name (str): Name of the target database
            item_id (str): Unique identifier for the item
            vector (np.ndarray): Vector representation of the item
            metadata (Optional[Dict[str, Any]]): Additional metadata for the item
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get the target database
            target_db = getattr(self, f"{database_name}_db")
            
            # Ensure vector is the correct dimension
            expected_dim = self.databases.get(database_name)
            if expected_dim is None:
                raise ValueError(f"Database '{database_name}' not found in configured databases")
            
            if vector.shape[-1] != expected_dim:
                print(f"Warning: Vector dimension {vector.shape[-1]} doesn't match expected {expected_dim}")
            
            # Initialize FAISS index if needed
            if self._faiss_indices[database_name] is None:
                self._initialize_faiss_index(database_name, vector.shape[-1])
            
            # Normalize vector for cosine similarity
            vector_norm = vector / self.np.linalg.norm(vector)
            
            # Add to FAISS index
            self._faiss_indices[database_name].add(vector_norm.reshape(1, -1).astype(self.np.float32))
            
            # Add to database with metadata
            target_db[item_id] = {
                'vector': vector_norm,
                'metadata': metadata or {},
                'index_position': self._faiss_indices[database_name].ntotal - 1
            }
            
            return True
            
        except Exception as e:
            print(f"Error adding item to database '{database_name}': {str(e)}")
            return False
    
    def similarity_search(self, query_vector: List[float], database_name: str, 
                         top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform similarity search in a specified database.
        
        Args:
            query_vector (np.ndarray): Query vector for similarity search
            database_name (str): Name of the database to search in
            top_k (int): Number of top similar items to return
            
        Returns:
            List[Tuple[str, float, Dict[str, Any]]]: List of (item_id, similarity_score, metadata) tuples
        """
        try:
            # Get the target database and FAISS index
            target_db = getattr(self, f"{database_name}_db")
            faiss_index = self._faiss_indices.get(database_name)
            
            print(f"Debug: Database '{database_name}' has {len(target_db)} items")
            print(f"Debug: FAISS index exists: {faiss_index is not None}")
            if faiss_index:
                print(f"Debug: FAISS index has {faiss_index.ntotal} vectors")
            
            if faiss_index is None:
                print(f"FAISS index for database '{database_name}' not initialized")
                # Try to rebuild index from existing data
                if target_db:
                    print(f"Attempting to rebuild FAISS index from {len(target_db)} items...")
                    self._rebuild_faiss_index(database_name, target_db)
                    faiss_index = self._faiss_indices.get(database_name)
                    if faiss_index is None:
                        print(f"Failed to rebuild FAISS index for '{database_name}'")
                        return []
                else:
                    print(f"No data in database '{database_name}' to rebuild index from")
                    return []
            
            if faiss_index.ntotal == 0:
                print(f"FAISS index for database '{database_name}' is empty")
                return []
            
            # Normalize query vector
            query_norm = query_vector / self.np.linalg.norm(query_vector)
            query_norm = query_norm.reshape(1, -1).astype(self.np.float32)
            
            # Perform search
            similarities, indices = faiss_index.search(query_norm, min(top_k, faiss_index.ntotal))
            
            # Map results back to item IDs and metadata
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                # Find item by index position
                for item_id, item_data in target_db.items():
                    if item_data.get('index_position') == idx:
                        results.append((item_id, float(similarity), item_data.get('metadata', {})))
                        break
            
            print(f"Found {len(results)} similar items")
            return results
            
        except Exception as e:
            logging.error(f"Error performing similarity search in database '{database_name}': {str(e)}")
            import traceback
            print(f"Search error traceback: {traceback.format_exc()}")
            return []
        
    def get_database_info(self, database_name: str) -> Dict[str, Any]:
        """
        Get information about a specific database.
        
        Args:
            database_name (str): Name of the database
            
        Returns:
            Dict[str, Any]: Information about the database
        """
        try:
            target_db = getattr(self, f"{database_name}_db")
            faiss_index = self._faiss_indices.get(database_name)
            
            info = {
                'name': database_name,
                'total_items': len(target_db),
                'faiss_index_size': faiss_index.ntotal if faiss_index else 0,
                'vector_dimension': self.databases.get(database_name, 'unknown'),
                'database_file': os.path.join(self.data_directory, f"{database_name}_db.pkl"),
                'index_initialized': faiss_index is not None,
                'has_data': len(target_db) > 0
            }
            
            return info
            
        except Exception as e:
            print(f"Error getting database info for '{database_name}': {str(e)}")
            return {}
    
    def list_databases(self) -> List[str]:
        """
        List all available databases.
        
        Returns:
            List[str]: List of database names
        """
        return list(self.databases.keys())
    
    def clear_database(self, database_name: str) -> bool:
        """
        Clear all data from a specific database.
        
        Args:
            database_name (str): Name of the database to clear
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Clear the database dictionary
            setattr(self, f"{database_name}_db", {})
            
            # Reset the FAISS index
            if database_name in self.databases:
                self._faiss_indices[database_name] = None
            else:
                raise ValueError(f"Database '{database_name}' not found in configured databases")
            
            # Save the empty database
            self.save_database(database_name, {})
            
            print(f"Database '{database_name}' cleared successfully")
            return True
            
        except Exception as e:
            print(f"Error clearing database '{database_name}': {str(e)}")
    
    def add_database(self, database_name: str, vector_dimension: int) -> bool:
        """
        Add a new database configuration dynamically.
        
        Args:
            database_name (str): Name of the new database
            vector_dimension (int): Vector dimension for this database
            
        Returns:
            bool: True if successful, False if database already exists
        """
        if database_name in self.databases:
            print(f"Database '{database_name}' already exists")
            return False
        
        # Add to databases configuration
        self.databases[database_name] = vector_dimension
        
        # Create database attribute
        setattr(self, f"{database_name}_db", {})
        
        # Initialize FAISS index slot
        self._faiss_indices[database_name] = None
        
        print(f"Database '{database_name}' added successfully with dimension {vector_dimension}")
        return True
    
    def remove_database(self, database_name: str) -> bool:
        """
        Remove a database configuration and clear its data.
        
        Args:
            database_name (str): Name of the database to remove
            
        Returns:
            bool: True if successful, False if database doesn't exist
        """
        if database_name not in self.databases:
            print(f"Database '{database_name}' doesn't exist")
            return False
        
        # Clear the database first
        self.clear_database(database_name)
        
        # Remove from configuration
        del self.databases[database_name]
        
        # Remove database attribute
        if hasattr(self, f"{database_name}_db"):
            delattr(self, f"{database_name}_db")
        
        # Remove FAISS index
        if database_name in self._faiss_indices:
            del self._faiss_indices[database_name]
        
        # Remove database file
        try:
            filename = os.path.join(self.data_directory, f"{database_name}_db.pkl")
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            print(f"Warning: Could not remove database file: {str(e)}")
        
        print(f"Database '{database_name}' removed successfully")
        return True
    
    def get_database_config(self) -> Dict[str, int]:
        """
        Get the current database configuration.
        
        Returns:
            Dict[str, int]: Dictionary of database names and their vector dimensions
        """
        return self.databases.copy()
    
    def update_database_dimension(self, database_name: str, new_dimension: int) -> bool:
        """
        Update the vector dimension for an existing database.
        Note: This will clear existing data in the database.
        
        Args:
            database_name (str): Name of the database to update
            new_dimension (int): New vector dimension
            
        Returns:
            bool: True if successful, False if database doesn't exist
        """
        if database_name not in self.databases:
            print(f"Database '{database_name}' doesn't exist")
            return False
        
        # Clear existing data since dimension is changing
        self.clear_database(database_name)
        
        # Update dimension
        self.databases[database_name] = new_dimension
        
        print(f"Database '{database_name}' dimension updated to {new_dimension}. ALL existing data cleared.")
        return True
    

if __name__ == "__main__":
    #note: This block is for testing and example usage only. numpy is required for this example., but auto installs in the MemRecall class.
    import numpy as np

    # Initialize the similarity search system with default databases
    faiss_search = MemRecall()
    
    # Or initialize with custom databases
    # custom_dbs = {'documents': 768, 'embeddings': 1024, 'features': 256}
    # faiss_search = MemRecall(databases=custom_dbs)
    
    # Show current database configuration
    print("Database configuration:", faiss_search.get_database_config())
    print("Available databases:", faiss_search.list_databases())
    
    # Add a new database dynamically
    faiss_search.add_database('custom_vectors', 512)
    print("After adding custom database:", faiss_search.list_databases())
    
    # Example: Add some dummy data to conversations database
    dummy_vectors = np.random.random((3, 384)).astype(np.float32)
    
    faiss_search.add_to_database('conversations', 'conv_1', dummy_vectors[0], 
                                {'text': 'Hello world', 'timestamp': '2024-01-01'})
    faiss_search.add_to_database('conversations', 'conv_2', dummy_vectors[1], 
                                {'text': 'How are you?', 'timestamp': '2024-01-02'})
    faiss_search.add_to_database('conversations', 'conv_3', dummy_vectors[2], 
                                {'text': 'Goodbye!', 'timestamp': '2024-01-03'})
    
    # Save the database (no need to pass the data)
    faiss_search.save_database('conversations')
    
    # Or save all databases at once
    save_results = faiss_search.save_all_databases()
    print("Save results:", save_results)
    
    # Perform similarity search
    query_vector = np.random.random(384).astype(np.float32)
    results = faiss_search.similarity_search(query_vector, 'conversations', top_k=2)
    
    print("Search results:", results)
    
    # Get database info
    info = faiss_search.get_database_info('conversations')
    print("Database info:", info)