import os
import sys
import pickle
import subprocess
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Set
import json
from collections import defaultdict, Counter



class rdfMemRecall:
    """
    A comprehensive RDF-based knowledge graph system with vector embeddings for predicates.
    
    This class provides functionality to:
    - Create and manage RDF knowledge graphs using rdflib
    - Embed predicates using vector embeddings for semantic similarity
    - Handle directional vs symmetric relationships based on connection patterns
    - Perform vector similarity searches on predicates
    - Store and retrieve RDF data with vector-enhanced querying
    
    The system uses a mixed approach:
    - RDF triples for structural relationships (subject, predicate, object)
    - Vector embeddings for predicates to enable semantic similarity
    - Directional relationship detection through connection patterns
    """
    
    def __init__(self, data_directory: str = "./rdf_data", 
                 graph_name: str = "knowledge_graph",
                 cuda_device: Optional[int] = None):
        """
        Initialize the RDF Memory Recall system.
        
        Args:
            data_directory (str): Directory path to store RDF and vector data
            graph_name (str): Name for the RDF graph
            cuda_device (Optional[int]): CUDA device for embeddings (None for CPU)
        """
        self.data_directory = data_directory
        self.graph_name = graph_name
        self.cuda_device = cuda_device
        
        # Create data directory
        os.makedirs(self.data_directory, exist_ok=True)
        
        # Check and install dependencies
        self._check_and_install_dependencies()
        
        # Initialize RDF graph
        self.graph = self.rdflib.Graph()
        self.namespace = None # will be rdflib.Namespace(namespace_uri)
        
        # Initialize embedder
        self._initialize_embedder()
        
        # Predicate vector storage
        self.predicate_vectors = {}  # Maps predicate URIs to vectors
        self.vector_to_predicate = {}  # Maps vector positions to predicate URIs
        
        # FAISS index for predicate similarity search
        self.predicate_index = None
        self.predicate_dimension = None
        self.predicate_similarity_search_threshold = 0.85 #are predicates symanticaly similar threshold
        
        # Load existing data
        self._load_graph_data()
        
        print(f"RDF Memory Recall initialized with graph '{graph_name}'")
        print(f"Graph contains {len(self.graph)} triples")
        print(f"Predicate vectors: {len(self.predicate_vectors)}")
    
    def _check_and_install_dependencies(self) -> None:
        """
        Check if required dependencies are installed and install if missing.
        """
        dependencies = [
            ('rdflib', 'rdflib'),
            ('faiss-cpu', 'faiss'),
            ('numpy', 'numpy')
        ]
        
        for pip_name, import_name in dependencies:
            try:
                if import_name == 'rdflib':
                    import rdflib
                    self.rdflib = rdflib
                elif import_name == 'faiss':
                    import faiss
                    self.faiss = faiss
                elif import_name == 'numpy':
                    import numpy as np
                    self.np = np
            except ImportError:
                print(f"{import_name} not found. Installing {pip_name}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                
                # Re-import after installation
                if import_name == 'rdflib':
                    import rdflib
                    self.rdflib = rdflib
                elif import_name == 'faiss':
                    import faiss
                    self.faiss = faiss
                elif import_name == 'numpy':
                    import numpy as np
                    self.np = np
        
        # Try to import MxBaiEmbedder (assuming it's available)
        try:
            from mxbai_embed import MxBaiEmbedder
            self.MxBaiEmbedder = MxBaiEmbedder
        except ImportError:
            print("Warning: MxBaiEmbedder not found. Please install mxbai_embed package.")
            self.MxBaiEmbedder = None
    
    def check_dependencies(self) -> Dict[str, bool]:
        """
        Check if required dependencies are installed.
        
        Returns:
            Dict[str, bool]: Dictionary with dependency names and installation status
        """
        dependencies = {
            'rdflib': hasattr(self, 'rdflib') and self.rdflib is not None,
            'faiss': hasattr(self, 'faiss') and self.faiss is not None,
            'numpy': hasattr(self, 'np') and self.np is not None,
            'mxbai_embed': hasattr(self, 'MxBaiEmbedder') and self.MxBaiEmbedder is not None
        }
        
        return dependencies
    
    def _initialize_embedder(self) -> None:
        """
        Initialize the MxBai embedder for predicate vectors.
        """
        try:
            if self.MxBaiEmbedder is not None:
                logging.debug("Initializing embedders...")
                self.embedders = self.MxBaiEmbedder(cuda_device=self.cuda_device)
                print("MxBai embedder initialized successfully")
            else:
                print("Warning: MxBai embedder not available")
                self.embedders = None
        except Exception as e:
            print(f"Error initializing embedder: {str(e)}")
            self.embedders = None
    
    def create_database_instance(self, namespace_uri: str = "http://example.org/") -> bool:
        """
        Create a new database instance with specified namespace.
        
        Args:
            namespace_uri (str): Base URI for the namespace
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create namespace
            self.namespace = self.rdflib.Namespace(namespace_uri)
            
            # Bind common prefixes
            #SPARQL queries or create triples, after binding can use rdf:type instead of the full URI
            self.graph.bind("ex", self.namespace) #"ex" → your custom namespace (http://example.org/)
            self.graph.bind("rdf", self.rdflib.RDF) # "rdf" → http://www.w3.org/1999/02/22-rdf-syntax-ns# (core RDF vocabulary)
            self.graph.bind("rdfs", self.rdflib.RDFS) # "rdfs" → http://www.w3.org/2000/01/rdf-schema# (RDF Schema)
            self.graph.bind("owl", self.rdflib.OWL) # "owl" → http://www.w3.org/2002/07/owl# (Web Ontology Language)
            
            print(f"Database instance created with namespace: {namespace_uri}")
            return True
            
        except Exception as e:
            print(f"Error creating database instance: {str(e)}")
            return False
    
    def _embed_predicate(self, predicate: str) -> Optional[List[float]]:
        """
        Create vector embedding for a predicate.
        
        Args:
            predicate (str): Predicate string to embed
            
        Returns:
            Optional[List[float]]: Vector embedding or None if failed
        """
        try:
            if self.embedders is None:
                print("Warning: Embedder not available")
                return None
                
            embedding = self.embedders.embed_string(predicate)
            return embedding 
            
        except Exception as e:
            print(f"Error embedding predicate '{predicate}': {str(e)}")
            return None
    
    def _initialize_predicate_index(self, dimension: int) -> None:
        """
        Initialize FAISS index for predicate similarity search.
        
        Args:
            dimension (int): Vector dimension
        """
        if self.predicate_index is None:
            self.predicate_index = self.faiss.IndexFlatIP(dimension)
            self.predicate_dimension = dimension
            print(f"Initialized predicate FAISS index with dimension {dimension}")
    
    def _clean_uri_component(self, text: str) -> str:
        """
        Clean a text string to be safe for use in URIs.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text safe for URI use
        """
        # Replace spaces and special characters with underscores
        import re
        cleaned = re.sub(r'[^\w\-_]', '_', text)
        # Remove multiple underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        return cleaned if cleaned else 'entity'
    
    def _entity_exists(self, entity_name: str) -> bool:
        """
        Check if an entity already exists in the graph.
        
        Args:
            entity_name (str): Name of the entity to check
            
        Returns:
            bool: True if entity exists, False otherwise
        """
        entity_clean = self._clean_uri_component(entity_name)
        entity_uri = self.rdflib.URIRef(self.namespace[entity_clean])
        
        # Check if entity appears as subject or object in any triple
        exists_as_subject = any(self.graph.triples((entity_uri, None, None)))
        exists_as_object = any(self.graph.triples((None, None, entity_uri)))
        
        #TODO: Name search isn't really sufficient, If for example there are two people named 'john' we would need to look at the family name connection
        #or have some way to determine what 'john' we are talking about.

        return exists_as_subject or exists_as_object
    
    def _has_reverse_connection(self, subject: str, predicate_vector: List[float], 
                               obj: str, similarity_threshold: float = False) -> bool:
        """
        Check if there's a reverse connection with a semantically similar predicate vector.
        Used for auto-detecting bidirectional relationships when predicates might be 
        semantically similar but not identical (e.g., "is family member" vs "family member").
        
        Args:
            subject (str): Subject entity
            predicate_vector (List[float]): Predicate vector to check
            obj (str): Object entity  
            similarity_threshold (float): Minimum cosine similarity for considering vectors similar
            
        Returns:
            bool: True if reverse connection with similar vector exists, False otherwise
        """
        if not similarity_threshold:
            similarity_threshold =  self.predicate_similarity_search_threshold
        try:
            if self.predicate_index is None or self.predicate_index.ntotal == 0:
                return False
            
            # Look for triples where obj is subject and subject is object
            obj_clean = self._clean_uri_component(obj)
            subject_clean = self._clean_uri_component(subject)
            obj_uri = self.rdflib.URIRef(self.namespace[obj_clean])
            subject_uri = self.rdflib.URIRef(self.namespace[subject_clean])
            
            # Find all predicates between obj and subject
            for triple in self.graph.triples((obj_uri, None, subject_uri)):
                predicate_uri = str(triple[1])
                
                if predicate_uri in self.predicate_vectors:
                    stored_vector = self.predicate_vectors[predicate_uri]
                    
                    # Calculate cosine similarity between vectors
                    vec1 = self.np.array(predicate_vector)
                    vec2 = self.np.array(stored_vector)
                    
                    # Normalize vectors
                    vec1_norm = vec1 / self.np.linalg.norm(vec1)
                    vec2_norm = vec2 / self.np.linalg.norm(vec2)
                    
                    # Cosine similarity
                    similarity = self.np.dot(vec1_norm, vec2_norm)
                    
                    if similarity >= similarity_threshold:
                        print(f"Found semantically similar reverse connection (similarity: {similarity:.3f})")
                        return True
            
            return False
            
        except Exception as e:
            print(f"Error checking reverse connection: {str(e)}")
            return False
    
    def add_connection(self, connection_data: Union[str, Dict[str, Any]]) -> bool:
        """
        Add a connection to the knowledge graph with predicate consolidation.
        
        Args:
            connection_data: JSON string or dict with format:
                {
                    "subject": "entity_name",
                    "predicate": "relationship_type", 
                    "object": "entity_name",
                    "directional": true/false (optional)
                }
                
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Parse input
            if isinstance(connection_data, str):
                data = json.loads(connection_data)
            else:
                data = connection_data
            
            subject = data.get("subject")
            predicate = data.get("predicate")
            obj = data.get("object")
            directional = data.get("directional", None)
            
            if not all([subject, predicate, obj]):
                print("Error: Missing required fields (subject, predicate, object)")
                return False
            
            # Find or create predicate
            predicate_uri, predicate_vector = self._get_or_create_predicate(predicate)
            if predicate_uri is None:
                return False
            
            # Create entity URIs
            subject_uri = self._create_entity_uri(subject)
            object_uri = self._create_entity_uri(obj)
            
            # Auto-detect directionality if not specified
            if directional is None:
                directional = self._has_reverse_connection(subject, predicate_vector, obj)
                print(f"Auto-detected as {'bidirectional' if directional else 'unidirectional'}")
            
            # Add main triple
            self.graph.add((subject_uri, predicate_uri, object_uri))
            
            # Add reverse connection for bidirectional relationships
            if directional:
                reverse_uri, _ = self._get_or_create_predicate(f"{predicate}_reverse")
                self.graph.add((object_uri, reverse_uri, subject_uri))
                print(f"Added bidirectional: {subject} ↔ {obj}")
            else:
                print(f"Added unidirectional: {subject} → {obj}")
            
            return True
            
        except Exception as e:
            print(f"Error adding connection: {str(e)}")
            return False

    def _get_or_create_predicate(self, predicate_text: str) -> Tuple[Optional[Any], Optional[List[float]]]:
        """
        Get existing similar predicate or create new one.
        
        Args:
            predicate_text (str): Predicate text to find or create
            
        Returns:
            Tuple[URIRef, List[float]]: Predicate URI and vector, or (None, None) if failed
        """
        # Search for similar existing predicates
        if self.predicate_vectors:
            similar = self.search_similar_predicates(predicate_text, top_k=1)
            if similar and similar[0][1] >= self.predicate_similarity_search_threshold:
                existing_uri = similar[0][0]
                existing_vector = self.predicate_vectors[existing_uri]
                print(f"Using existing predicate '{existing_uri}' (similarity: {similar[0][1]:.3f})")
                return self.rdflib.URIRef(existing_uri), existing_vector
        
        # Create new predicate
        vector = self._embed_predicate(predicate_text)
        if vector is None:
            print(f"Failed to embed predicate: {predicate_text}")
            return None, None
        
        # Initialize FAISS index if needed
        if self.predicate_index is None:
            self._initialize_predicate_index(len(vector))
        
        # Create URI and store
        clean_text = self._clean_uri_component(predicate_text)
        uri = self.rdflib.URIRef(self.namespace[clean_text])
        
        self.predicate_vectors[str(uri)] = vector
        self._add_to_faiss_index(vector, str(uri))
        
        print(f"Created new predicate: {clean_text}")
        return uri, vector

    def _create_entity_uri(self, entity_name: str) -> Any:
        """Create URI for entity."""
        clean_name = self._clean_uri_component(entity_name)
        return self.rdflib.URIRef(self.namespace[clean_name])

    def _add_to_faiss_index(self, vector: List[float], uri: str) -> None:
        """Add vector to FAISS index."""
        vector_array = self.np.array(vector).reshape(1, -1).astype(self.np.float32)
        vector_norm = vector_array / self.np.linalg.norm(vector_array)
        self.predicate_index.add(vector_norm)
        
        position = self.predicate_index.ntotal - 1
        self.vector_to_predicate[position] = uri

    def consolidate_existing_predicates(self) -> Dict[str, Any]:
        """
        Clean up existing graph by consolidating similar predicates.
        
        Returns:
            Dict[str, Any]: Consolidation results
        """
        results = {
            'predicates_before': len(self.predicate_vectors),
            'triples_updated': 0,
            'groups_merged': 0
        }
        
        # Find similar predicate groups
        processed = set()
        
        for pred_uri in list(self.predicate_vectors.keys()):
            if pred_uri in processed:
                continue
                
            # Find similar predicates
            pred_text = pred_uri.split('/')[-1].replace('_', ' ')
            similar = self.search_similar_predicates(pred_text, top_k=10)
            
            similar_group = [pred_uri]
            for uri, similarity in similar:
                if uri != pred_uri and similarity >= self.predicate_similarity_search_threshold:
                    similar_group.append(uri)
            
            if len(similar_group) > 1:
                # Use first predicate as canonical
                canonical = similar_group[0]
                canonical_uri = self.rdflib.URIRef(canonical)
                
                # Update all triples
                for uri in similar_group[1:]:
                    old_uri = self.rdflib.URIRef(uri)
                    triples_to_update = list(self.graph.triples((None, old_uri, None)))
                    
                    for s, p, o in triples_to_update:
                        self.graph.remove((s, p, o))
                        self.graph.add((s, canonical_uri, o))
                        results['triples_updated'] += 1
                    
                    # Remove old predicate
                    if uri in self.predicate_vectors:
                        del self.predicate_vectors[uri]
                    
                    processed.add(uri)
                
                results['groups_merged'] += 1
            
            processed.add(pred_uri)
        
        results['predicates_after'] = len(self.predicate_vectors)
        
        # Rebuild FAISS index
        if self.predicate_vectors:
            self._rebuild_predicate_index()
        
        print(f"Consolidated {results['groups_merged']} predicate groups")
        print(f"Updated {results['triples_updated']} triples")
        
        return results

    def search_similar_predicates(self, query_predicate: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find predicates similar to the query predicate.
        
        Args:
            query_predicate (str): Predicate to search for
            top_k (int): Number of similar predicates to return
            
        Returns:
            List[Tuple[str, float]]: List of (predicate_uri, similarity_score) tuples
        """
        try:
            if self.predicate_index is None or self.predicate_index.ntotal == 0:
                print("No predicate index available")
                return []
            
            # Embed query predicate
            query_vector = self._embed_predicate(query_predicate)
            if query_vector is None:
                return []
            
            # Normalize query vector
            query_array = self.np.array(query_vector).reshape(1, -1).astype(self.np.float32)
            query_norm = query_array / self.np.linalg.norm(query_array)
            
            # Search for similar vectors
            similarities, indices = self.predicate_index.search(query_norm, min(top_k, self.predicate_index.ntotal))
            
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx != -1 and idx in self.vector_to_predicate:
                    predicate_uri = self.vector_to_predicate[idx]
                    results.append((predicate_uri, float(similarity)))
            
            return results
            
        except Exception as e:
            print(f"Error searching similar predicates: {str(e)}")
            return []
    
    def query_graph(self, query_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph with flexible search patterns and semantic predicate matching.
        
        Args:
            query_dict (Dict[str, Any]): Query parameters with keys:
                - subject (str, optional): Subject entity name
                - predicate (str, optional): Predicate/relationship type (uses semantic matching)
                - object (str, optional): Object entity name
                
        Four query patterns:
        1. Subject only: {'subject': 'john'} 
        → Returns all connections from john (one level out)
        2. Predicate only: {'predicate': 'similar historical event'}
        → Returns all connections using semantically similar predicates
        3. Subject + Predicate: {'subject': 'john', 'predicate': 'children'} 
        → Returns all objects connected to john via 'children' relationship
        4. Full Triple: {'subject': 'john', 'predicate': 'children', 'object': 'sarah'} 
        → Checks if specific connection exists
        
        Returns:
            List[Dict[str, Any]]: List of result dictionaries with keys:
                - subject: Subject entity (cleaned name)
                - predicate: Predicate (cleaned name) 
                - object: Object entity (cleaned name)
                - subject_uri: Full subject URI
                - predicate_uri: Full predicate URI
                - object_uri: Full object URI
                - similarity_score: Predicate semantic similarity score (if applicable)
        """
        try:
            subject = query_dict.get('subject')
            predicate = query_dict.get('predicate') 
            obj = query_dict.get('object')
            
            # Clean and validate inputs
            if not subject and not predicate:
                print("Error: Either subject or predicate is required")
                return []
            
            results = []
            
            # Handle subject-based queries
            if subject:
                # Create subject URI
                subject_clean = self._clean_uri_component(subject)
                subject_uri = self.rdflib.URIRef(self.namespace[subject_clean])
                
                # Check if subject exists in graph
                if not any(self.graph.triples((subject_uri, None, None))):
                    print(f"Subject '{subject}' not found in graph")
                    return []
            
            # Pattern 1: Subject only - return all connections (one level out)
            if subject and not predicate and not obj:
                print(f"Finding all connections for subject '{subject}'")
                
                for s, p, o in self.graph.triples((subject_uri, None, None)):
                    result = {
                        'subject': self._extract_local_name(str(s)),
                        'predicate': self._extract_local_name(str(p)),
                        'object': self._extract_local_name(str(o)),
                        'subject_uri': str(s),
                        'predicate_uri': str(p),
                        'object_uri': str(o),
                        'similarity_score': 1.0  # Exact match
                    }
                    results.append(result)
                
                return results
            
            # Pattern 2: Predicate only - find all connections with semantically similar predicates
            if predicate and not subject and not obj:
                print(f"Finding all connections using predicate '{predicate}' or semantically similar")
                
                # Find semantically similar predicates
                similar_predicates = self.search_similar_predicates(predicate, top_k=10)
                
                if not similar_predicates:
                    print(f"No similar predicates found for '{predicate}'")
                    return []
                
                print(f"Found {len(similar_predicates)} similar predicates for '{predicate}':")
                for pred_uri, similarity in similar_predicates[:3]:  # Show top 3
                    pred_clean = self._extract_local_name(pred_uri)
                    print(f"  - {pred_clean} (similarity: {similarity:.3f})")
                
                # Find all connections using these predicates
                for pred_uri, similarity in similar_predicates:
                    predicate_uri_ref = self.rdflib.URIRef(pred_uri)
                    
                    # Find all triples with this predicate
                    for s, p, o in self.graph.triples((None, predicate_uri_ref, None)):
                        result = {
                            'subject': self._extract_local_name(str(s)),
                            'predicate': self._extract_local_name(str(p)),
                            'object': self._extract_local_name(str(o)),
                            'subject_uri': str(s),
                            'predicate_uri': str(p),
                            'object_uri': str(o),
                            'similarity_score': similarity
                        }
                        results.append(result)
                
                return results
            
            # Pattern 3 & 4: Use semantic predicate matching with subject
            if predicate and subject:
                # Find semantically similar predicates
                similar_predicates = self.search_similar_predicates(predicate, top_k=10)
                
                if not similar_predicates:
                    print(f"No similar predicates found for '{predicate}'")
                    return []
                
                print(f"Found {len(similar_predicates)} similar predicates for '{predicate}':")
                for pred_uri, similarity in similar_predicates[:3]:  # Show top 3
                    pred_clean = self._extract_local_name(pred_uri)
                    print(f"  - {pred_clean} (similarity: {similarity:.3f})")
                
                # Pattern 4: Full triple check - subject + predicate + object
                if obj:
                    print(f"Checking specific connection: '{subject}' -[{predicate}]-> '{obj}'")
                    
                    obj_clean = self._clean_uri_component(obj)
                    obj_uri = self.rdflib.URIRef(self.namespace[obj_clean])
                    
                    # Check each similar predicate
                    for pred_uri, similarity in similar_predicates:
                        predicate_uri_ref = self.rdflib.URIRef(pred_uri)
                        
                        # Check if the specific triple exists
                        if (subject_uri, predicate_uri_ref, obj_uri) in self.graph:
                            result = {
                                'subject': self._extract_local_name(str(subject_uri)),
                                'predicate': self._extract_local_name(pred_uri),
                                'object': self._extract_local_name(str(obj_uri)),
                                'subject_uri': str(subject_uri),
                                'predicate_uri': pred_uri,
                                'object_uri': str(obj_uri),
                                'similarity_score': similarity
                            }
                            results.append(result)
                    
                    return results
                
                # Pattern 3: Subject + predicate - find all connected objects
                else:
                    print(f"Finding all objects connected to '{subject}' via '{predicate}'")
                    
                    # Search through similar predicates
                    for pred_uri, similarity in similar_predicates:
                        predicate_uri_ref = self.rdflib.URIRef(pred_uri)
                        
                        # Find all objects connected via this predicate
                        for s, p, o in self.graph.triples((subject_uri, predicate_uri_ref, None)):
                            result = {
                                'subject': self._extract_local_name(str(s)),
                                'predicate': self._extract_local_name(str(p)),
                                'object': self._extract_local_name(str(o)),
                                'subject_uri': str(s),
                                'predicate_uri': str(p),
                                'object_uri': str(o),
                                'similarity_score': similarity
                            }
                            results.append(result)
                    
                    return results
            
            return results
            
        except Exception as e:
            print(f"Error in query_graph: {str(e)}")
            return []
    
    def _extract_local_name(self, uri: str) -> str:
        """
        Extract the local name from a URI for display purposes.
        
        Args:
            uri (str): Full URI
            
        Returns:
            str: Local name (cleaned)
        """
        try:
            if '/' in uri:
                local_name = uri.split('/')[-1]
            elif '#' in uri:
                local_name = uri.split('#')[-1]
            else:
                local_name = uri
            
            # Convert underscores to spaces for readability
            return local_name.replace('_', ' ')
        
        except Exception:
            return uri
    
    def get_all_triples(self) -> List[Tuple[str, str, str]]:
        """
        Get all triples in the graph.
        
        Returns:
            List[Tuple[str, str, str]]: List of (subject, predicate, object) tuples
        """
        return [(str(s), str(p), str(o)) for s, p, o in self.graph]
    
    def save_graph_data(self) -> bool:
        """
        Save RDF graph and vector data to files.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Save RDF graph with better error handling
            graph_file = os.path.join(self.data_directory, f"{self.graph_name}.ttl")
            
            # Try to serialize with different formats if turtle fails
            try:
                self.graph.serialize(destination=graph_file, format="turtle")
                print(f"Graph saved in Turtle format: {graph_file}")
            except Exception as turtle_error:
                print(f"Turtle serialization failed: {turtle_error}")
                # Try N-Triples format as fallback
                graph_file_nt = os.path.join(self.data_directory, f"{self.graph_name}.nt")
                self.graph.serialize(destination=graph_file_nt, format="nt")
                print(f"Graph saved in N-Triples format: {graph_file_nt}")
            
            # Save predicate vectors
            vectors_file = os.path.join(self.data_directory, f"{self.graph_name}_vectors.pkl")
            vector_data = {
                'predicate_vectors': self.predicate_vectors,
                'vector_to_predicate': self.vector_to_predicate,
                'predicate_dimension': self.predicate_dimension
            }
            
            with open(vectors_file, 'wb') as f:
                pickle.dump(vector_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"Vector data saved to {vectors_file}")
            return True
            
        except Exception as e:
            print(f"Error saving graph data: {str(e)}")
            return False
    
    def _load_graph_data(self) -> None:
        """
        Load RDF graph and vector data from files.
        """
        try:
            # Load RDF graph with multiple format support
            graph_file_ttl = os.path.join(self.data_directory, f"{self.graph_name}.ttl")
            graph_file_nt = os.path.join(self.data_directory, f"{self.graph_name}.nt")
            
            if os.path.exists(graph_file_ttl):
                try:
                    self.graph.parse(graph_file_ttl, format="turtle")
                    print(f"Loaded RDF graph from {graph_file_ttl}")
                except Exception as turtle_error:
                    print(f"Error parsing Turtle format: {turtle_error}")
                    if os.path.exists(graph_file_nt):
                        self.graph.parse(graph_file_nt, format="nt")
                        print(f"Loaded RDF graph from {graph_file_nt}")
            elif os.path.exists(graph_file_nt):
                self.graph.parse(graph_file_nt, format="nt")
                print(f"Loaded RDF graph from {graph_file_nt}")
            
            # Load predicate vectors
            vectors_file = os.path.join(self.data_directory, f"{self.graph_name}_vectors.pkl")
            if os.path.exists(vectors_file):
                with open(vectors_file, 'rb') as f:
                    vector_data = pickle.load(f)
                
                self.predicate_vectors = vector_data.get('predicate_vectors', {})
                self.vector_to_predicate = vector_data.get('vector_to_predicate', {})
                self.predicate_dimension = vector_data.get('predicate_dimension', None)
                
                # Rebuild FAISS index
                if self.predicate_vectors and self.predicate_dimension:
                    self._rebuild_predicate_index()
                
                print(f"Loaded vector data from {vectors_file}")
            
        except Exception as e:
            print(f"Error loading graph data: {str(e)}")
    
    def _rebuild_predicate_index(self) -> None:
        """
        Rebuild FAISS index from loaded predicate vectors.
        """
        try:
            if not self.predicate_vectors or self.predicate_dimension is None:
                return
            
            self._initialize_predicate_index(self.predicate_dimension)
            
            vectors = []
            position_mapping = {}
            
            for i, (predicate_uri, vector) in enumerate(self.predicate_vectors.items()):
                vector_array = self.np.array(vector).astype(self.np.float32)
                vector_norm = vector_array / self.np.linalg.norm(vector_array)
                vectors.append(vector_norm)
                position_mapping[i] = predicate_uri
            
            if vectors:
                vectors_array = self.np.vstack(vectors)
                self.predicate_index.add(vectors_array)
                self.vector_to_predicate = position_mapping
                print(f"Rebuilt predicate index with {len(vectors)} vectors")
            
        except Exception as e:
            print(f"Error rebuilding predicate index: {str(e)}")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dict[str, Any]: Graph statistics
        """
        stats = {
            'total_triples': len(self.graph),
            'total_predicates': len(self.predicate_vectors),
            'unique_subjects': len(set(str(s) for s, p, o in self.graph)),
            'unique_objects': len(set(str(o) for s, p, o in self.graph)),
            'vector_dimension': self.predicate_dimension,
            'faiss_index_size': self.predicate_index.ntotal if self.predicate_index else 0
        }
        
        return stats
    
    def clear_graph(self) -> bool:
        """
        Clear all data from the graph.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.graph = self.rdflib.Graph()
            self.predicate_vectors = {}
            self.vector_to_predicate = {}
            self.predicate_index = None
            self.predicate_dimension = None
            
            print("Graph cleared successfully")
            return True
            
        except Exception as e:
            print(f"Error clearing graph: {str(e)}")
            return False


# Demo and testing section
if __name__ == "__main__":
    print("="*60)
    print("RDF Memory Recall Demo")
    print("="*60)
    
    # Initialize the system
    print("\n1. Initializing RDF Memory Recall...")
    rdf_memory = rdfMemRecall(data_directory="./demo_rdf_data", graph_name="demo_graph")
    
    # Create database instance
    print("\n2. Creating database instance...")
    rdf_memory.create_database_instance("http://demo.example.org/")
    
    # Check dependencies
    print("\n3. Checking dependencies...")
    deps = rdf_memory.check_dependencies()
    for dep, status in deps.items():
        print(f"   {dep}: {'✓' if status else '✗'}")
    
    print("\n4. Adding connections to demonstrate different relationship types...")
    
    # Example 1: Explicit bidirectional relationship
    print("\n--- Example 1: Explicit Bidirectional Relationship ---")
    connection1 = {
        "subject": "Tiananmen_Protest_1989",
        "predicate": "similar_historical_event",
        "object": "Michigan_Protest_1965",
        "directional": True
    }
    result1 = rdf_memory.add_connection(connection1)
    print(f"Connection added: {result1}")
    
    # Example 2: Explicit unidirectional relationship (hierarchical)
    print("\n--- Example 2: Explicit Unidirectional Relationship ---")
    connection2 = {
        "subject": "Apple_Inc",
        "predicate": "owns",
        "object": "iPhone_Division",
        "directional": False
    }
    result2 = rdf_memory.add_connection(connection2)
    print(f"Connection added: {result2}")
    
    # Example 3: Another hierarchical relationship
    print("\n--- Example 3: Parent-Child Relationship ---")
    connection3 = {
        "subject": "John_Smith",
        "predicate": "parent_of",
        "object": "Sarah_Smith",
        "directional": False
    }
    result3 = rdf_memory.add_connection(connection3)
    print(f"Connection added: {result3}")
    
    # Example 4: Auto-detect - first connection (should be unidirectional)
    print("\n--- Example 4: Auto-detect - First Family Connection ---")
    connection4 = {
        "subject": "Bob Johnson",
        "predicate": "is family member of",
        "object": "Alice Johnson"
        # No directional field - will auto-detect
    }
    result4 = rdf_memory.add_connection(connection4)
    print(f"Connection added: {result4}")
    
    # Example 5: Auto-detect - reverse connection (should become bidirectional)
    print("\n--- Example 5: Auto-detect - Reverse Family Connection ---")
    connection5 = {
        "subject": "Alice_Johnson",
        "predicate": "family member",  # Semantically similar to "is family member of"
        "object": "Bob Johnson"
        # No directional field - should detect reverse and become bidirectional
    }
    result5 = rdf_memory.add_connection(connection5)
    print(f"Connection added: {result5}")
    
    # Example 6: Work relationship
    print("\n--- Example 6: Work Relationship ---")
    connection6 = {
        "subject": "Microsoft",
        "predicate": "employs",
        "object": "Jane Developer",
        "directional": False
    }
    result6 = rdf_memory.add_connection(connection6)
    print(f"Connection added: {result6}")
    
    # Example 7: Temporal connection (events)
    print("\n--- Example 7: Temporal Event Connection ---")
    connection7 = {
        "subject": "World War 2",
        "predicate": "preceded by",
        "object": "World War 1",
        "directional": False  # Temporal relationships are typically unidirectional
    }
    result7 = rdf_memory.add_connection(connection7)
    print(f"Connection added: {result7}")
    
    print("\n5. Graph Statistics:")
    print("="*40)
    stats = rdf_memory.get_graph_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n6. All Triples in Graph:")
    print("="*40)
    
    all_triples = rdf_memory.get_all_triples()

    for i, (s, p, o) in enumerate(all_triples, 1):
        # Clean up the URIs for display
        s_clean = s.split('/')[-1] if '/' in s else s
        p_clean = p.split('/')[-1] if '/' in p else p  
        o_clean = o.split('/')[-1] if '/' in o else o
        print(f"   {i:2d}. {s_clean} --[{p_clean}]--> {o_clean}")
    
    print("\n7. Testing Predicate Similarity Search:")
    print("="*40)
    if rdf_memory.embedders:  # Only if embedder is available
        similar_predicates = rdf_memory.search_similar_predicates("family relationship", top_k=3)
        if similar_predicates:
            print("   Similar predicates to 'family relationship':")
            for pred_uri, similarity in similar_predicates:
                pred_clean = pred_uri.split('/')[-1] if '/' in pred_uri else pred_uri
                print(f"   - {pred_clean} (similarity: {similarity:.3f})")
        else:
            print("   No similar predicates found")
    else:
        print("   Embedder not available - skipping similarity search")

    #####################################################################
    """
    Demonstrate the unified query_graph method with all three patterns.
    """
    print("\n" + "="*60)
    print("UNIFIED QUERY_GRAPH METHOD DEMO")
    print("="*60)
    
    # Pattern 1: Subject only
    print("\n" + "="*50)
    print("PATTERN 1: Subject Only - Find all connections")
    print("="*50)
    
    query1 = {'subject': 'John Smith'}  # Note: spaces will be cleaned to underscores
    print(f"Query: {query1}")
    
    results1 = rdf_memory.query_graph(query1)
    print(f"\nResults ({len(results1)} found):")
    
    for i, result in enumerate(results1, 1):
        print(f"  {i}. {result['subject']} -[{result['predicate']}]-> {result['object']}")
        print(f"     (Similarity: {result['similarity_score']:.3f})")
    
    # Pattern 2: Predicate only
    print("\n" + "="*50)
    print("PATTERN 2: Predicate Only - Find all connections with similar predicates")
    print("="*50)
    
    query2 = {'predicate': 'similar historical event'}
    print(f"Query: {query2}")
    
    results2 = rdf_memory.query_graph(query2)
    print(f"\nResults ({len(results2)} found):")
    
    for i, result in enumerate(results2, 1):
        print(f"  {i}. {result['subject']} -[{result['predicate']}]-> {result['object']}")
        print(f"     (Similarity: {result['similarity_score']:.3f})")
    
    # Test with a more common predicate
    query2b = {'predicate': 'parent of'}
    print(f"\nQuery: {query2b}")
    
    results2b = rdf_memory.query_graph(query2b)
    print(f"\nResults ({len(results2b)} found):")
    
    for i, result in enumerate(results2b, 1):
        print(f"  {i}. {result['subject']} -[{result['predicate']}]-> {result['object']}")
        print(f"     (Similarity: {result['similarity_score']:.3f})")
    
    # Pattern 3: Subject + Predicate
    print("\n" + "="*50)
    print("PATTERN 3: Subject + Predicate - Find connected objects")
    print("="*50)
    
    # Test semantic matching with "children" (should match "parent_of" semantically)
    query3 = {'subject': 'John Smith', 'predicate': 'children'}
    print(f"Query: {query3}")
    
    results3 = rdf_memory.query_graph(query3)
    print(f"\nResults ({len(results3)} found):")
    
    for i, result in enumerate(results3, 1):
        print(f"  {i}. {result['subject']} -[{result['predicate']}]-> {result['object']}")
        print(f"     (Similarity: {result['similarity_score']:.3f})")
    
    # Test with exact predicate match
    query3b = {'subject': 'John Smith', 'predicate': 'parent of'}
    print(f"\nQuery: {query3b}")
    
    results3b = rdf_memory.query_graph(query3b)
    print(f"\nResults ({len(results3b)} found):")
    
    for i, result in enumerate(results3b, 1):
        print(f"  {i}. {result['subject']} -[{result['predicate']}]-> {result['object']}")
        print(f"     (Similarity: {result['similarity_score']:.3f})")
    
    # Pattern 4: Full Triple
    print("\n" + "="*50)
    print("PATTERN 4: Full Triple - Check specific connection")
    print("="*50)
    
    query4 = {'subject': 'John Smith', 'predicate': 'children', 'object': 'Sarah Smith'}
    print(f"Query: {query4}")
    
    results4 = rdf_memory.query_graph(query4)
    print(f"\nResults ({len(results4)} found):")
    
    if results4:
        for i, result in enumerate(results4, 1):
            print(f"  {i}. Connection exists: {result['subject']} -[{result['predicate']}]-> {result['object']}")
            print(f"     (Similarity: {result['similarity_score']:.3f})")
    else:
        print("  No matching connection found")
    
    # Test non-existent connection
    query4b = {'subject': 'John Smith', 'predicate': 'children', 'object': 'Random Person'}
    print(f"\nQuery: {query4b}")
    
    results4b = rdf_memory.query_graph(query4b)
    print(f"\nResults ({len(results4b)} found):")
    
    if results4b:
        for result in results4b:
            print(f"  Connection exists: {result['subject']} -[{result['predicate']}]-> {result['object']}")
    else:
        print("  No matching connection found (as expected)")
    
    # Test directional behavior
    print("\n" + "="*50)
    print("DIRECTIONAL BEHAVIOR TEST")
    print("="*50)
    
    # This should NOT return Jane -> children -> John
    # because we only search in the specified direction
    query_direction = {'subject': 'Sarah Smith', 'predicate': 'children'}
    print(f"Query: {query_direction}")
    print("(This should NOT return John Smith as Sarah's child)")
    
    results_direction = rdf_memory.query_graph(query_direction)
    print(f"\nResults ({len(results_direction)} found):")
    
    if results_direction:
        for result in results_direction:
            print(f"  {result['subject']} -[{result['predicate']}]-> {result['object']}")
    else:
        print("  No results (correct - Sarah is not a parent in our data)")
    
    print("\n" + "="*60)
    print("UNIFIED QUERY DEMO COMPLETED")
    print("="*60)


    ######################################################

    print("\n9. Saving Graph Data:")
    print("="*40)
    save_success = rdf_memory.save_graph_data()
    print(f"   Graph data saved: {save_success}")
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)
    
    # Optional: Test loading in a new instance
    print("\n10. Testing Data Persistence:")
    print("="*40)
    print("   Creating new instance and loading saved data...")
    new_instance = rdfMemRecall(data_directory="./demo_rdf_data", graph_name="demo_graph")
    new_stats = new_instance.get_graph_statistics()
    print(f"   Loaded graph has {new_stats['total_triples']} triples")
    print(f"   Loaded graph has {new_stats['total_predicates']} predicates")
    
    if new_stats['total_triples'] > 0:
        print("   ✓ Data persistence working correctly!")
    else:
        print("   ✗ Data persistence issue detected")
    


    