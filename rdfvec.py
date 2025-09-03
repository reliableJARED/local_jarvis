import os
import sys
import pickle
import subprocess
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import json
from collections import defaultdict, Counter
from datetime import datetime



class rdfMemRecall:
    """
    A comprehensive RDF-based knowledge graph system with vector embeddings for predicates
    and temporal validity tracking.
    
    This class provides functionality to:
    - Create and manage RDF knowledge graphs using rdflib
    - Embed predicates using vector embeddings for semantic similarity
    - Handle directional vs symmetric relationships based on connection patterns
    - Track temporal validity of relationships with 'valid_from' and 'valid_to' timestamps
    - Perform vector similarity searches on predicates
    - Store and retrieve RDF data with vector-enhanced querying
    - Query relationships at specific points in time or across time ranges
    
    The system uses a mixed approach:
    - RDF triples for structural relationships (subject, predicate, object)
    - Vector embeddings for predicates to enable semantic similarity
    - Directional relationship detection through connection patterns
    - Temporal validity periods for each relationship
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
        self.predicate_similarity_search_threshold = 0.7 #are predicates symanticaly similar threshold
        
        # Temporal relationship tracking
        self.temporal_relationships = {}  # Maps relationship_id to temporal metadata
        self.relationship_counter = 0  # Counter for unique relationship IDs
        
        # Load existing data
        self._load_graph_data()
        
        print(f"RDF Memory Recall initialized with graph '{graph_name}'")
        print(f"Graph contains {len(self.graph)} triples")
        print(f"Predicate vectors: {len(self.predicate_vectors)}")
        print(f"Temporal relationships: {len(self.temporal_relationships)}")
    
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
            self.graph.bind("time", self.rdflib.Namespace("http://www.w3.org/2006/time#")) # time ontology
            
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
            
            # Normalize predicate text for consistent embedding
            #example "works_at", "works at", and "works-at" all get embedded as "works at",
            normalized_predicate = predicate.replace('_', ' ').replace('-', ' ').strip()
            # Could also handle other cases like multiple spaces
            normalized_predicate = ' '.join(normalized_predicate.split())  # Collapse multiple spaces
            
            embedding = self.embedders.embed_string(normalized_predicate)
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
        FIXED: Clean a text string to be safe for use in URIs.
        Now maintains consistent cleaning across all operations.
        """
        import re
        # Normalize spaces first
        text = text.strip()
        # Replace spaces with underscores, but be consistent
        cleaned = re.sub(r'[^\w\-]', '_', text)
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
    
    def _parse_timestamp(self, timestamp: Union[str, datetime, None]) -> Optional[datetime]:
        """
        Parse timestamp into datetime object.
        
        Args:
            timestamp: String, datetime object, or None
            
        Returns:
            Optional[datetime]: Parsed datetime or None
        """
        if timestamp is None:
            return None
        
        if isinstance(timestamp, datetime):
            return timestamp
            
        if isinstance(timestamp, str):
            try:
                # Try various formats
                formats = [
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%f",
                    "%Y-%m-%dT%H:%M:%SZ"
                ]
                
                for fmt in formats:
                    try:
                        return datetime.strptime(timestamp, fmt)
                    except ValueError:
                        continue
                
                print(f"Could not parse timestamp: {timestamp}")
                return None
                
            except Exception as e:
                print(f"Error parsing timestamp {timestamp}: {str(e)}")
                return None
        
        return None
    
    def _get_current_timestamp(self) -> datetime:
        """Get current timestamp."""
        return datetime.now()
    
    def _invalidate_conflicting_relationships(self, subject: str, predicate: str, obj: str, 
                                       valid_from: datetime, 
                                       predicate_vector: List[float]) -> List[str]:
        """
        FIXED: Find and invalidate relationships that conflict with the new one.
        This now properly removes RDF triples and updates temporal intervals.
        """
        invalidated = []
        
        try:
            similar_predicates = self.search_similar_predicates(predicate, top_k=10)
            subject_clean = self._clean_uri_component(subject)
            subject_uri = str(self.rdflib.URIRef(self.namespace[subject_clean]))
            
            for rel_id, temporal_data in self.temporal_relationships.items():
                if temporal_data.get('valid_to') is not None:
                    continue
                
                if temporal_data['subject_uri'] == subject_uri:
                    rel_predicate_uri = temporal_data['predicate_uri']
                    
                    for similar_pred_uri, similarity in similar_predicates:
                        if similar_pred_uri == rel_predicate_uri and similarity >= self.predicate_similarity_search_threshold:
                            # CRITICAL FIX: Actually invalidate the RDF triples
                            old_subject_uri = self.rdflib.URIRef(temporal_data['subject_uri'])
                            old_predicate_uri = self.rdflib.URIRef(temporal_data['predicate_uri'])
                            old_object_uri = self.rdflib.URIRef(temporal_data['object_uri'])
                            
                            # Remove the main triple
                            self.graph.remove((old_subject_uri, old_predicate_uri, old_object_uri))
                            
                            # Update temporal metadata
                            temporal_data['valid_to'] = valid_from
                            temporal_data['invalidated_by'] = f"New relationship: {subject} -> {predicate} -> {obj}"
                            
                            # Update RDF temporal information
                            time_ns = self.rdflib.Namespace("http://www.w3.org/2006/time#")
                            temporal_uri = self.rdflib.URIRef(self.namespace[f"temporal_{rel_id}"])
                            
                            # Remove old hasEnd and add new one
                            for triple in list(self.graph.triples((temporal_uri, time_ns.hasEnd, None))):
                                self.graph.remove(triple)
                            
                            self.graph.add((temporal_uri, time_ns.hasEnd, 
                                        self.rdflib.Literal(valid_from.isoformat(), 
                                                            datatype=self.rdflib.XSD.dateTime)))
                            
                            invalidated.append(rel_id)
                            print(f"Invalidated and removed conflicting relationship {rel_id} (similarity: {similarity:.3f})")
                            break
            
            return invalidated
            
        except Exception as e:
            print(f"Error invalidating conflicting relationships: {str(e)}")
            return []
    
    def _check_for_duplicates(self, subject: str, predicate: str, obj: str, 
                        valid_from_dt: datetime, valid_to_dt: datetime = None) -> Dict[str, Any]:
        """
        FIXED: Check if a connection with the same subject, predicate, object already exists.
        Now properly matches URIs and checks temporal relationships correctly.
        
        Args:
            subject: Subject entity name
            predicate: Predicate/relationship type
            obj: Object entity name  
            valid_from_dt: Start of validity period
            valid_to_dt: End of validity period (None for open-ended)
            
        Returns:
            Dict with duplicate relationship details if found, None otherwise
        """
        try:
            # FIXED: Create clean URIs to match what will be created
            subject_clean = self._clean_uri_component(subject)
            obj_clean = self._clean_uri_component(obj)
            subject_uri = str(self.rdflib.URIRef(self.namespace[subject_clean]))
            object_uri = str(self.rdflib.URIRef(self.namespace[obj_clean]))
            
            # FIXED: Check temporal relationships directly using URIs
            for rel_id, rel_data in self.temporal_relationships.items():
                # Match by URIs (which is what gets stored)
                if (rel_data['subject_uri'] == subject_uri and 
                    rel_data['object_uri'] == object_uri):
                    
                    # FIXED: Check predicate similarity properly
                    rel_predicate_text = rel_data['predicate']
                    
                    # Check if predicates are exactly the same or semantically similar
                    is_same_predicate = False
                    
                    # First check exact text match
                    if rel_predicate_text.lower() == predicate.lower():
                        is_same_predicate = True
                    else:
                        # Check semantic similarity if embedder available
                        if self.embedders and rel_data['predicate_uri'] in self.predicate_vectors:
                            similar_predicates = self.search_similar_predicates(predicate, top_k=10)
                            for pred_uri, similarity in similar_predicates:
                                if (pred_uri == rel_data['predicate_uri'] and 
                                    similarity >= self.predicate_similarity_search_threshold):
                                    is_same_predicate = True
                                    break
                    
                    if is_same_predicate:
                        # FIXED: Check temporal overlap properly
                        existing_valid_from = rel_data['valid_from']
                        existing_valid_to = rel_data.get('valid_to')
                        
                        # Check for exact temporal match first
                        from_match = (existing_valid_from == valid_from_dt)
                        to_match = (existing_valid_to == valid_to_dt)
                        
                        if from_match and to_match:
                            return {
                                'relationship_id': rel_id,
                                'valid_from': existing_valid_from,
                                'valid_to': existing_valid_to,
                                'is_currently_valid': rel_data.get('valid_to') is None or rel_data['valid_to'] > self._get_current_timestamp(),
                                'match_type': 'exact_temporal_match'
                            }
                        
                        # FIXED: Check for overlapping periods (optional - can be enabled)
                        if self._temporal_periods_overlap(valid_from_dt, valid_to_dt, 
                                                        existing_valid_from, existing_valid_to):
                            # Decide policy: prevent overlaps or just warn
                            # For now, let's prevent overlapping identical relationships
                            return {
                                'relationship_id': rel_id,
                                'valid_from': existing_valid_from,
                                'valid_to': existing_valid_to,
                                'is_currently_valid': rel_data.get('valid_to') is None or rel_data['valid_to'] > self._get_current_timestamp(),
                                'match_type': 'overlapping_temporal_match'
                            }
            
            return None
            
        except Exception as e:
            print(f"Error checking for duplicates: {str(e)}")
            return None

    def _temporal_periods_overlap(self, start1: datetime, end1: datetime, 
                                start2: datetime, end2: datetime) -> bool:
        """
        Check if two temporal periods overlap.
        
        Args:
            start1, end1: First period (end1 can be None for open-ended)
            start2, end2: Second period (end2 can be None for open-ended)
            
        Returns:
            bool: True if periods overlap, False otherwise
        """
        try:
            # Convert string dates to datetime objects if needed
            if isinstance(start2, str):
                start2 = self._parse_timestamp(start2)
            if isinstance(end2, str):
                end2 = self._parse_timestamp(end2)
            
            # Handle open-ended periods (None end dates)
            if end1 is None and end2 is None:
                return True  # Both are open-ended, they overlap
            if end1 is None:
                return start1 <= (end2 if end2 else datetime.max)
            if end2 is None:
                return start2 <= end1
            
            # Check for overlap: periods overlap if start1 <= end2 and start2 <= end1
            return start1 <= end2 and start2 <= end1
            
        except Exception as e:
            print(f"Error checking temporal overlap: {str(e)}")
            return False
        
    def add_connection(self, connection_data: Union[str, Dict[str, Any]]) -> bool:
        """
        FIXED: Add a connection to the knowledge graph with proper duplicate detection.
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
            valid_from = data.get("valid_from")
            valid_to = data.get("valid_to")
            auto_invalidate = data.get("auto_invalidate_conflicts", True)
            allow_duplicates = data.get("allow_duplicates", False)
            
            if not all([subject, predicate, obj]):
                print("Error: Missing required fields (subject, predicate, object)")
                return False
            
            # Parse timestamps
            valid_from_dt = self._parse_timestamp(valid_from) or self._get_current_timestamp()
            valid_to_dt = self._parse_timestamp(valid_to)
            
            # Validate timestamp logic
            if valid_to_dt and valid_from_dt >= valid_to_dt:
                print("Error: valid_from must be before valid_to")
                return False
            
            # FIXED: Check for duplicates unless explicitly allowed
            if not allow_duplicates:
                duplicate_check = self._check_for_duplicates(subject, predicate, obj, valid_from_dt, valid_to_dt)
                if duplicate_check:
                    print(f"Duplicate connection found: {subject} -> {predicate} -> {obj}")
                    print(f"Existing relationship ID: {duplicate_check['relationship_id']}")
                    print(f"Existing valid period: {duplicate_check['valid_from']} to {duplicate_check['valid_to'] or 'indefinite'}")
                    print(f"Match type: {duplicate_check.get('match_type', 'unknown')}")
                    return False
            
            # Rest of the method remains the same...
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
            
            # Handle conflicting relationships
            invalidated_relationships = []
            if auto_invalidate:
                invalidated_relationships = self._invalidate_conflicting_relationships(
                    subject, predicate, obj, valid_from_dt, predicate_vector)
            
            # Generate unique relationship ID
            self.relationship_counter += 1
            relationship_id = f"rel_{self.relationship_counter}_{int(valid_from_dt.timestamp())}"
            
            # Add temporal metadata
            temporal_data = {
                'relationship_id': relationship_id,
                'subject': subject,
                'predicate': predicate,
                'object': obj,
                'subject_uri': str(subject_uri),
                'predicate_uri': str(predicate_uri),
                'object_uri': str(object_uri),
                'valid_from': valid_from_dt,
                'valid_to': valid_to_dt,
                'directional': directional,
                'created_timestamp': self._get_current_timestamp(),
                'invalidated_relationships': invalidated_relationships
            }
            
            self.temporal_relationships[relationship_id] = temporal_data
            
            # Add main triple to RDF graph with temporal annotation
            self.graph.add((subject_uri, predicate_uri, object_uri))
            
            # Add temporal triples using time ontology concepts
            time_ns = self.rdflib.Namespace("http://www.w3.org/2006/time#")
            temporal_uri = self.rdflib.URIRef(self.namespace[f"temporal_{relationship_id}"])
            
            # Link the relationship to its temporal information
            self.graph.add((temporal_uri, self.rdflib.RDF.type, time_ns.Interval))
            self.graph.add((temporal_uri, time_ns.hasBeginning, 
                        self.rdflib.Literal(valid_from_dt.isoformat(), datatype=self.rdflib.XSD.dateTime)))
            
            if valid_to_dt:
                self.graph.add((temporal_uri, time_ns.hasEnd, 
                            self.rdflib.Literal(valid_to_dt.isoformat(), datatype=self.rdflib.XSD.dateTime)))
            
            # Link temporal interval to the main triple
            statement_uri = self.rdflib.URIRef(self.namespace[f"statement_{relationship_id}"])
            self.graph.add((statement_uri, self.rdflib.RDF.type, self.rdflib.RDF.Statement))
            self.graph.add((statement_uri, self.rdflib.RDF.subject, subject_uri))
            self.graph.add((statement_uri, self.rdflib.RDF.predicate, predicate_uri))
            self.graph.add((statement_uri, self.rdflib.RDF.object, object_uri))
            self.graph.add((statement_uri, self.namespace.validDuring, temporal_uri))
            
            # Add reverse connection for bidirectional relationships
            if directional:
                reverse_uri, _ = self._get_or_create_predicate(f"{predicate}_reverse")
                self.graph.add((object_uri, reverse_uri, subject_uri))
                
                # Add reverse temporal data
                reverse_rel_id = f"{relationship_id}_reverse"
                reverse_temporal_data = temporal_data.copy()
                reverse_temporal_data['relationship_id'] = reverse_rel_id
                reverse_temporal_data['subject'] = obj
                reverse_temporal_data['object'] = subject
                reverse_temporal_data['subject_uri'] = str(object_uri)
                reverse_temporal_data['object_uri'] = str(subject_uri)
                reverse_temporal_data['predicate_uri'] = str(reverse_uri)
                self.temporal_relationships[reverse_rel_id] = reverse_temporal_data
                
                print(f"Added bidirectional: {subject} ↔ {obj} (valid from: {valid_from_dt.strftime('%Y-%m-%d %H:%M:%S')})")
            else:
                print(f"Added unidirectional: {subject} → {obj} (valid from: {valid_from_dt.strftime('%Y-%m-%d %H:%M:%S')})")
            
            if valid_to_dt:
                print(f"  Valid until: {valid_to_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"  Valid indefinitely (until updated)")
            
            if invalidated_relationships:
                print(f"  Invalidated {len(invalidated_relationships)} conflicting relationships")
            
            return True
            
        except Exception as e:
            print(f"Error adding connection: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def update_relationship_validity(self, relationship_id: str, 
                               valid_to: Union[str, datetime, None] = None,
                               extend_to: Union[str, datetime, None] = None) -> bool:
        """
        FIXED: Update the validity period of an existing relationship.
        Now properly handles both temporal metadata and RDF triples.
        """
        try:
            if relationship_id not in self.temporal_relationships:
                print(f"Relationship {relationship_id} not found")
                return False
            
            temporal_data = self.temporal_relationships[relationship_id]
            
            # Determine new valid_to date
            new_valid_to = None
            if valid_to is not None:
                new_valid_to = self._parse_timestamp(valid_to)
                if new_valid_to and new_valid_to <= temporal_data['valid_from']:
                    print("Error: valid_to must be after valid_from")
                    return False
            elif extend_to is not None:
                new_valid_to = self._parse_timestamp(extend_to)
                if new_valid_to and new_valid_to <= temporal_data['valid_from']:
                    print("Error: extend_to must be after valid_from")
                    return False
            
            # FIXED: Handle relationship reactivation
            if temporal_data.get('valid_to') and new_valid_to is None:
                # Reactivating relationship - add back to RDF graph
                subject_uri = self.rdflib.URIRef(temporal_data['subject_uri'])
                predicate_uri = self.rdflib.URIRef(temporal_data['predicate_uri'])
                object_uri = self.rdflib.URIRef(temporal_data['object_uri'])
                
                # Re-add the main triple
                self.graph.add((subject_uri, predicate_uri, object_uri))
            elif new_valid_to and not temporal_data.get('valid_to'):
                # Ending active relationship - remove from RDF graph
                subject_uri = self.rdflib.URIRef(temporal_data['subject_uri'])
                predicate_uri = self.rdflib.URIRef(temporal_data['predicate_uri'])
                object_uri = self.rdflib.URIRef(temporal_data['object_uri'])
                
                # Remove the main triple
                self.graph.remove((subject_uri, predicate_uri, object_uri))
            
            # Update temporal metadata
            temporal_data['valid_to'] = new_valid_to
            
            # Update RDF temporal information
            time_ns = self.rdflib.Namespace("http://www.w3.org/2006/time#")
            temporal_uri = self.rdflib.URIRef(self.namespace[f"temporal_{relationship_id}"])
            
            # Remove old hasEnd if it exists
            for triple in list(self.graph.triples((temporal_uri, time_ns.hasEnd, None))):
                self.graph.remove(triple)
            
            # Add new hasEnd if specified
            if new_valid_to:
                self.graph.add((temporal_uri, time_ns.hasEnd, 
                            self.rdflib.Literal(new_valid_to.isoformat(), 
                                                datatype=self.rdflib.XSD.dateTime)))
            
            print(f"Updated relationship {relationship_id} validity")
            return True
            
        except Exception as e:
            print(f"Error updating relationship validity: {str(e)}")
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
            similarities, indices = self.predicate_index.search(query_norm, min(top_k * 2, self.predicate_index.ntotal))  # Get more to account for duplicates
            
            results = []
            seen_predicates = set()  # Track seen predicates (case-insensitive)
            
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx != -1 and idx in self.vector_to_predicate:
                    predicate_uri = self.vector_to_predicate[idx]
                    
                    # Extract predicate name from URI for duplicate checking
                    predicate_name = predicate_uri.split('/')[-1].split('#')[-1].lower()
                    
                    # Skip if we've already seen this predicate (case-insensitive)
                    if predicate_name in seen_predicates:
                        continue
                        
                    seen_predicates.add(predicate_name)
                    results.append((predicate_uri, float(similarity)))
                    
                    # Stop when we have enough unique results
                    if len(results) >= top_k:
                        break
            
            return results
            
        except Exception as e:
            print(f"Error searching similar predicates: {str(e)}")
            return []

    def _build_sparql_query(self, subject: str = None, predicate: str = None, obj: str = None, 
                   at_time: datetime = None, time_range: tuple = None, 
                   include_invalid: bool = False, similar_predicates: list = None) -> str:
        """
        FIXED: Build SPARQL query with proper URI consistency and temporal filtering.
        """
        try:
            select_vars = "?subject ?predicate ?object ?temporal ?validFrom ?validTo ?statement"
            where_clauses = []
            
            # FIXED: Consistent URI building
            if subject:
                subject_clean = self._clean_uri_component(subject)
                subject_uri = f"ex:{subject_clean}"
                where_clauses.append(f"?statement rdf:subject {subject_uri} .")
                where_clauses.append(f"BIND({subject_uri} AS ?subject)")
            else:
                where_clauses.append("?statement rdf:subject ?subject .")
                
            if obj:
                obj_clean = self._clean_uri_component(obj)
                obj_uri = f"ex:{obj_clean}"
                where_clauses.append(f"?statement rdf:object {obj_uri} .")
                where_clauses.append(f"BIND({obj_uri} AS ?object)")
            else:
                where_clauses.append("?statement rdf:object ?object .")
            
            # Always get predicate from statement
            where_clauses.append("?statement rdf:predicate ?predicate .")
            
            # Add predicate filtering if similar predicates provided
            if similar_predicates:
                predicate_uris = []
                for pred_uri in similar_predicates:
                    if pred_uri.startswith('http'):
                        # Extract local name from full URI
                        if str(self.namespace) in pred_uri:
                            local_name = pred_uri.replace(str(self.namespace), '')
                        else:
                            local_name = pred_uri.split('/')[-1].split('#')[-1]
                    else:
                        local_name = pred_uri
                    predicate_uris.append(f"ex:{local_name}")
                
                if predicate_uris:
                    predicate_filter = " || ".join([f"?predicate = {uri}" for uri in predicate_uris])
                    where_clauses.append(f"FILTER({predicate_filter})")
            
            # FIXED: Proper temporal constraint structure
            where_clauses.append("?statement rdf:type rdf:Statement .")
            where_clauses.append("?statement ex:validDuring ?temporal .")
            where_clauses.append("?temporal rdf:type time:Interval .")
            where_clauses.append("?temporal time:hasBeginning ?validFrom .")
            where_clauses.append("OPTIONAL { ?temporal time:hasEnd ?validTo }")
            
            # FIXED: Proper temporal filtering logic
            if at_time:
                at_time_str = at_time.isoformat()
                where_clauses.append(f'FILTER(?validFrom <= "{at_time_str}"^^xsd:dateTime)')
                where_clauses.append(f'FILTER(!BOUND(?validTo) || ?validTo > "{at_time_str}"^^xsd:dateTime)')
            elif time_range:
                start_dt, end_dt = time_range
                start_str = start_dt.isoformat()
                end_str = end_dt.isoformat()
                where_clauses.append(f'FILTER(!BOUND(?validTo) || ?validTo > "{start_str}"^^xsd:dateTime)')
                where_clauses.append(f'FILTER(?validFrom < "{end_str}"^^xsd:dateTime)')
            elif not include_invalid:
                # Only currently valid relationships
                current_time_str = self._get_current_timestamp().isoformat()
                where_clauses.append(f'FILTER(?validFrom <= "{current_time_str}"^^xsd:dateTime)')
                where_clauses.append(f'FILTER(!BOUND(?validTo) || ?validTo > "{current_time_str}"^^xsd:dateTime)')
            
            # Build complete query
            prefixes = f"""
            PREFIX ex: <{str(self.namespace)}>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX time: <http://www.w3.org/2006/time#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            """
            
            where_clause = "WHERE {\n    " + "\n    ".join(where_clauses) + "\n}"
            query = f"{prefixes}\nSELECT {select_vars}\n{where_clause}\nORDER BY ?validFrom"
            
            return query
            
        except Exception as e:
            print(f"Error building SPARQL query: {str(e)}")
            return ""
    
    def _execute_sparql_query(self, query: str) -> list:
        """
        Execute SPARQL query and return results.
        
        Args:
            query (str): SPARQL query string
            
        Returns:
            list: Query results
        """
        try:
            if not query:
                return []
                
            results = self.graph.query(query)
            return list(results)
            
        except Exception as e:
            print(f"Error executing SPARQL query: {str(e)}")
            print(f"Query was: {query}")
            return []

    def _process_sparql_results(self, sparql_results: list, predicate_query: str = None) -> List[Dict[str, Any]]:
        """
        Process SPARQL query results into standardized format.
        
        Args:
            sparql_results (list): Raw SPARQL results
            predicate_query (str, optional): Original predicate query for similarity scoring
            
        Returns:
            List[Dict[str, Any]]: Processed results
        """
        try:
            results = []
            current_time = self._get_current_timestamp()
            
            for row in sparql_results:
                # Extract values from SPARQL result row
                subject_uri = str(row.subject) if row.subject else ""
                predicate_uri = str(row.predicate) if row.predicate else ""
                object_uri = str(row.object) if row.object else ""
                valid_from_str = str(row.validFrom) if row.validFrom else ""
                valid_to_str = str(row.validTo) if row.validTo else ""
                
                # Parse timestamps
                valid_from = self._parse_timestamp(valid_from_str) if valid_from_str else None
                valid_to = self._parse_timestamp(valid_to_str) if valid_to_str else None
                
                # Calculate similarity score if predicate query provided
                similarity_score = 1.0
                if predicate_query and predicate_uri in self.predicate_vectors:
                    similar_predicates = self.search_similar_predicates(predicate_query, top_k=10)
                    for pred_uri, similarity in similar_predicates:
                        if pred_uri == predicate_uri:
                            similarity_score = similarity
                            break
                
                # Determine current validity
                is_currently_valid = False
                if valid_from:
                    is_currently_valid = (valid_from <= current_time and 
                                        (valid_to is None or valid_to > current_time))
                
                # Find relationship ID from temporal relationships
                relationship_id = None
                for rel_id, rel_data in self.temporal_relationships.items():
                    if (rel_data['subject_uri'] == subject_uri and 
                        rel_data['predicate_uri'] == predicate_uri and 
                        rel_data['object_uri'] == object_uri and
                        rel_data['valid_from'] == valid_from):
                        relationship_id = rel_id
                        break
                
                # Create result dictionary
                result = {
                    'subject': self._extract_local_name(subject_uri),
                    'predicate': self._extract_local_name(predicate_uri),
                    'object': self._extract_local_name(object_uri),
                    'subject_uri': subject_uri,
                    'predicate_uri': predicate_uri,
                    'object_uri': object_uri,
                    'similarity_score': similarity_score,
                    'valid_from': valid_from,
                    'valid_to': valid_to,
                    'relationship_id': relationship_id,
                    'is_currently_valid': is_currently_valid
                }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error processing SPARQL results: {str(e)}")
            return []

    def query_graph(self, query_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph using SPARQL with flexible search patterns, semantic predicate matching,
        and temporal filtering.
        
        Args:
            query_dict (Dict[str, Any]): Query parameters with keys:
                - subject (str, optional): Subject entity name
                - predicate (str, optional): Predicate/relationship type (uses semantic matching)
                - object (str, optional): Object entity name
                - at_time (str/datetime, optional): Query relationships valid at specific time
                - time_range (tuple, optional): Query relationships valid within time range (start, end)
                - include_invalid (bool, optional): Include invalidated relationships (default: False)
                
        Returns:
            List[Dict[str, Any]]: List of result dictionaries with relationship details and temporal info formated the same as Args
        """
        try:
            # Ensure namespace is initialized
            if not self.ensure_namespace_initialized():
                print("Error: Could not initialize namespace for querying")
                return []
            
            subject = query_dict.get('subject')
            predicate = query_dict.get('predicate')
            obj = query_dict.get('object')
            at_time = query_dict.get('at_time')
            time_range = query_dict.get('time_range')
            include_invalid = query_dict.get('include_invalid', False)
            
            # Validate inputs
            if not subject and not predicate and not obj:
                print("Error: At least one of subject, predicate, or object is required")
                return []
            
            # Parse temporal constraints
            at_time_dt = self._parse_timestamp(at_time) if at_time else None
            time_range_dt = None
            if time_range:
                start_dt = self._parse_timestamp(time_range[0])
                end_dt = self._parse_timestamp(time_range[1])
                if start_dt and end_dt:
                    time_range_dt = (start_dt, end_dt)
            
            # Handle semantic predicate matching
            similar_predicates = []
            if predicate:
                similar_predicates_with_scores = self.search_similar_predicates(predicate, top_k=10)
                # Filter by similarity threshold and extract URIs
                similar_predicates = [pred_uri for pred_uri, similarity in similar_predicates_with_scores 
                                    if similarity >= self.predicate_similarity_search_threshold]
                
                if not similar_predicates:
                    print(f"No predicates found similar to '{predicate}' above threshold {self.predicate_similarity_search_threshold}")
                    return []
            
            # Build and execute SPARQL query
            sparql_query = self._build_sparql_query(
                subject=subject,
                predicate=predicate,
                obj=obj,
                at_time=at_time_dt,
                time_range=time_range_dt,
                include_invalid=include_invalid,
                similar_predicates=similar_predicates
            )
            
            if not sparql_query:
                print("Error: Could not build SPARQL query")
                return []
            
            # Execute query
            sparql_results = self._execute_sparql_query(sparql_query)
            
            if not sparql_results:
                print("No results found for query")
                return []
            
            # Process results
            results = self._process_sparql_results(sparql_results, predicate)
            
            # Apply additional similarity filtering if needed
            if predicate:
                results = [r for r in results if r['similarity_score'] >= self.predicate_similarity_search_threshold]
            
            # Sort results by similarity score (highest first), then by current validity, then by valid_from date
            results.sort(key=lambda x: (-x['similarity_score'], -int(x['is_currently_valid']), 
                                    x['valid_from'] if x['valid_from'] else datetime.min))
            
            print(f"Found {len(results)} matching relationships")
            return results
            
        except Exception as e:
            print(f"Error in query_graph: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def get_relationship_history(self, subject: str, predicate: str = None, obj: str = None) -> List[Dict[str, Any]]:
        """
        Get the complete history of relationships for a subject, optionally filtered by predicate or object.
        
        Args:
            subject (str): Subject entity
            predicate (str, optional): Filter by predicate
            obj (str, optional): Filter by object
            
        Returns:
            List[Dict[str, Any]]: Chronologically ordered relationship history
        """
        try:
            query_dict = {'subject': subject, 'include_invalid': True}
            if predicate:
                query_dict['predicate'] = predicate
            if obj:
                query_dict['object'] = obj
            
            relationships = self.query_graph(query_dict)
            
            # Sort by valid_from timestamp
            relationships.sort(key=lambda x: x['valid_from'])
            
            print(f"Found {len(relationships)} relationships in history for '{subject}'")
            
            return relationships
            
        except Exception as e:
            print(f"Error getting relationship history: {str(e)}")
            return []
    
    def get_current_relationships(self, subject: str = None, predicate: str = None, obj: str = None) -> List[Dict[str, Any]]:
        """
        Get only currently valid relationships.
        
        Args:
            subject (str, optional): Filter by subject
            predicate (str, optional): Filter by predicate
            obj (str, optional): Filter by object
            
        Returns:
            List[Dict[str, Any]]: Currently valid relationships
        """
        try:
            current_time = self._get_current_timestamp()
            query_dict = {'at_time': current_time}
            
            if subject:
                query_dict['subject'] = subject
            if predicate:
                query_dict['predicate'] = predicate
            if obj:
                query_dict['object'] = obj
            
            return self.query_graph(query_dict)
            
        except Exception as e:
            print(f"Error getting current relationships: {str(e)}")
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
        Save RDF graph, vector data, and temporal relationship data to files.
        
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
            
            # Save temporal relationships and namespace information
            temporal_file = os.path.join(self.data_directory, f"{self.graph_name}_temporal.pkl")
            temporal_data = {
                'temporal_relationships': self.temporal_relationships,
                'relationship_counter': self.relationship_counter,
                'namespace_uri': str(self.namespace) if self.namespace else None
            }
            
            with open(temporal_file, 'wb') as f:
                pickle.dump(temporal_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"Vector data saved to {vectors_file}")
            print(f"Temporal data saved to {temporal_file}")
            return True
            
        except Exception as e:
            print(f"Error saving graph data: {str(e)}")
            return False
    
    def _load_graph_data(self) -> None:
        """
        Load RDF graph, vector data, and temporal relationship data from files.
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
            
            # Load temporal relationships and namespace
            temporal_file = os.path.join(self.data_directory, f"{self.graph_name}_temporal.pkl")
            if os.path.exists(temporal_file):
                with open(temporal_file, 'rb') as f:
                    temporal_data = pickle.load(f)
                
                self.temporal_relationships = temporal_data.get('temporal_relationships', {})
                self.relationship_counter = temporal_data.get('relationship_counter', 0)
                
                # Restore namespace if it was saved
                saved_namespace_uri = temporal_data.get('namespace_uri')
                if saved_namespace_uri:
                    self.namespace = self.rdflib.Namespace(saved_namespace_uri)
                    # Rebind the namespace prefixes
                    self.graph.bind("ex", self.namespace)
                    self.graph.bind("rdf", self.rdflib.RDF)
                    self.graph.bind("rdfs", self.rdflib.RDFS)
                    self.graph.bind("owl", self.rdflib.OWL)
                    self.graph.bind("time", self.rdflib.Namespace("http://www.w3.org/2006/time#"))
                    print(f"Restored namespace: {saved_namespace_uri}")
                
                print(f"Loaded temporal data from {temporal_file}")
            
        except Exception as e:
            print(f"Error loading graph data: {str(e)}")
    
    def ensure_namespace_initialized(self, default_uri: str = "http://example.org/") -> bool:
        """
        Ensure namespace is initialized, creating a default one if needed.
        
        Args:
            default_uri (str): Default namespace URI to use if none exists
            
        Returns:
            bool: True if namespace is available, False otherwise
        """
        if self.namespace is None:
            print(f"Warning: Namespace not initialized. Creating default namespace: {default_uri}")
            return self.create_database_instance(default_uri)
        return True
    
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
        Get statistics about the knowledge graph including temporal information.
        
        Returns:
            Dict[str, Any]: Graph statistics
        """
        current_time = self._get_current_timestamp()
        currently_valid = sum(1 for rel_data in self.temporal_relationships.values() 
                            if rel_data['valid_from'] <= current_time and 
                            (rel_data.get('valid_to') is None or rel_data['valid_to'] > current_time))
        
        invalidated = sum(1 for rel_data in self.temporal_relationships.values() 
                        if rel_data.get('valid_to') is not None)
        
        stats = {
            'total_triples': len(self.graph),
            'total_predicates': len(self.predicate_vectors),
            'unique_subjects': len(set(str(s) for s, p, o in self.graph)),
            'unique_objects': len(set(str(o) for s, p, o in self.graph)),
            'vector_dimension': self.predicate_dimension,
            'faiss_index_size': self.predicate_index.ntotal if self.predicate_index else 0,
            'total_temporal_relationships': len(self.temporal_relationships),
            'currently_valid_relationships': currently_valid,
            'invalidated_relationships': invalidated,
            'relationship_counter': self.relationship_counter
        }
        
        return stats
    
    def clear_graph(self) -> bool:
        """
        Clear all data from the graph including temporal relationships.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.graph = self.rdflib.Graph()
            self.predicate_vectors = {}
            self.vector_to_predicate = {}
            self.predicate_index = None
            self.predicate_dimension = None
            self.temporal_relationships = {}
            self.relationship_counter = 0
            
            print("Graph cleared successfully")
            return True
            
        except Exception as e:
            print(f"Error clearing graph: {str(e)}")
            return False

# Demo and testing section with temporal features
if __name__ == "__main__":
    print("="*60)
    print("RDF Memory Recall Demo - Enhanced with Temporal Features")
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
    
    # Example 1: Explicit bidirectional relationship with temporal info
    print("\n--- Example 1: Historical Event Connection (with temporal validity) ---")
    connection1 = {
        "subject": "Tiananmen Protest 1989",
        "predicate": "similar historical event",
        "object": "Michigan Protest 1965",
        "directional": True,
        "valid_from": "2023-01-01",  # When this comparison became recognized
        "valid_to": None  # Still valid
    }
    result1 = rdf_memory.add_connection(connection1)
    print(f"Connection added: {result1}")
    
    # Example 2: Company ownership (temporal - companies change)
    print("\n--- Example 2: Company Ownership (temporal relationship) ---")
    connection2 = {
        "subject": "Apple Inc",
        "predicate": "owns",
        "object": "iPhone Division", 
        "directional": False,
        "valid_from": "2007-06-29"  # iPhone launch date
    }
    result2 = rdf_memory.add_connection(connection2)
    print(f"Connection added: {result2}")
    
    # Example 3: Employment relationship (will change over time)
    print("\n--- Example 3: Employment Relationship (Jane at Apple) ---")
    connection3 = {
        "subject": "Jane Doe",
        "predicate": "works at", 
        "object": "Apple Inc",
        "directional": False,
        "valid_from": "2022-03-15"
    }
    result3 = rdf_memory.add_connection(connection3)
    print(f"Connection added: {result3}")
    
    # Example 4: Family relationship (permanent)
    print("\n--- Example 4: Parent-Child Relationship (permanent) ---")
    connection4 = {
        "subject": "John Smith",
        "predicate": "parent of",
        "object": "Sarah Smith",
        "directional": False,
        "valid_from": "1990-05-12"  # Sarah's birth date
    }
    result4 = rdf_memory.add_connection(connection4)
    print(f"Connection added: {result4}")
    
    # Example 5: Marriage relationship (can change)
    print("\n--- Example 5: Marriage Relationship ---")
    connection5 = {
        "subject": "Alice Johnson",
        "predicate": "married to",
        "object": "Bob Johnson", 
        "directional": True,  # Marriage is bidirectional
        "valid_from": "2018-07-20",
        "valid_to": "2023-12-15"  # Divorced
    }
    result5 = rdf_memory.add_connection(connection5)
    print(f"Connection added: {result5}")
    
    print("\n" + "="*60)
    print("TEMPORAL FUNCTIONALITY DEMONSTRATION")
    print("="*60)
    
    # Demonstrate job change scenario
    print("\n--- SCENARIO: Jane Changes Jobs ---")
    print("Current situation: Jane works at Apple (since 2022-03-15)")

    # Query Jane
    jane_doe = rdf_memory.query_graph({
        "subject": "Jane Doe",
        "include_invalid": True
    })
    print(f"\nJane's Data:")
    for rel in jane_doe:
        print(f"  {rel['subject']} -> {rel['predicate']} -> {rel['object']}")
    
    # Query Jane's current job
    current_job = rdf_memory.get_current_relationships(subject="Jane_Doe", predicate="works at")
    print(f"\nJane's current job:")
    for rel in current_job:
        print(f"  {rel['subject']} works at {rel['object']} (since {rel['valid_from'].strftime('%Y-%m-%d')})")
    
    # Now Jane gets a new job at Google
    print(f"\n--- Jane gets hired at Google (2024-08-01) ---")
    new_job = {
        "subject": "Jane Doe", 
        "predicate": "works at",
        "object": "Google Inc",
        "valid_from": "2024-08-01",
        "auto_invalidate_conflicts": True  # This will auto-invalidate the Apple job
    }
    rdf_memory.add_connection(new_job)
    
    # Query Jane's current job again
    print(f"\nJane's job after the change:")
    current_job_after = rdf_memory.get_current_relationships(subject="Jane Doe", predicate="works at")
    for rel in current_job_after:
        print(f"  {rel['subject']} works at {rel['object']} (since {rel['valid_from'].strftime('%Y-%m-%d')})")
        print(f"  Currently valid: {rel['is_currently_valid']}")
    
    # Get Jane's complete job history
    print(f"\nJane's complete employment history:")
    job_history = rdf_memory.get_relationship_history("Jane Doe", "works at")
    for i, rel in enumerate(job_history, 1):
        valid_until = rel['valid_to'].strftime('%Y-%m-%d') if rel['valid_to'] else "present"
        status = "ACTIVE" if rel['is_currently_valid'] else "ENDED"
        print(f"  {i}. {rel['object']} ({rel['valid_from'].strftime('%Y-%m-%d')} to {valid_until}) [{status}]")
    
    print("\n--- TEMPORAL QUERIES ---")
    
    # Query relationships at specific point in time
    print("\n1. What was Jane's job in March 2023?")
    job_in_march_2023 = rdf_memory.query_graph({
        "subject": "Jane Doe",
        "predicate": "works at", 
        "at_time": "2023-03-15"
    })
    
    if job_in_march_2023:
        for rel in job_in_march_2023:
            print(f"   Jane worked at {rel['object']} in March 2023")
    else:
        print("   No job found for that time period")
    
    # Query relationships within time range
    print("\n2. All of Jane's jobs between 2022 and 2024:")
    jobs_2022_2024 = rdf_memory.query_graph({
        "subject": "Jane Doe",
        "predicate": "works at",
        "time_range": ("2022-01-01", "2024-12-31"),
        "include_invalid": True
    })
    
    for rel in jobs_2022_2024:
        valid_until = rel['valid_to'].strftime('%Y-%m-%d') if rel['valid_to'] else "ongoing"
        print(f"   {rel['object']} ({rel['valid_from'].strftime('%Y-%m-%d')} to {valid_until})")
    
    # Query who was married in 2020
    print("\n3. Who was married in 2020?")
    married_2020 = rdf_memory.query_graph({
        "predicate": "married to",
        "at_time": "2020-06-15"
    })
    
    for rel in married_2020:
        print(f"   {rel['subject']} was married to {rel['object']} in 2020,  {rel['similarity_score']}")
    
    # Query who is currently married
    print("\n4. Who is currently married?")
    currently_married = rdf_memory.get_current_relationships(predicate="married to")
    
    if currently_married:
        for rel in currently_married:
            print(f"   {rel['subject']} is married to {rel['object']}")
    else:
        print("   No current marriages in the database")
    
    print("\n--- UPDATING RELATIONSHIP VALIDITY ---")
    
    # Find Alice and Bob's marriage relationship ID
    alice_bob_marriage = rdf_memory.query_graph({
        "subject": "Alice Johnson",
        "predicate": "married to", 
        "object": "Bob Johnson",
        "include_invalid": True
    })
    
    if alice_bob_marriage:
        rel_id = alice_bob_marriage[0]['relationship_id']
        print(f"\nFound marriage relationship ID: {rel_id}")
        
        # Let's say they reconciled - extend the marriage
        print("Scenario: Alice and Bob reconciled, extending marriage validity...")
        rdf_memory.update_relationship_validity(rel_id, valid_to=None)  # Remove end date
        
        # Check if they're married now
        current_marriage = rdf_memory.get_current_relationships(
            subject="Alice Johnson", predicate="married to"
        )
        
        if current_marriage:
            print("   Alice and Bob are now married again!")
        else:
            print("   Update didn't work as expected")
    
    print("\n5. Enhanced Graph Statistics (with temporal info):")
    print("="*50)
    stats = rdf_memory.get_graph_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n6. All Temporal Relationships:")
    print("="*40)
    
    # Show all temporal relationships in a nice format
    from datetime import datetime
    current_time = datetime.now()
    
    print("\n   Current Relationships:")
    current_rels = rdf_memory.get_current_relationships(predicate="owner")
    for i, rel in enumerate(current_rels, 1):
        print(f"   {i:2d}. {rel['subject']} -[{rel['predicate']}]-> {rel['object']}")
        print(f"       Valid since: {rel['valid_from'].strftime('%Y-%m-%d %H:%M:%S')}")
        if rel['valid_to']:
            print(f"       Valid until: {rel['valid_to'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"       Valid until: ongoing")
        print(f"       Relationship ID: {rel['relationship_id']}")
        print()
    
    print("\n   Historical (Invalid) Relationships:")
    all_rels = rdf_memory.query_graph({"subject": "Jane_Doe", "include_invalid": True})
    invalid_rels = [rel for rel in all_rels if not rel['is_currently_valid']]
    
    for i, rel in enumerate(invalid_rels, 1):
        print(f"   {i:2d}. {rel['subject']} -[{rel['predicate']}]-> {rel['object']}")
        print(f"       Was valid: {rel['valid_from'].strftime('%Y-%m-%d')} to {rel['valid_to'].strftime('%Y-%m-%d') if rel['valid_to'] else 'unknown'}")
        print(f"       Relationship ID: {rel['relationship_id']}")
        print()
    
    print("\n7. Testing Predicate Similarity Search with Temporal Context:")
    print("="*40)
    if rdf_memory.embedders:  # Only if embedder is available
        # Test employment-related predicate similarity
        similar_predicates = rdf_memory.search_similar_predicates("employed by", top_k=3)
        if similar_predicates:
            print("   Similar predicates to 'employed by':")
            for pred_uri, similarity in similar_predicates:
                pred_clean = pred_uri.split('/')[-1] if '/' in pred_uri else pred_uri
                print(f"   - {pred_clean} (similarity: {similarity:.3f})")
                
                # Show relationships using this predicate
                rels_with_pred = rdf_memory.query_graph({"predicate": pred_clean})
                for rel in rels_with_pred:
                    status = "ACTIVE" if rel['is_currently_valid'] else "ENDED"
                    print(f"     * {rel['subject']} -> {rel['object']} [{status}]")
        else:
            print("   No similar predicates found")
    else:
        print("   Embedder not available - skipping similarity search")

    print("\n" + "="*60)
    print("ENHANCED UNIFIED QUERY_GRAPH METHOD DEMO (with temporal)")
    print("="*60)
    
    # Pattern 1: Subject only (current relationships)
    print("\n" + "="*50)
    print("PATTERN 1: Subject Only - Current relationships")
    print("="*50)
    
    query1 = {'subject': 'Jane Doe'}
    print(f"Query: {query1}")
    
    results1 = rdf_memory.query_graph(query1)
    print(f"\nCurrent relationships ({len(results1)} found):")
    
    for i, result in enumerate(results1, 1):
        status = "ACTIVE" if result['is_currently_valid'] else "ENDED"
        print(f"  {i}. {result['subject']} -[{result['predicate']}]-> {result['object']} [{status}]")
        print(f"     Valid: {result['valid_from'].strftime('%Y-%m-%d')} to {'ongoing' if not result['valid_to'] else result['valid_to'].strftime('%Y-%m-%d')}")
        print(f"     Similarity: {result['similarity_score']:.3f}")
    
    # Pattern 2: Subject with time constraint
    print("\n" + "="*50)
    print("PATTERN 2: Subject at specific time")
    print("="*50)
    
    query2 = {'subject': 'Jane_Doe', 'at_time': '2023-06-15'}
    print(f"Query: {query2}")
    
    results2 = rdf_memory.query_graph(query2)
    print(f"\nRelationships valid on 2023-06-15 ({len(results2)} found):")
    
    for i, result in enumerate(results2, 1):
        print(f"  {i}. {result['subject']} -[{result['predicate']}]-> {result['object']}")
        print(f"     Valid: {result['valid_from'].strftime('%Y-%m-%d')} to {'ongoing' if not result['valid_to'] else result['valid_to'].strftime('%Y-%m-%d')}")
    
    # Pattern 3: Predicate with temporal context
    print("\n" + "="*50)
    print("PATTERN 3: All employment relationships (current and historical)")
    print("="*50)
    
    query3 = {'predicate': 'employed at', 'include_invalid': True}
    print(f"Query: {query3}")
    
    results3 = rdf_memory.query_graph(query3)
    print(f"\nAll employment relationships ({len(results3)} found):")
    
    for i, result in enumerate(results3, 1):
        status = "ACTIVE" if result['is_currently_valid'] else "ENDED"
        print(f"  {i}. {result['subject']} -[{result['predicate']}]-> {result['object']} [{status}]")
        print(f"     Period: {result['valid_from'].strftime('%Y-%m-%d')} to {'ongoing' if not result['valid_to'] else result['valid_to'].strftime('%Y-%m-%d')}")
    
    print("\n" + "="*60)
    print("TEMPORAL QUERY DEMO COMPLETED")
    print("="*60)

    print("\n8. Saving Graph Data (including temporal information):")
    print("="*40)
    save_success = rdf_memory.save_graph_data()
    print(f"   Graph data saved: {save_success}")
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)
    
    # Test loading in a new instance with temporal data
    print("\n9. Testing Data Persistence (including temporal data):")
    print("="*40)
    print("   Creating new instance and loading saved data...")
    new_instance = rdfMemRecall(data_directory="./demo_rdf_data", graph_name="demo_graph")

    new_instance.create_database_instance("http://demo.example.org/")  # MUST add this line when reloading a database

    new_stats = new_instance.get_graph_statistics()
    print(f"   Loaded graph has {new_stats['total_triples']} triples")
    print(f"   Loaded graph has {new_stats['total_predicates']} predicates")
    print(f"   Loaded graph has {new_stats['total_temporal_relationships']} temporal relationships")
    print(f"   Currently valid: {new_stats['currently_valid_relationships']}")
    print(f"   Invalidated: {new_stats['invalidated_relationships']}")
    
    if new_stats['total_temporal_relationships'] > 0:
        print("   ✓ Temporal data persistence working correctly!")
        
        # Test a temporal query on the new instance
        print("\n   Testing temporal query on loaded data...")
        test_query = new_instance.get_current_relationships(subject="Jane_Doe")
        if test_query:
            print(f"   ✓ Found {len(test_query)} current relationships for Jane_Doe")
            for rel in test_query:
                print(f"     - Works at: {rel['object']}")
        else:
            print("   ✗ No current relationships found")
    else:
        print("   ✗ Temporal data persistence issue detected")

    ######################
    # Enhanced visualization to show temporal aspects
    print("\n10. Enhanced Visualization (with temporal information):")
    print("="*40)
    
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import numpy as np
        
        # Create enhanced visualization showing current vs historical relationships
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left plot: Current relationships only
        current_rels = rdf_memory.get_current_relationships()
        
        if current_rels:
            G_current = nx.DiGraph()
            for rel in current_rels:
                G_current.add_edge(
                    rel['subject'], 
                    rel['object'], 
                    label=rel['predicate'],
                    weight=rel['similarity_score']
                )
            
            pos1 = nx.spring_layout(G_current, k=2, iterations=50)
            nx.draw(G_current, pos1, ax=ax1, with_labels=True, 
                   node_color='lightgreen', node_size=1500, 
                   font_size=8, arrows=True, edge_color='green')
            
            # Add edge labels
            edge_labels1 = nx.get_edge_attributes(G_current, 'label')
            nx.draw_networkx_edge_labels(G_current, pos1, edge_labels1, ax=ax1, font_size=6)
            ax1.set_title("Current Relationships", fontsize=14, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, "No current relationships", ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title("Current Relationships", fontsize=14, fontweight='bold')
        
        # Right plot: All relationships (including historical)
        all_rels = []
        for subject in ['Jane_Doe', 'Alice_Johnson', 'John_Smith']:
            all_rels.extend(rdf_memory.query_graph({'subject': subject, 'include_invalid': True}))
        
        if all_rels:
            G_all = nx.DiGraph()
            for rel in all_rels:
                color = 'green' if rel['is_currently_valid'] else 'red'
                style = 'solid' if rel['is_currently_valid'] else 'dashed'
                G_all.add_edge(
                    rel['subject'], 
                    rel['object'], 
                    label=rel['predicate'],
                    color=color,
                    style=style,
                    current=rel['is_currently_valid']
                )
            
            pos2 = nx.spring_layout(G_all, k=2, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G_all, pos2, ax=ax2, node_color='lightblue', node_size=1500)
            nx.draw_networkx_labels(G_all, pos2, ax=ax2, font_size=8)
            
            # Draw edges with different colors for current vs historical
            current_edges = [(u, v) for u, v, d in G_all.edges(data=True) if d['current']]
            historical_edges = [(u, v) for u, v, d in G_all.edges(data=True) if not d['current']]
            
            nx.draw_networkx_edges(G_all, pos2, edgelist=current_edges, 
                                 edge_color='green', style='solid', ax=ax2, arrows=True)
            nx.draw_networkx_edges(G_all, pos2, edgelist=historical_edges, 
                                 edge_color='red', style='dashed', ax=ax2, arrows=True)
            
            # Add edge labels
            edge_labels2 = nx.get_edge_attributes(G_all, 'label')
            nx.draw_networkx_edge_labels(G_all, pos2, edge_labels2, ax=ax2, font_size=6)
            
            ax2.set_title("All Relationships (Green=Current, Red=Historical)", fontsize=14, fontweight='bold')
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='green', lw=2, label='Current'),
                Line2D([0], [0], color='red', lw=2, linestyle='--', label='Historical')
            ]
            ax2.legend(handles=legend_elements, loc='upper right')
        else:
            ax2.text(0.5, 0.5, "No relationships found", ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("All Relationships", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.suptitle("RDF Knowledge Graph - Temporal Visualization", fontsize=16, fontweight='bold', y=0.98)
        plt.show()
        
        print("   ✓ Enhanced visualization generated successfully!")
        
    except ImportError as e:
        print(f"   Visualization libraries not available: {e}")
    except Exception as e:
        print(f"   Error generating visualization: {e}")

    print("\n" + "="*60)
    print("ENHANCED TEMPORAL DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)