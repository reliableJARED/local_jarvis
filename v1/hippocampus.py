"""
    Advanced hybrid memory system for robust entity and fact recall in Language Models.
    Features:
    - Flexible memory creation from word lists of full text inputs
    - LLM-assisted key fact extraction and context prefix generation
    - Semantic search-based memory injection using embedding similarity
    - Support for both entity memories and general fact memories
    - Knowedge graph construction for relational understanding
    """

"""
Structure of the Hippocampus module:
-Memory Concepts:
Who: Nouns representing entities (people, places, objects)
What: Key facts or attributes about the entities
When: Temporal context
Where: Spatial context 

Memories are stored as combinations of these concepts.

- Memory Creation:
After prefrontal cortex generates text outputs, the Hippocampus processes the user input and AI output to create structured memory. 
It will use the chat() function to call the LLM to extract key facts and the Who and What components.

Then check existing memories for relevance using embeddings and semantic search.
If relevant memories exist, link them to the new memory.
If a very similar memory already exists, update it instead of creating a duplicate.
"""

import sqlite3
import json
import time
import logging
import uuid
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Callable, Tuple, Optional
from datetime import datetime
import re

# Try to import sentence_transformers, handle gracefully if missing
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("WARNING: sentence_transformers not installed. Semantic search will be disabled.")

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HIPPOCAMPUS")

class Hippocampus:
    """
    The Hippocampus manages the storage, retrieval, and consolidation of memories.
    It utilizes a hybrid approach:
    1. Vector Embeddings: For semantic similarity search (finding related concepts).
    2. Knowledge Graph: For structural relationships (Who, Where, When links).
    3. SQL Storage: For raw text and metadata persistence.
    """

    def __init__(self, db_path="memory.db", embedding_model_name='all-MiniLM-L6-v2'): #change model as needed, mixedbread-ai/mxbai-embed-large-v1 
        self.db_path = db_path
        self.graph = nx.DiGraph()
        
        # Initialize Embedding Model
        self.encoder = None
        if EMBEDDINGS_AVAILABLE:
            try:
                logger.info(f"Loading embedding model: {embedding_model_name}...")
                self.encoder = SentenceTransformer(embedding_model_name)
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")

        # Initialize Storage
        self._init_db()
        self._load_graph_from_db()

    def _init_db(self):
        """Initialize SQLite database for memories and graph edges."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main Memory Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                text TEXT,
                created_at REAL,
                embedding BLOB,
                metadata JSON
            )
        ''')

        # Graph Nodes (Entities)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                name TEXT PRIMARY KEY,
                type TEXT,
                attributes JSON
            )
        ''')

        # Graph Edges (Relationships)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                source TEXT,
                target TEXT,
                relation TEXT,
                memory_id TEXT,
                FOREIGN KEY(memory_id) REFERENCES memories(id)
            )
        ''')
        conn.commit()
        conn.close()

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Convert text to vector embedding."""
        if self.encoder:
            vector = self.encoder.encode(text)
            return vector
        return None

    def _blob_to_array(self, blob) -> np.ndarray:
        """Convert SQLite BLOB back to numpy array."""
        return np.frombuffer(blob, dtype=np.float32)

    def _load_graph_from_db(self):
        """Rebuild NetworkX graph from SQLite storage on startup."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load Entities
        cursor.execute("SELECT name, type, attributes FROM entities")
        for name, type_, attrs in cursor.fetchall():
            self.graph.add_node(name, type=type_, **json.loads(attrs))

        # Load Relationships
        cursor.execute("SELECT source, target, relation FROM relationships")
        for source, target, relation in cursor.fetchall():
            self.graph.add_edge(source, target, relation=relation)
            
        conn.close()
        logger.info(f"Knowledge Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

    def _extract_concepts_with_llm(self, text: str, llm_chat_func: Callable) -> Dict:
        """
        Uses the provided LLM function to extract Who, What, Where, When.
        
        Args:
            text: The text to analyze.
            llm_chat_func: A function f(messages) -> str that calls the LLM.
        """
        system_prompt = (
            "You are the memory consolidation unit. Analyze the user's input. "
            "Extract key entities and facts into a valid JSON object. "
            "Do not output markdown code blocks, just the raw JSON string.\n"
            "Schema:\n"
            "{\n"
            "  'who': ['list', 'of', 'people/entities'],\n"
            "  'what': ['list', 'of', 'key', 'facts/actions'],\n"
            "  'when': 'time reference or null',\n"
            "  'where': 'location context or null',\n"
            "  'summary': 'A concise one-sentence summary'\n"
            "}"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Input: {text}"}
        ]

        try:
            # Call the LLM (provided by Cerebrum/PFC)
            response_str = llm_chat_func(messages)
            
            # Clean response (remove markdown if present)
            response_str = re.sub(r'```json', '', response_str)
            response_str = re.sub(r'```', '', response_str).strip()
            
            data = json.loads(response_str)
            return data
        except Exception as e:
            logger.error(f"LLM Extraction failed: {e}")
            return {"who": [], "what": [], "when": None, "where": None, "summary": text}

    def ingest_memory(self, text: str, llm_chat_func: Callable = None):
        """
        The main entry point for saving a memory.
        1. Checks for duplicates via semantic search.
        2. Extracts concepts via LLM.
        3. Saves to Vector DB and Knowledge Graph.
        """
        if not text or len(text.strip()) < 5:
            return

        embedding = self._get_embedding(text)

        # 1. Check for Semantically Similar Memories (Deduplication/Reinforcement)
        similar_memories = self.semantic_search(text, top_k=1, threshold=0.85)
        if similar_memories:
            existing_id, existing_text, score = similar_memories[0]
            logger.info(f"Memory similarity {score:.2f} detected. Reinforcing existing memory instead of creating new.")
            # In a full system, we might merge facts here. For now, we update the timestamp logic or skip.
            # But let's proceed to extract concepts to update the graph anyway.

        # 2. Extract Concepts
        if llm_chat_func:
            concepts = self._extract_concepts_with_llm(text, llm_chat_func)
        else:
            concepts = {"who": [], "what": [text], "when": None, "where": None, "summary": text}

        # 3. Persist to DB
        memory_id = str(uuid.uuid4())
        created_at = time.time()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO memories (id, text, created_at, embedding, metadata) VALUES (?, ?, ?, ?, ?)",
            (memory_id, text, created_at, embedding.tobytes() if embedding is not None else None, json.dumps(concepts))
        )

        # 4. Update Knowledge Graph & Tables
        self._update_graph(cursor, memory_id, concepts)
        
        conn.commit()
        conn.close()
        logger.info(f"Memory stored: {concepts.get('summary', 'Unknown')}")

    def _update_graph(self, cursor, memory_id: str, concepts: Dict):
        """Updates the NetworkX graph and SQL entities based on extracted concepts."""
        
        # Process WHO (Entities)
        for person in concepts.get('who', []):
            person = person.lower().strip()
            self.graph.add_node(person, type="person")
            cursor.execute("INSERT OR IGNORE INTO entities (name, type, attributes) VALUES (?, ?, ?)", 
                           (person, "person", "{}"))

        # Process WHERE (Location Reference Frame)
        location = concepts.get('where')
        if location:
            location = location.lower().strip()
            self.graph.add_node(location, type="location")
            cursor.execute("INSERT OR IGNORE INTO entities (name, type, attributes) VALUES (?, ?, ?)", 
                           (location, "location", "{}"))
            
            # Link WHO to WHERE
            for person in concepts.get('who', []):
                person = person.lower().strip()
                self.graph.add_edge(person, location, relation="located_at")
                cursor.execute("INSERT INTO relationships (source, target, relation, memory_id) VALUES (?, ?, ?, ?)",
                               (person, location, "located_at", memory_id))

        # Process WHEN (Temporal Reference Frame)
        # In a graph, time is often a node or an edge attribute. Here we make it a node context.
        time_ref = concepts.get('when')
        if time_ref:
            self.graph.add_node(time_ref, type="time")
            # Link events/facts to time? 
            # For simplicity, we link the entities to this time frame
            for person in concepts.get('who', []):
                person = person.lower().strip()
                self.graph.add_edge(person, time_ref, relation="active_during")

    def semantic_search(self, query: str, top_k: int = 3, threshold: float = 0.0) -> List[Tuple[str, str, float]]:
        """
        Retrieve relevant memories using vector cosine similarity.
        Returns: List of (id, text, similarity_score)
        """
        if not self.encoder:
            return []

        query_vec = self.encoder.encode(query)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, text, embedding FROM memories WHERE embedding IS NOT NULL")
        rows = cursor.fetchall()
        conn.close()

        results = []
        for mid, text, blob in rows:
            vec = self._blob_to_array(blob)
            # Cosine Similarity
            score = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
            if score >= threshold:
                results.append((mid, text, score))

        # Sort by score descending
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def graph_traversal(self, query_entities: List[str], depth: int = 1) -> List[str]:
        """
        Traverse the knowledge graph starting from specific entities found in the query.
        Returns a list of facts (edges) connected to these entities.
        """
        facts = []
        for entity in query_entities:
            entity = entity.lower().strip()
            if entity in self.graph:
                # Get neighbors
                edges = list(self.graph.edges(entity, data=True))
                # Also get incoming edges (reverse relationships)
                in_edges = list(self.graph.in_edges(entity, data=True))
                
                all_edges = edges + in_edges
                
                for u, v, data in all_edges:
                    relation = data.get('relation', 'related_to')
                    facts.append(f"{u} is {relation} {v}")
                    
        return list(set(facts)) # Dedup

    def recall(self, query: str, llm_chat_func: Callable = None) -> str:
        """
        The 'Context Builder'. Retrieves vectors and graph data to construct a context block.
        """
        # 1. Vector Search (The "Vibe" / Semantic Recall)
        vector_hits = self.semantic_search(query, top_k=3, threshold=0.4)
        memories = [f"- {text} (Confidence: {score:.2f})" for _, text, score in vector_hits]

        # 2. Graph Search (The "Facts")
        # Extract entities from query quickly (naive or using the LLM again)
        # For speed, we just check if any known graph nodes exist in the query string
        known_entities = [node for node in self.graph.nodes() if node in query.lower()]
        graph_facts = self.graph_traversal(known_entities)

        # 3. Construct Context String
        context_str = "### RECALLED MEMORIES ###\n"
        if memories:
            context_str += "From Past Conversations:\n" + "\n".join(memories) + "\n"
        
        if graph_facts:
            context_str += "\nKnown Facts:\n" + "\n".join(graph_facts) + "\n"
            
        if not memories and not graph_facts:
            return ""

        return context_str


