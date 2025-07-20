"""
Jarvis Memory Viewer and Manager
A companion tool to view, search, and manage memories saved by the Jarvis voice assistant
"""

import os
import pickle
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import asdict
import argparse

# Import the embedding system and memory structures
try:
    from text_embed import MxBaiEmbedder
    from local_llm import MemoryPage, JarvisConfig
except ImportError as e:
    print(f"Warning: Could not import Jarvis modules: {e}")
    print("Make sure this script is in the same directory as your Jarvis files")


class JarvisMemoryViewer:
    """
    Tool for viewing and managing Jarvis memories
    """
    
    def __init__(self, embeddings_file: str = "jarvis_memory.pkl"):
        self.embeddings_file = embeddings_file
        self.embedder: Optional[MxBaiEmbedder] = None
        self.memory_data: Dict = {}
        
        # Try to load the embedding system
        self._load_embedding_system()
    
    def _load_embedding_system(self):
        """Load the embedding system and memory data"""
        try:
            if os.path.exists(self.embeddings_file):
                self.embedder = MxBaiEmbedder(pickle_file=self.embeddings_file)
                if self.embedder.load_model():
                    print(f"✅ Loaded embedding system with {self.embedder.get_stored_count()} memories")
                    self._load_memory_data()
                else:
                    print("❌ Failed to load embedding model")
            else:
                print(f"❌ Memory file not found: {self.embeddings_file}")
        except Exception as e:
            print(f"Error loading embedding system: {e}")
    
    def _load_memory_data(self):
        """Load raw memory data from pickle file"""
        try:
            with open(self.embeddings_file, 'rb') as f:
                data = pickle.load(f)
                self.memory_data = data
                print(f"📊 Loaded raw memory data structure")
        except Exception as e:
            print(f"Error loading raw memory data: {e}")
    
    def list_all_memories(self, limit: Optional[int] = None, show_embeddings: bool = False):
        """List all stored memories"""
        if not self.embedder:
            print("❌ No embedding system loaded")
            return
        
        print("\n" + "="*80)
        print("📚 JARVIS MEMORY STORE")
        print("="*80)
        
        # Get all stored embeddings
        stored_count = self.embedder.get_stored_count()
        print(f"Total memories: {stored_count}")
        
        if stored_count == 0:
            print("No memories found.")
            return
        
        # Access the internal stores
        embeddings_store = getattr(self.embedder, 'embeddings_store', {})
        metadata_store = getattr(self.embedder, 'metadata_store', {})
        
        memories = []
        for memory_id in embeddings_store.keys():
            if memory_id in metadata_store:
                metadata = metadata_store[memory_id]
                memories.append({
                    'id': memory_id,
                    'text': metadata.get('text', 'No text available'),
                    'created_at': metadata.get('created_at', 'Unknown'),
                    'embedding_shape': metadata.get('embedding_shape', 'Unknown')
                })
        
        # Sort by creation time (most recent first)
        try:
            memories.sort(key=lambda x: float(x['created_at']), reverse=True)
        except:
            pass  # If sorting fails, just show in original order
        
        # Apply limit
        if limit:
            memories = memories[:limit]
        
        for i, memory in enumerate(memories, 1):
            print(f"\n🧠 Memory #{i}")
            print(f"ID: {memory['id']}")
            print(f"Created: {memory['created_at']}")
            print(f"Embedding Shape: {memory['embedding_shape']}")
            print("Content:")
            print("-" * 40)
            
            # Parse the memory content to show it nicely
            self._display_memory_content(memory['text'])
            
            if show_embeddings:
                embedding = embeddings_store.get(memory['id'])
                if embedding is not None:
                    print(f"Embedding (first 10 values): {embedding[:10]}")
            
            print("-" * 40)
    
    def _display_memory_content(self, text: str):
        """Parse and display memory content in a readable format"""
        # Try to parse the structured memory content
        lines = text.split(' | ')
        
        for line in lines:
            if line.startswith('Date: '):
                print(f"📅 {line}")
            elif line.startswith('Time: '):
                print(f"⏰ {line}")
            elif line.startswith('Speaker: '):
                print(f"👤 {line}")
            elif line.startswith('Conversation: '):
                # This is the main content - parse it further
                conversation = line.replace('Conversation: ', '')
                self._display_conversation(conversation)
            elif line.startswith('Image: '):
                print(f"🖼️  {line}")
            elif line.startswith('Audio: '):
                print(f"🔊 {line}")
            else:
                print(f"ℹ️  {line}")
    
    def _display_conversation(self, conversation: str):
        """Display the conversation in a readable format"""
        print("💬 Conversation:")
        
        # Split by User: and Jarvis: markers
        parts = conversation.split('\n')
        for part in parts:
            part = part.strip()
            if part.startswith('User: '):
                print(f"   👤 User: {part[6:]}")
            elif part.startswith('Jarvis: '):
                print(f"   🤖 Jarvis: {part[8:]}")
            elif part:
                print(f"   📝 {part}")
    
    def search_memories(self, query: str, n: int = 5, threshold: float = 0.7):
        """Search memories by similarity"""
        if not self.embedder:
            print("❌ No embedding system loaded")
            return
        
        print(f"\n🔍 Searching for: '{query}'")
        print(f"Similarity threshold: {threshold}")
        print("="*60)
        
        try:
            results = self.embedder.search_by_text(query, n=n)
            
            if not results:
                print("No results found.")
                return
            
            filtered_results = [(mid, score, text) for mid, score, text in results if score >= threshold]
            
            if not filtered_results:
                print(f"No results above similarity threshold of {threshold}")
                print("All results:")
                for memory_id, similarity_score, stored_text in results:
                    print(f"  📉 {similarity_score:.3f} - {stored_text[:100]}...")
                return
            
            print(f"Found {len(filtered_results)} results above threshold:")
            
            for i, (memory_id, similarity_score, stored_text) in enumerate(filtered_results, 1):
                print(f"\n🎯 Result #{i} (Similarity: {similarity_score:.3f})")
                print(f"ID: {memory_id}")
                print("Content:")
                print("-" * 40)
                self._display_memory_content(stored_text)
                print("-" * 40)
                
        except Exception as e:
            print(f"Error searching memories: {e}")
    
    def get_memory_by_id(self, memory_id: str):
        """Get a specific memory by ID"""
        if not self.embedder:
            print("❌ No embedding system loaded")
            return
        
        try:
            metadata_store = getattr(self.embedder, 'metadata_store', {})
            embeddings_store = getattr(self.embedder, 'embeddings_store', {})
            
            if memory_id not in metadata_store:
                print(f"❌ Memory ID '{memory_id}' not found")
                return
            
            metadata = metadata_store[memory_id]
            embedding = embeddings_store.get(memory_id)
            
            print(f"\n🧠 Memory Details")
            print("="*50)
            print(f"ID: {memory_id}")
            print(f"Created: {metadata.get('created_at', 'Unknown')}")
            print(f"Embedding Shape: {metadata.get('embedding_shape', 'Unknown')}")
            print("\nContent:")
            print("-" * 40)
            self._display_memory_content(metadata.get('text', 'No text available'))
            print("-" * 40)
            
            if embedding is not None:
                print(f"\nEmbedding Statistics:")
                print(f"  Shape: {embedding.shape}")
                print(f"  Mean: {np.mean(embedding):.6f}")
                print(f"  Std: {np.std(embedding):.6f}")
                print(f"  Min: {np.min(embedding):.6f}")
                print(f"  Max: {np.max(embedding):.6f}")
                print(f"  First 10 values: {embedding[:10]}")
            
        except Exception as e:
            print(f"Error retrieving memory: {e}")
    
    def get_stats(self):
        """Get comprehensive statistics about the memory store"""
        if not self.embedder:
            print("❌ No embedding system loaded")
            return
        
        print("\n📊 MEMORY STATISTICS")
        print("="*50)
        
        try:
            embeddings_store = getattr(self.embedder, 'embeddings_store', {})
            metadata_store = getattr(self.embedder, 'metadata_store', {})
            
            print(f"Total memories: {len(embeddings_store)}")
            print(f"Metadata entries: {len(metadata_store)}")
            
            if embeddings_store:
                # Analyze embeddings
                all_embeddings = list(embeddings_store.values())
                embedding_shapes = [emb.shape for emb in all_embeddings]
                embedding_sizes = [emb.size for emb in all_embeddings]
                
                print(f"\nEmbedding Statistics:")
                print(f"  Embedding dimension: {embedding_shapes[0] if embedding_shapes else 'N/A'}")
                print(f"  Total embedding elements: {sum(embedding_sizes)}")
                print(f"  Average embedding size: {np.mean(embedding_sizes):.1f}")
                
                # Memory usage estimation
                total_bytes = sum(emb.nbytes for emb in all_embeddings)
                print(f"  Memory usage: {total_bytes / 1024 / 1024:.2f} MB")
            
            # Analyze metadata
            if metadata_store:
                texts = [meta.get('text', '') for meta in metadata_store.values()]
                text_lengths = [len(text) for text in texts]
                
                print(f"\nText Statistics:")
                print(f"  Average text length: {np.mean(text_lengths):.1f} characters")
                print(f"  Longest text: {max(text_lengths) if text_lengths else 0} characters")
                print(f"  Shortest text: {min(text_lengths) if text_lengths else 0} characters")
                
                # Count conversation types
                conversations = 0
                for text in texts:
                    if 'User: ' in text and 'Jarvis: ' in text:
                        conversations += 1
                
                print(f"  Conversations detected: {conversations}")
                print(f"  Non-conversation entries: {len(texts) - conversations}")
            
            # File information
            if os.path.exists(self.embeddings_file):
                file_size = os.path.getsize(self.embeddings_file)
                file_modified = datetime.fromtimestamp(os.path.getmtime(self.embeddings_file))
                print(f"\nFile Information:")
                print(f"  File: {self.embeddings_file}")
                print(f"  Size: {file_size / 1024 / 1024:.2f} MB")
                print(f"  Last modified: {file_modified}")
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
    
    def export_memories_json(self, output_file: str = None):
        """Export all memories to a JSON file for backup/analysis"""
        if not self.embedder:
            print("❌ No embedding system loaded")
            return
        
        if output_file is None:
            output_file = f"jarvis_memories_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            embeddings_store = getattr(self.embedder, 'embeddings_store', {})
            metadata_store = getattr(self.embedder, 'metadata_store', {})
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_memories': len(embeddings_store),
                'memories': []
            }
            
            for memory_id in embeddings_store.keys():
                if memory_id in metadata_store:
                    metadata = metadata_store[memory_id]
                    memory_export = {
                        'id': memory_id,
                        'text': metadata.get('text', ''),
                        'created_at': metadata.get('created_at', ''),
                        'embedding_shape': str(metadata.get('embedding_shape', '')),
                        'has_embedding': True
                    }
                    export_data['memories'].append(memory_export)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Exported {len(export_data['memories'])} memories to {output_file}")
            
        except Exception as e:
            print(f"Error exporting memories: {e}")
    
    def delete_memory(self, memory_id: str, confirm: bool = False):
        """Delete a specific memory (requires confirmation)"""
        if not confirm:
            print("⚠️  Memory deletion requires confirmation. Use --confirm flag")
            return
        
        if not self.embedder:
            print("❌ No embedding system loaded")
            return
        
        try:
            embeddings_store = getattr(self.embedder, 'embeddings_store', {})
            metadata_store = getattr(self.embedder, 'metadata_store', {})
            
            if memory_id not in embeddings_store:
                print(f"❌ Memory ID '{memory_id}' not found")
                return
            
            # Remove from both stores
            del embeddings_store[memory_id]
            if memory_id in metadata_store:
                del metadata_store[memory_id]
            
            # Save the updated pickle file
            self.embedder._save_to_pickle()
            
            print(f"✅ Deleted memory: {memory_id}")
            
        except Exception as e:
            print(f"Error deleting memory: {e}")
    
    def clear_all_memories(self, confirm: bool = False):
        """Clear all memories (requires confirmation)"""
        if not confirm:
            print("⚠️  Clearing all memories requires confirmation. Use --confirm flag")
            return
        
        if not self.embedder:
            print("❌ No embedding system loaded")
            return
        
        try:
            self.embedder.clear_all_embeddings()
            print("✅ All memories cleared")
            
        except Exception as e:
            print(f"Error clearing memories: {e}")


def main():
    """CLI interface for the memory viewer"""
    parser = argparse.ArgumentParser(description="Jarvis Memory Viewer and Manager")
    parser.add_argument("--file", "-f", default="jarvis_memory.pkl", 
                        help="Path to memory pickle file")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all memories")
    list_parser.add_argument("--limit", "-l", type=int, help="Limit number of results")
    list_parser.add_argument("--embeddings", "-e", action="store_true", 
                            help="Show embedding vectors")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--count", "-n", type=int, default=5, 
                              help="Number of results to return")
    search_parser.add_argument("--threshold", "-t", type=float, default=0.7,
                              help="Similarity threshold")
    
    # Get command
    get_parser = subparsers.add_parser("get", help="Get specific memory by ID")
    get_parser.add_argument("memory_id", help="Memory ID to retrieve")
    
    # Stats command
    subparsers.add_parser("stats", help="Show memory statistics")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export memories to JSON")
    export_parser.add_argument("--output", "-o", help="Output file name")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a specific memory")
    delete_parser.add_argument("memory_id", help="Memory ID to delete")
    delete_parser.add_argument("--confirm", action="store_true", 
                              help="Confirm deletion")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear all memories")
    clear_parser.add_argument("--confirm", action="store_true", 
                             help="Confirm clearing all memories")
    
    args = parser.parse_args()
    
    # Create viewer instance
    viewer = JarvisMemoryViewer(args.file)
    
    # Execute command
    if args.command == "list":
        viewer.list_all_memories(limit=args.limit, show_embeddings=args.embeddings)
    elif args.command == "search":
        viewer.search_memories(args.query, n=args.count, threshold=args.threshold)
    elif args.command == "get":
        viewer.get_memory_by_id(args.memory_id)
    elif args.command == "stats":
        viewer.get_stats()
    elif args.command == "export":
        viewer.export_memories_json(args.output)
    elif args.command == "delete":
        viewer.delete_memory(args.memory_id, confirm=args.confirm)
    elif args.command == "clear":
        viewer.clear_all_memories(confirm=args.confirm)
    else:
        # If no command specified, show stats by default
        viewer.get_stats()


if __name__ == "__main__":
    main()