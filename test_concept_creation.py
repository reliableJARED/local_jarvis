#!/usr/bin/env python3
"""
Test script to demonstrate the create_memory_concepts() functionality
"""

from prefrontal2 import Hippocampus, MemoryPage, pseudo_vggish_embed, pseudo_beit_embed
from qwen3_emotion_memory import MxBaiEmbedder
import time
import numpy as np

def create_sample_memories():
    """Create some sample memories for testing concept formation"""
    
    # Initialize the hippocampus
    hippocampus = Hippocampus(
        audio_embedder=pseudo_vggish_embed(),
        img_embedder=pseudo_beit_embed(),
        txt_embedder=MxBaiEmbedder(),
        memory_store_file="test_memory_store.pkl",
        concept_store_file="test_concept_store.pkl"
    )
    
    # Create diverse sample memories covering multiple topics and timeframes
    sample_memories = [
        # John's dog Spike storyline
        {
            "text": "John told me about his dog Spike today. Spike is a golden retriever who loves to play fetch.",
            "timestamp": time.time() - (21 * 24 * 3600)  # 21 days ago
        },
        {
            "text": "I saw John walking Spike in the park. The dog looked very energetic and happy.",
            "timestamp": time.time() - (18 * 24 * 3600)  # 18 days ago
        },
        {
            "text": "Spike got into John's trash again. John found coffee grounds all over the kitchen floor.",
            "timestamp": time.time() - (15 * 24 * 3600)  # 15 days ago
        },
        {
            "text": "John took Spike to obedience training. The trainer said Spike is very smart but stubborn.",
            "timestamp": time.time() - (12 * 24 * 3600)  # 12 days ago
        },
        {
            "text": "John mentioned that Spike has been sick lately. He took him to the vet yesterday.",
            "timestamp": time.time() - (3 * 24 * 3600)  # 3 days ago
        },
        {
            "text": "Good news! John said Spike is feeling much better after the vet visit. The medication worked.",
            "timestamp": time.time() - (1 * 24 * 3600)  # 1 day ago
        },
        
        # Cooking and recipe storyline
        {
            "text": "I tried making pasta today. The recipe called for fresh basil and garlic.",
            "timestamp": time.time() - (20 * 24 * 3600)  # 20 days ago
        },
        {
            "text": "My pasta cooking skills are improving. I learned how to make a good marinara sauce.",
            "timestamp": time.time() - (17 * 24 * 3600)  # 17 days ago
        },
        {
            "text": "Made homemade pizza tonight. The dough recipe I found online turned out perfectly crispy.",
            "timestamp": time.time() - (14 * 24 * 3600)  # 14 days ago
        },
        {
            "text": "Tried baking bread for the first time. It didn't rise properly, need to check yeast expiration.",
            "timestamp": time.time() - (11 * 24 * 3600)  # 11 days ago
        },
        {
            "text": "Successfully made sourdough starter. It's been bubbling for three days now.",
            "timestamp": time.time() - (8 * 24 * 3600)  # 8 days ago
        },
        {
            "text": "Baked my first successful sourdough loaf! The crust was perfect and the inside had great texture.",
            "timestamp": time.time() - (5 * 24 * 3600)  # 5 days ago
        },
        
        # Work and career storyline
        {
            "text": "Started a new project at work involving machine learning and natural language processing.",
            "timestamp": time.time() - (25 * 24 * 3600)  # 25 days ago
        },
        {
            "text": "The ML model training is taking longer than expected. Need to optimize the hyperparameters.",
            "timestamp": time.time() - (22 * 24 * 3600)  # 22 days ago
        },
        {
            "text": "Had a breakthrough with the NLP pipeline. The accuracy improved from 75% to 89%.",
            "timestamp": time.time() - (19 * 24 * 3600)  # 19 days ago
        },
        {
            "text": "Presented the ML project to the team. Everyone was impressed with the results.",
            "timestamp": time.time() - (16 * 24 * 3600)  # 16 days ago
        },
        {
            "text": "Got approval to deploy the model to production. DevOps team is setting up the infrastructure.",
            "timestamp": time.time() - (13 * 24 * 3600)  # 13 days ago
        },
        {
            "text": "The model went live today! Monitoring shows it's performing well in production.",
            "timestamp": time.time() - (10 * 24 * 3600)  # 10 days ago
        },
        
        # Health and fitness storyline
        {
            "text": "Decided to start running again after months of being sedentary. Ran 2 miles today.",
            "timestamp": time.time() - (30 * 24 * 3600)  # 30 days ago
        },
        {
            "text": "My legs are so sore from yesterday's run. Need to remember to stretch more.",
            "timestamp": time.time() - (29 * 24 * 3600)  # 29 days ago
        },
        {
            "text": "Bought proper running shoes today. The store did a gait analysis - I overpronate.",
            "timestamp": time.time() - (26 * 24 * 3600)  # 26 days ago
        },
        {
            "text": "Ran 3 miles without stopping! The new shoes make a huge difference.",
            "timestamp": time.time() - (23 * 24 * 3600)  # 23 days ago
        },
        {
            "text": "Signed up for a 5K race next month. Need to keep training consistently.",
            "timestamp": time.time() - (20 * 24 * 3600)  # 20 days ago
        },
        {
            "text": "Hit a personal best today: 5 miles in 45 minutes. Training is really paying off.",
            "timestamp": time.time() - (7 * 24 * 3600)  # 7 days ago
        },
        {
            "text": "Completed my first 5K race! Finished in 24:30, better than my goal of 25 minutes.",
            "timestamp": time.time() - (2 * 24 * 3600)  # 2 days ago
        },
        
        # Family and relationships
        {
            "text": "Mom called today. She's planning a family reunion for next summer.",
            "timestamp": time.time() - (28 * 24 * 3600)  # 28 days ago
        },
        {
            "text": "My sister Sarah got engaged! The proposal was at the beach where they first met.",
            "timestamp": time.time() - (24 * 24 * 3600)  # 24 days ago
        },
        {
            "text": "Helped Dad fix his computer today. He had somehow installed 5 different antivirus programs.",
            "timestamp": time.time() - (21 * 24 * 3600)  # 21 days ago
        },
        {
            "text": "Had dinner with my college friends. It's amazing how we can pick up right where we left off.",
            "timestamp": time.time() - (18 * 24 * 3600)  # 18 days ago
        },
        {
            "text": "Grandma's 85th birthday party was wonderful. Four generations all together.",
            "timestamp": time.time() - (9 * 24 * 3600)  # 9 days ago
        },
        {
            "text": "Sarah asked me to be her maid of honor. I'm so excited and honored!",
            "timestamp": time.time() - (6 * 24 * 3600)  # 6 days ago
        },
        
        # Travel and exploration
        {
            "text": "Booked a trip to Japan for next month. Always wanted to visit Tokyo and Kyoto.",
            "timestamp": time.time() - (35 * 24 * 3600)  # 35 days ago
        },
        {
            "text": "Started learning basic Japanese phrases. 'Arigato gozaimasu' is harder to pronounce than I thought.",
            "timestamp": time.time() - (32 * 24 * 3600)  # 32 days ago
        },
        {
            "text": "Reading about Japanese culture and etiquette. Don't want to accidentally offend anyone.",
            "timestamp": time.time() - (27 * 24 * 3600)  # 27 days ago
        },
        {
            "text": "Got my passport renewed and travel insurance sorted. Trip is getting real!",
            "timestamp": time.time() - (15 * 24 * 3600)  # 15 days ago
        },
        
        # Hobbies and interests
        {
            "text": "Started learning guitar again. My fingers hurt but I managed to play 3 chords.",
            "timestamp": time.time() - (33 * 24 * 3600)  # 33 days ago
        },
        {
            "text": "Can now play 'Wonderwall' somewhat recognizably. Classic beginner song achieved!",
            "timestamp": time.time() - (28 * 24 * 3600)  # 28 days ago
        },
        {
            "text": "Joined a local photography club. They're organizing a nature walk photo session.",
            "timestamp": time.time() - (25 * 24 * 3600)  # 25 days ago
        },
        {
            "text": "The photography walk was amazing. Got some great shots of birds and flowers.",
            "timestamp": time.time() - (22 * 24 * 3600)  # 22 days ago
        },
        {
            "text": "One of my photos won third place in the club's monthly contest!",
            "timestamp": time.time() - (12 * 24 * 3600)  # 12 days ago
        },
        
        # Random life events and observations
        {
            "text": "Saw the most beautiful sunset today. The sky was painted in shades of orange and pink.",
            "timestamp": time.time() - (31 * 24 * 3600)  # 31 days ago
        },
        {
            "text": "Coffee shop ran out of my usual order. Tried something new and actually liked it better.",
            "timestamp": time.time() - (26 * 24 * 3600)  # 26 days ago
        },
        {
            "text": "Power went out during the storm last night. Ended up having a great conversation by candlelight.",
            "timestamp": time.time() - (19 * 24 * 3600)  # 19 days ago
        },
        {
            "text": "Found a $20 bill on the sidewalk today. Decided to donate it to the local food bank.",
            "timestamp": time.time() - (14 * 24 * 3600)  # 14 days ago
        },
        {
            "text": "Neighbor's cat keeps coming to my window. I think it wants to be friends.",
            "timestamp": time.time() - (8 * 24 * 3600)  # 8 days ago
        },
        {
            "text": "Finally finished reading that book I started months ago. The ending was worth the wait.",
            "timestamp": time.time() - (4 * 24 * 3600)  # 4 days ago
        },
        {
            "text": "Watched a documentary about ocean conservation. Really made me think about plastic usage.",
            "timestamp": time.time()  # Today
        }
    ]
    
    # Create MemoryPage objects and add them to hippocampus
    for mem_data in sample_memories:
        memory = MemoryPage(
            text=mem_data["text"],
            text_embedding=hippocampus.txt_embedder.embed_text_string(mem_data["text"]),
            timestamp=mem_data["timestamp"]
        )
        hippocampus.save_memory_page(memory)
    
    print(f"Created {len(sample_memories)} sample memories")
    return hippocampus

def test_concept_creation():
    """Test the concept creation process"""
    
    print("=== Testing Memory Concept Creation ===\n")
    
    # Create sample memories
    hippocampus = create_sample_memories()
    
    print(f"Initial state: {len(hippocampus.memories)} memories, {len(hippocampus.memory_concepts)} concepts\n")
    
    # Run concept creation
    print("Running create_memory_concepts()...")
    hippocampus.create_memory_concepts()
    
    print(f"\nFinal state: {len(hippocampus.memories)} memories, {len(hippocampus.memory_concepts)} concepts\n")
    
    # Display created concepts
    print("=== Created Concepts ===")
    for i, concept in enumerate(hippocampus.memory_concepts, 1):
        print(f"\nConcept {i}: {concept.theme}")
        print(f"  Summary: {concept.summary}")
        print(f"  Memory IDs: {len(concept.memory_ids)} memories")
        print(f"  Key Entities: {concept.key_entities}")
        print(f"  Key Nouns: {concept.key_nouns}")
        print(f"  Temporal Span: {concept.temporal_span_days:.1f} days")
        print(f"  Importance Score: {concept.importance_score:.3f}")
        print(f"  Consolidation Strength: {concept.consolidation_strength:.3f}")
    
    # Test updating concepts with new information
    print("\n=== Testing Concept Updates ===")
    
    # Add a new memory about Spike
    new_memory = MemoryPage(
        text="John brought Spike to the dog park today. Spike made friends with a poodle named Max.",
        text_embedding=hippocampus.txt_embedder.embed_text_string("John brought Spike to the dog park today. Spike made friends with a poodle named Max."),
        timestamp=time.time() + 3600  # 1 hour from now
    )
    
    hippocampus.save_memory_page(new_memory)
    print(f"Added new memory: '{new_memory.text}'")
    
    # Run concept creation again to update existing concepts
    print("\nRunning create_memory_concepts() again to update concepts...")
    hippocampus.create_memory_concepts()
    
    # Display updated concepts
    print("\n=== Updated Concepts ===")
    for i, concept in enumerate(hippocampus.memory_concepts, 1):
        if "spike" in concept.theme.lower() or "john" in concept.theme.lower():
            print(f"\nUpdated Concept {i}: {concept.theme}")
            print(f"  Summary: {concept.summary}")
            print(f"  Memory IDs: {len(concept.memory_ids)} memories")
            print(f"  Key Entities: {concept.key_entities}")
            print(f"  Temporal Span: {concept.temporal_span_days:.1f} days")
            print(f"  Last Updated: {time.ctime(concept.last_updated)}")

if __name__ == "__main__":
    test_concept_creation()
