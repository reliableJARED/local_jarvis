import re
import spacy
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
from collections import defaultdict
import uuid
# Import your existing classes (assuming they're available)
# from your_modules import rdfMemRecall, Qwen, SpacyNounExtractor, faiss_search
from qwen_ import Qwen
from noun_finder import SpacyNounExtractor
from faiss_db import MemRecall
from rdfvec import rdfMemRecall
from mxbai_embed import MxBaiEmbedder


import logging
logging.basicConfig(level=logging.INFO)


# Import your classes (assuming they're available)
# from your_modules import rdfMemRecall, SpacyNounExtractor, MemRecall, Qwen

@dataclass
class Subject:
    """Represents an active subject in the focus manager."""
    name: str
    relevance_score: float
    first_mentioned: int
    last_mentioned: int
    entity_type: str = "UNKNOWN"
    mentions_count: int = 1
    
    def decay(self, decay_factor: float = 0.9):
        """Apply decay to relevance score."""
        self.relevance_score *= decay_factor

class ContextualProcessor:
    """
    Main processing system that integrates knowledge graph, vector DB, 
    focus management, and LLM processing for incremental document analysis.
    """
    
    def __init__(self, 
                 knowledge_graph: 'rdfMemRecall',
                 vector_db: 'MemRecall', 
                 llm: 'Qwen',
                 text_analyzer: 'SpacyNounExtractor',
                 embedding_model: 'MxBaiEmbedder',
                 max_context_sentences: int = 15):
        
        self.kg = knowledge_graph
        self.db = vector_db
        self.llm = llm
        self.nlp = text_analyzer
        self.embeder = embedding_model
        self.max_context_sentences = max_context_sentences

    @staticmethod
    def split_sentences(text):
        """
        Split text into sentences, handling common prefixes and quoted punctuation.
        
        Args:
            text (str): The input text to split
            
        Returns:
            list: A list of sentences with whitespace stripped
        """
        # Common prefixes that end with periods but aren't sentence endings
        prefixes = r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|Inc|Corp|Ltd|Co)\.'
        
        # Replace these prefixes temporarily with a placeholder that won't interfere
        placeholder = "XXXXPREFIXXXXX"
        text_protected = re.sub(prefixes, lambda m: m.group().replace('.', placeholder), text)
        
        # Split on sentence-ending punctuation followed by whitespace or end of string
        # This pattern looks for [.!?] followed by optional quotes/punctuation, then whitespace
        sentences = re.split(r'[.!?]+["\']*\s+', text_protected)
        
        # Handle the case where text ends with punctuation but no whitespace
        # The split will leave the final sentence intact
        
        # Restore the original periods and clean up
        sentences = [
            sentence.replace(placeholder, '.').strip() 
            for sentence in sentences 
            if sentence.strip()
        ]
        
        # Remove any remaining placeholder artifacts and empty sentences
        sentences = [s for s in sentences if s and placeholder not in s]
        
        return sentences
    
    @staticmethod
    def fix_malformed_json(text):
        """
        Attempts to fix common malformed JSON issues from LLM output
        """
        # Strip whitespace
        text = text.strip()
        
        # Remove any prefix before the first '{' or '['
        # This handles cases like "18:{'read_more': False, ...}"
        json_start = max(text.find('{'), text.find('['))
        if json_start > 0:
            text = text[json_start:]
        
        # Replace single quotes with double quotes (most common issue)
        text = text.replace("'", '"')
        
        # Fix common boolean/null issues
        text = re.sub(r'\bTrue\b', 'true', text)
        text = re.sub(r'\bFalse\b', 'false', text)
        text = re.sub(r'\bNone\b', 'null', text)
        
        # Try to parse and return
        try:
            print(f"FIX: {text}")
            return json.loads(text)
        except json.JSONDecodeError as e:
            # If still failing, try some more aggressive fixes
            
            # Remove trailing commas
            text = re.sub(r',(\s*[}\]])', r'\1', text)
            
            # Add missing quotes around unquoted keys (basic cases)
            text = re.sub(r'(\w+):', r'"\1":', text)
            
            # Try again
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Last resort - return a default dict with error info
                print(f"Warning: Could not parse JSON: {text[:100]}...")
                return {"error": "malformed_json", "original": text}
        
       
# Example usage function
def create_processor_pipeline(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Create a complete processing pipeline with all components."""
    
    # Initialize components
    folder = "./demo_rdf_data"
    kg = rdfMemRecall(data_directory=folder, graph_name="demo_graph")
    db = MemRecall(data_directory=folder)
    llm = Qwen(model_name=model_name, auto_append_conversation=False)
    nlp = SpacyNounExtractor()
    embeder = MxBaiEmbedder()
    # Create processor
    processor = ContextualProcessor(
        knowledge_graph=kg,
        vector_db=db,
        llm=llm,
        text_analyzer=nlp,
        embedding_model = embeder
    )
    
    return processor

# Example usage
if __name__ == "__main__":
   
    # Process a sample document
    sample_text = """
    John walked into the coffee shop on Main Street. He ordered his usual latte from Sarah, the barista.
    Sarah had been working there for three years. She remembered that John always came in at 9 AM sharp.
    Today was different though - John seemed worried about something. He kept checking his phone.
    It is a truth universally acknowledged, that a single man in possession
of a good fortune must be in want of a wife.

However little known the feelings or views of such a man may be on his
first entering a neighbourhood, this truth is so well fixed in the minds
of the surrounding families, that he is considered as the rightful
property of some one or other of their daughters.

“My dear Mr. Bennet,” said his lady to him one day, “have you heard that
Netherfield Park is let at last?”

Mr. Bennet replied that he had not.

“But it is,” returned she; “for Mrs. Long has just been here, and she
told me all about it.”

Mr. Bennet made no answer.

“Do not you want to know who has taken it?” cried his wife, impatiently.

“_You_ want to tell me, and I have no objection to hearing it.”
xyz.
hiccup.
This Agreement...constitutes the sole and entire agreement of the Parties with respect to the subject matter contained herein, and supersedes all prior and contemporaneous understandings, agreements, representations, and warranties, both written and oral, with respect to such subject matter.
The Receiving Party shall maintain the confidentiality...and shall not disclose or use such Confidential Information for any purpose other than as explicitly permitted under this Agreement.
Explains how to do something, usually with a step-by-step format, like a recipe.
Types of Narration
These are variations of how the story is told: 
First-Person: The story is told by a character using "I" or "we". 
Second-Person: The narrator directly addresses the reader using "you". 
Third-Person: An external narrator tells the story using "he,his," "she,her," or "they,their,". 
Omniscient Narration: The narrator knows everything about all characters and events. 
Limited Omniscient Narration: The narrator knows the thoughts and feelings of only one or a few characters.
3.1 Formula Mass and the Mole ConceptThe Mole: A Counting Unit for ChemistsImagine you are baking a cake. 
The recipe calls for two cups of flour, but you wouldn't count out every single grain of flour, would you? 
Similarly, when chemists work with elements and compounds, they need a convenient way to count the enormous number of atoms, ions, and molecules involved in a reaction. 
This is where the concept of the mole comes in.A mole (mol) is a fundamental unit in chemistry that represents a specific number of particles, much like a dozen represents 12 items. 
For any substance, one mole is defined as the number of atoms in exactly 12 grams of pure carbon-12 (\({}^{12}\text{C}\)). 
This number is known as Avogadro's number (\(N_{A}\)).\(N_{A}=6.022\times 10^{23}\text{\ particles\ per\ mole}\)So, one mole of carbon atoms contains \(6.022\times 10^{23}\) carbon atoms. 
One mole of water molecules contains \(6.022\times 10^{23}\) water molecules, and one mole of sodium chloride (NaCl) formula units contains \(6.022\times 10^{23}\) formula units of NaCl.
Molar Mass: Mass per MoleThe molar mass of a substance is the mass in grams of one mole of that substance. For an element, the molar mass is numerically equal to its average atomic mass in atomic mass units (amu), which you can find on the periodic table.
For example, the periodic table lists the atomic mass of oxygen as 16.00 amu. Therefore, the molar mass of oxygen atoms is 16.00 g/mol.
Similarly, the atomic mass of aluminum is 26.98 amu, so its molar mass is 26.98 g/mol.
The molar mass of a compound is the sum of the molar masses of all the elements in its chemical formula, taking into account the number of atoms of each element. 
    """
    
    # Process the document
    results = ContextualProcessor.split_sentences(sample_text)
    
    # Print results
    for i,result in enumerate(results):
        print(f"{i}:{result}\n")

    #Instance of our processor
    mind = create_processor_pipeline(model_name = "Qwen/Qwen2.5-7B-Instruct")

    #Classification System Prompt
    sp_subject_ID ="YOU MUST respond with JSON schema. You are an editor and you classify sentences\
    Attempt to determine what type of text the input sentnece is. If you can not, and need more context set read_more True\
        Types of Text:\
            \nNarrative:story element told from First, Second or Third person. Can also be narration or character monologue.\
            \nDescriptive:Focuses on describing a person, place, or thing using vivid language.\
            \nExpository:Explains or informs by presenting facts, definitions, or processes clearly and concisely. Can also explain how to do something.\
            \nArgumentative:Persuades the reader to agree with a particular viewpoint by using logic and evidence.\
            \nUnknown: Insuffiecient information or context to classify.\
            \nYou MUST be certain of your selection, if it's possible to classify in multiple buckets read more to narrow down to a single selection.\
            \nYour responses should ALWAYS have format {'read_more':True/False,'type_of_text':'narrative/descriptive/expository/argumentative/unknown'}\
                                    RESPOND WITH JSON"
    mind.llm._update_system_prompt(sp_subject_ID)

    #classify text
    i = 0
    classification_results = {'narrative':[],'descriptive':[],'expository':[],'argumentative':[]}
    subject = ['none']
    context = 'none'
    while i < len(results):
        #reset system prompt
        mind.llm._update_system_prompt(sp_subject_ID)

        sentence = results[i]  # Don't pop yet - just read
        res = mind.llm.generate_response(sentence)
        print(f"LLM RESPNOSE: {res}")
        res_dict = mind.fix_malformed_json(res)#ensure no JSON issues
        print(f"\n{i}:{res}\n")

        if res_dict.get('read_more', False) or (res_dict.get('type_of_text').lower() == 'unknown'):
            print("\n\nREADING AHEAD")
            print("+"*50)
            
            # Start building the combined sentence
            combined_sentences = sentence
            read_ahead_count = 1  # How many additional sentences we've read
            
            print(f"Starting at sentence {i}")
            
            while ( res_dict.get('read_more', False) or (res_dict.get('type_of_text').lower() == 'unknown')) and (i + read_ahead_count) < len(results):
                # Add the next sentence
                next_sentence_idx = i + read_ahead_count
                print(f"Reading ahead to sentence {next_sentence_idx}")
                combined_sentences += f". {results[next_sentence_idx]}"
                read_ahead_count += 1
                
                # Check if we want to read even more
                res = mind.llm.generate_response(combined_sentences)
                res_dict = mind.fix_malformed_json(res)#ensure no JSON issues
            
            # Now we've processed sentences i through i+read_ahead_count-1
            # Skip ahead to the next unprocessed sentence
            i += read_ahead_count
            print(f"Read ahead complete. Next sentence to process: {i}")
            classification_results[res_dict.get('type_of_text').lower()].append(combined_sentences)

            sp_focus = """User has provided a {type_} sentence or passage. You need to determine the input text establishes or changes the subject or context. \nSUBJECT(S):{subject} \nCONTEXT:{context}\
            \nYour JSON responses should ALWAYS have format {{'subjects':['string name','string name',...]','context':'string context of text up to this point'}}\
            RESPOND WITH JSON" """.format(
                                                        type_=res_dict.get('type_of_text').lower(),
                                                        subject=str(subject),
                                                        context=context
                                                    )

            mind.llm.generate_response(sentence)
            res = mind.llm.generate_response(combined_sentences)
            res_dict = mind.fix_malformed_json(res)#ensure no JSON issues
            
            print("="*50)
            print(res)
            subject = res_dict.get('subject')
            context = res_dict.get('context')
            print(res_dict)
            print("="*50)

            
        else:
            # No read ahead needed, just move to next sentence
            classification_results[res_dict.get('type_of_text').lower()].append(sentence)
            i += 1

            sp_focus = """User has provided a {type_} sentence or passage. You need to determine the input text establishes or changes the subject or context. \nSUBJECT(S):{subject} \nCONTEXT:{context}\
            \nYour responses should ALWAYS have format {{'subjects':['string name','string name',...]','context':'string context of text up to this point'}}\
                                        RESPOND WITH JSON" """.format(
                                                        type_=res_dict.get('type_of_text').lower(),
                                                        subject=str(subject),
                                                        context=context
                                                    )


            mind.llm._update_system_prompt(sp_focus)
            res = mind.llm.generate_response(sentence)
            res_dict = mind.fix_malformed_json(res)#ensure no JSON issues
            print("="*50)
            print(res)
            subject = res_dict.get('subject')
            context = res_dict.get('context')
            print(res_dict)
            print("="*50)

        
 
        
        

    print(classification_results)
    