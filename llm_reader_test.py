import re
import spacy
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import time
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
            return text
        except json.JSONDecodeError as e:
            # If still failing, try some more aggressive fixes

            # Remove trailing commas
            text = re.sub(r',(\s*[}\]])', r'\1', text)

            # Add missing quotes around unquoted keys (basic cases)
            text = re.sub(r'(\w+):', r'"\1":', text)

            # Try again
            try:
                return text
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
   
    # Process a sample document (random)
    sample_text = """
    John walked into the coffee shop on Main Street. He ordered his usual latte from Sarah, the barista.
    Sarah had been working there for three years. She remembered that John always came in at 9 AM sharp.
    Today was different though - John seemed worried about something. He kept checking his phone. It is a truth universally acknowledged, that a single man in possession
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
End of chapter
When Jane and Elizabeth were alone, the former, who had been cautious in
her praise of Mr. Bingley before, expressed to her sister how very much
she admired him.

“He is just what a young-man ought to be,” said she, “sensible,
good-humoured, lively; and I never saw such happy manners! so much ease,
with such perfect good breeding!”

“He is also handsome,” replied Elizabeth, “which a young man ought
likewise to be if he possibly can. His character is thereby complete.”

“I was very much flattered by his asking me to dance a second time. I
did not expect such a compliment.”

“Did not you? _I_ did for you. But that is one great difference between
us. Compliments always take _you_ by surprise, and _me_ never. What
could be more natural than his asking you again? He could not help
seeing that you were about five times as pretty as every other woman in
the room. No thanks to his gallantry for that. Well, he certainly is
very agreeable, and I give you leave to like him. You have liked many a
stupider person.”

“Dear Lizzy!”

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
3.1 Formula Mass and the Mole ConceptThe Mole: A Counting Unit for Chemists Imagine you are baking a cake. 
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
    
    #pride and prejudice (literature)
    sample_text1 = """
Chapter I.]


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

[Illustration:

“He came down to see the place”

[_Copyright 1894 by George Allen._]]

This was invitation enough.

“Why, my dear, you must know, Mrs. Long says that Netherfield is taken
by a young man of large fortune from the north of England; that he came
down on Monday in a chaise and four to see the place, and was so much
delighted with it that he agreed with Mr. Morris immediately; that he is
to take possession before Michaelmas, and some of his servants are to be
in the house by the end of next week.”

“What is his name?”

“Bingley.”

“Is he married or single?”

“Oh, single, my dear, to be sure! A single man of large fortune; four or
five thousand a year. What a fine thing for our girls!”

“How so? how can it affect them?”

“My dear Mr. Bennet,” replied his wife, “how can you be so tiresome? You
must know that I am thinking of his marrying one of them.”

“Is that his design in settling here?”

“Design? Nonsense, how can you talk so! But it is very likely that he
_may_ fall in love with one of them, and therefore you must visit him as
soon as he comes.”

“I see no occasion for that. You and the girls may go--or you may send
them by themselves, which perhaps will be still better; for as you are
as handsome as any of them, Mr. Bingley might like you the best of the
party.”

“My dear, you flatter me. I certainly _have_ had my share of beauty, but
I do not pretend to be anything extraordinary now. When a woman has five
grown-up daughters, she ought to give over thinking of her own beauty.”

“In such cases, a woman has not often much beauty to think of.”

“But, my dear, you must indeed go and see Mr. Bingley when he comes into
the neighbourhood.”

“It is more than I engage for, I assure you.”

“But consider your daughters. Only think what an establishment it would
be for one of them. Sir William and Lady Lucas are determined to go,
merely on that account; for in general, you know, they visit no new
comers. Indeed you must go, for it will be impossible for _us_ to visit
him, if you do not.”

“You are over scrupulous, surely. I dare say Mr. Bingley will be very
glad to see you; and I will send a few lines by you to assure him of my
hearty consent to his marrying whichever he chooses of the girls--though
I must throw in a good word for my little Lizzy.”

“I desire you will do no such thing. Lizzy is not a bit better than the
others: and I am sure she is not half so handsome as Jane, nor half so
good-humoured as Lydia. But you are always giving _her_ the preference.”

“They have none of them much to recommend them,” replied he: “they are
all silly and ignorant like other girls; but Lizzy has something more of
quickness than her sisters.”

“Mr. Bennet, how can you abuse your own children in such a way? You take
delight in vexing me. You have no compassion on my poor nerves.”

“You mistake me, my dear. I have a high respect for your nerves. They
are my old friends. I have heard you mention them with consideration
these twenty years at least.”

“Ah, you do not know what I suffer.”

“But I hope you will get over it, and live to see many young men of four
thousand a year come into the neighbourhood.”

“It will be no use to us, if twenty such should come, since you will not
visit them.”

“Depend upon it, my dear, that when there are twenty, I will visit them
all.”

Mr. Bennet was so odd a mixture of quick parts, sarcastic humour,
reserve, and caprice, that the experience of three-and-twenty years had
been insufficient to make his wife understand his character. _Her_ mind
was less difficult to develope. She was a woman of mean understanding,
little information, and uncertain temper. When she was discontented, she
fancied herself nervous. The business of her life was to get her
daughters married: its solace was visiting and news."""
    
    #psycology of managment (text book)
    sample_text2 = """THE PSYCHOLOGY OF MANAGEMENT


                             CHAPTER I

                 DESCRIPTION AND GENERAL OUTLINE OF

    DEFINITION OF PSYCHOLOGY OF MANAGEMENT.--The Psychology of
Management, as here used, means,--the effect of the mind that is
directing work upon that work which is directed, and the effect of
this undirected and directed work upon the mind of the worker.

    IMPORTANCE OF THE SUBJECT.--Before defining the terms that will
be used more in detail, and outlining the method of treatment to be
followed, it is well to consider the importance of the subject
matter of this book, for upon the reader's interest in the subject,
and his desire, from the outset, to follow what is said, and to
respond to it, rests a large part of the value of this book.

    VALUE OF PSYCHOLOGY.--First of all, then, what is there in the
subject of psychology to demand the attention of the manager?

    Psychology, in the popular phrase, is "the study of the mind."
It has for years been included in the training of all teachers, and
has been one of the first steps for the student of philosophy; but
it has not, usually, been included among the studies of the young
scientific or engineering student, or of any students in other lines
than Philosophy and Education. This, not because its value as a
"culture subject" was not understood, but because the course of the
average student is so crowded with technical preparation necessary
to his life work, and because the practical value of psychology has
not been recognized. It is well recognized that the teacher must
understand the working of the mind in order best to impart his
information in that way that will enable the student to grasp it
most readily. It was not recognized that every man going out into
the world needs all the knowledge that he can get as to the working
of the human mind in order not only to give but to receive
information with the least waste and expenditure of energy, nor was
it recognized that in the industrial, as well as the academic world,
almost every man is a teacher.

    VALUE OF MANAGEMENT.--The second question demanding attention
is;--Of what value is the study of management?

    The study of management has been omitted from the student's
training until comparatively recently, for a very different reason
than was psychology. It was never doubted that a knowledge of
management would be of great value to anyone and everyone, and many
were the queer schemes for obtaining that knowledge after
graduation. It was doubted that management could be studied
otherwise than by observation and practice.[1] Few teachers, if any,
believed in the existence, or possibility, of a teaching science of
management. Management was assumed by many to be an art, by even
more it was thought to be a divinely bestowed gift or talent, rather
than an acquired accomplishment. It was common belief that one could
learn to manage only by going out on the work and watching other
managers, or by trying to manage, and not by studying about
management in a class room or in a text book; that watching a good
manager might help one, but no one could hope really to succeed who
had not "the knack born in him."

    With the advent of "Scientific Management," and its
demonstration that the best management is founded on laws that have
been determined, and can be taught, the study of management in the
class room as well as on the work became possible and actual.[2]

    VALUE OF PSYCHOLOGY OF MANAGEMENT.--Third, we must consider the
value of the study of the psychology of management.[3]

    This question, like the one that precedes it, is answered by
Scientific Management. It has demonstrated that the emphasis in
successful management lies on the _man_, not on the _work_; that
efficiency is best secured by placing the emphasis on the man, and
modifying the equipment, materials and methods to make the most of
the man. It has, further, recognized that the man's mind is a
controlling factor in his efficiency, and has, by teaching, enabled
the man to make the most of his powers.[4] In order to understand
this teaching element that is such a large part of management, a
knowledge of psychology is imperative; and this study of psychology,
as it applies to the work of the manager or the managed, is exactly
what the "psychology of management" is.

    FIVE INDICATIONS OF THIS VALUE.--In order to realize the
importance of the psychology of management it is necessary to
consider the following five points:--

    1. Management is a life study of every man who works with other
men. He must either manage, or be managed, or both; in any case, he
can never work to best advantage until he understands both the
psychological and managerial laws by which he governs or is governed.

    2. A knowledge of the underlying laws of management is the most
important asset that one can carry with him into his life work, even
though he will never manage any but himself. It is useful,
practical, commercially valuable.

    3. This knowledge is to be had _now_. The men who have it are
ready and glad to impart it to all who are interested and who will
pass it on.[5] The text books are at hand now. The opportunities for
practical experience in Scientific Management will meet all demands
as fast as they are made.

    4. The psychology of, that is, the mind's place in management is
only one part, element or variable of management; one of numerous,
almost numberless, variables.

    5. It is a division well fitted to occupy the attention of the
beginner, as well as the more experienced, because it is a most
excellent place to start the study of management. A careful study of
the relations of psychology to management should develop in the
student a method of attack in learning his selected life work that
should help him to grasp quickly the orderly array of facts that the
other variables, as treated by the great managers, bring to him.

    PURPOSE OF THIS BOOK.--It is scarcely necessary to mention that
this book can hope to do little more than arouse an interest in the
subject and point the way to the detailed books where such an
interest can be more deeply aroused and more fully satisfied.

    WHAT THIS BOOK WILL NOT DO.--It is not the purpose of this book
to give an exhaustive treatment of psychology. Neither is it
possible in this book to attempt to give a detailed account of
management in general, or of the Taylor plan of "Scientific
Management" so-called, in particular. All of the literature on the
subject has been carefully studied and reviewed for the purpose of
writing this book,--not only what is in print, but considerable that
is as yet in manuscript. No statement has been made that is not
along the line of the accepted thought and standardized practice of
the authorities. The foot notes have been prepared with great care.
By reading the references there given one can verify statements in
the text, and can also, if he desires, inform himself at length on
any branch of the subject that especially interests him.

    WHAT THIS BOOK WILL DO.--This book aims not so much to
instruct as to arouse an interest in its subject, and to point
the way whence instruction comes. If it can serve as an
introduction to psychology and to management, can suggest the
relation of these two fields of inquiries and can ultimately
enroll its readers as investigators in a resultant great field of
inquiry, it will have accomplished its aim.

    DEFINITION OF MANAGEMENT.--To discuss this subject more
in detail--

    First: What is "Management"?

    "Management," as defined by the Century Dictionary, is "the
art of managing by direction or regulation."

    Successful management of the old type was an art based on no
measurement. Scientific Management is an art based upon a
science,--upon laws deducted from measurement. Management continues
to be what it has always been,--the _art_ of directing activity.

    CHANGE IN THE ACCEPTED MEANING.--"Management," until recent
years, and the emphasis placed on Scientific Management was
undoubtedly associated, in the average mind, with the _managing_
part of the organization only, neglecting that vital part--the best
interests of the managed, almost entirely. Since we have come to
realize that management signifies the relationship between the
managing and the managed in doing work, a new realization of its
importance has come about.[6]

    INADEQUACY OF THE TERMS USED.--It is unfortunate that the
English language is so poor in synonyms in this field that the same
word must have two such different and conflicting meanings, for,
though the new definition of management be accepted, the "Fringe" of
associations that belong to the old are apt to remain.[7] The
thoughts of "knack, aptitude, tact, adroitness,"--not to speak of
the less desirable "Brute Force," "shrewdness, subtlety, cunning,
artifice, deceit, duplicity," of the older idea of management remain
in the background of the mind and make it difficult, even when one
is convinced that management is a science, to think and act as if
it were.

    It must be noticed and constantly remembered that one of the
greatest difficulties to overcome in studying management and its
development is the meaning of the terms used. It is most
unfortunate that the new ideas have been forced to content
themselves with old forms as best they may.

    PSYCHOLOGICAL INTEREST OF THE TERMS.--Psychology could ask no
more interesting subject than a study of the mental processes that
lie back of many of these terms. It is most unfortunate for the
obtaining of clearness, that new terms were not invented for the new
ideas. There is, however, an excellent reason for using the old
terms. By their use it is emphasized that the new thought is a
logical outgrowth of the old, and experience has proved that this
close relationship to established ideas is a powerful argument for
the new science; but such terms as "task," "foreman," "speed boss,"
"piece-rate" and "bonus," as used in the science of management,
suffer from misunderstanding caused by old and now false
associations. Furthermore, in order to compare old and new
interpretations of the ideas of management, the older terms of
management should have their traditional meanings only. The two sets
of meanings are a source of endless confusion, unwarranted
prejudice, and worse. This is well recognized by the authorities
on Management.

    THE THREE TYPES OF MANAGEMENT.--We note this inadequacy of
terms again when we discuss the various _types_ of Management.

    We may divide all management into three types--
      (1) Traditional
      (2) Transitory
      (3) Scientific, or measured functional.[8]

    Traditional Management, the first, has been variously called
"Military," "Driver," the "Marquis of Queensberry type," "Initiative
and Incentive Management," as well as "Traditional" management.

    DEFINITION OF THE FIRST TYPE.--In the first type, the power of
managing lies, theoretically at least, in the hands of one man, a
capable "all-around" manager. The line of authority and of
responsibility is clear, fixed and single. Each man comes in direct
contact with but one man above him. A man may or may not manage more
than one man beneath him, but, however this may be, he is managed by
but one man above him.

    PREFERABLE NAME FOR THE FIRST TYPE.--The names "Traditional," or
"Initiative and Incentive," are the preferable titles for this form
of management. It is true they lack in specificness, but the other
names, while aiming to be descriptive, really emphasize one feature
only, and in some cases with unfortunate results.

    THE NAME "MILITARY" INADVISABLE.--The direct line of authority
suggested the name "Military,"[9] and at the time of the adoption of
that name it was probably appropriate as well as complimentary.[10]
Appropriate in the respect referred to only, for the old type of
management varied so widely in its manifestations that the
comparison to the procedure of the Army was most inaccurate.
"Military" has always been a synonym for "systematized", "orderly,"
"definite," while the old type of management was more often quite
the opposite of the meaning of all these terms. The term "Military
Management" though often used in an uncomplimentary sense would,
today, if understood, be more complimentary than ever it was in the
past. The introduction of various features of Scientific Management
into the Army and Navy,--and such features are being incorporated
steadily and constantly,--is raising the standard of management
there to a high degree. This but renders the name "Military"
Management for the old type more inaccurate and misleading.

    It is plain that the stirring associations of the word
"military" make its use for the old type, by advocates of the old
type, a weapon against Scientific Management that only the careful
thinker can turn aside.

    THE NAMES "DRIVER" AND "MARQUIS OF QUEENSBERRY"
UNFORTUNATE.--The name "Driver" suggests an opposition between the
managers and the men, an opposition which the term "Marquis of
Queensberry" emphasizes. This term "Marquis of Queensberry" has been
given to that management which is thought of as a mental and
physical contest, waged "according to the rules of the game." These
two names are most valuable pictorially, or in furnishing oratorical
material. They are constant reminders of the constant desire of the
managers to get all the work that is possible out of the men, but
they are scarcely descriptive in any satisfactory sense, and the
visions they summon, while they are perhaps definite, are certainly,
for the inexperienced in management, inaccurate. In other words,
they usually lead to imagination rather than to perception.

    THE NAME "INITIATIVE AND INCENTIVE" AUTHORITATIVE.--The term
"Initiative and Incentive" is used by Dr. Taylor, and is fully
described by him.[11] The words themselves suggest, truly, that he
gives the old form of management its due. He does more than this. He
points out in his definition of the terms the likenesses between the
old and new forms.

    THE NAME "TRADITIONAL" BRIEF AND DESCRIPTIVE.--The only excuses
for the term "Traditional," since Dr. Taylor's term is available,
are its brevity and its descriptiveness. The fact that it is
indefinite is really no fault in it, as the subject it describes is
equally indefinite. The "fringe"[12] of this word is especially
good. It calls up ideas of information handed down from generation
to generation orally, the only way of teaching under the old type of
management. It recalls the idea of the inaccurate perpetuation of
unthinking custom, and the "myth" element always present in
tradition,--again undeniable accusations against the old type of
management. The fundamental idea of the tradition, that it is
_oral_, is the essence of the difference of the old type of
management from science, or even system, which must be written.

    It is not necessary to make more definite here the content of
this oldest type of management, rather being satisfied with the
extent, and accepting for working use the name "Traditional" with
the generally accepted definition of that name.

    DEFINITION OF THE SECOND TYPE OF MANAGEMENT.--The second type of
management is called "Interim" or "Transitory" management. It
includes all management that is consciously passing into Scientific
Management and embraces all stages, from management that has
incorporated one scientifically derived principle, to management
that has adopted all but one such principle.

    PREFERABLE NAME FOR SECOND TYPE OF MANAGEMENT.--Perhaps the name
"Transitory" is slightly preferable in that, though the element of
temporariness is present in both words, it is more strongly
emphasized in the latter. The usual habit of associating with it the
ideas of "fleeting, evanescent, ephemeral, momentary, short-lived,"
may have an influence on hastening the completion of the installing
of Scientific Management.

    DEFINITION OF THE THIRD TYPE OF MANAGEMENT.--The third form of
management is called "Ultimate," "measured Functional," or
"Scientific," management, and might also be called,--but for the
objection of Dr. Taylor, the "Taylor Plan of Management." This
differs from the first two types mentioned in that it is a definite
plan of management synthesized from scientific analysis of the data
of management. In other words, Scientific Management is that
management which is a science, i.e., which operates according to
known, formulated, and applied laws.[13]

    PREFERABLE NAME OF THE THIRD TYPE OF MANAGEMENT.--The name
"Ultimate" has, especially to the person operating under the
transitory stage, all the charm and inspiration of a goal. It has
all the incentives to accomplishment of a clearly circumscribed
task. Its very definiteness makes it seem possible of attainment. It
is a great satisfaction to one who, during a lifetime of managing
effort, has tried one offered improvement after another to be
convinced that he has found the right road at last. The name is,
perhaps, of greatest value in attracting the attention of the
uninformed and, as the possibilities of the subject can fulfill the
most exacting demands, the attention once secured can be held.

    The name "measured functional" is the most descriptive, but
demands the most explanation. The principle of functionalization is
one of the underlying, fundamental principles of Scientific
Management. It is not as necessary to stop to define it here, as it
is necessary to discuss the definition, the principle, and the
underlying psychology, at length later.

    The name "scientific" while in some respects not as appropriate
as are any of the other names, has already received the stamp of
popular approval. In derivation it is beyond criticism. It also
describes exactly, as has been said, the difference between the
older forms of management and the new. Even its "fringe" of
association is, or at least was when first used, all that could be
desired; but the name is, unfortunately, occasionally used
indiscriminately for any sort of system and for schemes of operation
that are not based on time study. It has gradually become identified
more or less closely with

    1. the Taylor Plan of Management
    2. what we have defined as the "Transitory" plan of
       management
    3. management which not only is not striving to be
       scientific, but which confounds "science" with "system."
       Both its advocates and opponents have been guilty of
       misuse of the word. Still, in spite of this, the very fact
       that the word has had a wide use, that it has become
       habitual to think of the new type of management as
       "Scientific," makes its choice advisable. We shall use it,
       but restrict its content. With us "Scientific Management"
       is used to mean the complete Taylor plan of management,
       with no modifications and no deviations.

    We may summarize by saying that:

    1. the popular name is Scientific Management,
    2. the inspiring name is Ultimate management,
    3. the descriptive name is measured Functional management,
    4. the distinctive name is the Taylor Plan of Management.

    For the purpose of this book, Scientific Management is, then,
the most appropriate name. Through its use, the reader is enabled to
utilize all his associations, and through his study he is able to
restrict and order the content of the term.

    RELATIONSHIP BETWEEN THE THREE TYPES OF MANAGEMENT.--From the
foregoing definitions and descriptions it will be clear that the
three types of management are closely related. Three of the names
given bring out this relationship most clearly. These are
Traditional (i.e., Primitive), Interim, and Ultimate. These show,
also, that the relationship is genetic, i.e., that the second form
grows out of the first, but passes through to the third. The growth
is evolutional.

    Under the first type, or in the first stage of management, the
laws or principles underlying right management are usually unknown,
hence disregarded.

    In the second stage, the laws are known and installed as fast as
functional foremen can be taught their new duties and the
resistances of human nature can be overcome.[14]

    In the third stage the managing is operated in accordance with
the recognized laws of management.

    PSYCHOLOGICAL SIGNIFICANCE OF THIS RELATIONSHIP.--The importance
of the knowledge and of the desire for it can scarcely be
overestimated. This again makes plain the value of the psychological
study of management.

    POSSIBLE PSYCHOLOGICAL STUDIES OF MANAGEMENT.--In making this
psychological study of management, it would be possible to take up
the three types as defined above, separately and in order, and to
discuss the place of the mind in each, at length; but such a
method would not only result in needless repetition, but also in
most difficult comparisons when final results were to be deduced
and formulated.

    It would, again, be possible to take up the various elements or
divisions of psychological study as determined by a consensus of
psychologists, and to illustrate each in turn from the three types
of management; but the results from any such method would be apt
to seem unrelated and impractical, i.e., it would be a lengthy
process to get results that would be of immediate, practical use
in managing.

    PLAN OF PSYCHOLOGICAL STUDY USED HERE.--It has, therefore,
seemed best to base the discussion that is to follow upon arbitrary
divisions of scientific management, that is--

    1. To enumerate the underlying principles on which scientific
       management rests.
    2. To show in how far the other two types of management vary
       from Scientific Management.
    3. To discuss the psychological aspect of each principle.

    ADVANTAGES OF THIS PLAN OF STUDY.--In this way the reader can
gain an idea of

    1. The relation of Scientific Management to the other types
       of management.
    2. The structure of Scientific Management.
    3. The relation between the various elements of Scientific
       Management.
    4. The psychology of management in general, and of the three
       types of management in particular.

    UNDERLYING IDEAS AND DIVISIONS OF SCIENTIFIC MANAGEMENT.--These
underlying ideas are grouped under nine divisions, as follows:--

    1. Individuality.
    2. Functionalization.
    3. Measurement.
    4. Analysis and Synthesis.
    5. Standardization.
    6. Records and Programmes.
    7. Teaching.
    8. Incentives.
    9. Welfare.

    It is here only necessary to enumerate these divisions. Each
will be made the subject of a chapter.

    DERIVATION OF THESE DIVISIONS.--These divisions lay no claim to
being anything but underlying ideas of Scientific Management, that
embrace varying numbers of established elements that can easily be
subjected to the scrutiny of psychological investigation.

    The discussion will be as little technical as is possible, will
take nothing for granted and will cite references at every step.
This is a new field of investigation, and the utmost care is
necessary to avoid generalizing from insufficient data.

    DERIVATION OF SCIENTIFIC MANAGEMENT.--There has been much
speculation as to the age and origin of Scientific Management. The
results of this are interesting, but are not of enough practical
value to be repeated here. Many ideas of Scientific Management can
be traced back, more or less clearly and directly, to thinkers of
the past; but the Science of Management, as such, was discovered,
and the deduction of its laws, or "principles," made possible when
Dr. Frederick W. Taylor discovered and applied Time Study. Having
discovered this, he constructed from it and the other fundamental
principles a complete whole.

    Mr. George Iles in that most interesting and instructive of
books, "Inventors at Work,"[15] has pointed out the importance, to
development in any line of progress or science, of measuring devices
and methods. Contemporaneous with, or previous to, the discovery of
the device or method, must come the discovery or determination of
the most profitable unit of measurement which will, of itself, best
show the variations in efficiency from class. When Dr. Taylor
discovered units of measurement for determining, _prior to
performance_, the amount of any kind of work that a worker could do
and the amount of rest he must have during the performance of that
work, then, and not until then, did management become a science. On
this hangs the science of management.[16]

    OUTLINE OF METHOD OF INVESTIGATION.--In the discussion of each
of the nine divisions of Scientific Management, the following topics
must be treated:

    1. Definition of the division and its underlying idea.
    2. Appearance and importance of the idea in Traditional and
       Transitory Management.
    3. Appearance and importance of the idea in Scientific
       Management.
    4. Elements of Scientific Management which show the effects
       of the idea.
    5. Results of the idea upon work and workers.

    These topics will be discussed in such order as the particular
division investigated demands. The psychological significance of the
appearance or non-appearance of the idea, and of the effect of the
idea, will be noted. The results will be summarized at the close of
each chapter, in order to furnish data for drawing conclusions at
the close of the discussion.

    CONCLUSIONS TO BE REACHED.--These conclusions will include
the following:--

    1. "Scientific Management" is a science.
    2. It alone, of the Three Types of Management, is a science.
    3. Contrary to a widespread belief that Scientific Management
       kills individuality, it is built on the basic principle of
       recognition of the individual, not only as an economic
       unit but also as a personality, with all the
       idiosyncrasies that distinguish a person.
    4. Scientific Management fosters individuality by
       functionalizing work.
    5. Measurement, in Scientific Management, is of ultimate
       units of subdivision.
    6. These measured ultimate units are combined into methods of
       least waste.
    7. Standardization under Scientific Management applies to all
       elements.
    8. The accurate records of Scientific Management make
       accurate programmes possible of fulfillment.
    9. Through the teaching of Scientific Management the
       management is unified and made self-perpetuating.
   10. The method of teaching of Scientific Management is a
       distinct and valuable contribution to Education.
   11. Incentives under Scientific Management not only stimulate
       but benefit the worker.
   12. It is for the ultimate as well as immediate welfare of
       the worker to work under Scientific Management.
   13. Scientific Management is applicable to all fields of
       activity, and to mental as well as physical work.
   14. Scientific Management is applicable to self-management as
       well as to managing others.
   15. It teaches men to coöperate with the management as well
       as to manage.
   16. It is a device capable of use by all.
   17. The psychological element of Scientific Management is the
       most important element.
   18. Because Scientific Management is psychologically right it
       is the ultimate form of management.
   19. This psychological study of Scientific Management
       emphasizes especially the teaching features.
   20. Scientific Management simultaneously

       a. increases output and wages and lowers costs.
       b. eliminates waste.
       c. turns unskilled labor into skilled.
       d. provides a system of self-perpetuating welfare.
       e. reduces the cost of living.
       f. bridges the gap between the college trained and
          the apprenticeship trained worker.
       g. forces capital and labor to coöperate and to
          promote industrial peace.
"""
    
    #legal
    sample_text3 = """The above-entitled matter came before the Court on Lighthouse Management Group, Inc's (the
“Receiver”) communication to the Court requesting changes to the June 7, 2024 order appointing the
Receiver (the “Receivership Order”).

CONCLUSIONS OF LAW

1. The Court appointed Lighthouse Management Group, Inc as the Receiver over Defendant 2Bros
Collectibles LLC (2Bros) on June 7, 2024.
2. Defendants have claimed to the Receiver that the amount of 2Bros inventory and assets is
essentially negligible.
3. Defendants have failed to produce satisfactory documentation showing the disposition of
inventory.
4. The Receiver filed a correspondence with the Court documenting its research into a related
company named Remember When Collectibles LLC (“RWC”). Based on its extensive
investigation, the Receiver has found that RWC is nothing more than a name change to 2Bros and
is selling Receivership Property outside of the Receivership.
5. Amending the Receivership Order to include RWC will allow the Receiver to gain control over
the Receivership Assets and thus, allow the Receiver to fulfill its duties and obligations to this
Court.

27-CV-24-7991

6. As such, the Court is adding the entity Remember When Collectibles LLC as a Respondent in its
Receivership Order.
7. Consequently, in the Receivership Order, “Respondent” shall now be defined collectively as
2Bros Sports Collectibles LLC and Remember When Collectibles LLC.
8. The result of this revised definition is that Lighthouse Management Group, Inc., is receiver over
both 2Bros Sports Collectibles LLC and Remember When Collectibles LLC.
9. Other than the revised definition of Respondent, there are no other changes to the Receivership
Order and it shall remain in full force and effect."""
    
    # Process the document
    sentence_list = ContextualProcessor.split_sentences(sample_text)
    
    # Print results
    print("==== SENTENCES ====\n")
    print("="*50,"\n")
    
    for i,result in enumerate(sentence_list):
        print(f"{i}:{result}\n")
    print("="*50,"\n")

    #Instance of our processor
    mind = create_processor_pipeline(model_name = "Qwen/Qwen2.5-7B-Instruct")

    #context extraction system prompt
    sp1 = """Use the function as a guide to read and understand the user text. Deterime if you need more more text from the document to produce output.

                text_evaluation(TEXT,pro_nouns,noun_phrases):
                    results = {who:False, #named people or entities

                                role:False, #what that named entity or person does

                                connection:False,#connections between who entities and pronouns references

                                type:False, #type of text

                                when:False, #time
                                
                                context:False, #short summary }
                    #Text type
                    if (type can be determined from TEXT):
                            #if the text type is not related or not consistent then it is unknown
                            results[type] =  "conversation, scientific, instructional, legal, unknown”
                    else:
                            results[type] = 'read more'
                            
                    #Text entities
                    if (who/proper nouns can be determined from TEXT):
                            results[who] = [people, organizations, etc]#(MUST be a named or specific entity)

                            #match pronoun references to named who entities
                            for pro_nouns and noun_phrases
                                if(pro_noun is equal to or subsitution for a noun_phrase)
                                    noun_phrase = who
                                    results[connection] += {{noun_phrase:pro_noun}}
                                else
                                    #unmatched or identified pronoun reference: he/his/she/her/it/you/they/their,etc
                                    results[connection] += {{pro_noun:"to be determined"}}
                    else:
                            results[who] = 'read more'

                    #What is the role of the who/named entity
                    if (role can be determined for who in TEXT):
                            results[role] += {{who:role}}
                    else:
                        results[role] = 'read more'

                    #What is happening
                    if (context can be determined from TEXT):
                        results[context] = “text summary”; #(a brief summary to contextualize ideas, concepts, scene, situation, explination)
                    else:
                        #insufficient information for a summary of the context
                        results[context] = 'read more'

                    #Time period
                    if (when/time can be determined):
                        results[when] = “Time frame”; #(Is this one of: recurring pattern,  one-time event, present day, historical-past, time ambiguous)
                    else:
                        results[when] = 'read more'


                        return results

                        

                MUST RETURN RESULTS JSON ONLY"""
    
    #below was 288 token
    sp1 = """Analyze any given text and extract the following information:
1. Named Entities (WHO)

Identify specific people, organizations, places, or other named entities
Must be proper nouns or specific identifiable entities
If unclear, mark as "read more"

2. Roles (WHAT THEY DO)

Determine what role or function each named entity serves
Examples: "CEO," "witness," "defendant," "researcher"
If unclear, mark as "read more"

3. Pronoun Connections

Match pronouns (he, she, it, they, etc.) to their corresponding named entities
Note any unclear pronoun references
Format as {pronoun: entity it refers to}

4. Text Type

Categorize as: conversation, scientific, instructional, legal, or unknown
If multiple types or unclear, mark as "unknown"
If insufficient information, mark as "read more"

5. Time Frame (WHEN)

Identify temporal context: recurring pattern, one-time event, present day, historical past, or time ambiguous
If unclear, mark as "read more"

6. Context Summary

Provide a brief summary explaining the situation, scene, or main concepts
Should help contextualize the text's purpose and content
If insufficient information, mark as "read more"

Output Format:
Return results as a JSON object with keys: who, role, connection, type, when, context"""
    
    #https://quizgecko.com/tools/token-counter
    #below was 512 token
    sp1 = """RETURN JSON. Use the function as a guide to read and understand the user text. Deterime if you need more more text from the document to produce output.

    text_evaluation(TEXT, pronouns, noun_phrases):
   results = {
       "who": None,
       "role": None, 
       "connection": None,
       "type": None,
       "when": None,
       "context": None
   }
   
   # Identify text type
   if text_type_determinable(TEXT):
       results["type"] = identify_category(TEXT)  # returns: "conversation", "scientific", "instructional", "legal", or "unknown"
   else:
       results["type"] = "read more"
   
   # Extract named entities
   if named_entities_present(TEXT):
       results["who"] = extract_entity_names(TEXT,noun_phrases)  # returns: list of named people, organizations, places. MUST be specific name
       
       # Match pronouns to entities
       connections = {}
       for pronoun in pronouns:
           matched_entity = match_pronoun_to_entity(pronoun, results["who"])
           if matched_entity:
               connections[pronoun] = matched_entity
           else:
               connections[pronoun] = "to be determined"
       results["connection"] = connections
   else:
       results["who"] = "read more"
       results["connection"] = "read more"
   
   # Determine roles of named entities
   if results["who"] != "read more" and roles_determinable(TEXT):
       role_mappings = {}
       for entity in results["who"]:
           role = extract_role(entity, TEXT)
           role_mappings[entity] = role
       results["role"] = role_mappings
   else:
       results["role"] = "read more"
   
   # Extract temporal information
   if temporal_markers_present(TEXT):
       results["when"] = categorize_timeframe(TEXT)  # returns: "recurring pattern", "one-time event", "present day", "historical past", or "time ambiguous"
   else:
       results["when"] = "read more"
   
   # Generate context summary
   if sufficient_context_available(TEXT):
       results["context"] = generate_summary(TEXT)  # returns: brief summary string
   else:
       results["context"] = "read more"
   
   return results
   
   RETURN RESULTS JSON ONLY"""

    sp2 = """RETURN JSON. You read and comprehend text. Use the text_evaluation() function as a framework to understand the user text. Determine if you need more text from the document to produce output.

text_evaluation(PREVIOUS_RESULTS, NEXT_SENTENCE, pronouns, noun_phrases):
   updated_results = copy(PREVIOUS_RESULTS)  # start with previous analysis
   changes_detected = False
   
   # Update text type classification
   if PREVIOUS_RESULTS["type"] == "read more":
       if text_type_determinable(NEXT_SENTENCE):
           updated_results["type"] = identify_category(NEXT_SENTENCE)
           changes_detected = True
   else:
       # Check if new sentence suggests different text type
       new_type = identify_category(NEXT_SENTENCE)
       if new_type != PREVIOUS_RESULTS["type"] and new_type != "unknown":
           updated_results["type"] = resolve_type_conflict(PREVIOUS_RESULTS["type"], new_type)
           changes_detected = True
   
   # Update named entities
   new_entities = extract_proper_nouns(NEXT_SENTENCE)
   if PREVIOUS_RESULTS["who"] == "read more":
       if new_entities:
           updated_results["who"] = new_entities
           changes_detected = True
   else:
       # Merge new entities with existing ones
       if new_entities:
           combined_entities = merge_entity_lists(PREVIOUS_RESULTS["who"], new_entities)
           if combined_entities != PREVIOUS_RESULTS["who"]:
               updated_results["who"] = combined_entities
               changes_detected = True
   
   # Update pronoun connections
   if updated_results["who"] != "read more":
       current_connections = PREVIOUS_RESULTS["connection"] if PREVIOUS_RESULTS["connection"] != "read more" else {}
       
       for pronoun in pronouns:
           if pronoun in current_connections and current_connections[pronoun] == "to be determined":
               # Try to resolve previously unresolved pronouns
               matched_entity = match_pronoun_to_entity(pronoun, updated_results["who"], NEXT_SENTENCE)
               if matched_entity:
                   current_connections[pronoun] = matched_entity
                   changes_detected = True
           elif pronoun not in current_connections:
               # New pronoun found
               matched_entity = match_pronoun_to_entity(pronoun, updated_results["who"], NEXT_SENTENCE)
               current_connections[pronoun] = matched_entity if matched_entity else "to be determined"
               changes_detected = True
       
       updated_results["connection"] = current_connections
   
   # Update roles
   if PREVIOUS_RESULTS["role"] == "read more":
       if updated_results["who"] != "read more" and roles_determinable(NEXT_SENTENCE):
           role_mappings = {}
           for entity in updated_results["who"]:
               role = extract_role(entity, NEXT_SENTENCE)
               if role:
                   role_mappings[entity] = role
           if role_mappings:
               updated_results["role"] = role_mappings
               changes_detected = True
   else:
       # Update existing roles or add new ones
       if updated_results["who"] != "read more":
           current_roles = PREVIOUS_RESULTS["role"]
           for entity in updated_results["who"]:
               new_role = extract_role(entity, NEXT_SENTENCE)
               if new_role and (entity not in current_roles or current_roles[entity] != new_role):
                   current_roles[entity] = new_role
                   changes_detected = True
           updated_results["role"] = current_roles
   
   # Update temporal information
   if PREVIOUS_RESULTS["when"] == "read more":
       if temporal_markers_present(NEXT_SENTENCE):
           updated_results["when"] = categorize_timeframe(NEXT_SENTENCE)
           changes_detected = True
   else:
       # Check for temporal shift or contradiction
       new_timeframe = categorize_timeframe(NEXT_SENTENCE)
       if new_timeframe and new_timeframe != PREVIOUS_RESULTS["when"]:
           updated_results["when"] = resolve_temporal_conflict(PREVIOUS_RESULTS["when"], new_timeframe)
           changes_detected = True
   
   # Update context summary
   if PREVIOUS_RESULTS["context"] == "read more":
       if sufficient_context_available(NEXT_SENTENCE):
           updated_results["context"] = generate_summary(NEXT_SENTENCE)
           changes_detected = True
   else:
       # Expand or refine existing context
       expanded_context = update_context_summary(PREVIOUS_RESULTS["context"], NEXT_SENTENCE)
       if expanded_context != PREVIOUS_RESULTS["context"]:
           updated_results["context"] = expanded_context
           changes_detected = True
   
   # Track what changed
   updated_results["recent_change"] = 'brief note on how the new sentence changed subject/context of previous - or no material change'
   
   return updated_results
   
   RETURN updated_results JSON ONLY"""
    
    sp3 = """def extract_knowledge_nodes(text, noun_phrases, context):
    ""
    Evaluates text to create Subject-Predicate-Object connections for knowledge graph. 
    Subjects and Objects are both specific entities or concept nodes, . 
    Predicates are the long form statements,references and context that connects the nodes.

    If pronoun references are unclear, return a clarification request instead of triples.
    
    Args:
        text (str): The text to analyze
        noun_phrases (list): Isolated noun phrases from the text
        context (str): Background context of the text
        
    
    Returns:
        dict: Either knowledge graph triples or clarification request
    ""
    results = {
        "triples": [],
        "clarification_needed": None
    }
    
    # Step 1: Identify pronouns in text
    pronouns = find_pronouns(text)  # Returns: He/His/She/Her/They/Their/We/I
    
    # Step 2: Attempt to resolve pronouns to specific entities
    if pronouns:
        for pronoun in pronouns:
            referent = resolve_pronoun(pronoun, text, noun_phrases, context)
            if referent == "unclear":
                results["clarification_needed"] = f"Cannot determine what '{pronoun}' refers to. Need additional sentences to contextualize."
                return results
    
    # Step 3: Extract Subject-Predicate-Object triples
    for noun_phrase in noun_phrases:
        # Find relationships involving this noun phrase
        connections = find_connections(noun_phrase, text, context)
        
        for connection in connections:
            #subject and object must be specific entities or concepts, no pronouns. 
            subject = connection["subject"]
            predicate = connection["predicate"] #predicates can be actions, relations, short statements, facts, ideas, connections
            object = connection["object"]
            
            # Ensure no unresolved pronouns in the triple
            if contains_unresolved_pronoun(subject, object):
                results["clarification_needed"] = "Pronoun reference unclear. Need additional sentences to contextualize."
                return results
            
            # Add valid triple
            results["triples"].append({
                "subject": subject,# (plain text, entity or concept no pronouns)
                "predicate": predicate,# (plain text, long form text relation and connection and action)
                "object": object #(plain text, entity or concept no pronouns)
            })
    
    return results

    RETURN results JSON ONLY"""
    
    sp3 = """You are tasked with analyzing text to create Subject-Predicate-Object knowledge graph connections.
User Input: The text to analyze
noun_phrases: List of noun phrases found in the text to help with entity identification
context: Background context about the text and subject/predicate/object hints

Your Task: Extract knowledge triples where:

Subjects and Objects: Must be specific named entities or concepts (no pronouns)
Predicates: Long-form statements, references, actions, relations, facts, ideas, or connections that link the subject to the object

Process:

Check for pronouns or ambiguous references in the text (He, His, She, Her, They, Their, We, I, It, It's etc.)
Resolve the references using the text, noun phrases, and context provided

If any pronoun cannot be clearly resolved to a specific entity, stop and request clarification


Extract triples by examining each noun phrase and finding its relationships within the text

Each triple must have: specific subject → descriptive predicate → specific object
No unresolved pronouns allowed in subjects or objects
Predicates should be descriptive and capture the full relationship context



Output Format: Return JSON only with this structure:
{
  "triples": [
    {
      "subject": "specific entity or concept name",
      "predicate": "plain text descriptive relationship/action/connection",
      "object": "specific entity or concept name"
    }
  ],
  "clarification_needed": null
}
If pronoun resolution fails, return:
{
  "triples": [],
  "clarification_needed": "Explanation of which pronoun is unclear and request for more context"
}"""
    
    sp3 = """You are tasked with analyzing text to create Subject-Predicate-Object knowledge graph node and edge connections.
User Input: The text to analyze
noun_phrases: List of noun phrases found in the text to help with entity identification
context: Background context about the text and subject/predicate/object hints

Your Task: Extract knowledge triples where:

Subjects and Objects: Must be specific named entities or concepts (no pronouns)
Predicates: Long-form statements, references, actions, relations, facts, ideas, or connections that link the subject to the object
Abstract concepts: A node can represent an abstract idea, such as a concept, event, or situation. For instance, a knowledge graph could include the node (Theory of Relativity) to represent the concept itself, which is a noun phrase.
Events: A node can represent an event rather than a physical object. For example, in the triple (Hurricane Sandy) -[occurred in]-> (2012), "Hurricane Sandy" is a named event. A more complex example could be (the launch of SpaceX's Dragon capsule) -[was a part of]-> (the Commercial Crew Program).
Adjectival concepts: While less common, certain graph models can use concepts that are grammatically adjectives. A graph might include the triple (The T-shirt) -[has color]-> (white), where "white" is a literal value but acts as the object. A more advanced approach might model the concept (White) as its own node.

Output Format: Return JSON only with this structure:
{
  "triples": [
    {
      "subject": "specific entity or concept name",
      "predicate": "plain text descriptive relationship/action/connection",
      "object": "specific entity or concept name"
    }
  ],
  "clarification_needed": null
}
If pronoun resolution fails, return:
{
  "triples": [],
  "clarification_needed": "Explanation of which pronoun is unclear and request for more context"
}
"""
    sp4 = """Create a summary of the text and context (dialogue, facts, information) with a specific focus on facts and statements AND a list of named unique Subjects or Objects.
    If the text is too short or lacks sufficient information, request more text by returning ONLY 'read more'. 
    RETURN 'read more' OR the summary text with context."""
    #set the system prompt
    #mind.llm._update_system_prompt(sp1)
    

    passage = ""
    summary = ""
    total_passages = len(sentence_list)
    res = None

    while len(sentence_list)>1:
        loop_start = time.time()  # Start timing the loop
        mind.llm._update_system_prompt(sp4)
        print("#"*50)
        while True:      
            #get a sentence
            passage += sentence_list.pop(0)+". "#add period back for context

            rez = mind.nlp.find_proper_nouns(passage)
            #print("found proper nouns:", [x[0] for x in rez] )
            propnouns = [x[0] for x in rez]
            #if none, keep reading
            if len(propnouns) > 0:
                summary = mind.llm.generate_response(passage)
                if summary.strip().lower() == "read more":
                    print("NEED MORE TEXT FOR CONTEXT...")
                    continue
                else:
                    
                    break
        mind.llm._update_system_prompt(sp3)
        print(f"TEXT INPUT:{passage}")
        print("SUMMARY:",summary)
        
        #now get our pronouns, noun phrases
        pron =  mind.nlp.find_pro_nouns(passage)
        pron =  [x[0] for x in pron]#isolate the pronouns for the token tags returned from Spacy
        nonphrs = mind.nlp.find_noun_phrases(passage)
        nonphrs =  [x for x in nonphrs]
        print("FOUND NOUN PHRASES:",nonphrs)

        
        res = mind.llm.generate_response(f"TEXT:{passage},\n noun_phrases:{nonphrs} {pron},\n context:{summary}")
        print("."*50)
        print("KNOWLEDGE NODES EXTRACTION RESULT:\n",res,"\n")
        print("."*50)
        loop_end = time.time()
        elapsed = loop_end - loop_start
        print(f"review completed in {elapsed:.2f} seconds")
        #reset passage
        passage = ""
        print("#"*50)
            
            

    #Original
    """while len(sentence_list)>1:
        loop_start = time.time()  # Start timing the loop
        
        print("="*50)
        while True:      
            #get a sentence
            passage += sentence_list.pop(0)+". "#add period back for context

            rez = mind.nlp.find_proper_nouns(passage)
            #print("found proper nouns:", [x[0] for x in rez] )
            propnouns = [x[0] for x in rez]
            #if none, keep reading
            if len(propnouns) > 0:
                break
            
        
        print(f"INPUT:{passage}")
        #now get our pronouns, noun phrases
        pron =  mind.nlp.find_pro_nouns(passage)
        pron =  [x[0] for x in pron]#isolate the pronouns for the token tags returned from Spacy

        nonphrs = mind.nlp.find_noun_phrases(passage)
        nonphrs =  [x for x in nonphrs]

        #first passage only:
        if len(sentence_list) == total_passages-1:
            res = mind.llm.generate_response(f"text_evaluation(TEXT:{passage},pro_nouns: {pron}, noun_phrases:{nonphrs})")
            #going forward use the progressive prompt
            mind.llm._update_system_prompt(sp2)
            res = json.loads(res)
        else:
            print("PREVIOUS.......",res,"."*25)
            res = mind.llm.generate_response(f"text_evaluation(PREVIOUS_RESULTS:{res}, \n NEXT_SENTENCE:{passage},\n pronouns:{pron}, \n noun_phrases:{nonphrs})")
            res = json.loads(res)
        res_json = res

        #finish this analysis if we don't need to read more
        if res_json.get('context',False) != "read more":
            print("+"*50,'\n',res,'\n',"="*50,'\n')
            loop_end = time.time()context:
            elapsed = loop_end - loop_start
            print(f"Loop completed in {elapsed:.2f} seconds")
            #reset passage
            passage = ""
        else:
            print("READING MORE...")
     """