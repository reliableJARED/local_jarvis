
import subprocess,sys
from typing import List, Tuple
import logging
logging.basicConfig(level=logging.INFO)

class PrefrontalCortex:
        """
        Specialized class to use Qwen to summarize conversations as well as determine the focus 
        """

        import json
        def __init__(self,llm):
                self.llm = llm #instance of Qwen llm
                self.spacy = None #will hold reference to the spaCy noun finding library
                self._check_and_install_dependencies()

        def json_to_string(self,json_data):
                """
                Convert JSON data to a formatted string.
                
                Args:
                    json_data: Can be a JSON string, list, or dict
                
                Returns:
                    str: Formatted JSON string
                """
                # If it's already a string, parse it first
                if isinstance(json_data, str):
                    json_data = json.loads(json_data)
                
                # Convert to formatted string with indentation
                return json.dumps(json_data, indent=2, ensure_ascii=False)
        
        def _check_and_install_dependencies(self) -> list:
            """
            Check if required dependencies are installed and install if missing.
           
            """
            # Check and install PyTorch
            try:
                import spacy
                self.spacy = spacy.load("en_core_web_sm")#en_core_web_trf  or en_core_web_lg - trf is the transformer and is best but most computationally expensive
                logging.debug("spaCy already installed and imported successfully")
            except ImportError:
                logging.debug("spaCy not found. Installing torch...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
                    import spacy
                    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                    self.spacy = spacy.load("en_core_web_sm")

                    logging.debug("spaCy and en_core_web_sm installed and imported successfully")
                except (subprocess.CalledProcessError, ImportError) as e:
                    logging.error(f"Failed to install or import spaCy: {e}")
                    raise ImportError("WARNING! Failed to install or import spacy")
            
        def extract_entities_and_nouns(self, txt: str) -> Tuple[List[str], List[str]]:
            """
            Extract named entities and nouns from text using spaCy.
            
            Args:
                txt: Input text to analyze
                
            Returns:
                Tuple[List[str], List[str]]: (entities, nouns)
            """
            entities = []
            nouns = []
            text = self.json_to_string(txt)

            # Clean up newlines and extra whitespace, spaCy likes clean text else special chars return with the nouns
            text = text.replace('\n', ' ')  # Replace newlines with spaces
            text = ' '.join(text.split())   # Normalize whitespace

            try:
                doc = self.spacy(text)
                print(f"Extracting entities and nouns from text: {text[:50]}...")
                
                # Extract named entities (people, places, organizations, etc.)
                for ent in doc.ents:
                    print(f"Found entity: {ent.text} ({ent.label_})")
                    if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT"]:
                        entities.append(ent.text.lower())
                
                # Extract important nouns
                for token in doc:
                    if (token.pos_ == "NOUN" and 
                        not token.is_stop and 
                        len(token.text) > 2 and
                        token.text.lower() not in entities):
                        nouns.append(token.text.lower())
                
            except Exception as e:
                print(f"Error extracting entities: {e}")
            
            return list(set(entities)), list(set(nouns))

        def summarize_conversation(self,messages=None):
            #sp ="YOU MUST respond with JSON schema.  Your responses should ALWAYS have format {'response':'the response to the user prompt','focus':'what the subject or focus of the conversation is'} Return JSON"
            #sp = "YOU MUST respond with JSON schema.  Your responses should ALWAYS have format {'response':'the response to the user prompt','focus':'detailed description of what the subject and focus of the conversation is'} Return JSON"
            sp = "YOU MUST respond with JSON schema.  Your responses should ALWAYS have format {'focus':'detailed description of the focus of the conversation so that when we embed this focus as a vector it has captured key language for similarity search'} Return JSON"
            result = self.generate_response(sp=sp,messages=messages)
            return result
       
        def conversation_flow(self,messages=None):
            sp ="YOU MUST respond with JSON schema.  Your responses should ALWAYS have format {'continuation':True/False,'alteration':'what is changing'} where continuation indicates if the conversation topic has changed and 'alteration' is what is different, even if conversation doesn't change the focus or alteration can slightly change. Return JSON"
            result = self.generate_response(sp=sp,messages=messages)
            return result

        def generate_response(self,sp="you are an assistant",messages=None):
            prompt = ""
            #store the existing system prompt
            current_sp = self.llm.messages[0]['content']#make a method to get the sp in Qwen
            #update sp to the new one
            self.llm._update_system_prompt(sp)

            if messages is None:
                messages = self.llm.messages[1:]

            response = self.llm.generate_response(self.json_to_string(messages),use_message_history=False)

           
            #restore the original System Prompt
            self.llm._update_system_prompt(current_sp)
            resp_dict = json.loads(response)
            return resp_dict
       
             


if __name__ == "__main__":
    import json
    from qwen_ import Qwen
    chat = Qwen(auto_append_conversation = True)
    prefrontal = PrefrontalCortex(chat)


    cooking_convo = [
        {
            "role": "user",
            "content": "Hi! I have chicken breast, bell peppers, onions, garlic, rice, and some basic spices. What can I make for dinner?"
        },
        {
            "role": "assistant",
            "content": "Great ingredients! You have several delicious options:\n\n1. **Chicken Stir-fry** - Quick and colorful with the peppers and onions over rice\n2. **One-pot Chicken and Rice** - Everything cooked together for easy cleanup\n3. **Stuffed Bell Peppers** - Fill the peppers with seasoned chicken and rice\n4. **Chicken Fajita Bowls** - Mexican-style with seasoned chicken over rice\n\nWhat style of cuisine are you in the mood for?"
        },
        {
            "role": "user",
            "content": "The one-pot chicken and rice sounds perfect! Can you give me the full recipe?"
        },
        {
            "role": "assistant",
            "content": "# One-Pot Chicken and Rice\n\n**Ingredients:**\n- 2 chicken breasts, cut into bite-sized pieces\n- 1 cup long-grain white rice\n- 1 bell pepper, diced\n- 1 medium onion, diced\n- 3 cloves garlic, minced\n- 2 cups chicken broth (or water)\n- Salt, pepper, paprika, and garlic powder to taste\n- 2 tbsp oil\n\n**Instructions:**\n1. Heat oil in a large pot over medium-high heat\n2. Season chicken with salt, pepper, and paprika\n3. Cook chicken pieces until golden (5-6 mins), remove and set aside\n4. Sauté onions until translucent (3-4 mins)\n5. Add bell peppers and garlic, cook 2 more minutes\n6. Add rice, stir for 1 minute\n7. Pour in broth, add chicken back, bring to boil\n8. Reduce heat, cover, simmer 18-20 minutes\n9. Let rest 5 minutes before serving\n\nReady to start cooking?"
        },
        {
            "role": "user",
            "content": "Yes! I just heated the oil and seasoned my chicken. Should I cook all the chicken pieces at once or in batches?"
        },
        {
            "role": "assistant",
            "content": "Cook in batches if your pot is getting crowded! Overcrowding will steam the chicken instead of giving it that nice golden sear. You want the pieces to have space around them.\n\nIf you can fit all the chicken in a single layer with some space between pieces, go ahead and cook it all at once. Otherwise, do it in 2 batches - it only takes an extra 2-3 minutes and you'll get much better browning.\n\nHow does it look in your pot?"
        },
        {
            "role": "user",
            "content": "Good call, I'm doing it in batches. The first batch is getting nice and golden. How do I know when it's done?"
        },
        {
            "role": "assistant",
            "content": "Perfect! Look for these signs:\n- Golden brown color on the outside (which it sounds like you're getting!)\n- The chicken should feel firm when you press it gently with your spatula\n- It should release easily from the bottom of the pot when you try to flip it\n\nDon't worry about cooking it completely through at this stage - it'll finish cooking when you simmer everything together with the rice. You're just looking for that golden sear for flavor.\n\nOnce the first batch looks good, remove it and start the second batch!"
        },
        {
            "role": "user",
            "content": "Great! Both batches are done and set aside. Now I'm sautéing the onions. They're starting to look translucent."
        },
        {
            "role": "assistant",
            "content": "Excellent! You're right on track. Once those onions are nicely translucent (usually takes about 3-4 minutes), go ahead and add your diced bell peppers and minced garlic.\n\nThe peppers will add a nice pop of color and sweetness. Cook everything together for about 2 minutes - you want the peppers to start softening just slightly and the garlic to become fragrant.\n\nBe careful not to let the garlic burn since you added it with the peppers - it cooks quickly!"
        },
        {
            "role": "user",
            "content": "Added the peppers and garlic. It smells amazing! The garlic is definitely fragrant. What's next?"
        },
        {
            "role": "assistant",
            "content": "That aroma is the best part! Now for the rice:\n\n1. Add your 1 cup of rice to the pot with the vegetables\n2. Stir everything together for about 1 minute - this lightly toasts the rice and helps prevent it from getting mushy\n3. The rice might look like it's absorbing some of the flavors from the pan, which is perfect!\n\nAfter that minute of stirring, you'll add your liquid (broth or water) and the chicken back in. Are you using chicken broth or water?"
        },
        {
            "role": "user",
            "content": "I'm using chicken broth. Just added the rice and I'm stirring it around. It's picking up all those good flavors from the bottom of the pot!"
        },
        {
            "role": "assistant",
            "content": "Perfect! That's exactly what you want - the rice is picking up all that delicious fond (those browned bits) from cooking the chicken and vegetables.\n\nAfter your minute of stirring is up:\n1. Pour in 2 cups of the chicken broth\n2. Add the cooked chicken pieces back to the pot\n3. Give it a good stir to distribute everything evenly\n4. Bring it up to a boil\n\nOnce it's boiling, you'll reduce the heat to low, cover, and let it simmer. The broth should just cover everything - let me know when you reach the boiling point!"
        },
        {
            "role": "user",
            "content": "Just added the broth and chicken back in. It's starting to bubble around the edges. Should I stir it while it's coming to a boil?"
        },
        {
            "role": "assistant",
            "content": "Give it one good stir now to make sure everything is evenly distributed, then try to avoid stirring too much once you add the lid. Too much stirring can make the rice gummy.\n\nThose bubbles around the edges mean you're almost there! Once you see it bubbling more actively across the surface (a good rolling boil), immediately:\n\n1. Reduce heat to LOW\n2. Cover with a tight-fitting lid\n3. Set a timer for 18-20 minutes\n4. Resist the urge to peek! Let the steam do its work\n\nHow's the bubbling looking now?"
        },
        {
            "role": "user",
            "content": "It's at a full boil now! Just reduced to low heat about 200 degrees and covered it. Timer set for 18 minutes. Should I hear anything while it's cooking?"
        },
        {
            "role": "assistant",
            "content": "Great job! Yes, you should hear gentle, quiet bubbling or simmering sounds - that's perfect. It should sound like a soft, steady simmer, not aggressive bubbling.\n\nIf you hear loud bubbling or see steam escaping rapidly from under the lid, your heat might be a bit too high - just turn it down a notch.\n\nIf it gets too quiet (completely silent), the heat might be too low and the rice won't cook properly.\n\nThe goal is that gentle, consistent simmer. And remember - no peeking! Every time you lift the lid, you lose steam and it affects the cooking time. Trust the process and let that timer do its job!"
        },
        {
            "role": "user",
            "content": "Perfect, I can hear that gentle simmering. It's been about 10 minutes so far. I'm resisting the urge to check!"
        },
        {
            "role": "assistant",
            "content": "You're doing great! That's exactly the right sound, and good for you for not peeking - I know it's tempting!\n\nYou're more than halfway there. In the last few minutes of cooking, you might notice the simmering sounds start to quiet down a bit as the liquid gets absorbed. That's totally normal.\n\nOnce your timer goes off, the key is to let it rest for 5 minutes OFF the heat (still covered) before you open it. This final rest period lets the rice finish steaming and absorb any remaining moisture.\n\nHang in there - 8 more minutes to go!"
        },
        {
            "role": "user",
            "content": "Timer just went off! Turning off the heat but keeping it covered for the 5 minute rest. The kitchen smells incredible."
        },
        {
            "role": "assistant",
            "content": "Excellent! You nailed the timing. That amazing smell means everything is working perfectly.\n\nNow comes the hardest part - waiting those 5 minutes! But it's so important. During this rest period:\n- The rice finishes absorbing any remaining liquid\n- The steam continues cooking everything gently\n- The flavors meld together beautifully\n- The rice becomes perfectly fluffy instead of sticky\n\nSet another timer for 5 minutes if it helps. Once that's up, you can finally lift the lid and fluff everything gently with a fork. You're almost there!"
        },
        {
            "role": "user",
            "content": "The 5 minutes felt like forever! Just lifted the lid and wow - it looks perfect! The rice is fluffy and the chicken looks tender. Should I fluff it with a fork now?"
        },
        {
            "role": "assistant",
            "content": "Yes! Gently fluff it with a fork - not a spoon, as that can mash the rice. Just lightly fold and fluff to separate the grains and distribute everything evenly.\n\nYou should see:\n- Fluffy, separate rice grains\n- Tender, juicy chicken pieces\n- Colorful bell peppers that still have a slight bite\n- Everything nicely incorporated together\n\nTaste it and adjust seasoning if needed - maybe a pinch more salt or pepper. You did an amazing job following the technique! How does it taste?"
        },
        {
            "role": "user",
            "content": "Just tasted it and it's delicious! The rice absorbed all the flavors beautifully. The chicken is perfectly tender and the vegetables have great texture. This was so much easier than I expected!"
        },
        {
            "role": "assistant",
            "content": "That's fantastic! You absolutely nailed it. One-pot meals like this are such great weeknight dinners because they're forgiving, flavorful, and create minimal cleanup.\n\nYou followed the technique perfectly - getting that good sear on the chicken, building flavors with the aromatics, toasting the rice, and most importantly, resisting the urge to peek during cooking!\n\nThis recipe is very adaptable too. Next time you could try:\n- Different vegetables (carrots, peas, mushrooms)\n- Different seasonings (curry powder, Italian herbs, or cajun spices)\n- Different proteins (thighs, shrimp, or sausage)\n\nEnjoy your delicious dinner - you earned it!"
        },
        {
            "role": "user",
            "content": "Thank you so much for walking me through this step by step! I definitely want to try some of those variations next time. This will definitely become a regular in my dinner rotation."
        },
        {
            "role": "assistant",
            "content": "You're so welcome! It was my pleasure helping you through it. You were a great student - following directions carefully and asking all the right questions along the way.\n\nOne-pot meals really are perfect for busy weeknights, and now that you understand the basic technique, you can get creative with whatever ingredients you have on hand.\n\nEnjoy your dinner, and feel free to come back anytime you want to tackle a new recipe or cooking technique. Happy cooking!"
        }
        ]
            
    response = prefrontal.summarize_conversation(messages=cooking_convo)
    print(f"\nprefrontal: {response}")
    prefrontal.llm.print_token_stats()

    #push the cooking convo to the messages
    chat.messages += cooking_convo

    print(chat.messages)


    while True:
        
        try:
            user_input = input("\nYou: ").strip()
            print("\nFocus Analysis...")
            messages = chat.messages[1:] + [{'role':'user','content':user_input}]
            response = prefrontal.conversation_flow(messages=messages)#skip system prompt
            print(f"prefrontal analysis: {response}")
            nouns = prefrontal.extract_entities_and_nouns(messages)
            print(f"\n\nNOUNS: {nouns}")
            print("\nThinking...")
            response = chat.generate_response(user_input)#skip system prompt
            print(f"\nQwen: {response}")
            chat.print_token_stats()

        except KeyboardInterrupt:
                    print("\n\nChat interrupted. Goodbye!")
                    break
        except Exception as e:
                    print(f"\nError: {e}")
                    print("Please try again.")