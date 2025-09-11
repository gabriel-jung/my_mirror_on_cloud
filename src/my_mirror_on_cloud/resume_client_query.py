import os
import json
import re
import numpy as np
from dotenv import load_dotenv
from time import perf_counter
from mistralai import Mistral

import loguru
logger = loguru.logger

def llm_query_openai(role_system: dict, user_query: dict)-> np.array:
    load_dotenv()
    api_key = os.environ.get("MISTRAL_API_KEY" )

    model = "mistral-small-latest" 
    
    client = Mistral(api_key=api_key)
    chat_response = client.chat.complete(
        model=model,
        messages=[role_system, user_query]
    )
    logger.debug(chat_response.choices[0].message.content)
    return chat_response.choices[0].message.content


def define_role_system()-> dict:
    """
    Define the system role for the LLM.
    output: role system
    """
    role_system = {
        "role": "system",
        "content": '''
        You are a fashion request classifier.

        Your task:
        - Analyze the user request and generate:
        - If the request is not clear,  Return a JSON array: `["Need clarification"]`.
        1. A single concise sentence in the format:
            "flow1: a [gender] outfit for [event] during [season] with a specific [color] [style]"
            "flow2: a [item] that goes well with [color] [style] [existing_item] for a [gender]"
            "flow3: a [color] [style] [item] for a [gender]"
        2. Extract the color tag (from [color_tags] allowed) and style tag (from [style_tags] allowed).

        - Return EXACTLY a JSON array: `[sentence, color_tag, style_tag]`
        - If no color or style can be detected, return `"N/A"` in that position.
        - Use only the allowed words. Do not add any extra text, explanation, or punctuation outside the format.
        - Be concise, Keep the exact same structure. 
        - Do not give recommendation or example.
        - Ensure the sentence is natural, readable English, without abbreviations or slang.
        - the name of the flow is really important. Do not forget it.

        Rules:
        - Use only these values:
        [item]: Dress, Long-Skirt, Short-Skirt, Top, Pants, Shorts, Pullover, Sweatshirt
        [gender]: Woman, Man
        [event]: Party, Dinner, Wedding, Casual, Rock
        [season]: Winter, Spring, Summer, Fall
        [color_tags]: Red, Blue, Green (Be free to add others colors depending on the request)
        [style-tags]: Casual, Chic, Sporty (Be free to add others styles depending on the request)

        '''
    }
    return role_system




def reformulation_query(query:str)-> dict:
    """
    Reformulate the user query to a structured description using the LLM.   
    input: user query, role system
    output: structured description
    """
    role_system = define_role_system()
    user_query = {
                "role": "user",
                'content': query,
            }

    logger.info(user_query)
    logger.info(role_system)

    t1 = perf_counter()
    response = llm_query_openai(
            role_system,
            user_query        
    )
    t2 = perf_counter()
    clean_response = re.sub(r'^```json\s*|```$', '', response, flags=re.MULTILINE).strip()
    arr = json.loads(clean_response)
    logger.info(arr)
    logger.info(f"Response time: {t2 - t1:.2f} seconds")
    return arr


if __name__ == "__main__":
    query = "I want a nice outfit for this summer weekend. I'm a woman. I want to walk through paris probably in a park"
    reformulation_query(query)
    
    
    
