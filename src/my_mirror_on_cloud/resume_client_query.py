from time import perf_counter
#from openai import OpenAI
from mistralai import Mistral
from .params import MISTRAL_API_KEY

import loguru
logger = loguru.logger

def llm_query_openai(role_system, user_query):
    
    api_key = MISTRAL_API_KEY

    model = "mistral-small-latest"  # vérifie le modèle que tu veux
    
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
                You are a fashion assistant. 
                Follow these steps:
                1- Understand the request among one of the three possibilities:
                - [flow1]: recommend outfit (ex: recommend an outfit for a friend party during winter )
                - [flow2]: recommend an associated clothing items (ex: recommend a skirt going well with my top)
                - [flow3]: find a specific clothing item (ex: find a red dress)

                2- If the request is not clear, ask for clarification.
                3- Depending on the identified flow, return the following structured description. Replace each [] using the possible values described below:
                - For [flow1], Output: "flow1: a [gender] outfit for [event] during [season] with a specific [color] [style]".
                - For [flow2], Output: "flow2: a [item] that goes well with [color] [style] [existing_item] for a [gender]".
                - For [flow3], Output: "flow3: a [color] [style] [item] for a [gender]".

                Be concise, Keep the exact same structure. Do not add any extra information. Do not give recommendation or example.

                Use this exhaustive list of possible values:
                [item]: Dress, Long-Skirt, Short-Skirt, Top, Pants, Shorts, Pullover, Sweatshirt
                [gender]: Woman, Man
                [event]: Party, Dinner, Wedding, Casual, Rock 
                [season]: Winter, Spring, Summer, Fall
                [color]: Red, Blue, Green 
                [style]: Casual, Chic, Sporty
                '''
            }
    return role_system




def reformulation_query(query:str)-> str:
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
    logger.info(response)
    logger.info(f"Response time: {t2 - t1:.2f} seconds")
    return response


if __name__ == "__main__":
    query = "I want a nice outfit for this summer weekend. I'm a woman. I want to walk through paris probably in a park"
    reformulation_query(query)
    
    
    
