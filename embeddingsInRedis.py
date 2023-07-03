import logging
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
import redis
import pandas as pd
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
# Load pre-trained universal sentence encoder model
import numpy as np
from redis.commands.search.query import Query
import os
os.environ["openapikey"] = "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"
logging.info(os.environ['openapikey'])
#print(os.environ['openapikey'])
api_key=os.environ['openapikey']
logging.info(api_key)
#os.environ["OPENAI_API_KEY"] = api_key
#os.environ['OPENAI_API_KEY'].split(os.pathsep)
openai.api_key=api_key
# models

GPT_MODEL = "gpt-3.5-turbo"


def search_vectors(query_vector, client, top_k=5):
    base_query = "*=>[KNN 5 @embedding $vector AS vector_score]"
    query = Query(base_query).return_fields("text", "vector_score").sort_by("vector_score").dialect(2)    
 
    try:
        results = client.ft("doc:").search(query, query_params={"vector": query_vector})
    except Exception as e:
        print("Error calling Redis search: ", e)
        return None
 
    return results

def ask(
    query: str,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""

    conn = redis.Redis(host='localhost', port=6379, password='', encoding='utf-8', decode_responses=True)
    if conn.ping():
        print("Connected to Redis")
 
    print("Vectorizing query...")
    query_vector=sbert_model.encode(query) 
    #print(query_vector)
    query_vector = np.array(query_vector).astype(np.float32).tobytes()
 
    # Perform the similarity search
    print("Searching for similar docs...")
    results = search_vectors(query_vector, conn)
 
    if results:
        print(f"Found {results.total} results:")
        for i, post in enumerate(results.docs):
            score = 1 - float(post.vector_score)
            print(f"\t{i}. {post.text} (Score: {round(score ,3) })")
    else:
        print("No results found")

    conn.close()




    
    # message = query_message(query,lines, df, model=model, token_budget=token_budget)
    # if print_message:
    #     print(message)
    # messages = [
    #     {"role": "system", "content": "You answer questions about EXL Financial Reports 2023."},
    #     {"role": "user", "content": message},
    # ]
    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=messages,
    #     temperature=0.7
    # )
    # response_message = response["choices"][0]["message"]["content"]
    response_message="done"
    return response_message


print(ask("skills"))