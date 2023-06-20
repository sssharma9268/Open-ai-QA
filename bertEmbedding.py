import logging
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

import pandas as pd
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
# Load pre-trained universal sentence encoder model
import numpy as np

import os
#os.environ["openapikey"] = "--old-key--sk--aynEQ3OSpdlAnTmc7bNsT3BlbkFJv6EbHLufeblxJwknLeaU"
logging.info(os.environ['openapikey'])
#print(os.environ['openapikey'])
api_key=os.environ['openapikey']
logging.info(api_key)
#os.environ["OPENAI_API_KEY"] = api_key
#os.environ['OPENAI_API_KEY'].split(os.pathsep)
openai.api_key=api_key
# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

file = open("./TextFiles/TextFile.txt",encoding="ISO-8859-1")
s=file.readlines()
lines=[]
# Replaces escape character with space
for l in s:


    l=l.replace("\n", " ").strip()

    
    lines.append(l)
ln=len(lines)

embeddings = sbert_model.encode(lines)

myDict={"text":[],"embedding":[]}

# Printing embeddings of each sentence
  
# To print each embeddings along with its corresponding 
# sentence below code can be used.
for i in range(len(lines)):

    myDict["text"].append(lines[i])
    myDict["embedding"].append(embeddings[i])

df=pd.DataFrame(myDict)
#df.to_csv("EXL_Embeddings.csv")
#print(df.head(2))


def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 5
):
    query_embedding=sbert_model.encode(query)
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding[0], row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    #print(strings_and_relatednesses)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below articles on EXL Financial Reports to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        section=string
        i=0        
        while i<len(lines):            
            if lines[i]==string:               
                i+=1                
                if i<len(lines)-1:                    
                    while lines[i]!='' :
                        string=string+lines[i]
                        i+=1
                        if i==len(lines):
                            break
            i+=1
                
        next_article = f'\n\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question



def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    print(df)
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about EXL Financial Reports 2023."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.7
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message


#print(ask(["what is the business highlights of 2023?"]))