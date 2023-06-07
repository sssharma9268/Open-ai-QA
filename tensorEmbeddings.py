# import necessary libraries
import tensorflow_hub as hub
import pandas as pd
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
# Load pre-trained universal sentence encoder model


import os
api_key="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"
api_key="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"
os.environ["OPENAI_API_KEY"] = api_key
os.environ['OPENAI_API_KEY'].split(os.pathsep)
openai.api_key=api_key
# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

query = "What is the name of the Author of Alice’s Adventures in Wonderland?"

response = openai.ChatCompletion.create(
    messages=[
        {'role': 'system', 'content': 'You answer questions about the Alice’s Adventures in Wonderland.'},
        {'role': 'user', 'content': query},
    ],
    model=GPT_MODEL,
    temperature=0,
)

#print(response['choices'][0]['message']['content'])



# Sentences for which you want to create embeddings,
# passed as an array in embed()
embed = hub.load("tensor")
file = open("alice.txt",encoding="utf8")
s=file.readlines()
lines=[]
# Replaces escape character with space
for l in s:


    l=l.replace("\n", " ")
    if l!=" ":
        lines.append(l)


embeddings = embed(lines)
#print(type(embeddings))
myDict={"text":[],"embedding":[]}

# Printing embeddings of each sentence
  
# To print each embeddings along with its corresponding 
# sentence below code can be used.
for i in range(len(lines)):

    myDict["text"].append(lines[i])
    myDict["embedding"].append(embeddings[i].numpy())

df=pd.DataFrame(myDict)
print(df.head(2))

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    #query_embedding_response = openai.Embedding.create(
        #model=EMBEDDING_MODEL,
        #input=query,
    #)
    #query_embedding = query_embedding_response["data"][0]["embedding"]
    query_embedding=embed(query)
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding.numpy()[0], row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

#strings, relatednesses = strings_ranked_by_relatedness(["Alice’s Adventures in Wonderland"], df, top_n=5)
#query_embedding=embed(["Alice"])
#print(query_embedding.numpy()[0])
#for string, relatedness in zip(strings, relatednesses):
    #print(f"{relatedness=:.3f}")
    #print(string)


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
    introduction = 'Use the below articles on Alice’s Adventures in Wonderland to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
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
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about Alice’s Adventures in Wonderland."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.7
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message


print(ask(["What is the name of the Author?"]))