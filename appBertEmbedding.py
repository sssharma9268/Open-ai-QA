import fitz
import bertEmbedding
import embeddingsInRedis
from flask import Flask,request, jsonify
import os
from flask_cors import CORS
import glob
import redis
import pandas as pd
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
import numpy as np
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

app = Flask(__name__)
CORS(app)

os.environ["PORT"] = "5000"

@app.route('/')
def index():
    return 'Server Works'

@app.route("/upload",methods=["POST"])
def upload():
    uploaded_files = request.files.getlist('files')

    for file in uploaded_files:
        file.save(file.filename)                                      
        
    processFiles(uploaded_files)
    print("Files Uploaded")
    files=glob.glob('*.pdf')
    #print(files)
    for filename in files:
        os.unlink(filename)

    createEmbeddings(uploaded_files)
   
    return jsonify({"success":True})

def processFiles(uploaded_files): 
    for file in uploaded_files:
        fname=file.filename.split('.')[0].strip() 
        file_to_delete = open("./TextFiles/"+fname+".txt",'w')
        file_to_delete.close()
    
        print("Filename:-----"+fname)
        with fitz.open(file.filename) as doc:
            for page in doc:
                text = page.get_text()
            #print("text---->"+text)
                with open("./TextFiles/"+fname+".txt", "a",encoding="utf-8") as f:
                    f.writelines(text)
                f.close()
    print("Converted files to Text")
    
  
    
def createEmbeddings(uploaded_files):
    print("Creating Embeddings")
    #redis_client = redis.Redis(host='localhost', port=6379, db=0) #reddis connection details 
    conn = redis.Redis(host='localhost', port=6379, password='', encoding='utf-8', decode_responses=True)
    p = conn.pipeline(transaction=False)

    SCHEMA = [
        TextField("text"),
        VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": 768, "DISTANCE_METRIC": "COSINE"}),
    ]

    try:
        
        conn.ft("docs").create_index(fields=SCHEMA, definition=IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH))
    except Exception as e:
        print("Index already exists")

    for file in uploaded_files:
        fname=file.filename.split('.')[0].strip() 
        readFile = open("./TextFiles/"+fname+".txt",'r',encoding="ISO-8859-1")
    
        print(fname)
  
        s=readFile.readlines()
        lines=[]

        for l in s:
            l=l.replace("\n", " ").strip()    
            lines.append(l)
        ln=len(lines)

        embeddings = sbert_model.encode(lines)

        myDict={"text":[],"embedding":[]}


        for i in range(len(lines)):
            myDict["text"].append(lines[i])
            myDict["embedding"].append(embeddings[i])

        df=pd.DataFrame(myDict)
        #df.to_csv("EXL_Embeddings.csv")
        #print(df.head(2))
        vector = np.array(embeddings).astype(np.float32).tobytes()

        doc_hash = {
            "text": fname,
            "embedding": vector
        }

        res = conn.hset(name=f"doc:"+fname, mapping=doc_hash)
        if res:
            print('data stored in redis')
        else:
            print('Error in storing data in redis ')


        
        p.execute()

        df_string = df.to_csv(index=False) #needs to convert to string before storing to reddis
        print("Storing in Redis")
        #res = redis_client.set(fname, df_string)	#This will save the data to redis database - 0 with name as masterdataset
        #read_raw_data = conn.hgetall("doc:"+fname)
        #clean_data = read_raw_data.decode('utf-8')
        #print(read_raw_data) #printing data read from redis database - 0 with name masterdataset    
    conn.close()
    print("Embeddings created and stored in Redis")

@app.route('/askQuestion',methods=['POST'])
def askQuestions():
    questions_list=[]
    req_data=request.get_json()
    questions = req_data['questions']
    questions_list.append(questions)
    answer=embeddingsInRedis.ask(questions_list)
    print(questions)
    return jsonify({"Answer":answer})


if __name__ == '__main__':
    port=int(os.environ["PORT"])
    print(port)
    app.run(debug=True,host="0.0.0.0",port=port)
