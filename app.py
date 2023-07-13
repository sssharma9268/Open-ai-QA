from flask import Flask,request, jsonify
import os
from flask_cors import CORS

import pandas as pd

import pypdf
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

app = Flask(__name__)
CORS(app)
global vectorDB

os.environ["PORT"] = "5000"
os.environ["OPENAI_API_KEY"] = "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"

class DB:
    
    #Declaring a constructor, taking length and colour as a parameter
    #Remember 'self'(object) is passed to each method
    def __init__(self, db):
        self.db = db

    def getDB(self):
        return self.db



@app.route('/')
def index():
    return 'Server Works'

@app.route("/upload",methods=["POST"])
def upload():
    uploaded_files = request.files.getlist('files')

    for file in uploaded_files:
        file.save("./MyDrive/"+file.filename)                                      
        
    db=processFiles(uploaded_files)
    
    #print("Files Uploaded")
    #files=glob.glob('*.pdf')
    #print(files)
    #for filename in files:
        #os.unlink(filename)
   
    return jsonify({"success":True})

def processFiles(uploaded_files):
    loader = PyPDFDirectoryLoader("./MyDrive/")
    docs = loader.load()
    embeddings=OpenAIEmbeddings()
    vectordb = Chroma.from_documents(docs,embedding=embeddings)
    db = DB(vectordb)
    print("Embeddings stored in vectorDB")
    return db
  

@app.route('/askQuestion',methods=['POST'])
def askQuestions():
    db=DB()
    vectordb = db.getDB()
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.9),vectordb.as_retriever(),memory=memory)
    
    req_data=request.get_json()
    questions = req_data['questions']
    result = pdf_qa({"question": questions})
   
    return jsonify({"Answer":result})


if __name__ == '__main__':
    port=int(os.environ["PORT"])
    print(port)
    app.run(debug=True,host="0.0.0.0",port=port)
