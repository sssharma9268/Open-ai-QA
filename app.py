from flask import Flask,request, jsonify
import os
from flask_cors import CORS
import glob
import pandas as pd


from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

app = Flask(__name__)
CORS(app)

#os.environ["PORT"] = "5000"
#os.environ["OPENAI_API_KEY"] = "old-key-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"


@app.route('/')
def index():
    return 'Server Works'

@app.route("/upload",methods=["POST"])
def upload():
    uploaded_files = request.files.getlist('files')

    for file in uploaded_files:
        file.save("./MyDrive/"+file.filename)                                      
        
    processFiles(uploaded_files)
    
    print("Files Uploaded")
    files=glob.glob('./MyDrive/*.pdf')
    print(files)
    for filename in files:
        os.unlink(filename)
   
    return jsonify({"success":True})

def processFiles(uploaded_files):
    loader = PyPDFDirectoryLoader("./MyDrive/")
    docs = loader.load()
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #embeddings=OpenAIEmbeddings()
    global vectorDB
    vectorDB = Chroma.from_documents(docs,embedding_function)
 
    print("Embeddings stored in vectorDB")
  
  

@app.route('/askQuestion',methods=['POST'])
def askQuestions():
    vectordb = vectorDB
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0),vectordb.as_retriever(),memory=memory)
    
    req_data=request.get_json()
    questions = req_data['questions']
    result = pdf_qa({"question": questions})
    print(result)
   
    return jsonify({"Answer":result["answer"]})
    #return result["answer"]


if __name__ == '__main__':
    port=int(os.environ["PORT"])
    print(port)
    app.run(debug=True,host="0.0.0.0",port=port)
