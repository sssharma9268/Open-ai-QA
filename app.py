import fitz
import bertEmbedding
from flask import *
import os

app = Flask(__name__)
#os.environ["PORT"] = "5000"

@app.route('/')
def index():
    return 'Server Works'

@app.route("/upload",methods=["POST"])
def upload():
    uploaded_files = flask.request.files.getlist("files[]")
    processFiles(uploaded_files)
    print("Files Uploaded")
    return "Success"

def processFiles(files):
    for file in files:    
        with fitz.open(file) as doc:
            for page in doc:
                text = page.get_text()
                with open("./TextFile.txt", "a") as f:
                    f.writelines(text)
                f.close()

@app.route('/askQuestion',methods=['POST'])
def askQuestions():
    questions_list=[]
    req_data=request.get_json()
    questions = req_data['questions']
    questions_list.append(questions)
    answer=bertEmbedding.ask(questions_list)
    print(questions)
    return "This is your question-"+questions+"and answer:--/n"+answer


if __name__ == '__main__':
    port=int(os.environ["PORT"])
    print(port)
    app.run(debug=True,host="0.0.0.0",port=port)
