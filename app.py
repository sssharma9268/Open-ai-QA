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
    uploaded_file = request.files['files']
    processFiles(uploaded_file.filename)
    print("Files Uploaded")
    return "Success"

def processFiles(filename):      
    with fitz.open(filename) as doc:
        for page in doc:
            text = page.get_text()
            print("text---->"+text)
            with open("./TextFiles/TextFile.txt", "a") as f:
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
    return answer


if __name__ == '__main__':
    port=int(os.environ["PORT"])
    print(port)
    app.run(debug=True,host="0.0.0.0",port=port)
