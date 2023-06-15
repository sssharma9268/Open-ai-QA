import fitz
import bertEmbedding
from flask import *

app = Flask(__name__)


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
    app.run()

# lines=[]
#,encoding="ISO-8859-1"
# with open("EXL_Reports_2023.txt","r") as f:
    
#     lines = f.readlines()

# print(lines)