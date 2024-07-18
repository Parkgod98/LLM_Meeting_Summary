import speech_recognition as sr
from flask import Flask, render_template, request, redirect, url_for

from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

from openai import AzureOpenAI

load_dotenv()

import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024  # 최대 파일 크기: 16MB

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def GetWhisperText(path) : # 문자열 형태의 음성파일 경로 받아.
    load_dotenv()
    client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),  # 키 호출
        api_version= os.getenv("OPENAI_API_VERSION"), # api 버전 호출
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") # 앤드포인트 호출
    )

    deployment_id = 'whisper' #whisper
    audio_test_file = path # 음성 파일의 경로.
    result = client.audio.transcriptions.create( # 음성 
        file=open(audio_test_file, "rb"),            
        model=deployment_id
    )
    
    return result.text
    

def ORGANIZATION(text):
    load_dotenv()

    model = AzureChatOpenAI(
        azure_deployment = 'gpt-4o',
        temperature=1.0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 회의록 요약 전문가야. 내가 주는 대화 내용을 회의록 형식으로 정리/요약하고 그것을 HTML 코드 형태로 나에게 전달해줘."),
        ("system", "회의 일시는 한국의 현재 날짜와 시간, 참석자는 박현성,윤아영,이현수,경세진, 회의주제는 너가 잘 추정해"),
        ("system", "또한 주어진 회의내용에 뜻을 제대로 알수 없는 단어가 있거나, 맥락에 비춰봤을때 잘못 입력된것 같으면 적절하게 바꿔서 요약해줘."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    output_parser = StrOutputParser()

    chain = prompt | model | output_parser


    # 메모리 초기화
    memory = ConversationBufferMemory(
                chat_memory=InMemoryChatMessageHistory(),
                return_messages=True #대화 기록이 메시지 객체(HumanMessage, AIMessage등)의 리스트로 반환
            )

    chat_history = memory.chat_memory.messages
    output = chain.invoke({
                "input": text,
                "chat_history": chat_history        
        })
    
    memory.chat_memory.add_user_message(text)
    memory.chat_memory.add_ai_message(output)
    
    return output 

total_script = ""
transcript = ""


@app.route("/", methods=["GET", "POST"])
def index():
    global transcript, total_script
    
    pattern = re.compile(r'<!DOCTYPE html>.*?</html>', re.DOTALL)
    match = pattern.search(transcript)

    if match:
        transcript = match.group(0)
    else:
        transcript = ''
        
    if request.method == "POST":
        recognizer = sr.Recognizer()
        
        with sr.Microphone() as source:
            audio = recognizer.listen(source)
            try:
                transcript = recognizer.recognize_google(audio, language="ko-KR")
                print("Google Speech Recognition thinks you said " + transcript)
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
    
    return render_template('index.html', total_script=total_script, transcript=transcript)


@app.route('/upload', methods=['POST'])
def upload_file():
    global total_script,transcript
    
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        total_script = GetWhisperText(file_path)  # total_script에 할당
        transcript = ORGANIZATION(total_script)
        
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
