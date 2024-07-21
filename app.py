import streamlit as st
from openai import AzureOpenAI
import os
import re

from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv


def GetWhisperText(audio_test_file) : # 문자열 형태의 음성파일 경로 받아.
    load_dotenv()
    client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),  # 키 호출
        api_version= os.getenv("OPENAI_API_VERSION"), # api 버전 호출
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") # 앤드포인트 호출
    )

    deployment_id = 'whisper' #whisper
    # audio_test_file = path # 음성 파일의 경로.
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
        ("system", "너는 회의록 요약 전문가야. 내가 주는 대화 내용을 회의록 형식으로 정리/요약하고 그것을 마크다운 형태로 나에게 전달해줘."),
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


#
st.title("회의록 요약 서비스")

st.write("이 서비스를 사용하여 회의록을 요약할 수 있습니다. 음성 파일을 업로드해주세요.")

uploaded_file = st.file_uploader("파일 업로드")

if uploaded_file is not None:
    # 업로드된 파일 변환
    with st.spinner("음성 파일 변환 중..."):
        transcription = GetWhisperText(uploaded_file)
    
    st.subheader("회의록 전문", anchor=None, help=None, divider=True)
    st.markdown(transcription, unsafe_allow_html=False)
    
    
    # 요약 생성
    with st.spinner("요약 생성 중..."):
        summary = ORGANIZATION(transcription)
        pattern = re.compile(r'```markdown', re.DOTALL)
        summary = re.sub(pattern, '', summary)

    
    # 요약 결과 출력
    st.subheader("회의 요약", anchor=None, help=None, divider=True)
    st.markdown(summary, unsafe_allow_html=False)
