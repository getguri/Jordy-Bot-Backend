import asyncio
from datetime import datetime, timedelta,UTC
from io import BytesIO
import os
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import status, HTTPException
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv, dotenv_values 
import queue
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document
from langchain_community.document_loaders.blob_loaders.schema import Blob
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
import sys
from typing import Any, List, Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.tools.tavily_search import TavilySearchResults

from agent import Agent
from essay_agent import EssayAgent
from helper import ExpertWriter
from investment_management_agent import InvestmentManagementAgent
from sales_agent import SalesAgent
from sql_app import models, crud, schemas
from sql_app.database import SessionLocal, engine
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt

# loading variables from .env file
load_dotenv() 

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Message(BaseModel):
    prompt_key: str
    prompt: str

@app.get("/")
async def root():
    return {"message": "Hello World"}


q = queue.Queue()
stop_item = "###finish###"

class Prompt(BaseModel):
    key: str
    text: str

prompts = List[schemas.PromptBase]

prompts = [
    Prompt.model_validate({
        "key": "sales_pitch",
        "text": """You are a good assistant that validates a sales pitch for the product. Extend your knowledge and use following piece of context to know more about the product. Use it to validate the sales pitch provided in the prompt.
    If the answer can't be found in the context, just say you don't know. *DO NOT* try to make up an answer.
    If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
    Give a response in the same language as the question.""",
        
    }),
    Prompt.model_validate({
        "key": "generate_questions",
        "text": """You are a good assistant that creates quiz questions for the product. Extend your knowledge and use following piece of context to know more about the product.
    If the answer can't be found in the context, just say you don't know. *DO NOT* try to make up an answer.
    If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
    Give a response in the same language as the question.""",
    }),
    Prompt.model_validate({
        "key": "jordy",
        "text": """You are a good assistant that answers any questions. Extend your knowledge and use following piece of context to know more about the product.
    If the answer can't be found, then just politely respond that you don't know. *DO NOT* try to make up an answer.
    Give a response in English.""",
    })
]


class StreamingStdOutCallbackHandlerYield(StreamingStdOutCallbackHandler):
    def on_llm_start(
        self, serialized, prompts, **kwargs
    ) -> None:
        """Run when LLM starts running."""
        with q.mutex:
            q.queue.clear()
    def on_llm_new_token(self, token, **kwargs) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        sys.stdout.write(token)
        sys.stdout.flush()
        q.put(token)
    def on_llm_end(self, response, **kwargs) -> None:
        """Run when LLM ends running."""
        q.put(stop_item)


# Some basic configurations
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Security configuration
security = HTTPBasic()


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Function to verify password
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


# Function to create access token
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Function to authenticate user
def authenticate_user(username: str, password: str, db: Session = Depends(get_db)):
    if not verify_password(password,crud.get_user_by_username(db, username=username).hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return username

# Login route
@app.post("/login")
def login(currentUser: schemas.UserCreateFromAPI, db: Session = Depends(get_db)):
    username = authenticate_user(currentUser.username, currentUser.password, db)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Protected route
@app.get("/protected")
def protected(username: str = Depends(authenticate_user)):
    return {"username": username}

# Endpoint to create a new user
@app.post("/users/")
def create_user(username: str, password: str, db: Session = Depends(get_db)):
    # You can implement logic to store the user in the database here
    if crud.get_user_by_username(db, username=username):
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Hash the password before storing it
    hashed_password = pwd_context.hash(password)

    crud.create_user(db, schemas.UserCreate(username=username, hashed_password=hashed_password))
    return {"username": username}

@app.post("/prompt/")
async def send_prompt(message: Message):

    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o',
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandlerYield()]
        )

    DB_PATH = "vectorstores/db/"
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=GPT4AllEmbeddings(model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"))

    chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectorstore.as_retriever(),
            condense_question_llm=llm,
            return_source_documents=True,
            chain_type="stuff",
            get_chat_history=lambda h : h
        )

    PROMPT_TEMPLATE = """You are a good assistant that answers questions. Your knowledge is strictly limited to the following piece of context. Use it to answer the question at the end.
        If the answer can't be found in the context, just say you don't know. *DO NOT* try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
        Give a response in the same language as the question.
        
        Context: {context}
        """
    PROMPT_TEMPLATE = """You are a good assistant that answers questions. Extend your knowledge and use following piece of context. Use it to answer the question at the end.
    If the answer can't be found in the context, just say you don't know. *DO NOT* try to make up an answer.
    If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
    Give a response in the same language as the question.
    
    Context: {context}
    """
    system_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = system_prompt
    ai_response = chain({"question": message.prompt, 'chat_history': ''})
    return StreamingResponse(ai_response['answer'], media_type="application/octet-stream")


@app.post("/rag_based_chat/")
async def rag_based_chat(message: Message, db: Session = Depends(get_db)):

    print("message===",message.prompt)
    print("message===",message.prompt_key)
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o',
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandlerYield()]
        )

    DB_PATH = "vectorstores/db/"
    if message.prompt_key:
        DB_PATH =  DB_PATH + message.prompt_key + "/"
    print("DB_PATH===",DB_PATH)
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=GPT4AllEmbeddings(model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"))

    chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectorstore.as_retriever(),
            condense_question_llm=llm,
            return_source_documents=True,
            chain_type="stuff",
            get_chat_history=lambda h : h
        )

    PROMPT_TEMPLATE = """You are a good assistant that validates a sales pitch for the product. Extend your knowledge and use following piece of context to know more about the product. Use it to validate the sales pitch provided in the prompt.
    If the answer can't be found in the context, just say you don't know. *DO NOT* try to make up an answer.
    If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
    Give a response in the same language as the question.
    
    Context: {context}
    """

    USER_PROMPT = """
    You are a good assistant that answers questions. Extend your knowledge and use following piece of context to know more about the product. Use it to answer the question at the end.
    If the answer can't be found, just say you don't know. *DO NOT* try to make up an answer.
    Give a response in the same language as the question.
    """

    

    PROMPT_TEMPLATE = """
    {user_prompt}
    Context: {context}
    """

    current_prompt = crud.get_prompt(db, key=message.prompt_key)
    if current_prompt:
        USER_PROMPT = current_prompt.text
    system_prompt = ChatPromptTemplate.from_template(USER_PROMPT) + "\n\nContext: {context}"
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = system_prompt
    ai_response = chain({"question": message.prompt, 'chat_history': ''})
    return StreamingResponse(ai_response['answer'], media_type="application/octet-stream")


@app.post("/sales_pitch/")
async def sales_pitch(message: Message, db: Session = Depends(get_db)):

    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o',
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandlerYield()]
        )

    DB_PATH = "vectorstores/db/"
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=GPT4AllEmbeddings(model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"))

    chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectorstore.as_retriever(),
            condense_question_llm=llm,
            return_source_documents=True,
            chain_type="stuff",
            get_chat_history=lambda h : h
        )

    PROMPT_TEMPLATE = """You are a good assistant that validates a sales pitch for the product. Extend your knowledge and use following piece of context to know more about the product. Use it to validate the sales pitch provided in the prompt.
    If the answer can't be found in the context, just say you don't know. *DO NOT* try to make up an answer.
    If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
    Give a response in the same language as the question.
    
    Context: {context}
    """

    USER_PROMPT = """
    You are a good assistant that answers questions. Extend your knowledge and use following piece of context to know more about the product. Use it to answer the question at the end.
    If the answer can't be found, just say you don't know. *DO NOT* try to make up an answer.
    Give a response in the same language as the question.
    """

    

    PROMPT_TEMPLATE = """
    {user_prompt}
    Context: {context}
    """

    current_prompt = crud.get_prompt(db, key="sales_pitch")
    if current_prompt:
        USER_PROMPT = current_prompt.text
    system_prompt = ChatPromptTemplate.from_template(USER_PROMPT) + "\n\nContext: {context}"
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = system_prompt
    ai_response = chain({"question": message.prompt, 'chat_history': ''})
    return StreamingResponse(ai_response['answer'], media_type="application/octet-stream")

        
@app.post("/generate_questions/")
async def generate_questions(message: Message, db: Session = Depends(get_db)):

    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o',
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandlerYield()]
        )

    DB_PATH = "vectorstores/db/"
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=GPT4AllEmbeddings(model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"))

    chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectorstore.as_retriever(),
            condense_question_llm=llm,
            return_source_documents=True,
            chain_type="stuff",
            get_chat_history=lambda h : h
        )



    USER_PROMPT = """
    You are a good assistant that answers questions. Extend your knowledge and use following piece of context to know more about the product. Use it to answer the question at the end.
    If the answer can't be found, just say you don't know. *DO NOT* try to make up an answer.
    Give a response in the same language as the question.
    """


    current_prompt = crud.get_prompt(db, key="generate_questions")
    if current_prompt:
        USER_PROMPT = current_prompt.text
    system_prompt = ChatPromptTemplate.from_template(USER_PROMPT) + "\n\nContext: {context}"
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = system_prompt
    ai_response = chain({"question": message.prompt, 'chat_history': ''})
    return StreamingResponse(ai_response['answer'], media_type="application/octet-stream")


@app.post("/jordy/")
async def jordy(message: Message, db: Session = Depends(get_db)):

    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o',
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandlerYield()]
        )

    DB_PATH = "vectorstores/db/"
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=GPT4AllEmbeddings(model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"))

    chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectorstore.as_retriever(),
            condense_question_llm=llm,
            return_source_documents=True,
            chain_type="stuff",
            get_chat_history=lambda h : h
        )



    USER_PROMPT = """
    You are a good assistant that answers questions. Extend your knowledge and use following piece of context to know more about the product. Use it to answer the question at the end.
    If the answer can't be found, just say you don't know. *DO NOT* try to make up an answer.
    Give a response in the same language as the question.
    """

    current_prompt = crud.get_prompt(db, key="jordy")
    if current_prompt:
        USER_PROMPT = current_prompt.text

    system_prompt = ChatPromptTemplate.from_template(USER_PROMPT) + "\n\nContext: {context}"
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = system_prompt
    ai_response = chain({"question": message.prompt, 'chat_history': ''})
    return StreamingResponse(ai_response['answer'], media_type="application/octet-stream")

@app.post("/multi_agent/")
async def multi_agent(message: Message, db: Session = Depends(get_db)):
    messages = [HumanMessage(content=message.prompt)]
    streamedResponse = '' 
    tool = TavilySearchResults(max_results=2)
    memory = SqliteSaver.from_conn_string("agent_state.db")
    #memory = AsyncSqliteSaver.from_conn_string("agent_state.db")
    prompt = """You are a smart research assistant. Use the search engine to look up information. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """
    model = ChatOpenAI(model="gpt-4o")
    abot = SalesAgent(model, [tool])
    thread = {"configurable": {"thread_id": "1"}}
    # for event in abot.graph.stream({"messages": messages}, thread):
    #     for v in event.values():
    #         print(v)

    # ai_response = abot.graph.stream({
    #         'task': message.prompt,
    #         "max_revisions": 2,
    #         "revision_number": 1,
    # }, thread)
    
    max_revisions = 1
    for event in abot.graph.stream({
            'task': message.prompt,
            "max_revisions": max_revisions,
            "revision_number": 1,
    }, thread):
        print("type of event==", type(event))

        if "generate" in event:
            for key in event.keys():
                print(key) 
            print("revision==", event["generate"]["revision_number"])
            print("type of==", type(event["generate"]["draft"]))
            #event_data = event["generate"]
            # for key in event_data.keys():
            #     print(key) 
            if event["generate"]["revision_number"] == max_revisions+1:
                streamedResponse = streamedResponse + event["generate"]["draft"]
                
    return StreamingResponse(streamedResponse, media_type="application/octet-stream")
    
    #return {"message": "multi_agent ran successfully"}

# @app.post("/multi_agent/")
# async def multi_agent(message: Message, db: Session = Depends(get_db)):
#     messages = [HumanMessage(content=message.prompt)]
#     streamedResponse = '' 
#     tool = TavilySearchResults(max_results=2)
#     #memory = SqliteSaver.from_conn_string(":memory:")
#     memory = AsyncSqliteSaver.from_conn_string("agent_state.db")
#     prompt = """You are a smart research assistant. Use the search engine to look up information. \
#     You are allowed to make multiple calls (either together or in sequence). \
#     Only look up information when you are sure of what you want. \
#     If you need to look up some information before asking a follow up question, you are allowed to do that!
#     """
#     model = ChatOpenAI(model="gpt-4o")
#     abot = Agent(model, [tool], system=prompt, checkpointer=memory)
#     thread = {"configurable": {"thread_id": "1"}}
#     # for event in abot.graph.stream({"messages": messages}, thread):
#     #     for v in event.values():
#     #         print(v)

    
#     async for event in abot.graph.astream_events({"messages": messages}, thread, version="v1"):
#         kind = event["event"]
#         if kind == "on_chat_model_stream":
#             content = event["data"]["chunk"].content
#             if content:
#                 # Empty content in the context of OpenAI means
#                 # that the model is asking for a tool to be invoked.
#                 # So we only print non-empty content
#                 print(content, end="|")
#                 streamedResponse = streamedResponse + content
#     #events = abot.graph.astream_events({"messages": messages}, thread, version="v1")
#     return StreamingResponse(streamedResponse, media_type="application/octet-stream")
    
#     #return {"message": "multi_agent ran successfully"}


@app.post("/embed/")
#async def extract_text(Files: List[UploadFile] = File(...)):
async def extract_text(myfile: UploadFile):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=[
            "\n\n",
            "\n",
            ]
        )
        print("file extension==", myfile.content_type)
        if myfile.content_type == 'application/pdf':
            rag_file_temp_dir=os.getenv('RAG_FILE_TEMP_DIR')
            full_file_path = f"{rag_file_temp_dir}/{myfile.filename}"
            with open(full_file_path, "wb") as f:
                f.write(myfile.file.read())
            loader = PyPDFLoader(full_file_path)
            pages = loader.load_and_split()
            for page in pages:
                #print("PDF page===", page.page_content)
                #texts=text_splitter.create_documents([page.decode()])
                texts=text_splitter.create_documents([page.page_content])
                DB_PATH = "vectorstores/db/"
                vectorstore = Chroma.from_documents(documents=texts, embedding=GPT4AllEmbeddings(model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"), persist_directory=DB_PATH)      
                vectorstore.persist()
            os.remove(full_file_path)
        else:
            file_content = await myfile.read()   
            texts=text_splitter.create_documents([file_content.decode()])
            DB_PATH = "vectorstores/db/"
            vectorstore = Chroma.from_documents(documents=texts, embedding=GPT4AllEmbeddings(model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"), persist_directory=DB_PATH)      
            vectorstore.persist()

        return {"message": "File uploaded successfully"}
    except Exception as e:
        return {"message": e.args}
    


@app.post("/embed_multiple_files/{prompt_key}")
async def extract_text_multiple_files(files: List[UploadFile] = File(...), prompt_key:str="default"):
#async def extract_text(myfile: UploadFile):
    print("prompt_key==", prompt_key)
    print("embeding files==", len(files))
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=[
            "\n\n",
            "\n",
            ]
        )
        for myfile in files:
            print("file extension==", myfile.content_type)
            if myfile.content_type == 'application/pdf':
                rag_file_temp_dir=os.getenv('RAG_FILE_TEMP_DIR')
                full_file_path = f"{rag_file_temp_dir}/{myfile.filename}"
                with open(full_file_path, "wb") as f:
                    f.write(myfile.file.read())
                loader = PyPDFLoader(full_file_path)
                pages = loader.load_and_split()
                for page in pages:
                    #print("PDF page===", page.page_content)
                    #texts=text_splitter.create_documents([page.decode()])
                    texts=text_splitter.create_documents([page.page_content])
                    DB_PATH = f"vectorstores/db/{prompt_key}"
                    vectorstore = Chroma.from_documents(documents=texts, embedding=GPT4AllEmbeddings(model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"), persist_directory=DB_PATH)      
                    vectorstore.persist()
                os.remove(full_file_path)
            else:
                file_content = await myfile.read()   
                texts=text_splitter.create_documents([file_content.decode()])
                DB_PATH = f"vectorstores/db/{prompt_key}"
                vectorstore = Chroma.from_documents(documents=texts, embedding=GPT4AllEmbeddings(model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"), persist_directory=DB_PATH)      
                vectorstore.persist()

        return {"message": "File uploaded successfully"}
    except Exception as e:
        return {"message": e.args}


@app.delete("/vector_stores/{prompt_key}")
async def delete_vector_store(prompt_key:str):
#async def extract_text(myfile: UploadFile):
    print("prompt_key==", prompt_key)
    try:
        DB_PATH = f"vectorstores/db/{prompt_key}"
        print("DB_PATH===",DB_PATH)
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=GPT4AllEmbeddings(model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"))
        vectorstore.delete_collection()
        vectorstore.persist()

        return {"message": "Vector store deleted successfully"}
    except Exception as e:
        return {"message": e.args}


@app.post("/prompts/")
async def create_prompt(currentPrompt:schemas.PromptBase, db: Session = Depends(get_db)):
    print("currentPrompt==", currentPrompt)
    crud.create_prompt(db, currentPrompt)
    return {"message": "Prompt created successfully"}
    
@app.put("/prompts/{prompt_key}")
async def update_prompt(prompt_key:str, currentPrompt:schemas.PromptBase, db: Session = Depends(get_db)):
    print("prompt_key==", prompt_key)
    print("currentPrompt==", currentPrompt)
    crud.update_prompt(db, currentPrompt)
    return {"message": "Prompt text updated successfully"}

@app.get("/prompts/")
async def prompt_list(db: Session = Depends(get_db)):
    return crud.get_prompts(db)



@app.delete("/prompts/{prompt_key}")
async def delete_prompt(prompt_key:str, db: Session = Depends(get_db)):
    crud.delete_prompt(db, prompt_key)
    return {"message": "Prompt deleted successfully"}

@app.post("/investment_management/")
async def multi_agent(message: Message, db: Session = Depends(get_db)):
    messages = [HumanMessage(content=message.prompt)]
    streamedResponse = '' 
    tool = TavilySearchResults(max_results=2)
    memory = SqliteSaver.from_conn_string("agent_state.db")
    #memory = AsyncSqliteSaver.from_conn_string("agent_state.db")
    prompt = """You are a smart research assistant. Use the search engine to look up information. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """
    model = ChatOpenAI(model="gpt-4o")
    abot = InvestmentManagementAgent(model, [tool])
    thread = {"configurable": {"thread_id": "1"}}
    # for event in abot.graph.stream({"messages": messages}, thread):
    #     for v in event.values():
    #         print(v)

    # ai_response = abot.graph.stream({
    #         'task': message.prompt,
    #         "max_revisions": 2,
    #         "revision_number": 1,
    # }, thread)
    
    max_revisions = 1
    for event in abot.graph.stream({
            'task': message.prompt,
            "max_revisions": max_revisions,
            "revision_number": 1,
    }, thread):
        print("type of event==", type(event))

        if "generate" in event:
            for key in event.keys():
                print(key) 
            print("revision==", event["generate"]["revision_number"])
            print("type of==", type(event["generate"]["draft"]))
            #event_data = event["generate"]
            # for key in event_data.keys():
            #     print(key) 
            if event["generate"]["revision_number"] == max_revisions+1:
                streamedResponse = streamedResponse + event["generate"]["draft"]
                
    return StreamingResponse(streamedResponse, media_type="application/octet-stream")
    
    #return {"message": "multi_agent ran successfully"}
