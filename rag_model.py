# 1. WebBaseLoader to read the web page
# 2. RecuresiveCharacterTextSplitter to chink the content inti documents
# 3. Convert the documents into embeddings and store into as FAISS DB
# 4. Createa stuff documents chain, create a retreival chain from the FAISS DB
# 5. Create a Retreival Chain using the FAISS retriver and docuent chian

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os

load_dotenv()

# Get the API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')

if api_key is None:
    raise ValueError("No OPENAI_API_KEY found in environment variables")

loader = WebBaseLoader("https://code4x.dev/courses/chat-app-using-langchain-openai-gpt-api-pinecone-vector-database/")

docs = loader.load()

# this RecursiveCharacterTextSplitter takes a large text and splits it based on a specified chunk size
# it does this bt using a set of characters . the default characters provided to it are ["\n\n","\n"," ",""]

text_splitter = RecursiveCharacterTextSplitter()

documents = text_splitter.split_documents(docs)

llm = ChatOpenAI(openai_api_key=api_key)

embeddings = OpenAIEmbeddings()

# FAISS (Facebook ai similarity search) is a library that allows developers to stire and search for embedding of docments that are similar to each other

vector = FAISS.from_documents(documents,embeddings)

# we have taken the document and  takent he info from the webpage and then spit it multiple doc and then convert it into language embedding of openai and then store them into the faiss vecotr databse 


prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the pervious context:
                                         
<context>
                                          {context}
                                          </context>
                                          Question:{input}""" )

document_chain = create_stuff_documents_chain(llm,prompt)

retriever = vector.as_retriever()

retrieval_chain = create_retrieval_chain(retriever,document_chain) # document chian beign part of the retrival chain

response = retrieval_chain.invoke({"context":"You are the trainer who is teaching the given course and you are to suggest to potential learners",
                                   "input":"What are the kew takeaways for learners from the Course?"})

print(response["answer"])