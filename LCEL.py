# LCEL : Langchain Expression language
# comprises of a prompt , llm object and stringout parser

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages([
    {"system","You are world class technical write"},
    {"user","{input}"}
])

output_parser= StrOutputParser()

chain = prompt |llm |output_parser

output = chain.invoke({"input":"how can langsmith help with testing?"})

print(output)