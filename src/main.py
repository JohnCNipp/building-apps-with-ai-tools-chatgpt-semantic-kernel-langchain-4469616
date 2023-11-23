from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.memory.vectorstore import VectorStoreRetrieverMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from qdrant_client.http import models as rest
from pydantic import BaseModel, Field
from langchain.document_loaders.csv_loader import CSVLoader
from OPEN_AI_KEY import OPENAI_API_KEY

import csv
from typing import Dict, List, Optional
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document

## Goal: Create a librarian with a library.
# Extend the librarian app from chapter 3 to include vector DB.
# Work with .csv datasets
# etch results with relevant book data and meta data


# What am I going to need?
# Use a CharacterTextSplitter from langchain.text_splitter
# Maybe extract objects straight from the csv files
# We want our answer to refer to the data in the csv file
# and inform the user with the most important information
# We want to extract the books based on the meta data from
# what we have found in the dataset

# class Book(BaseModel):
#     isbn10: Field(description= "isbn10 number")
#     isbn13: Field(description= "isbn13 number")
#     title: Field(description= "Title of the book")
#     subtitle: Field(description= "Subtitle of the book, if there is one")
#     authors: Field(description= "The author or authors of the book")
#     categories: Field(description= "The genres the book falls into")
#     thumbnail: Field(description= "A url to the thumnail of the book")
#     description: Field(description= "Description of the book")
#     published_year: Field(description= "Year the book was published")
#     average_rating: Field(description= "The average rating of the book")
#     num_pages: Field(description= "The number of pages in the book")
#     ratings_count: Field(description= "The number of times the book has been rated")

# parser = PydanticOutputParser(pydantic_object = Book)
loader = CSVLoader(
    file_path="./src/dataset_small.csv", source_column="title")

data = loader.load()
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, request_timeout=120)
vectorstore = FAISS.from_documents(data, embeddings)

# embeddings = OpenAIEmbeddings()
# qa = RetrievalQA.from_chain_type(llm = OpenAI, from_chain_type = "stuff",
#     retriever=vectorstore.as_retriever(), return_source_documents = True)

visitor_inquiry = input("""Hello, please tell me a little bit about yourself",
so that I can help you pick out a book that you might like.\n""")
# Need to connect the loading of the csv file with the pydantic objects
#
system_template = """You are a virtual librarian, who is
            helping visitors find a book that they might enjoy reading. You are
            knowledgeable and give reccomendations based on personal preferences
            as well as their character traits. Return your reccomendation as a
            python object with all of the relevant meta data from the csv
            included. Don't ask for more context, just give a reccommendation.
            Please include information such as average_rating, published year,
            and number of pages. Reference the csv provided as embeddings in
            memory."""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")
chat_prompt = ChatPromptTemplate.from_messages(
[system_message_prompt, human_message_prompt]
)
retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)

chain = LLMChain(
    llm=ChatOpenAI(temperature = .5,
                    openai_api_key = OPENAI_API_KEY),
    prompt=chat_prompt,
    memory = memory,
    # output_parser=parser
)
print(chain.run(visitor_inquiry))
