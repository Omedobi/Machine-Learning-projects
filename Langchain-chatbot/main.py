from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import SeleniumURLLoader
from langchain import PromptTemplate
from getpass import getpass
import openai
from dotenv import load_dotenv
import os


openai.api_key = os.getenv("OPENAI_API_KEY")

articles = [
    'https://www.digitaltrends.com/computing/claude-sonnet-vs-gpt-4o-comparison/',
    'https://www.digitaltrends.com/computing/apple-intelligence-proves-that-macbooks-need-something-more/',
    'https://www.digitaltrends.com/computing/how-to-use-openai-chatgpt-text-generation-chatbot/', 
    'https://www.digitaltrends.com/computing/character-ai-how-to-use/', 
    'https://www.digitaltrends.com/computing/how-to-upload-pdf-to-chatgpt/',
]

# load the documents using Selenium loader
loader = SeleniumURLLoader(urls=articles)
docs_not_splitted = loader.load()

# split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs_not_splitted)

#make embedding from the content
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

#Adding Activeloop ID
my_activeloop_org_id = "erder"
my_activeloop_dataset_name = "chatbot_article_dataset"
path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=path, embedding_function= embeddings, verbose=False)

#Add the documents to Deeplake dataset
db.add_documents(docs)

#template

template = "You are an exceptional customer support chatbot, that gently answer questions"