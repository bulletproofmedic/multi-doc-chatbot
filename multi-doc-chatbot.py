import os
import sys
import warnings
from pprint import pprint
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from dotenv import load_dotenv
from bs4 import BeautifulSoup as Soup
from chromadb.config import Settings

from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser, OpenAIWhisperParserLocal
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders.chatgpt import ChatGPTLoader # Import ChatGPT conversation history
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import 

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredURLLoader,
    WikipediaLoader,
    UnstructuredHTMLLoader,
    )

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

from langchain.text_splitter import (
    Language,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    )

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain import OpenAI, VectorDBQA

llm = OpenAI(temperature=0)
warnings.filterwarnings("ignore")
client = chromadb.HttpClient(host='localhost', port=8000)

load_dotenv('.env')
API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = API_KEY

# Define variables
dir_path = "C:/AI/Web_App"

doc_names = []
documents = []
yt_vids = ["

]

urls = [
    "https://www.innovaitivesolutions.ca",
    "https://www.innovaitivesolutions.ca",
]

recursive_url = "https://docs.python.org/3.9/"

gpt_log = "./example_data/fake_conversations.json"

wiki = "<search term>"

### DEFINE FUNCTIONS ###

def doc_combine():
combined_docs = [doc.page_content for doc in documents]
text = " ".join(combined_docs)

def char_split():
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=75)
    documents = text_splitter.split_documents(documents)

    
def rec_char_split():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_text(text)


def store(db_name):
    # Save to Vectorstore
    collection_name = db_name  # You can choose any valid name based on the naming restrictions
    collection = client.create_collection(name=collection_name, embedding_function=openAIEmbeddings())
    collection.add(
        embeddings=documents
    )


def process_docs(split_type)
    # Load Documents
    documents = loader.load()
    
    # Combine Documents
    doc_combine()
    
    # Split Documents
    if split_type = "Character":
        char_split()
    elif split_type = "Recursive":
        rec_char_split()
    elif split_type = "":
        
    # Index Documents
    index()
    

def qa_chain():


def single_vectorstore(vs_name, vs_desc):
    vectorstore_info = VectorStoreInfo(
        name=vs_name,
        description=vs_desc,
        vectorstore=name+"_store",
    )
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
    agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)


# Function to ingest ChatGPT conversation history log
def gpt_history():
    loader = ChatGPTLoader(log_file=gpt_log, num_logs=1)
    documents = loader.load()
    
    
def yt_transcription():
    # set a flag to switch between local and remote parsing
    # change this to True if you want to use local parsing
    local = False
    
    # Directory to save audio files
    save_dir = "~/Downloads/YouTube"

    # Transcribe the videos to text
    if local:
        loader = GenericLoader(YoutubeAudioLoader(yt_vids, save_dir), OpenAIWhisperParserLocal())
    else:
        loader = GenericLoader(YoutubeAudioLoader(yt_vids, save_dir), OpenAIWhisperParser())
        
    # Load Documents
    documents = loader.load()
    
    # Combine Documents
    doc_combine()
    
    # Split Documents
    rec_char_split()
    
    # Index Documents
    index()
    
    
def wikipedia():
    
    """
    WikipediaLoader has these arguments:

        query: free text which used to find documents in Wikipedia
        optional lang: default="en". Use it to search in a specific language part of Wikipedia
        optional load_max_docs: default=100. Use it to limit number of downloaded documents. It takes time to download all 100 documents, so use a small number for experiments. There is a hard limit of 300 for now.
        optional load_all_available_meta: default=False. By default only the most important fields downloaded: Published (date when document was published/last updated), title, Summary. If True, other fields also downloaded.
    """
    loader = WikipediaLoader(query=wiki, load_max_docs=2).load()
    documents = loader.load()
    
    
# Function to ingest web page and all URLs under a root directory
def recursive_web():

    """
    Parameters
        url: str, the target url to crawl.
        exclude_dirs: Optional[str], webpage directories to exclude.
        use_async: Optional[bool], wether to use async requests, using async requests is usually faster in large tasks. However, async will disable the lazy loading feature(the function still works, but it is not lazy). By default, it is set to False.
        extractor: Optional[Callable[[str], str]], a function to extract the text of the document from the webpage, by default it returns the page as it is. It is recommended to use tools like goose3 and beautifulsoup to extract the text. By default, it just returns the page as it is.
        max_depth: Optional[int] = None, the maximum depth to crawl. By default, it is set to 2. If you need to crawl the whole website, set it to a number that is large enough would simply do the job.
        timeout: Optional[int] = None, the timeout for each request, in the unit of seconds. By default, it is set to 10.
        prevent_outside: Optional[bool] = None, whether to prevent crawling outside the root url. By default, it is set to True.
    """
    
    loader = RecursiveUrlLoader(url=url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text)
    documents = loader.load()

def ingest_dir():
    # Load documents from all of files in the dir_path
    for file in dir_path:
        if file not in doc_names:
            if file.endswith(".csv"):
                csv_path = dir_path + "/" + file
                loader = CSVLoader(csv_path) # Add source_column="<name>" for chains that provide sources
                documents = loader.load()
            elif file.endswith(".doc"):
                doc_path = dir_path + "/" + file
                loader = Docx2txtLoader(doc_path)
                documents = loader.load()
            elif file.endswith(".html"):
                html_path = dir_path + "/" + file
                loader = UnstructuredHTMLLoader(html_path)
                documents = loader.load()
                html_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.HTML, chunk_size=60, chunk_overlap=0
                )
                html_docs = html_splitter.create_documents([documents])
            elif file.endswith(".js", ".py"):
                code_path = dir_path + "/" + file
                loader = GenericLoader.from_filesystem(
                    dir_path,
                    glob="*",
                    suffixes=[".py", ".js"],
                    parser=LanguageParser(),
                )
                documents = loader.load()
                if file.endswith(".py"):
                    python_splitter = RecursiveCharacterTextSplitter.from_language(
                        language=Language.PYTHON, chunk_size=50, chunk_overlap=0
                    )
                    python_docs = python_splitter.create_documents([documents])
                elif file.endswith(".js"):
                    js_splitter = RecursiveCharacterTextSplitter.from_language(
                        language=Language.JS, chunk_size=60, chunk_overlap=0
                    )
                    js_docs = js_splitter.create_documents([documents])
                    
            else:
                document_path = dir_path + "/" + file
                loader = DirectoryLoader(doc_path, show_progress=True, use_multithreading=True)
                documents = loader.load()
                rec_char_split()
        else:
            pass
        


# Create our Q&A chain
pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo-16k'),
    retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the DocBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = pdf_qa(
        {"question": query, "chat_history": chat_history})
    print(f"{white}Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))