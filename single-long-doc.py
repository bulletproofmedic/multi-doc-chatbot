import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

OpenAI.api_key = os.getenv('OPENAI_API_KEY')

# load the document as before
loader = PyPDFLoader(input('Provide full file path: '))
documents = loader.load()

# we split the data into chunks of 1,000 characters, with an overlap
# of 200 characters between the chunks, which helps to give better results
# and contain the context of the information between chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

# we create our vectorDB, using the OpenAIEmbeddings tranformer to create
# embeddings from our text chunks. We set all the db information to be stored
# inside the ./data directory, so it doesn't clutter up our source files
vectordb = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
    persist_directory='./data'
)
vectordb.persist()

# we create the RetrievalQA chain, passing in the vectorstore as our source of
# information. Behind the scenes, this will only retrieve the relevant
# data from the vectorstore, based on the semantic similiarity between
# the prompt and the stored information
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(model_name="gpt-4"),
    retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True
)

# we can now exectute queries againse our Q&A chain
while True:
    print("Choose an option:")
    choice = input("1. Query document \n 2. Exit")
    if choice == "1":
        query = input("Enter your query")
        result = qa_chain({'query': f'{query}'})
        print(result['result'])
    elif choice == "2":
        break
    else:
        print("Invalid choice.")
        continue