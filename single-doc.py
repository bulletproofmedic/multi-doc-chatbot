from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader

load_dotenv('.env')

pdf_loader = PyPDFLoader(input('Provide full file path:'))
documents = pdf_loader.load()

chain = load_qa_chain(llm=OpenAI())

while True:
    print("Choose an option:")
    choice = input("1. Query document \n 2. Exit")
        if choice == "1":
            query = input("Enter your query")
            response = chain.run(input_documents=documents, question=query)
            print(response)
        elif choice == "2":
            break
        else:
            print("Invalid choice.")
            continue