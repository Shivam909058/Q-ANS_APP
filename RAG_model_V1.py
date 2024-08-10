import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from tqdm import tqdm

# Set API key directly (Note: This is not recommended for production use)
os.environ[
    "OPENAI_API_KEY"] = "sk-Tjq8_LHf2OkRViIKwtd29WS2EYhXQiIoNdNPvx06h8T3BlbkFJBAbx-iT9Y_LUcg4-hR-gilTUEj3ml8k1VkjyNT2WAA"

# Initialize components
loader = DirectoryLoader('./datasets', glob="**/*.txt", loader_cls=TextLoader)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def process_documents():
    print("Loading and processing documents...")
    try:
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        return None

    print("Splitting documents into chunks...")
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    vectorstore.persist()
    print("Vector store created and persisted")

    return vectorstore


def setup_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain


def main():
    vectorstore = process_documents()
    if vectorstore is None:
        print("Failed to process documents. Exiting.")
        return

    qa_chain = setup_qa_chain(vectorstore)

    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        result = qa_chain({"query": question})
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['result']}")
        print("\nSources:")
        for doc in result['source_documents']:
            print(f"- {doc.metadata['source']}")
        print()


if __name__ == "__main__":
    main()