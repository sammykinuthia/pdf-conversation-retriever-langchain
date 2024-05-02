import os
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import secret


os.environ["OPENAI_API_KEY"] = secret.OPENAI_API_KEY
# Enable to save to disk & reuse the model (for repeated queries on the same data)
# PERSIST = False

# if PERSIST and os.path.exists("persist"):
#     print("Reusing index...\n")
#     vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
#     index = VectorStoreIndexWrapper(vectorstore=vectorstore)
# else:
# loader = UnstructuredFileLoader('data.txt')
# raw_documents = loader.load()
#   loader = TextLoader("data.txt",) # Use this line if you only need data.txt
#   loader = DirectoryLoader("data/")
#   if PERSIST:
#     index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
#   else:


# while True:
#   if not query:
#     query = input("Prompt: ")
#   if query in ['quit', 'q', 'exit']:
#     sys.exit()
#   result = chain({"question": query, "chat_history": chat_history})
#   print(result['answer'])

#   chat_history.append((query, result['answer']))
#   query = None


def myChatGPT(query, chat_history):
    loader = UnstructuredFileLoader("data.txt")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    documents = text_splitter.split_documents(loader.load())
    embeddings = OpenAIEmbeddings()
    index = VectorstoreIndexCreator(embedding=embeddings).from_documents(documents)

    # index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
        
    )

    result = chain({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    # print(result)
    return result["answer"]


