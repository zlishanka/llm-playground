from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from utils import print_info

print_info("Start to load llama2...")
llm = Ollama(model="llama2")

# Retrieval Chain
# Retrieval is useful when you have too much data to pass to the LLM directly. 
# You can then use a retriever to fetch only the most relevant pieces and pass those in.
#(1) populate a vector store and use that as a retriever
print_info("Start to load doc...")
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()

#(2) index it into a vectorstore. This requires a few components, namely an embedding model and a vectorstore.
print_info("Start to create embeddings...")
embeddings = OllamaEmbeddings()

#(3) use this embedding model to ingest documents into a vectorstore. use a simple local vectorstore, FAISS to build our index
text_splitter = RecursiveCharacterTextSplitter()

print_info("Start to use embeddings model to ingest text into vectorstore...")
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

#This chain will take an incoming question, look up relevant documents, then pass those documents 
# along with the original question into an LLM and ask it to answer the original question.

#(4)set up the chain that takes a question and the retrieved documents and generates an answer.
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

print_info("Start to create retrieval chain ...")
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

print_info("Start to take a question and the retrieved documents and generates an answer ...")
response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])
print_info("Done.")