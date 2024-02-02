import faiss
from langchain.vectorstores import FAISS
import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader

# https://docs.tigergraph.com/sitemap.xml
'''
urls = [
    "https://docs.tigergraph.com/tigergraph-server/current/api/built-in-endpoints"
]

loader = UnstructuredURLLoader(urls=urls)
documents = loader.load()
'''
#loader = SitemapLoader("https://docs.tigergraph.com/sitemap-pytigergraph.xml")
#documents = loader.load()
loader = DirectoryLoader('/Users/parkererickson/pytigergraph-docs', glob="**/*.adoc", loader_cls=TextLoader)
documents = loader.load()
#loader = DirectoryLoader('/Users/parkererickson/pytg-examples', glob="**/*.py", loader_cls=TextLoader)
#documents += loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key="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")

store = FAISS.from_documents(texts, embeddings)
faiss.write_index(store.index, "pytg.index")
store.index = None
with open("pytg_faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)