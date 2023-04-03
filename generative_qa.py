from langchain.chains import LLMChain
import faiss
from langchain import OpenAI
from langchain.prompts import PromptTemplate
import pickle

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index

prompt_template = """Use the context below to write a 400 word answer to the question below, using GSQL code examples when appropriate:
    Context: {context}
    Question: {question}
    Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

llm = OpenAI(temperature=0)

chain = LLMChain(llm=llm, prompt=PROMPT)

def generate_answer(question):
    docs = store.similarity_search(question, k=5)
    inputs = [{"context": doc.page_content, "question": question} for doc in docs]
    print(chain.apply(inputs)[0]["text"])

while True:
    q = input("User: ")
    generate_answer(q)