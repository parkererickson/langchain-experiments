from langchain.chains import LLMChain
import faiss
from langchain import OpenAI
from langchain.prompts import PromptTemplate
import pickle
from langchain.utilities import PythonREPL
import pyTigerGraph as tg
import config as cfg

conn = tg.TigerGraphConnection(cfg.host, graphname=cfg.graph, username=cfg.user, password=cfg.pw)
conn.getToken(conn.createSecret())

# Load the LangChain.
index = faiss.read_index("pytg.index")

with open("pytg_faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index

prompt_template = """Use the context, vertex types, edge types to write the pyTigerGraph function call to answer the question using a pyTigerGraph connection.
For example, if a count of vertices is asked for, use getVertexCount(). If multiple entites are referred to in the question, use runInstalledQuery() when necessary.
Parameters of queries to be replaced are denoted <INSERT_ID_HERE>.
Vertex Types: {vertices}
Edge Types: {edges}
Queries: {queries}
Context: {context}
Question: {question}
Python Call: conn."""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["question", "context", "vertices", "queries", "edges"]
)

question_restate = """Replace the entites mentioned in the question to one of these choices: {vertices}.
                      Replace the relationships mentioned in the question to one of these choices: {edges}.
                      Generate the complete question with the appropriate replacements.
                      If there are no replacements to be made, restate the question.
                      Example: How many universities are there?
                      Response: How many vertices are University Vetexes?
                      Example: What is the schema?
                      Response: What is the schema?
                      Example: How many transactions are there?
                      Response: How many Transaction Edges are there?
                      QUESTION: {question}
                      RESTATED: """

RESTATE_QUESTION_PROMPT = PromptTemplate(
    template=question_restate, input_variables=["question", "vertices", "edges"]
)

llm = OpenAI(temperature=0, model_name="text-davinci-003")

chain = LLMChain(llm=llm, prompt=PROMPT)
restate_chain = LLMChain(llm=llm, prompt=RESTATE_QUESTION_PROMPT)

def generate_answer(question):
    restate_q = restate_chain.apply([{"vertices": [x + " Vertex" for x in conn.getVertexTypes()], # + [x + " Edge" for x in conn.getEdgeTypes()],
                                      "question": question,
                                      "edges": [x + " Edge" for x in conn.getEdgeTypes()]}])[0]["text"]

    print("RESTATED QUESTION:", restate_q)

    docs = store.similarity_search(restate_q, k=5)
    print()
    for doc in docs:
        print(doc)
    print()
    inputs = {"question": restate_q, 
                "vertices": conn.getVertexTypes(), 
                "edges": conn.getEdgeTypes(), 
                "queries": {"get_papers_of_author": {"auth": "<INSERT_ID_HERE>"},
                            "get_number_of_papers_for_author": {"auth": "<INSERT_ID_HERE>"},
                            "author_institutions": {"auth_id": "<INSERT_ID_HERE>"},
                            "num_authors_at_institution": {"inst": "<INSERT_ID_HERE>"}
                            }}
    generated = chain.apply(inputs)[0]["text"]
    loc = {}
    try:
        print(generated)
        exec("res = conn."+generated, {"conn": conn}, loc)
        print(loc["res"])
    except Exception as e:
        print("Failed getting result for generated:", generated)
        print("Exception was:", e)

while True:
    q = input("User: ")
    generate_answer(q)
    print()