import streamlit as st
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, get_response_synthesizer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import PromptTemplate
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_parse import LlamaParse
from dotenv import load_dotenv #python-dotenv

load_dotenv()

st.title(":llama: Llama Tutorial :llama:")
st.subheader("Llama API is best for RAG solutions implementation.")

st.text("""To run this code you must have Python environment installed and all dependencies. \nYou also have Streamlit and llama_index installed and have an openai api key \nand Llama cloud key configured in an .env file LLAMA_CLOUD_API_KEY='', \nOPEN_AI_KEY=''...""")
st.text("""Create a 'data' folder with test files and an 'Index' folder to save vectorstores""")

### Line 1 ###
st.text("Line 1 : Quick and Simple OpenAI integration ")

st.code("from llama_index.llms.openai import OpenAI \n\nresponse = OpenAI().complete('Michael Jackson is ')")
if st.toggle("Run line 1 code"):
    response = OpenAI().complete("Michael Jackson is ")
    st.write(response)

st.text("Line 2 : Prompt Template for response customization")

### Line 2 ###
st.code("""template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
qa_template = PromptTemplate(template)
    prompt = qa_template.format(context_str="You are a loyal Michael Jackson fan", 
        query_str="Michael Jackson is ")
    st.text(prompt)
    st.write(OpenAI().complete(prompt))
    prompt = qa_template.format(context_str="You are not a fan of Michael Jackson", 
        query_str="Michael Jackson is ")
    st.text(prompt)
    st.write(OpenAI().complete(prompt))
           """)

if st.toggle("Run line 2 code"):
    template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
    qa_template = PromptTemplate(template)
    prompt = qa_template.format(context_str="You are a loyal Michael Jackson fan", query_str="Michael Jackson is ")
    st.text(prompt)
    st.write(OpenAI().complete(prompt))
    prompt = qa_template.format(context_str="You are not a fan of Michael Jackson", query_str="Michael Jackson is ")
    st.text(prompt)
    st.write(OpenAI().complete(prompt))

### Line 3 ###  
st.text("Line 3 : LLM Settings and simple Data Loading and Indexing")
st.code("""
        \nfrom llama_index.llms.openai import OpenAI\nfrom llama_index.core import Settings \nfrom llama_index.core import VectorStoreIndex, SimpleDirectoryReader 
        \nSettings.llm = OpenAI(temperature=0.2, model='gpt-3.5-turbo') 
        \ndocuments = SimpleDirectoryReader('data').load_data() \nindex = VectorStoreIndex.from_documents( documents,)
        \nst.write(documents)
        \nst.write(index)""")

if st.toggle("Run line 3 code"):
    Settings.llm = OpenAI(temperature=0.2, model='gpt-3.5-turbo') 
    documents = SimpleDirectoryReader('data').load_data() 
    index = VectorStoreIndex.from_documents( documents,)
    st.write(documents)
    st.write(index)

### Line 4 ###
st.text("Line 4 : Storing and Rebuilding Indexed Data")
st.code(""" 

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings 
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage
        
Settings.llm = OpenAI(temperature=0.2, model='gpt-3.5-turbo') 
    documents = SimpleDirectoryReader('data').load_data() 
    index = VectorStoreIndex.from_documents( documents,)
    index.storage_context.persist(persist_dir="Index")
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="Index")
    # load index
    index = load_index_from_storage(storage_context)
    st.write(index)

""")
if st.toggle("Run line 4 code"):

    Settings.llm = OpenAI(temperature=0.2, model='gpt-3.5-turbo') 
    documents = SimpleDirectoryReader('data').load_data() 
    index = VectorStoreIndex.from_documents( documents,)
    index.storage_context.persist(persist_dir="Index")
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="Index")
    # load index
    index = load_index_from_storage(storage_context)
    st.write(index)

### Line 5 ###
st.text("Line 5 : Querying the Created, Stored and Rebuilt Index")
st.code(""" 
    from llama_index.llms.openai import OpenAI
    from llama_index.core import Settings 
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core import StorageContext, load_index_from_storage
    
    Settings.llm = OpenAI(temperature=0.2, model='gpt-3.5-turbo') 
    storage_context = StorageContext.from_defaults(persist_dir="Index")
    # load index
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is the document about")
    st.write(response)

""")

if st.toggle("Run line 5 code"):
    Settings.llm = OpenAI(temperature=0.2, model='gpt-3.5-turbo') 
    storage_context = StorageContext.from_defaults(persist_dir="Index")
    # load index
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is document about?")
    st.write(response)



### Line 6 ###
st.text("Line 6 : Customized Querying Steps")
st.code(""" 
        
    from llama_index.core import VectorStoreIndex, get_response_synthesizer
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.postprocessor import SimilarityPostprocessor
        
    Settings.llm = OpenAI(temperature=0.2, model='gpt-3.5-turbo') 
    storage_context = StorageContext.from_defaults(persist_dir="Index")
    # load index
    index = load_index_from_storage(storage_context)
    # configure retriever
    retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
    )
    # configure response synthesizer
    response_synthesizer = get_response_synthesizer()

    query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    )
    
    response = query_engine.query("What is the document about")
    st.write(response)
""")
        
if st.toggle("Run line 6 code"):
    Settings.llm = OpenAI(temperature=0.2, model='gpt-3.5-turbo') 
    storage_context = StorageContext.from_defaults(persist_dir="Index")
    # load index
    index = load_index_from_storage(storage_context)
    # configure retriever
    retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
    )
    # configure response synthesizer
    response_synthesizer = get_response_synthesizer()

    # assemble query engine
    query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    )

    # query
    response = query_engine.query("What is the document about")
    st.write(response)

### Line 7 ###
st.text("Line 7 : Llama Parse of a Tuition Fees Tax Slip with Tables")
st.code(""" 
document = LlamaParse(result_type="markdown").load_data(".\data\T2202.pdf")
    index = VectorStoreIndex.from_documents(document)
    query_engine = index.as_query_engine()

    response = query_engine.query(
        "How much are the tuition fees?"
    )
    st.text(response)
""")
if st.toggle("Run line 7 code"):
    document = LlamaParse(result_type="markdown").load_data(".\data\T2202.pdf")
    index = VectorStoreIndex.from_documents(document)
    query_engine = index.as_query_engine()

    response = query_engine.query(
        "How much are the tuition fees?"
    )
    st.write(document)
    st.text(response)