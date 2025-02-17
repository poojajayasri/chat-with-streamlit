from __future__ import annotations
import re
from typing import Optional, Tuple, List, Union, Literal, Any, Dict
import base64
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
import os
import openai
import graphviz
from dataclasses import dataclass, asdict
from textwrap import dedent
from streamlit_agraph import agraph, Node, Edge, Config
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from streamlit_option_menu import option_menu
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredImageLoader,
    SeleniumURLLoader
)
from langchain.chains import ConversationalRetrievalChain
from PIL import Image
import mimetypes
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')
from langchain_community.document_loaders import CSVLoader
from langchain.llms import OpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
import streamlit.components.v1 as components
import re
import base64
import time
from io import BytesIO
import requests
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from pypdf import PdfReader
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from pathlib import Path

# Enhanced Flowchart and Mindmap Styles
FLOWCHART_STYLES = {
    'graph_attrs': {
        'rankdir': 'TB',
        'splines': 'ortho',
        'nodesep': '0.8',
        'ranksep': '1.0',
        'bgcolor': 'white',
        'concentrate': 'true'
    },
    'node_attrs': {
        'style': 'filled,rounded',
        'shape': 'box',
        'fillcolor': '#E8F0FE',
        'color': '#4285F4',
        'fontname': 'Arial',
        'fontsize': '12',
        'penwidth': '2',
        'margin': '0.3,0.2'
    },
    'edge_attrs': {
        'color': '#4285F4',
        'penwidth': '2',
        'arrowhead': 'vee',
        'arrowsize': '1.2',
        'fontname': 'Arial',
        'fontsize': '10',
        'fontcolor': '#666666'
    }
}

MINDMAP_STYLES = {
    'graph_attrs': {
        'rankdir': 'LR',
        'splines': 'spline',
        'overlap': 'false',
        'sep': '+25',
        'bgcolor': 'white',
        'concentrate': 'true'
    },
    'node_attrs': {
        'style': 'filled',
        'shape': 'ellipse',
        'fillcolor': '#FFF3E0',
        'color': '#FF9800',
        'fontname': 'Arial',
        'fontsize': '12',
        'penwidth': '2'
    },
    'edge_attrs': {
        'color': '#FF9800',
        'penwidth': '2',
        'arrowhead': 'none',
        'len': '2.0'
    },
    'central_node_attrs': {
        'shape': 'doubleoctagon',
        'style': 'filled',
        'fillcolor': '#FFA726',
        'color': '#E65100',
        'penwidth': '3',
        'fontsize': '14',
        'fontname': 'Arial Bold'
    }
}

st.set_page_config(page_title="DataMap AI", page_icon="frilogo.png", layout="wide", initial_sidebar_state="collapsed")

# --- PATH SETTINGS ---
THIS_DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
ASSETS_DIR = THIS_DIR / "assets"
STYLES_DIR = THIS_DIR / "styles"
CSS_FILE = STYLES_DIR / "main.css"

def load_css_file(css_file_path):
    with open(css_file_path) as f:
        return st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css_file(CSS_FILE)

# Initialize session states
if 'tab_selection' not in st.session_state:
    st.session_state.tab_selection = " "
if 'session_button' not in st.session_state:
    st.session_state.session_button = [False,False,False,False,False,False]
if 'video_url' not in st.session_state:
    st.session_state.video_url = ""
if 'web_url' not in st.session_state:
    st.session_state.web_url = "jsmastery.pro"
if 'book' not in st.session_state:
    st.session_state.book = " "
if 'api' not in st.session_state:
    st.session_state.api = " "
if 'agree' not in st.session_state:
    st.session_state.agree = " "
if 'pages' not in st.session_state:
    st.session_state.agree = []
if 'filenames' not in st.session_state:
    st.session_state.filenames = []
if 'pages' not in st.session_state:
    st.session_state.pages = []

COLOR = "#F7A7A6"
FOCUS_COLOR = "#84b3d1"

def enhance_flowchart_visualization(dot, content):
    """Enhance flowchart visualization with better styling"""
    # Parse input content
    graph_def = f'''digraph{{{content}}}'''
    
    # Create enhanced Graphviz graph
    enhanced_dot = graphviz.Digraph()
    enhanced_dot.attr(**FLOWCHART_STYLES['graph_attrs'])
    enhanced_dot.attr('node', **FLOWCHART_STYLES['node_attrs'])
    enhanced_dot.attr('edge', **FLOWCHART_STYLES['edge_attrs'])
    
    # Parse and add nodes/edges while maintaining the original structure
    current_dot = graphviz.Source(graph_def)
    enhanced_dot.body.extend(current_dot.body)
    
    return enhanced_dot

def create_mindmap(central_topic, topics, connections):
    """Create an enhanced mindmap visualization"""
    dot = graphviz.Graph(comment='Mind Map')
    dot.attr(**MINDMAP_STYLES['graph_attrs'])
    
    # Add central topic with special styling
    dot.node(central_topic, **MINDMAP_STYLES['central_node_attrs'])
    
    # Add other topics
    for topic in topics:
        dot.node(topic, **MINDMAP_STYLES['node_attrs'])
    
    # Add connections with enhanced styling
    for source, target in connections:
        dot.edge(source, target, **MINDMAP_STYLES['edge_attrs'])
    
    return dot

def process_mindmap_query(query: str, api_key: str) -> Tuple[str, List[str], List[Tuple[str, str]]]:
    """Process natural language query to extract mindmap structure"""
    prompt = f"""Generate a comprehensive mindmap structure for: {query}
    
    Respond in exactly three lines:
    1. Central topic (single main concept)
    2. Related topics (8-10 key concepts, comma-separated)
    3. Connection pairs as 'source:target' (logical relationships, comma-separated)
    
    Make the structure clear and cohesive.
    """
    
    llm = OpenAI(temperature=0.7, api_key=api_key)
    response = llm(prompt)
    
    # Parse response
    lines = response.strip().split('\n')
    central_topic = lines[0].strip()
    topics = [t.strip() for t in lines[1].split(',')]
    connections = [tuple(c.strip().split(':')) for c in lines[2].split(',')]
    
    return central_topic, topics, connections

def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output

def text_to_docs(text: str) -> List[Document]:
    """Converts text content to a list of Documents with metadata"""
    if isinstance(text, str):
        text = [text]
    
    page_docs = [Document(page_content=str(page)) for page in text]
    
    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1
    
    # Split pages into chunks
    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, 
                metadata={"page": doc.metadata["page"], "chunk": i}
            )
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    
    return doc_chunks

def test_embed():
    """Create and test embeddings index"""
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.api)
    if 'index' not in st.session_state:
        st.session_state.index = FAISS.from_documents(
            [Document(page_content='hi', metadata={'source': 'one).pdf', 'page': 1})], 
            embeddings
        )
    
    with st.spinner("Loading..."):
        index = FAISS.from_documents(st.session_state.pages, embeddings)
    st.header("Successfully uploaded")
    return index

def test_embed2():
    """Alternative embedding test"""
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.api)
    with st.spinner("Loading..."):
        index = FAISS.from_documents(st.session_state.pages, embeddings)
    st.success("Successfully loaded!", icon="✅")
    return index

# Main page sidebar
main_page_sidebar = st.empty()
with main_page_sidebar:
    st.session_state.tab_selection = option_menu(
        menu_title = '',
        menu_icon = 'list-columns-reverse',
        options = ['Home','Add Key','Upload files','Begin Chat','---','---','---','---','Contact us'],
        icons = ['house-door-fill','key-fill','database-fill-add','chat-fill','','','','','send-plus-fill','box-arrow-in-right'],
        orientation="horizontal",
        styles = {
            "container": {"background-color": "#403f3f", "height": "10% !important" ,"width": "100%", "padding": "1px", "border-radius": "30px"},
            "nav-link": {"padding": "5px", "height": "100%" ,"width": "100%" ,"font-weight": "bold","font-family": "Arial", "font-size": "65%", "color": "white", "text-align": "left", "margin":"0px",
                        "--hover-opacity": "0%","text-align": "centre"},
            "separator": {"opacity": "0% !important"},
            "nav-link-selected": {
                "padding": "5px",
                "opacity":"100%",
                "background": "#403f3f",
                "font-family": "Arial",
                "font-weight": "bold",
                "font-size": "75%",
                "color": "white",
                "margin": "0px",
                "position": "relative",
                "overflow": "hidden",
                "border-radius": "30px",
                "text-decoration": "underline",
                "font-weight": "bolder",
                "text-align": "centre"
            },
        }
    )

transformed_list = []

if st.session_state.tab_selection == "Upload files":
    try:
        if len(st.session_state.api) > 1:
            if not st.session_state.pages:
                embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.api)
                st.header("Supported: PDF, Docx, PNG, CSV, Youtube & Website URLs")
                
                with st.expander("Upload Files"):
                    urlinput = st.text_input("Youtube Links", placeholder="Enter Youtube links separated by spaces")
                    urlinput1 = st.text_input("Website Link", placeholder="jsmastery.pro", help='URL format: jsmastery.pro')
                    uploaded_file = st.file_uploader(
                        "**Supported Formats: PDF, Docx, PNG (Extracts text from image)**",
                        accept_multiple_files=True,
                        help='Uploaded files will not be stored',
                        type=['png', 'pdf', 'csv', 'docx']
                    )
                    
                    for file in uploaded_file:
                        st.session_state.filenames.append(file.name)
                
                pages = []
                datata = []
                
                if urlinput:
                    string_elements = urlinput.split(" ")
                    transformed_list = [
                        element for element in string_elements 
                        if element.startswith("www.") or element.startswith("https://")
                    ]
                
                if uploaded_file or transformed_list or urlinput1:
                    for file in uploaded_file:
                        if file.name.lower().endswith('.docx'):
                            xy = os.getcwd()
                            upload_dir = f"{xy}/uploads/"
                            os.makedirs(upload_dir, exist_ok=True)
                            file_path = os.path.join(upload_dir, file.name)
                            
                            with open(file_path, "wb") as filee:
                                filee.write(file.read())
                            
                            loader = Docx2txtLoader(f"{file_path}")
                            data = loader.load()
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
                            docss = text_splitter.split_documents(data)
                            pages.extend(docss)
                            
                        elif file.name.lower().endswith('.pdf'):
                            xy = os.getcwd()
                            upload_dir = f"{xy}/uploads/"
                            os.makedirs(upload_dir, exist_ok=True)
                            file_path = os.path.join(upload_dir, file.name)
                            
                            with open(file_path, "wb") as filee:
                                filee.write(file.read())
                            
                            loader = PyPDFLoader(f"{file_path}")
                            data = loader.load()
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                            docss = text_splitter.split_documents(data)
                            pages.extend(docss)
                            
                        elif file.name.lower().endswith('.csv'):
                            xy = os.getcwd()
                            upload_dir = f"{xy}/uploads/"
                            os.makedirs(upload_dir, exist_ok=True)
                            file_path = os.path.join(upload_dir, file.name)
                            
                            with open(file_path, "wb") as filee:
                                filee.write(file.read())
                            
                            loader = CSVLoader(f"{file_path}")
                            data = loader.load()
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                            docss = text_splitter.split_documents(data)
                            pages.extend(docss)
                            
                        elif file.name.lower().endswith('.png') or file.name.lower().endswith('.jpeg'):
                            xy = os.getcwd()
                            upload_dir = f"{xy}/uploads/"
                            os.makedirs(upload_dir, exist_ok=True)
                            file_path = os.path.join(upload_dir, file.name)
                            
                            with open(file_path, "wb") as filee:
                                filee.write(file.read())
                            
                            loader = UnstructuredImageLoader(f"{file_path}")
                            data = loader.load()
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                            docss = text_splitter.split_documents(data)
                            pages.extend(docss)
                    
                    if transformed_list:
                        urls = transformed_list
                        for yo in urls:
                            loader = YoutubeLoader.from_youtube_url(yo)
                            transcript = loader.load()
                            datata.extend(transcript)
                        
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        docss = text_splitter.split_documents(datata)
                        pages.extend(docss)
                        
                    if urlinput1:
                        st.session_state.web_url = urlinput1
                        
                        try:
                            HTTP_URL_PATTERN = r'^http[s]*://.+'
                            domain = st.session_state.web_url
                            full_url = "https://www." + st.session_state.web_url
                            
                            class HyperlinkParser(HTMLParser):
                                def __init__(self):
                                    super().__init__()
                                    self.hyperlinks = []
                                
                                def handle_starttag(self, tag, attrs):
                                    attrs = dict(attrs)
                                    if tag == "a" and "href" in attrs:
                                        self.hyperlinks.append(attrs["href"])
                            
                            def get_hyperlinks(url):
                                try:
                                    with urllib.request.urlopen(url) as response:
                                        if not response.info().get('Content-Type').startswith("text/html"):
                                            return []
                                        html = response.read().decode('utf-8')
                                except Exception as e:
                                    print(e)
                                    return []
                                
                                parser = HyperlinkParser()
                                parser.feed(html)
                                return parser.hyperlinks
                            
                            def get_domain_hyperlinks(local_domain, url):
                                clean_links = []
                                for link in set(get_hyperlinks(url)):
                                    clean_link = None
                                    
                                    if re.search(HTTP_URL_PATTERN, link):
                                        url_obj = urlparse(link)
                                        if url_obj.netloc == local_domain:
                                            clean_link = link
                                    else:
                                        if link.startswith("/"):
                                            link = link[1:]
                                        elif link.startswith("#") or link.startswith("mailto:"):
                                            continue
                                        clean_link = "https://" + local_domain + "/" + link
                                    
                                    if clean_link is not None:
                                        if clean_link.endswith("/"):
                                            clean_link = clean_link[:-1]
                                        clean_links.append(clean_link)
                                
                                return list(set(clean_links))
                            
                            def crawl(url):
                                local_domain = urlparse(url).netloc
                                queue = deque([url])
                                seen = set([url])
                                
                                if not os.path.exists("text/"):
                                    os.mkdir("text/")
                                if not os.path.exists("text/"+local_domain+"/"):
                                    os.mkdir("text/" + local_domain + "/")
                                if not os.path.exists("processed"):
                                    os.mkdir("processed")
                                
                                while queue:
                                    url = queue.pop()
                                    print(url)
                                    
                                    with open('text/'+local_domain+'/'+url[8:].replace("/", "_") + ".txt", "w") as f:
                                        soup = BeautifulSoup(requests.get(url).text, "html.parser")
                                        text = soup.get_text()
                                        print(text)
                                        
                                        if ("You need to enable JavaScript to run this app." in text):
                                            print("Unable to parse page " + url + " due to JavaScript being required")
                                        
                                        print(f)
                                        f.write(text)
                                    
                                    for link in get_domain_hyperlinks(local_domain, url):
                                        if link not in seen:
                                            queue.append(link)
                                            seen.add(link)
                            
                            crawl(full_url)
                            
                            def remove_newlines(serie):
                                serie = serie.str.replace('\n', ' ')
                                serie = serie.str.replace('\\n', ' ')
                                serie = serie.str.replace('  ', ' ')
                                serie = serie.str.replace('  ', ' ')
                                return serie
                            
                            texts = []
                            for file in os.listdir("text/www." + domain + "/"):
                                with open("text/www." + domain + "/" + file, "r") as f:
                                    text = f.read()
                                    print(text)
                                    texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))
                            print(texts)
                            
                            docococ = text_to_docs(texts)
                            pages.extend(docococ)
                            
                        except Exception as e:
                            st.info("Please enter a valid link")
                            print(e)
                
                st.session_state.agree = st.button('Upload')
                if st.session_state.agree:
                    try:
                        st.write(pages)
                        st.session_state.index = test_embed()
                        st.session_state.pages = pages
                    except Exception as e:
                        st.info("Loading Unsuccessful")
                        print(e)
            else:
                st.header("Successfully uploaded")
                editbtn = st.button("Restart")
                if editbtn:
                    st.session_state.pages = []
                    st.session_state.filenames = []
        else:
            st.header("Enter Your API Key to continue")
            st.markdown("[OpenAI API Key](%s)" % "https://platform.openai.com/account/api-keys")
    except Exception as e:
        st.info("Invalid API key/files")

if st.session_state.tab_selection == "Contact us":
    with st.container():
        contact_form = f"""<form action="https://formsubmit.co/zikkijapo@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value = "false">
            <div class="thisempty"><p></p><div>
            <input type="text" name="name" placeholder="Name" required>
            <div class="thisempty"><p></p><div>
            <input type="text" name="Subject" placeholder="Subject" required>
            <div class="thisempty"><p></p><div>
            <input type="email" name="email" placeholder="Email" required>
            <div class="thisempty"><p></p><div>
            <textarea name="message" placeholder="Enter your message" required></textarea>
            <button type="submit">Send</button>
        </form>"""
        one, two, three = st.columns([5,10,5])
        with two:
            st.header("Send us your feedback")
            st.markdown(contact_form,unsafe_allow_html=True)

if st.session_state.tab_selection == "Begin Chat":
    st.session_state.option = st.selectbox(
        '',
        ('General Query: Flowchart', 'General Query: Mindmap', 'Uploaded data: Flowchart', 'Uploaded data: Quick Query', 'Uploaded data: Conversational Chat'),
        label_visibility="collapsed"
    )
    
    if st.session_state.api:
        try:
            if st.session_state.agree:
                upfiles = ", ".join(st.session_state.filenames)
                st.caption(f"Uploaded files: {upfiles}")
                
                qa1 = RetrievalQA.from_chain_type(
                    llm = OpenAI(openai_api_key=st.session_state.api),
                    chain_type = "map_reduce",
                    retriever=st.session_state.index.as_retriever(),
                )
                
                tools5 = [Tool(
                    name="QuestionsAnswers",
                    func=qa1.run,
                    description="Useful for when you need to answer questions about the things asked. Input may be a partial or fully formed question.",
                )]
                
                prefix1 = """provide answers using only the tool provided. Answer in detail"""
                suffix1 = """Begin!"

                {chat_history1}
                Question: {input1}
                {agent_scratchpad}"""
                
                prompt1 = ZeroShotAgent.create_prompt(
                    tools5,
                    prefix=prefix1,
                    suffix=suffix1,
                    input_variables=["input1", "chat_history1", "agent_scratchpad"],
                )
                
                if "memory1" not in st.session_state:
                    st.session_state.memory1 = ConversationBufferMemory(
                        memory_key="chat_history1"
                    )
                
                llm_chain1 = LLMChain(
                    llm = OpenAI(openai_api_key=st.session_state.api),
                    prompt=prompt1,
                )
                agent1 = ZeroShotAgent(llm_chain=llm_chain1, tools=tools5, verbose=True)
                agent_chain1 = AgentExecutor.from_agent_and_tools(
                    agent=agent1,
                    tools=tools5,
                    verbose=True,
                    memory=st.session_state.memory1
                )
                
                try:
                    @dataclass
                    class Message:
                        """Class for keeping track of a chat message."""
                        origin: Literal["human", "ai"]
                        message: str
                    
                    def initialize_session_state():
                        if "history" not in st.session_state:
                            st.session_state.history = []
                        if "option" not in st.session_state:
                            st.session_state.option = ""
                        if "option1" not in st.session_state:
                            st.session_state.option1 = ""
                        if "history1" not in st.session_state:
                            st.session_state.history1 = []
                        if "token_count" not in st.session_state:
                            st.session_state.token_count = 0
                        if "token_count1" not in st.session_state:
                            st.session_state.token_count1 = 0
                        try:
                            if "conversation" not in st.session_state:
                                llm = OpenAI(
                                    temperature=0,
                                    openai_api_key=st.session_state.api,
                                    model_name="gpt-4o-mini"
                                )
                                st.session_state.conversation = qa1
                            if "conversation1" not in st.session_state:
                                llm = OpenAI(
                                    temperature=0,
                                    openai_api_key=st.session_state.api,
                                    model_name="gpt-4o-mini"
                                )
                                st.session_state.conversation1 = agent_chain1
                        except Exception as e:
                            st.write("")
                    
                    def on_click_callback():
                        with get_openai_callback() as cb:
                            human_prompt = st.session_state.human_prompt
                            print("hi3")
                            if len(human_prompt) > 5:
                                try:
                                    diag = """step1. rewrite the result as a directed graph  \n
                                    step2: make sure the result is exactly in the format as provided in the sample. 
                                    step 3: provide No other explanations. max no. of nodes: 10. sample format(take into consideration only the structure and format of the sample, not the actual words. make sure to include the quotes as shown in the sample. do not display the sample). 
                                    Sample: 
                                                "run" -> "intr"
                                                "intr" -> "runbl intr"
                                                "runbl" -> "run"
                                                "run" -> "kernel"
                                                "kernel" -> "zombie"
                                                "kernel" -> "sleep"
                                                "kernel" -> "runmem"
                                                """
                                    total = "the Question is:" + human_prompt + "\n answer the question and present the result it in this format:" + diag
                                    print(total)
                                    
                                    if st.session_state.option.strip() == 'Uploaded data: Flowchart':
                                        try:
                                            gen_response = st.session_state.conversation.run(
                                                "Answer in detail" + human_prompt 
                                            )
                                            print("llm response is 1" + gen_response)
                                            
                                            llm = OpenAI(openai_api_key=st.session_state.api)
                                            llm_response = llm(gen_response + diag)
                                            
                                            # Use enhanced flowchart visualization
                                            dot = enhance_flowchart_visualization(graphviz.Digraph(), llm_response)
                                            llm_response1 = dot.source + " diagramcreator"
                                            
                                        except Exception as e:
                                            llm_response1 = "Please Upload your files and try again"
                                            
                                    elif st.session_state.option.strip() == 'Uploaded data: Quick Query':
                                        try:
                                            llm_response = st.session_state.conversation.run(
                                                "Answer in detail" + human_prompt 
                                            )
                                            print("llm response is 1" + llm_response)
                                            llm_response1 = llm_response
                                            
                                        except Exception as e:
                                            llm_response1 = "Please Upload your files and try again"
                                            
                                    elif st.session_state.option.strip() == 'General Query: Flowchart':
                                        try:
                                            llm = OpenAI(openai_api_key=st.session_state.api)
                                            gen_response = llm(human_prompt + "   \n answer in detail")
                                            if len(gen_response) > 10:
                                                print(f"DETAILED{gen_response}")
                                                llm_response = llm(gen_response + diag)
                                                print(f"DETAILED{llm_response}")
                                                
                                                # Use enhanced flowchart visualization
                                                dot = enhance_flowchart_visualization(graphviz.Digraph(), llm_response)
                                                llm_response1 = dot.source + " diagramcreator"
                                                
                                        except Exception as e:
                                            st.write(e)
                                            llm_response1 = "Please enter a valid key and try again"
                                            
                                    elif st.session_state.option.strip() == 'General Query: Mindmap':
                                        try:
                                            central_topic, topics, connections = process_mindmap_query(human_prompt, st.session_state.api)
                                            mindmap = create_mindmap(central_topic, topics, connections)
                                            llm_response1 = mindmap.source + " mindmapcreator"
                                            gen_response = f"Created mindmap with central topic: {central_topic}"
                                            
                                        except Exception as e:
                                            llm_response1 = "Error creating mindmap. Please try again."
                                            gen_response = ""
                                except Exception as e:
                                    print(e)
                                            
                            else:
                                    human_prompt = "Please enter a valid query"
                                    
                                try:
                                    st.session_state.history.append(
                                        Message("human", human_prompt)
                                    )
                                    st.session_state.history.append(
                                        Message("ai", llm_response1)
                                    )
                                    if (st.session_state.option.strip() == 'General Query: Flowchart' or 
                                        st.session_state.option.strip() == 'Uploaded data: Flowchart' or
                                        st.session_state.option.strip() == 'General Query: Mindmap'):
                                        st.session_state.history.append(
                                            Message("ai", gen_response)
                                        )
                                    st.session_state.token_count += cb.total_tokens
                                except Exception as e:
                                    print(e)
                    
                    def on_click_callback1():
                        with get_openai_callback() as cb:
                            human_prompt = st.session_state.human_prompt1
                            if len(human_prompt) > 5:
                                try:
                                    llm_response = st.session_state.conversation1.run(
                                        human_prompt 
                                    )
                                except Exception as e:
                                    print(e)
                                    llm_response = "Please Upload your files and try again"
                            else:
                                human_prompt = "Please enter a valid query"
                            
                            try:
                                st.session_state.history1.append(
                                    Message("human", human_prompt)
                                )
                                st.session_state.history1.append(
                                    Message("ai", llm_response)
                                )
                                st.session_state.token_count1 += cb.total_tokens
                            except Exception as e:
                                print(e)
                    
                    initialize_session_state()
                    
                    chat_placeholder = st.container()
                    prompt_placeholder = st.form("chat-form")
                    credit_card_placeholder = st.empty()
                    chat_placeholder1 = st.container()
                    prompt_placeholder1 = st.form("chat-form1")
                    credit_card_placeholder1 = st.empty()
                    
                    if (st.session_state.option == 'Uploaded data: Quick Query' or 
                        st.session_state.option == 'Uploaded data: Flowchart' or 
                        st.session_state.option == 'General Query: Flowchart' or
                        st.session_state.option == 'General Query: Mindmap'):
                        
                        with chat_placeholder:
                            for chat in st.session_state.history:
                                words = chat.message.split()
                                last_word = words[-1] if words else ""
                                last_word = last_word.strip()
                                message_before_last_word = ' '.join(words[:-1]) if words else ""
                                
                                print(f"last word is {last_word}")
                                print(f"rest of it is {message_before_last_word}")
                                
                                st.session_state.isdiagram = (last_word == "diagramcreator" or last_word == "mindmapcreator")
                                
                                if chat.origin == 'ai' and st.session_state.isdiagram:
                                    content = st.graphviz_chart(message_before_last_word)
                                    content = ""
                                else:
                                    content = chat.message
                                
                                if content:
                                    div = f"""
                                    <div class="chat-row 
                                        {'' if chat.origin == 'ai' else 'row-reverse'}">
                                        <img class="chat-icon" src="{
                                            'https://cdn.discordapp.com/attachments/852337726904598574/1126648713788526722/ai.png' if chat.origin == 'ai' 
                                            else 'https://cdn.discordapp.com/attachments/852337726904598574/1126648675238682655/human.png'}"
                                            width=32 height=32>
                                        <div class="chat-bubble
                                        {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
                                            &#8203;{content}
                                        </div>
                                    </div>
                                    """
                                    st.markdown(div, unsafe_allow_html=True)
                            
                            for _ in range(3):
                                st.markdown("")
                            st.write("""<div class='PortMarker'/>""", unsafe_allow_html=True)
                        
                        with prompt_placeholder:
                            cols = st.columns((6, 1))
                            cols[0].text_input(
                                "Chat",
                                value=" ",
                                label_visibility="collapsed",
                                key="human_prompt",
                            )
                            cols[1].form_submit_button(
                                "Submit", 
                                type="primary", 
                                on_click=on_click_callback,
                            )
                            
                    elif st.session_state.option == 'Uploaded data: Conversational Chat':
                        with chat_placeholder1:
                            for chat in st.session_state.history1:
                                div = f"""
                                <div class="chat-row 
                                    {'' if chat.origin == 'ai' else 'row-reverse'}">
                                    <img class="chat-icon" src="{
                                        'https://cdn.discordapp.com/attachments/852337726904598574/1126648713788526722/ai.png' if chat.origin == 'ai' 
                                        else 'https://cdn.discordapp.com/attachments/852337726904598574/1126648675238682655/human.png'}"
                                        width=32 height=32>
                                    <div class="chat-bubble
                                    {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
                                        &#8203;{chat.message}
                                    </div>
                                </div>
                                """
                                st.markdown(div, unsafe_allow_html=True)
                            
                            for _ in range(3):
                                st.markdown("")
                            st.write("""<div class='PortMarker'/>""", unsafe_allow_html=True)
                        
                        with prompt_placeholder1:
                            cols = st.columns((6, 1))
                            cols[0].text_input(
                                "Chat",
                                value=" ",
                                label_visibility="collapsed",
                                key="human_prompt1",
                            )
                            cols[1].form_submit_button(
                                "Submit", 
                                type="primary", 
                                on_click=on_click_callback1,
                            )
                            
                except Exception as e:
                    st.header("Upload your files to begin")
                    st.subheader("Check your OpenAI Plan")
                    st.write(e)
                    
        except Exception as e:
            print(e)
            st.info("Add key and upload files/URL to continue")
    else:
        st.header("Enter Your API Key to continue")

if st.session_state.tab_selection == "Add Key":
    if len(st.session_state.api) > 1:
        st.header("API key loaded")
        if st.button("Edit Key"):
            st.session_state.api = ""
            if not len(st.session_state.api) < 2:
                st.rerun()
    else:
        st.header("Load Your API Key")
        st.markdown("[OpenAI API Key](%s)" % "https://platform.openai.com/account/api-keys")
        st.session_state.api = st.text_input("API Key ", placeholder="Enter your API Key", type='password', help='API Keys will not be stored')
        if st.session_state.api:
            st.rerun()

if st.session_state.tab_selection == "Home":
    co1, co2, co3 = st.columns([5,8,5])
    with co2:
        st.markdown('<div style="justify-content: center; text-align: center;"><span style="font-size: 2.5em; font-family: Helvetica Neue; font: 500; font-style: bolder; font-weight: 600; justify-content: center; text-align: center; border-radius: 30px; padding: 10px;"><span style="color: white; padding: 5px;"></span><img class="chat-icon" src="https://cdn.discordapp.com/attachments/852337726904598574/1126682090101035019/frilogo11.png" width=54 height=54 style="border-radius: 50px;border: 1px solid #515389; padding: 5px;"></img></span></span></div>', unsafe_allow_html=True)
        st.markdown('<div style="justify-content: center; text-align: center;"><span style="font-size: 3.3em; font-family: Helvetica Neue; font: 500; font-style: bolder; font-weight: 600; justify-content: center; text-align: center; border-radius: 30px; padding: 10px;">Deep Dive into your Data,<span style="color: #cc003d;font-family: Brush Script MT;"> Visually</span>.</span><span style="color: #898989; font-size: 1.5em; font-family: Helvetica Neue; font: 500; font-style: bolder; font-weight: 400; justify-content: center; text-align: center; border-radius: 30px; padding: 5px;"></span></div>', unsafe_allow_html=True)
        st.markdown('<div style="justify-content: center; text-align: center;"><span style="color: #a1a0a0; font-size: 1em; font-family: Helvetica Neue; font: 500; font-style: bolder; font-weight: 400; justify-content: center; text-align: center; border-radius: 30px; padding: 5px;">Generate Interactive Mindmaps and Dynamic Flowcharts with <span style="font-style: bolder;font-weight: 600;">Datamap AI.</span></span></div>', unsafe_allow_html=True)
        st.write(" ")

# Page styling
page_bg_img = """
<style>
@keyframes glowing {
    0% { background-position: 0 0; }
    50% { background-position: 400% 0; }
    100% { background-position: 0 0; }
}
[data-testid="stAppViewContainer"] > .main {
    background-size: 180%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
}

[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    background-position: left; 
    background-repeat: no-repeat;
    background-attachment: fixed;
    min-width: 100px;
}

[data-testid="stHeader"] {
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
