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

@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str

def enhance_flowchart_visualization(dot, content):
    """Enhance flowchart visualization with better styling"""
    graph_def = f'''digraph{{{content}}}'''
    enhanced_dot = graphviz.Digraph()
    enhanced_dot.attr(rankdir='TB', 
                     splines='ortho',
                     nodesep='0.8',
                     ranksep='1.0',
                     bgcolor='white')
    enhanced_dot.attr('node', 
                     style='filled,rounded',
                     shape='box',
                     fillcolor='#E8F0FE',
                     color='#4285F4',
                     fontname='Arial',
                     fontsize='12',
                     penwidth='2')
    enhanced_dot.attr('edge',
                     color='#4285F4',
                     penwidth='2',
                     arrowhead='vee',
                     fontname='Arial',
                     fontsize='10')
    current_dot = graphviz.Source(graph_def)
    enhanced_dot.body.extend(current_dot.body)
    return enhanced_dot

def create_mindmap(central_topic, topics, connections):
    """Create an enhanced mindmap visualization"""
    dot = graphviz.Graph(comment='Mind Map')
    dot.attr(rankdir='LR',
            splines='spline',
            overlap='false',
            sep='+25',
            bgcolor='white')
    
    # Add central topic with special styling
    dot.node(central_topic,
            shape='doubleoctagon',
            style='filled',
            fillcolor='#FFA726',
            color='#E65100',
            penwidth='3',
            fontsize='14',
            fontname='Arial Bold')
    
    # Add other topics
    for topic in topics:
        dot.node(topic,
                style='filled',
                shape='ellipse',
                fillcolor='#FFF3E0',
                color='#FF9800',
                fontname='Arial',
                fontsize='12',
                penwidth='2')
    
    # Add connections
    for source, target in connections:
        dot.edge(source, target,
                color='#FF9800',
                penwidth='2',
                len='2.0')
    
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
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output

def text_to_docs(text: str) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    
    page_docs = [Document(page_content=str(page)) for page in text]
    
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1
    
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

def get_diagram_prompt():
    return """
    step1. rewrite the result as a directed graph
    step2: make sure the result is exactly in the format as provided in the sample.
    step3: provide No other explanations. max no. of nodes: 10. sample format
    (take into consideration only the structure and format of the sample,
    not the actual words. make sure to include the quotes as shown in the sample.
    do not display the sample).
    Sample:
        "run" -> "intr"
        "intr" -> "runbl intr"
        "runbl" -> "run"
        "run" -> "kernel"
        "kernel" -> "zombie"
        "kernel" -> "sleep"
        "kernel" -> "runmem"
    """

def process_message(message):
    with get_openai_callback() as cb:
        try:
            if len(message) > 5:
                if st.session_state.option.strip() == 'Uploaded data: Flowchart':
                    gen_response = st.session_state.conversation.run(
                        "Answer in detail: " + message
                    )
                    llm = OpenAI(openai_api_key=st.session_state.api)
                    llm_response = llm(gen_response + get_diagram_prompt())
                    final_response = enhance_flowchart_visualization(graphviz.Digraph(), llm_response).source + " diagramcreator"
                
                elif st.session_state.option.strip() == 'Uploaded data: Quick Query':
                    final_response = st.session_state.conversation.run(
                        "Answer in detail: " + message
                    )
                
                elif st.session_state.option.strip() == 'General Query: Flowchart':
                    llm = OpenAI(openai_api_key=st.session_state.api)
                    gen_response = llm(message + " \n answer in detail")
                    if len(gen_response) > 10:
                        llm_response = llm(gen_response + get_diagram_prompt())
                        final_response = enhance_flowchart_visualization(graphviz.Digraph(), llm_response).source + " diagramcreator"
                
                elif st.session_state.option.strip() == 'General Query: Mindmap':
                    central_topic, topics, connections = process_mindmap_query(message, st.session_state.api)
                    mindmap = create_mindmap(central_topic, topics, connections)
                    final_response = mindmap.source + " mindmapcreator"
                
                st.session_state.history.append(Message("human", message))
                st.session_state.history.append(Message("ai", final_response))
                st.session_state.token_count += cb.total_tokens
                
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing message: {str(e)}")
            print(f"Message processing error: {e}")

def process_message1(message):
    with get_openai_callback() as cb:
        try:
            if len(message) > 5:
                response = st.session_state.conversation1.run(message)
                
                st.session_state.history1.append(Message("human", message))
                st.session_state.history1.append(Message("ai", response))
                st.session_state.token_count1 += cb.total_tokens
                
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing message: {str(e)}")
            print(f"Message processing error: {e}")

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
        'Select Visualization Type',
        ('General Query: Flowchart', 'General Query: Mindmap', 'Uploaded data: Flowchart', 'Uploaded data: Quick Query', 'Uploaded data: Conversational Chat'),
        label_visibility="visible"
    )
    
    if st.session_state.api:
        try:
            if st.session_state.agree:
                upfiles = ", ".join(st.session_state.filenames)
                st.caption(f"Uploaded files: {upfiles}")
                
                qa1 = RetrievalQA.from_chain_type(
                    llm=OpenAI(openai_api_key=st.session_state.api),
                    chain_type="map_reduce",
                    retriever=st.session_state.index.as_retriever(),
                )
                
                tools5 = [Tool(
                    name="QuestionsAnswers",
                    func=qa1.run,
                    description="Useful for when you need to answer questions about the things asked. Input may be a partial or fully formed question.",
                )]
                
                prefix1 = """provide answers using only the tool provided. Answer in detail"""
                suffix1 = """Begin!"\n\n{chat_history1}\nQuestion: {input1}\n{agent_scratchpad}"""
                
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
                    llm=OpenAI(openai_api_key=st.session_state.api),
                    prompt=prompt1,
                )
                agent1 = ZeroShotAgent(llm_chain=llm_chain1, tools=tools5, verbose=True)
                agent_chain1 = AgentExecutor.from_agent_and_tools(
                    agent=agent1,
                    tools=tools5,
                    verbose=True,
                    memory=st.session_state.memory1
                )
                
                # Initialize chat states
                if "history" not in st.session_state:
                    st.session_state.history = []
                if "history1" not in st.session_state:
                    st.session_state.history1 = []
                if "token_count" not in st.session_state:
                    st.session_state.token_count = 0
                if "token_count1" not in st.session_state:
                    st.session_state.token_count1 = 0
                
                try:
                    if "conversation" not in st.session_state:
                        st.session_state.conversation = qa1
                    if "conversation1" not in st.session_state:
                        st.session_state.conversation1 = agent_chain1
                except Exception as e:
                    print(f"Conversation initialization error: {e}")

                # Chat interface based on selected option
                if (st.session_state.option in ['Uploaded data: Quick Query', 'Uploaded data: Flowchart', 
                                              'General Query: Flowchart', 'General Query: Mindmap']):
                    
                    # Display chat history
                    for chat in st.session_state.history:
                        if chat.origin == 'ai' and chat.message.endswith((' diagramcreator', ' mindmapcreator')):
                            message_parts = chat.message.rsplit(' ', 1)
                            st.graphviz_chart(message_parts[0])
                        else:
                            div = f"""
                            <div class="chat-row {'' if chat.origin == 'ai' else 'row-reverse'}">
                                <img class="chat-icon" src="{'https://cdn.discordapp.com/attachments/852337726904598574/1126648713788526722/ai.png' if chat.origin == 'ai' else 'https://cdn.discordapp.com/attachments/852337726904598574/1126648675238682655/human.png'}" width=32 height=32>
                                <div class="chat-bubble {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
                                    &#8203;{chat.message}
                                </div>
                            </div>
                            """
                            st.markdown(div, unsafe_allow_html=True)
                    
                    # Chat input form
                    with st.form(key="chat_form", clear_on_submit=True):
                        cols = st.columns([6, 1])
                        with cols[0]:
                            user_input = st.text_input(
                                label="Your message",
                                placeholder="Type your message here...",
                                key="human_prompt",
                                label_visibility="collapsed"
                            )
                        with cols[1]:
                            submit_button = st.form_submit_button(
                                label="Send",
                                type="primary",
                                use_container_width=True
                            )
                            
                        if submit_button and user_input:
                            process_message(user_input)
                
                elif st.session_state.option == 'Uploaded data: Conversational Chat':
                    # Display chat history
                    for chat in st.session_state.history1:
                        div = f"""
                        <div class="chat-row {'' if chat.origin == 'ai' else 'row-reverse'}">
                            <img class="chat-icon" src="{'https://cdn.discordapp.com/attachments/852337726904598574/1126648713788526722/ai.png' if chat.origin == 'ai' else 'https://cdn.discordapp.com/attachments/852337726904598574/1126648675238682655/human.png'}" width=32 height=32>
                            <div class="chat-bubble {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
                                &#8203;{chat.message}
                            </div>
                        </div>
                        """
                        st.markdown(div, unsafe_allow_html=True)
                    
                    # Chat input form
                    with st.form(key="chat_form1", clear_on_submit=True):
                        cols = st.columns([6, 1])
                        with cols[0]:
                            user_input = st.text_input(
                                label="Your message",
                                placeholder="Type your message here...",
                                key="human_prompt1",
                                label_visibility="collapsed"
                            )
                        with cols[1]:
                            submit_button = st.form_submit_button(
                                label="Send",
                                type="primary",
                                use_container_width=True
                            )
                            
                        if submit_button and user_input:
                            process_message1(user_input)
            else:
                st.info("Please upload your files first.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            print(f"Chat interface error: {e}")
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
