
from __future__ import annotations
import re
from typing import Optional, Tuple, List, Union, Literal
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

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from typing import Any, Dict, List
from streamlit_option_menu import option_menu
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import SeleniumURLLoader
from typing import Literal
import streamlit as st
from dataclasses import dataclass
from langchain.chains import ConversationalRetrievalChain
from PIL import Image
import mimetypes
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')
from langchain.document_loaders.csv_loader import CSVLoader



from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
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
import streamlit as st
from pypdf import PdfReader


from langchain import LLMChain, OpenAI
from langchain.llms import OpenAI
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
#st.write(st.session_state.api)
#st.session_state.api = os.environ.get("OPENAI_API_KEY")
#st.session_state.api = st.text_input(" ",placeholder = "Enter your API Key", type = 'password' , help='API Keys will not be stored')
api = st.session_state.api
#pages = st.session_state.pages
main_page_sidebar = st.empty()
with main_page_sidebar:

    st.session_state.tab_selection = option_menu(
        menu_title = '',
        menu_icon = 'list-columns-reverse',
        options = ['Home','Add Key','Upload files','Begin Chat','---','---','---','---','Contact us','Logout'],
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
                "font-weight": "bolder","text-align": "centre"

                },
                } )
#if api:

transformed_list = []










#st.write(st.session_state.tab_selection)


def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    #st.write("WRITING")
    #st.write(pdf)
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


# Define a function to convert text content to a list of documents
def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
        #st.write("YOOO")
        print(text)
        print("yoo")
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
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    #st.write("wrotings")
    #st.write(doc_chunks)
    return doc_chunks


# Define a function for the embeddings
def test_embed():
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.api)
    if 'index' not in st.session_state:
        st.session_state.index = FAISS.from_documents([Document(page_content='hi', metadata={'source': 'one).pdf', 'page': 1})], embeddings)
    # Save in a Vector DB
    #index =[]
    with st.spinner("Loading..."):
        index = (FAISS.from_documents(pages, embeddings))
    st.header("Successfully uploaded")
    return index

def test_embed2():
    embeddings = OpenAIEmbeddings(openai_api_key=api)
    # Indexing
    # Save in a Vector DB
    with st.spinner("Loading..."):
        index = FAISS.from_documents(pages, embeddings)
    st.success("Successfully loaded!", icon="✅")
    return index

#-----------------------------------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------------------------------


if st.session_state.tab_selection == "Upload files":
    try:
        if len(st.session_state.api)>1:
            if not st.session_state.pages:
                embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.api)
                st.header("Supported: PDF, Docx, PNG, CSV, Youtube & Website URLs")
                with st.expander("Upload Files"):
                    urlinput = st.text_input("", placeholder="Enter Youtube links separated by spaces")
                    urlinput1 = st.text_input("", placeholder="jsmastery.pro", help='URL format: jsmastery.pro')
                    
                    uploaded_file = st.file_uploader("**Supported Formats: PDF, Docx, PNG (Extracts text from image)**", accept_multiple_files=True , help='Uploaded files will not be stored', type=['png', 'pdf', 'csv', 'docx']   )
                    #st.write(st.session_state.uploaded_file[0].name)
                    for file in uploaded_file:
                        st.session_state.filenames.append(file.name)
                pages = []
                datata = []
                #st.write(uploaded_file)
                
                if urlinput:
                    string_elements = urlinput.split(" ")

                    # Add "www." to the beginning of each element if it doesn't already start with "www."
                    transformed_list = []
                    for element in string_elements:
                        if element.startswith("www.") or element.startswith("https://"):
                            transformed_list.append(element)
                    #st.write("transformed_list")
                    #st.write(transformed_list)
                

                if uploaded_file or transformed_list or urlinput1:
                    #name_of_file = uploaded_file.name
                    for file in uploaded_file:
                        if file.name.lower().endswith('.docx'):
                            xy = os.getcwd()
                            upload_dir = f"{xy}/uploads/"
                            os.makedirs(upload_dir, exist_ok=True)  # Create the directory if it doesn't exist

                            file_path = os.path.join(upload_dir, file.name)

                            #file_path = "./uploads/" + uploaded_file.name  # Specify the desired directory
                            with open(file_path, "wb") as filee:
                                filee.write(file.read())
                            #st.write("File saved to:", file_path)
                            #st.write("Document TIME")

                            #st.write(file)

                            loader = Docx2txtLoader(f"{file_path}")
                            data = loader.load()
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
                            docss = text_splitter.split_documents(data)
                            pages.extend(docss)

                        elif file.name.lower().endswith('.pdf'):
                            xy = os.getcwd()
                            upload_dir = f"{xy}/uploads/"
                            os.makedirs(upload_dir, exist_ok=True)  # Create the directory if it doesn't exist

                            file_path = os.path.join(upload_dir, file.name)
                            #st.write(file_path)
                            #file_path = "./uploads/" + uploaded_file.name  # Specify the desired directory
                            with open(file_path, "wb") as filee:
                                filee.write(file.read())
                            #st.write("File saved to:", file_path)
                            #st.write("PDF TIME")
                            #st.write(file)

                            loader = PyPDFLoader(f"{file_path}")
                            data = loader.load()
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                            docss = text_splitter.split_documents(data)
                            pages.extend(docss)
                        elif file.name.lower().endswith('.csv'):
                            xy = os.getcwd()
                            upload_dir = f"{xy}/uploads/"
                            os.makedirs(upload_dir, exist_ok=True)  # Create the directory if it doesn't exist

                            file_path = os.path.join(upload_dir, file.name)
                            #st.write(file_path)
                            #file_path = "./uploads/" + uploaded_file.name  # Specify the desired directory
                            with open(file_path, "wb") as filee:
                                filee.write(file.read())
                            #st.write("File saved to:", file_path)
                            #st.write("PDF TIME")
                            #st.write(file)

                            loader = CSVLoader(f"{file_path}")

                            data = loader.load()
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                            docss = text_splitter.split_documents(data)
                            pages.extend(docss)



                        elif file.name.lower().endswith('.png') or file.name.lower().endswith('.jpeg') :
                            xy = os.getcwd()
                            upload_dir = f"{xy}/uploads/"
                            os.makedirs(upload_dir, exist_ok=True)  # Create the directory if it doesn't exist

                            file_path = os.path.join(upload_dir, file.name)
                            #st.write(file_path)
                            #file_path = "./uploads/" + uploaded_file.name  # Specify the desired directory
                            with open(file_path, "wb") as filee:
                                filee.write(file.read())
                            #st.write("File saved to:", file_path)
                            #st.write("PNG TIME")
                            #st.write(file)

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

                            # Define root domain to crawl
                            #domain = "jsmastery.pro"
                            #full_url = "https://www.jsmastery.pro"
                            domain = st.session_state.web_url
                            full_url = "https://www." + st.session_state.web_url

                            # Create a class to parse the HTML and get the hyperlinks
                            class HyperlinkParser(HTMLParser):
                                def __init__(self):
                                    super().__init__()
                                    # Create a list to store the hyperlinks
                                    self.hyperlinks = []

                                # Override the HTMLParser's handle_starttag method to get the hyperlinks
                                def handle_starttag(self, tag, attrs):
                                    attrs = dict(attrs)

                                    # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
                                    if tag == "a" and "href" in attrs:
                                        self.hyperlinks.append(attrs["href"])

                            # Function to get the hyperlinks from a URL
                            def get_hyperlinks(url):

                                # Try to open the URL and read the HTML
                                try:
                                    # Open the URL and read the HTML
                                    with urllib.request.urlopen(url) as response:

                                        # If the response is not HTML, return an empty list
                                        if not response.info().get('Content-Type').startswith("text/html"):
                                            return []

                                        # Decode the HTML
                                        html = response.read().decode('utf-8')
                                except Exception as e:
                                    print(e)
                                    return []

                                # Create the HTML Parser and then Parse the HTML to get hyperlinks
                                parser = HyperlinkParser()
                                parser.feed(html)

                                return parser.hyperlinks

                            # Function to get the hyperlinks from a URL that are within the same domain
                            def get_domain_hyperlinks(local_domain, url):
                                clean_links = []
                                for link in set(get_hyperlinks(url)):
                                    clean_link = None

                                    # If the link is a URL, check if it is within the same domain
                                    if re.search(HTTP_URL_PATTERN, link):
                                        # Parse the URL and check if the domain is the same
                                        url_obj = urlparse(link)
                                        if url_obj.netloc == local_domain:
                                            clean_link = link

                                    # If the link is not a URL, check if it is a relative link
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

                                # Return the list of hyperlinks that are within the same domain
                                return list(set(clean_links))


                            def crawl(url):
                                # Parse the URL and get the domain
                                local_domain = urlparse(url).netloc

                                # Create a queue to store the URLs to crawl
                                queue = deque([url])

                                # Create a set to store the URLs that have already been seen (no duplicates)
                                seen = set([url])

                                # Create a directory to store the text files
                                if not os.path.exists("text/"):
                                        os.mkdir("text/")

                                if not os.path.exists("text/"+local_domain+"/"):
                                        os.mkdir("text/" + local_domain + "/")

                                # Create a directory to store the csv files
                                if not os.path.exists("processed"):
                                        os.mkdir("processed")

                                # While the queue is not empty, continue crawling
                                while queue:

                                    # Get the next URL from the queue
                                    url = queue.pop()
                                    print(url) # for debugging and to see the progress

                                    # Save text from the url to a <url>.txt file
                                    with open('text/'+local_domain+'/'+url[8:].replace("/", "_") + ".txt", "w") as f:

                                        # Get the text from the URL using BeautifulSoup
                                        soup = BeautifulSoup(requests.get(url).text, "html.parser")

                                        # Get the text but remove the tags
                                        text = soup.get_text()
                                        print(text)
                                        # If the crawler gets to a page that requires JavaScript, it will stop the crawl
                                        if ("You need to enable JavaScript to run this app." in text):
                                            print("Unable to parse page " + url + " due to JavaScript being required")

                                        # Otherwise, write the text to the file in the text directory

                                        print(f)
                                        f.write(text)

                                    # Get the hyperlinks from the URL and add them to the queue
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

                            texts=[]

                        # Get all the text files in the text directory
                            for file in os.listdir("text/www." + domain + "/"):

                                # Open the file and read the text
                                with open("text/www." + domain + "/" + file, "r") as f:
                                    text = f.read()
                                    print(text)
                                    # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
                                    texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))
                            print(texts)

                            docococ = text_to_docs(texts)
                            pages.extend(docococ) 

                        except Exception as e:
                            st.info("Please enter a valid link")
                            print(e)

                


                st.session_state.agree = st.button('Upload')
                if st.session_state.agree:
                    
                    #st.write(st.session_state.pages)
                    try:
                        #st.write(pages)
                        st.session_state.index = test_embed()
                        st.session_state.pages = pages
                        #st.write(pages)
                        #st.session_state.pages = pages
                    except Exception as e:
                        st.info("Loading Unsuccessful")
                        print(e)
                    #st.write("Loaded")
            else:
                st.header("Successfully uploaded")
                editbtn = st.button("Restart")
                if editbtn:
                    st.session_state.pages = []       
                    st.session_state.filenames =[]     
                    
        else:
            #if st.session_state.tab_selection == "Begin Chat" or st.session_state.tab_selection == "Enter API":
        
            st.header("Enter Your API Key to continue")
            st.markdown("[OpenAI API Key](%s)" % "https://platform.openai.com/account/api-keys")
            
                
    except Exception as e:
        #st.write(e)
        st.info("Invalid API key/files")          
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
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
    ('General Query: Flowchart', 'General Query: Mindmap', 'Uploaded data: Flowchart', 'Uploaded data: Quick Query', 'Uploaded data: Conversational Chat'),label_visibility="collapsed", )
    if st.session_state.api:
        #if st.session_state.agree:
        try:
            if st.session_state.agree:
                #st.write("Pagesss")
                #st.write(f"YO{st.session_state.pages}")
                upfiles = ", ".join(st.session_state.filenames)
                st.caption(f"Uploaded files: {upfiles}")
                #st.write(", ".join(st.session_state.filenames))
                
                    
                qa1 = RetrievalQA.from_chain_type(
                        llm = OpenAI(openai_api_key=st.session_state.api),
                        chain_type = "map_reduce",
                        retriever=st.session_state.index.as_retriever(),
                    )
                #st.write(type(index[0]))
            
                tools5 = [Tool(
                            name="QuestionsAnswers",
                            func=qa1.run,
                            description="Useful for when you need to answer questions about the things asked. Input may be a partial or fully formed question.",
                        )]
                tools4 = Tool(
                            name="QuestionsAnswers",
                            func=qa1.run,
                            description="Useful for when you need to answer questions about the things asked. Input may be a partial or fully formed question.",
                        )
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
                    agent=agent1, tools=tools5, verbose=True, memory=st.session_state.memory1
                )
    
        
            try:
                @dataclass
                class Message:
                    """Class for keeping track of a chat message."""
                    origin: Literal["human", "ai"]
                    message: str

                def load_css():
                    with open("static/styles.css", "r") as f:
                        css = f"<style>{f.read()}</style>"
                        st.markdown(css, unsafe_allow_html=True)

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
                                model_name="gpt-3.5-turbo"
                            )
                            
                            st.session_state.conversation = qa1
                        if "conversation1" not in st.session_state:
                            llm = OpenAI(
                                temperature=0,
                                openai_api_key=st.session_state.api,
                                model_name="gpt-3.5-turbo"
                            )
                            
                            st.session_state.conversation1 = agent_chain1
                    except Exception as e:
                        st.write("")
                def on_click_callback():
                    
                    with get_openai_callback() as cb:
                        human_prompt = st.session_state.human_prompt
                        print("hi3")
                        if(len(human_prompt)>5):
                            try:
                                diag =  """step1. rewrite the result as a directed graph  \n
                                step2: make sure the resuly is exactly in the format as provided in the sample. 
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
                                
                                #print(st.session_state.option1.strip())
                                if st.session_state.option.strip() =='Uploaded data: Flowchart':
                                    try:
                                        gen_response = st.session_state.conversation.run(
                                            "Answer in detail" + human_prompt 
                                        )
                                        print("llm response is 1" + gen_response)

                                        llm = OpenAI(openai_api_key=st.session_state.api)
                                        llm_response = llm(gen_response + diag)

                                        llm_response1 = '''digraph{'''+llm_response+'''} diagramcreator'''
                                    except Exception as e:
                                        #st.write(e)
                                        llm_response1 = "Please Upload your files and try again"
                                elif st.session_state.option.strip() == 'Uploaded data: Quick Query':
                                    try:
                                        llm_response = st.session_state.conversation.run(
                                            "Answer in detail" + human_prompt 
                                        )
                                        print("llm response is 1" + llm_response)

                                        llm_response1 = llm_response
                                    except Exception as e:
                                        #st.write(e)
                                        llm_response1 = "Please Upload your files and try again"
                                elif st.session_state.option.strip() == 'General Query: Flowchart':
                                    try:
                                        st.write(human_prompt)
                                        st.write("yooooo" + st.session_state.api)
                                        llm = OpenAI(openai_api_key=st.session_state.api)
                                        gen_response=llm(human_prompt + "   \n answer in detail")
                                        if len(gen_response)>10:
                                            print(f"DETAILED{gen_response}")
                                            llm_response = llm(gen_response + diag)
                                            print(f"DETAILED{llm_response}")
                                            llm_response1 = '''digraph{'''+llm_response+'''} diagramcreator'''
                                    except Exception as e:
                                        st.write(e)
                                        llm_response1 = "Please enter a valid key and try again"
                            except Exception as e:
                                #st.write(e)
                                llm_response1 = " Please try again"
                        else:
                            human_prompt = "Please enter a valid query"
                        try:
                            st.session_state.history.append(
                                Message("human", human_prompt)
                            )
                            st.session_state.history.append(
                                Message("ai", llm_response1)
                            )
                            if st.session_state.option.strip() == 'General Query: Flowchart' or st.session_state.option.strip() =='Uploaded data: Flowchart':
                                st.session_state.history.append(
                                Message("ai", gen_response))
                            st.session_state.token_count += cb.total_tokens
                        except Exception as e:
                            print(e)


                def on_click_callback1():
                    with get_openai_callback() as cb:
                        human_prompt = st.session_state.human_prompt1
                        if(len(human_prompt)>5):
                            try:
                            
                                llm_response = st.session_state.conversation1.run(
                                    human_prompt 
                                )
                            except Exception as e:
                                print(e)
                                llm_response1 = " Please Upload your files and try again"
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
                            
                load_css()
                initialize_session_state()
                try:
                    st.session_state.conversation = qa1
                    #st.title("Hello Custom CSS Chatbot 🤖")

                    st.session_state.conversation1 = agent_chain1
                except Exception as e:
                    st.write("")
                chat_placeholder = st.container()
                prompt_placeholder = st.form("chat-form")
                credit_card_placeholder = st.empty()
                chat_placeholder1 = st.container()
                prompt_placeholder1 = st.form("chat-form1")
                credit_card_placeholder1 = st.empty()

                if st.session_state.option == 'Uploaded data: Quick Query' or st.session_state.option == 'Uploaded data: Flowchart' or st.session_state.option == 'General Query: Flowchart':

                    with chat_placeholder:
                        for chat in st.session_state.history:
                            words = chat.message.split()
                            last_word = words[-1]
                            last_word = last_word.strip()

                            message_before_last_word = ' '.join(words[:-1])

                            print(f"last word is {last_word}")
                            print(f"rest of it is {message_before_last_word}")
                            if last_word == "diagramcreator":
                                print(f"making it true {last_word}")
                                st.session_state.isdiagram = True
                            else:
                                st.session_state.isdiagram = False
                            if chat.origin == 'ai' and st.session_state.isdiagram == True:
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
                            #st.write("hi")
                            #st.graphviz_chart(chat.message)
                                st.markdown(div, unsafe_allow_html=True)
                        
                        for _ in range(3):
                            st.markdown("")
                        st.write("""<div class='PortMarker'/>""", unsafe_allow_html=True)
                    print(st.session_state.option1)
                    with prompt_placeholder:
                        #st.markdown("**Chat**")
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
                        #st.markdown("**Chat**")
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
                elif st.session_state.option == 'General Query: Mindmap':
                    openai.api_key = st.session_state.api
                    @dataclass
                    class Message:
                        """A class that represents a message in a ChatGPT conversation.
                        """
                        content: str
                        role: Literal["user", "system", "assistant"]

                        # is a built-in method for dataclasses
                        # called after the __init__ method
                        def __post_init__(self):
                            self.content = dedent(self.content).strip()

                    START_CONVERSATION = [
                        Message("""
                            You are a useful mind map/undirected graph-generating AI that can generate mind maps
                            based on any input or instructions.
                        """, role="system"),
                        Message("""
                            You have the ability to perform the following actions given a request
                            to construct or modify a mind map/graph:

                            1. add(node1, node2) - add an edge between node1 and node2
                            2. delete(node1, node2) - delete the edge between node1 and node2
                            3. delete(node1) - deletes every edge connected to node1

                            Note that the graph is undirected and thus the order of the nodes does not matter
                            and duplicates will be ignored. Another important note: the graph should be sparse,
                            with many nodes and few edges from each node. Too many edges will make it difficult 
                            to understand and hard to read. The answer should only include the actions to perform, 
                            nothing else. If the instructions are vague or even if only a single word is provided, 
                            still generate a graph of multiple nodes and edges that that could makes sense in the 
                            situation. Remember to think step by step and debate pros and cons before settling on 
                            an answer to accomplish the request as well as possible.

                            Here is my first request: Add a mind map about machine learning.
                        """, role="user"),
                        Message("""
                            add("Machine learning","AI")
                            add("Machine learning", "Reinforcement learning")
                            add("Machine learning", "Supervised learning")
                            add("Machine learning", "Unsupervised learning")
                            add("Supervised learning", "Regression")
                            add("Supervised learning", "Classification")
                            add("Unsupervised learning", "Clustering")
                            add("Unsupervised learning", "Anomaly Detection")
                            add("Unsupervised learning", "Dimensionality Reduction")
                            add("Unsupervised learning", "Association Rule Learning")
                            add("Clustering", "K-means")
                            add("Classification", "Logistic Regression")
                            add("Reinforcement learning", "Proximal Policy Optimization")
                            add("Reinforcement learning", "Q-learning")
                        """, role="assistant"),
                        Message("""
                            Remove the parts about reinforcement learning and K-means.
                        """, role="user"),
                        Message("""
                            delete("Reinforcement learning")
                            delete("Clustering", "K-means")
                        """, role="assistant")
                    ]

                    def ask_chatgpt(conversation: List[Message]) -> Tuple[str, List[Message]]:
                        client = OpenAI(api_key=st.session_state.api)

                        messages = [asdict(c) for c in conversation]

                        response = client.chat.create(
                                messages=messages,
                                )

                        #st.write(response)
                        # turn into a Message object
                        msg = Message(**response["choices"][0]["message"])
                        # return the text output and the new conversation
                        return msg.content, conversation + [msg]

                    class MindMap:
                        """A class that represents a mind map as a graph.
                        """
                        
                        def __init__(self, edges: Optional[List[Tuple[str, str]]]=None, nodes: Optional[List[str]]=None) -> None:
                            self.edges = [] if edges is None else edges
                            self.nodes = [] if nodes is None else nodes
                            self.save()

                        @classmethod
                        def load(cls) -> MindMap:
                            """Load mindmap from session state if it exists
                            
                            Returns: Mindmap
                            """
                            if "mindmap" in st.session_state:
                                return st.session_state["mindmap"]
                            return cls()

                        def save(self) -> None:
                            # save to session state
                            st.session_state["mindmap"] = self

                        def is_empty(self) -> bool:
                            return len(self.edges) == 0
                        
                        def ask_for_initial_graph(self, query: str) -> None:
                            """Ask GPT-3 to construct a graph from scrach.

                            Args:
                                query (str): The query to ask GPT-3 about.

                            Returns:
                                str: The output from GPT-3.
                            """

                            conversation = START_CONVERSATION + [
                                Message(f"""
                                    Great, now ignore all previous nodes and restart from scratch. I now want you do the following:    

                                    {query}
                                """, role="user")
                            ]

                            output, self.conversation = ask_chatgpt(conversation)
                            # replace=True to restart
                            self.parse_and_include_edges(output, replace=True)

                        def ask_for_extended_graph(self, selected_node: Optional[str]=None, text: Optional[str]=None) -> None:
                            """Cached helper function to ask GPT-3 to extend the graph.

                            Args:
                                query (str): query to ask GPT-3 about
                                edges_as_text (str): edges formatted as text

                            Returns:
                                str: GPT-3 output
                            """

                            # do nothing
                            if (selected_node is None and text is None):
                                return

                            # change description depending on if a node
                            # was selected or a text description was given
                            #
                            # note that the conversation is copied (shallowly) instead
                            # of modified in place. The reason for this is that if
                            # the chatgpt call fails self.conversation will not
                            # be updated
                            if selected_node is not None:
                                # prepend a description that this node
                                # should be extended
                                conversation = self.conversation + [
                                    Message(f"""
                                        add new edges to new nodes, starting from the node "{selected_node}"
                                    """, role="user")
                                ]
                                st.session_state.last_expanded = selected_node
                            else:
                                # just provide the description
                                conversation = self.conversation + [Message(text, role="user")]

                            # now self.conversation is updated
                            output, self.conversation = ask_chatgpt(conversation)
                            self.parse_and_include_edges(output, replace=False)

                        def parse_and_include_edges(self, output: str, replace: bool=True) -> None:
                            """Parse output from LLM (GPT-3) and include the edges in the graph.

                            Args:
                                output (str): output from LLM (GPT-3) to be parsed
                                replace (bool, optional): if True, replace all edges with the new ones, 
                                    otherwise add to existing edges. Defaults to True.
                            """

                            # Regex patterns
                            pattern1 = r'(add|delete)\("([^()"]+)",\s*"([^()"]+)"\)'
                            pattern2 = r'(delete)\("([^()"]+)"\)'

                            # Find all matches in the text
                            matches = re.findall(pattern1, output) + re.findall(pattern2, output)

                            new_edges = []
                            remove_edges = set()
                            remove_nodes = set()
                            for match in matches:
                                op, *args = match
                                add = op == "add"
                                if add or (op == "delete" and len(args)==2):
                                    a, b = args
                                    if a == b:
                                        continue
                                    if add:
                                        new_edges.append((a, b))
                                    else:
                                        # remove both directions
                                        # (undirected graph)
                                        remove_edges.add(frozenset([a, b]))
                                else: # must be delete of node
                                    remove_nodes.add(args[0])

                            if replace:
                                edges = new_edges
                            else:
                                edges = self.edges + new_edges

                            # make sure edges aren't added twice
                            # and remove nodes/edges that were deleted
                            added = set()
                            for edge in edges:
                                nodes = frozenset(edge)
                                if nodes in added or nodes & remove_nodes or nodes in remove_edges:
                                    continue
                                added.add(nodes)

                            self.edges = list([tuple(a) for a in added])
                            self.nodes = list(set([n for e in self.edges for n in e]))
                            self.save()

                        def _delete_node(self, node) -> None:
                            """Delete a node and all edges connected to it.

                            Args:
                                node (str): The node to delete.
                            """
                            self.edges = [e for e in self.edges if node not in frozenset(e)]
                            self.nodes = list(set([n for e in self.edges for n in e]))
                            self.conversation.append(Message(
                                f'delete("{node}")', 
                                role="user"
                            ))
                            self.save()

                        def _add_expand_delete_buttons(self, node) -> None:
                            with st.container():
                                cols = st.columns(6)
                                cols[0].subheader(node)
                                cols[1].button(
                                    label="Brainstorm", 
                                    on_click=self.ask_for_extended_graph,
                                    key=f"expand_{node}",
                                    # pass to on_click (self.ask_for_extended_graph)
                                    kwargs={"selected_node": node}
                                )
                                cols[2].button(
                                    label="Remove", 
                                    on_click=self._delete_node,
                                    type="primary",
                                    key=f"delete_{node}",
                                    # pass on to _delete_node
                                    args=(node,)
                                )

                        def visualize(self, graph_type: Literal["agraph", "networkx", "graphviz"]) -> None:
                            """Visualize the mindmap as a graph a certain way depending on the `graph_type`.

                            Args:
                                graph_type (Literal["agraph", "networkx", "graphviz"]): The graph type to visualize the mindmap as.
                            Returns:
                                Union[str, None]: Any output from the clicking the graph or 
                                    if selecting a node in the sidebar.
                            """

                            selected = st.session_state.get("last_expanded")
                            graph_type = "agraph"
                            if graph_type == "agraph":
                                vis_nodes = [
                                    Node(
                                        id=n, 
                                        label=n, 
                                        # a little bit bigger if selected
                                        size=10+10*(n==selected), 
                                        # a different color if selected
                                        color=COLOR if n != selected else FOCUS_COLOR
                                    ) 
                                    for n in self.nodes
                                ]
                                vis_edges = [Edge(source=a, target=b) for a, b in self.edges]
                                config = Config(width=2000,
                                        height=1000,
                                        directed=False, 
                                        physics=False,
                                        hierarchical=False,
                                        initialZoom=1
                                        )
                                # returns a node if clicked, otherwise None
                                clicked_node = agraph(nodes=vis_nodes, 
                                                edges=vis_edges, 
                                                config=config)
                                # if clicked, update the sidebar with a button to create it
                                if clicked_node is not None:
                                    self._add_expand_delete_buttons(clicked_node)
                                return
                            if graph_type == "networkx":
                                graph = nx.Graph()
                                for a, b in self.edges:
                                    graph.add_edge(a, b)
                                colors = [FOCUS_COLOR if node == selected else COLOR for node in graph]
                                fig, _ = plt.subplots(figsize=(16, 16))
                                pos = nx.spring_layout(graph, seed = 123)
                                nx.draw(graph, pos=pos, node_color=colors, with_labels=True)
                                st.pyplot(fig)
                            else: # graph_type == "graphviz":
                                graph = graphviz.Graph()
                                graph.attr(rankdir='LR')
                                for a, b in self.edges:
                                    graph.edge(a, b, dir="both")
                                for n in self.nodes:
                                    graph.node(n, style="filled", fillcolor=FOCUS_COLOR if n == selected else COLOR)
                                #st.graphviz_chart(graph, use_container_width=True)
                                b64 = base64.b64encode(graph.pipe(format='svg')).decode("utf-8")
                                html = f"<img style='width: 100%' src='data:image/svg+xml;base64,{b64}'/>"
                                st.write(html, unsafe_allow_html=True)
                            # sort alphabetically
                            for node in sorted(self.nodes):
                                self._add_expand_delete_buttons(node)

                    def main():
                        # will initialize the graph from session state
                        # (if it exists) otherwise will create a new one
                        mindmap = MindMap.load()


                        #graph_type = st.radio("Type of graph", options=["agraph", "networkx", "graphviz"])
                        graph_type = "agraph"
                        empty = mindmap.is_empty()
                        reset = True
                        with prompt_placeholder:
                        #st.markdown("**Chat**")
                            col1,col2 = st.columns([6, 1])
                            with col1:
                                query = st.text_input("", 
                                    value=st.session_state.get("mindmap-input", ""),
                                    key="mindmap-input",
                                    label_visibility="collapsed",
                                )
                            with col2:
                                submit = st.form_submit_button("Submit")

                        valid_submission = submit and query != ""

                        if empty and not valid_submission:
                            return

                        with st.spinner(text="Loading graph..."):
                            # if submit and non-empty query, then update graph
                            if valid_submission:
                                if reset:
                                    # completely new mindmap
                                    mindmap.ask_for_initial_graph(query=query)
                                else:
                                    # extend existing mindmap
                                    mindmap.ask_for_extended_graph(text=query)
                                # since inputs also have to be updated, everything
                                # is rerun
                                st.experimental_rerun()
                            else:
                                mindmap.visualize(graph_type)

                    #if __name__ == "__main__":
                    main()
            except Exception as e:
                st.header("Upload your files to begin/")
                st.subheader("Check your OpenAI Plan")
                st.write(e)
        except Exception as e:
            print(e)
            st.info("Add key and upload files/URL to continue")
    else:
        st.header("Enter Your API Key to continue")
if st.session_state.tab_selection == "Add Key":
    #st.write(f"SEssion:{st.session_state.api}")
    if len(st.session_state.api)>1:
        st.header("API key loaded")
        #st.write("yo")
        if(st.button("Edit Key")):
            #st.write("HEHE")
            st.session_state.api =""
            if not len(st.session_state.api)<2:
                st.experimental_rerun()
    else:
        st.header("Load Your API Key")
        st.markdown("[OpenAI API Key](%s)" % "https://platform.openai.com/account/api-keys")
        st.session_state.api = st.text_input(" ",placeholder = "Enter your API Key", type = 'password' , help='API Keys will not be stored')
        if st.session_state.api: 
            st.experimental_rerun()
    
if st.session_state.tab_selection == "Home":
    co1,co2,co3 = st.columns([5,8,5])
    with co2:
        #st.image("frilogo.png")
        st.markdown('<div style = "justify-content: center; text-align: center;"><span style="font-size: 2.5em; font-family: Helvetica Neue; font: 500; font-style: bolder; font-weight: 600; justify-content: center; text-align: center; border-radius: 30px; padding: 10px; "><span style = "color: white; padding: 5px;"></span><img class="chat-icon" src="https://cdn.discordapp.com/attachments/852337726904598574/1126682090101035019/frilogo11.png" width=54 height=54 style="border-radius: 50px;border: 1px solid #515389; padding: 5px;"></img></span></span></div>', unsafe_allow_html=True)

        st.markdown('<div style = "justify-content: center; text-align: center;"><span style="font-size: 3.3em; font-family: Helvetica Neue; font: 500; font-style: bolder; font-weight: 600; justify-content: center; text-align: center; border-radius: 30px; padding: 10px;">Deep Dive into your Data,<span style="color: #cc003d;font-family: Brush Script MT;"> Visually</span>.</span><span style="color: #898989; font-size: 1.5em; font-family: Helvetica Neue; font: 500; font-style: bolder; font-weight: 400; justify-content: center; text-align: center; border-radius: 30px; padding: 5px;"></span></div>', unsafe_allow_html=True)

        st.markdown('<div style = "justify-content: center; text-align: center;"><span style="color: #a1a0a0; font-size: 1em; font-family: Helvetica Neue; font: 500; font-style: bolder; font-weight: 400; justify-content: center; text-align: center; border-radius: 30px; padding: 5px;"> Generate Interactive Mindmaps and Dynamic Flowcharts with <span style="font-style: bolder;font-weight: 600;">Datamap AI.</span></span></div>', unsafe_allow_html=True)
        st.write(" ")
page_bg_img = f"""
<style>

@keyframes glowing {{
    0% {{ background-position: 0 0; }}
    50% {{ background-position: 400% 0; }}
    100% {{ background-position: 0 0; }}
}}
[data-testid="stAppViewContainer"] > .main {{

background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {{
background-position: left; 
background-repeat: no-repeat;
background-attachment: fixed;
min-width: 100px;

}}


[data-testid="stHeader"] {{
}}
<script>
window.parent.document.querySelectorAll('[data-testid="stButton"]')).backgroundcolor = yellow;
Array.from(window.parent.document.querySelectorAll('div[data-testid="stExpander"] div[role="button"] p')).find(el => el.innerText === 'label2').classList.add('label2css');
Array.from(window.parent.document.querySelectorAll('div[data-testid="stExpander"] div[role="button"] p')).find(el => el.innerText === 'label3').classList.add('label3css');
console.log("test");
</script>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
