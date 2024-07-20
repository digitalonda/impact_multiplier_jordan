from openai import OpenAI
import streamlit as st
from gdocs import gdocs
from streamlit_modal import Modal
from pinecone import Pinecone
import cohere
from anthropic import Anthropic  
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

from customcss.extras import custom_btn
 
import time
import tiktoken
from split_string import split_string_with_limit
import requests
import json 
import docx2txt
import pdfplumber


st.markdown(
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"/>',
    unsafe_allow_html=True,
)

ANTHROPIC_API_KEY = st.secrets['ANTHROPIC_API_KEY']   
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
COHERE_API_KEY = st.secrets['COHERE_API_KEY']
  

CHUNK_TOKEN_LEN = 1024  

cohere_client = cohere.Client(COHERE_API_KEY)
def cohere_rerank(query: str,docs, top_n=6):
    rerank_docs = cohere_client.rerank(
    query=query, documents=docs, top_n=top_n,return_documents=True, model="rerank-english-v2.0"
    ) 
    return [doc.document.text for doc in rerank_docs.results]

client_claude = Anthropic(
    api_key=ANTHROPIC_API_KEY
)

pc = Pinecone(PINECONE_API_KEY)
data_index = pc.Index("chatdoc")

model_name = "gpt-4o"
def send_llm(data,format_style):
    system_prompting,messages = get_llm_prompt(data,format_style)
    client = OpenAI(
        api_key=OPENAI_API_KEY,
    )
    our_sms = [{"role": "system", "content": system_prompting }]
    our_sms.extend(messages)
     
    chat_completion = client.chat.completions.create(
        messages=our_sms,
        model=model_name,
    )
    return chat_completion.choices[0].message.content

def send_llm_claude(data,format_style):
    system_prompting,our_sms = get_llm_prompt(data,format_style)
    message = client_claude.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=4096,
    system = system_prompting,
    messages=our_sms
    )
    return message.content[0].text

def get_llm_prompt(data,format_style):
    system_prompt = st.session_state.system_prompt
    
    if not system_prompt:
        system_prompting = "You are a helpful assistant.Based on these documents provided below, please complete the task requested by the user:"
    else:
        system_prompting = system_prompt
        if len(data): 
            system_prompting += "\n [VOICE] \n"
            system_prompting += "\n\n".join(data)

        if len(format_style):
            system_prompting += "\n [FORMAT & STYLE] \n"
            system_prompting += "\n\n".join(format_style)

    our_sms = st.session_state.chat_history["history"]
    our_sms = our_sms[-10:]
    return system_prompting,our_sms

def get_embedding(text,embed_model="text-embedding-3-small" ):
    client = OpenAI(api_key=OPENAI_API_KEY)
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=embed_model).data[0].embedding

default_vec_embedding = get_embedding("default")

def add_to_index(data,nsp="default"):
    data_index.upsert(vectors=data,namespace=nsp)

def get_from_index_raw(vec,top_k=20,nsp="default",filter={}):
    res = data_index.query(vector=vec,top_k=top_k,include_values=True,include_metadata=True,namespace=nsp,
                            filter=filter
                            )
    return res["matches"]
def get_from_index(vec,top_k=20,nsp="default",filter={}):
    res_matches = get_from_index_raw(vec,top_k,nsp,filter)
    docs = [x["metadata"]['text'] for x in res_matches]
    if nsp == "list" or nsp == "list_style" or nsp=="chat_history_list":
        docs = { x["metadata"]['doc_id']:x["metadata"]['text'] for i, x in enumerate(res_matches)}
    
    return docs


def get_filter_id(doc_ids):
    return {"doc_id": {"$in": doc_ids}}
 

def get_all_docs():
    docs = get_from_index(default_vec_embedding,1000,"list")
    return docs
def get_all_style_docs():
    docs = get_from_index(default_vec_embedding,1000,"list_style")
    return docs

def get_all_history_list():
    docs = get_from_index(default_vec_embedding,1000,"chat_history_list")
    return docs
    
def save_doc_to_db(document_id,title,nsp):
    metadata = {"doc_id": document_id,"text": title}
    data = [{ "id": document_id, "values":get_embedding(title), "metadata": metadata}]
    add_to_index(data, nsp)

def save_doc_to_vecdb(document_id,chunks,nsp="default"):
    data = []
    lim = 100
    for idx,chunk in enumerate(chunks):
        metadata = {"doc_id": document_id,"text": chunk}
        data.append({ "id": document_id+"_"+str(idx),"values":get_embedding(chunk),"metadata": metadata})
        if len(data) >= lim:
            add_to_index(data,nsp)
            data = []
                
    if len(data) > 0 :
        add_to_index(data,nsp)

def slugify(s):
  s = s.lower().strip()
  s = re.sub(r'[^\w\s-]', '', s)
  s = re.sub(r'[\s_-]+', '-', s)
  s = re.sub(r'^-+|-+$', '', s)
  return s

def get_gdoc(url):
    creds = gdocs.gdoc_creds()
    document_id = gdocs.extract_document_id(url)
    chunks = gdocs.read_gdoc_content(creds,document_id)
    title = gdocs.read_gdoc_title(creds,document_id)
    return document_id,title,chunks

def extract_youtube_id(url):
    pattern = (
        r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/shorts/|youtube\.com/clip/|youtube\.com/user/.*#p/u/\d+/|youtube\.com/attribution_link\?a=|youtube\.com/.*#.*|youtube\.com/live/|youtube\.com/video/|youtube\.com/clip/|youtube\.com/.*#.*|youtube\.com/user/[^/]+/.*#.*|youtube\.com/[^/]+/[^/]+/[^/]+/|youtube\.com/.*\?v=|youtube\.com/.*\?clip_id=)([a-zA-Z0-9_-]{11})'
    )
    # Search for the pattern in the URL
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None

def load_history(k):
    res_matches = get_from_index_raw(default_vec_embedding,1000,"chat_history",filter={"chat_id":k})
    new_history = {"id": k, "history": []}
    for x in res_matches:
        idx = int(x["metadata"]["order"]) - 1
        new_history["history"].insert(idx,{"role":x["metadata"]["role"],"content":x["metadata"]["text"]} )

    
    st.session_state.chat_history = new_history
     
def delete_docs(doc_id,doc_nsp="default",doc_list_nsp="list"):
    l1 = get_from_index_raw(default_vec_embedding,1000,doc_nsp,filter={"doc_id":doc_id}) 
    l2 = get_from_index_raw(default_vec_embedding,1000,doc_list_nsp,filter={"doc_id":doc_id}) 
     # delete from index
    d1 = [x["id"] for x in l1]
    d2 = [x["id"] for x in l2]
     
    #data_index.delete(d1, namespace=doc_nsp)
    #data_index.delete(d2, namespace=doc_list_nsp)
             
def delete_single_history(chat_id,nsp="chat_history",list_nsp="chat_history_list"):
    filter = {"chat_id": {"$in": chat_id}}
    l1 = get_from_index_raw(default_vec_embedding,10000,nsp,filter=filter) 
    d1 = [x["id"] for x in l1]
    data_index.delete(d1, namespace=nsp)
    data_index.delete([chat_id], namespace=list_nsp)

if not "all_docs" in st.session_state:
    st.session_state.all_docs = {}
if not "all_style_docs" in st.session_state:
    st.session_state.all_style_docs = {}

all_docs = get_all_docs() 
st.session_state.all_docs = all_docs
all_style_docs = get_all_style_docs() 
st.session_state.all_style_docs = all_style_docs

def retrive_selected_docs():
    sd = get_from_index_raw(default_vec_embedding,top_k=1,nsp="selected_doc")
    
    if len(sd) > 0:
        sd = sd[0]
        keys = sd["metadata"]["keys"].split(",")
        values = sd["metadata"]["values"].split(",")
        for idx,key in enumerate(keys):
            st.session_state.selected_docs[key] = values[idx]

def retrive_selected_style_docs():
    sd = get_from_index_raw(default_vec_embedding,top_k=1,nsp="selected_style_doc")
    
    if len(sd) > 0:
        sd = sd[0]
        keys = sd["metadata"]["keys"].split(",")
        values = sd["metadata"]["values"].split(",")
        for idx,key in enumerate(keys):
            st.session_state.selected_docs[key] = values[idx]

def save_selected_docs():
    metadata = {"keys": ",".join(st.session_state.selected_docs.keys()),"values": ",".join(st.session_state.selected_docs.values())}
    data = [{ "id": "selected_doc", "values":default_vec_embedding, "metadata": metadata}]
    add_to_index(data, "selected_doc") 
def save_selected_style_docs():
    metadata = {"keys": ",".join(st.session_state.selected_style_docs.keys()),"values": ",".join(st.session_state.selected_style_docs.values())}
    data = [{ "id": "selected_style_doc", "values":default_vec_embedding, "metadata": metadata}]
    add_to_index(data, "selected_style_doc") 

def add_selected_docs(idx,doc_title):
    st.session_state.selected_docs[idx] = doc_title
    save_selected_docs()

def add_selected_style_docs(idx,doc_title):
    st.session_state.selected_style_docs[idx] = doc_title
    save_selected_style_docs()

if not "selected_docs" in st.session_state:
    st.session_state.selected_docs = {}
retrive_selected_docs()

if not "selected_style_docs" in st.session_state:
    st.session_state.selected_style_docs = {}
retrive_selected_style_docs()


def retrive_system_prompt():
    sd = get_from_index_raw(default_vec_embedding,top_k=1,nsp="system_prompt")
    if len(sd) > 0:
        sd = sd[0]
        return sd["metadata"]["text"] 
    else:
        return '''You are an AI Assistant specialized in creating social media captions. Based on the style of the example posts provided in CONTEXT below, craft a caption using the content input by the user.'''    

new_doc_style_modal = Modal(
    "Add New Document", 
    key="new-doc-style-modal",
    padding=20,    # default value
    max_width=700  # default value
)
if new_doc_style_modal.is_open():
    with new_doc_style_modal.container():
        uploaded_file_style = st.file_uploader("Choose a file",type=["docx","doc","txt","rtf","pdf"])
        if uploaded_file_style is not None: 
            if uploaded_file_style.type == "text/plain":
                string_data = uploaded_file_style.read().decode("utf-8")
            elif uploaded_file_style.type == "application/pdf":
                pages = pdfplumber.open(uploaded_file_style).pages
                l = list(map(lambda page:page.extract_text(),pages))
                string_data = "\n\n".join(l)
            else:
                string_data =  docx2txt.process(uploaded_file_style)    
                
            title = uploaded_file_style.name
            document_id = slugify(title)
            tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
            chunks = split_string_with_limit(string_data, CHUNK_TOKEN_LEN*2,tiktoken_encoding)
            if document_id in st.session_state.all_style_docs.keys():
                st.write("Document already exists.")
            else:
                with st.spinner(text="Please patient,it may take some time to process the document."):
                    all_style_docs[document_id] = title
                    st.session_state.selected_style_docs[document_id] = title
                    st.session_state.all_style_docs = all_style_docs 
                    save_doc_to_vecdb(document_id,chunks)
                    save_doc_to_db(document_id,title,"list_style")
                    st.write("Document added successfully.")
                    new_doc_style_modal.close()


new_doc_modal = Modal(
    "Add New Document", 
    key="new-doc-modal",
    padding=20,    # default value
    max_width=700  # default value
)
if new_doc_modal.is_open():
    with new_doc_modal.container():
        tab1, tab2 = st.tabs(["Upload Document", "Youtube"])
         
        with tab1:
            uploaded_file = st.file_uploader("Choose a document file",type=["docx","doc","txt","rtf","pdf"])
            if uploaded_file is not None: 

                if uploaded_file.type == "text/plain":
                    string_data = uploaded_file.read().decode("utf-8")
                elif uploaded_file.type == "application/pdf":
                    pages = pdfplumber.open(uploaded_file).pages
                    l = list(map(lambda page:page.extract_text(),pages))
                    string_data = "\n\n".join(l)
                else:
                    string_data =  docx2txt.process(uploaded_file)    
                
                title = uploaded_file.name
                document_id = slugify(title)
                tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
                chunks = split_string_with_limit(string_data, CHUNK_TOKEN_LEN,tiktoken_encoding)
                if document_id in all_docs.keys():
                    st.write("Document already exists.")
                else:
                    with st.spinner(text="Please patient,it may take some time to process the document."):
                        all_docs[document_id] = title
                        st.session_state.selected_docs[document_id] = title
                        st.session_state.all_docs = all_docs 
                        save_doc_to_vecdb(document_id,chunks)
                        save_doc_to_db(document_id,title,"list")
                        st.write("Document added successfully.")
                        new_doc_modal.close()

        with tab2:
            vid_title = st.text_input("Youtube title:")
            vid_url = st.text_input("Enter your Youtube url, Ex: https://www.youtube.com/watch?v=xxxxxx")
            video_id = extract_youtube_id(vid_url)
            submit_video = st.button("Submit Video")

       
        if submit_video:
            with st.spinner(text="Please patient,it may take some time to process the document."):
                if not video_id or video_id in all_docs.keys():
                    st.write("Video already exists.")
                else:            
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])  
                    formatter = TextFormatter()
                    formatted_transcript = formatter.format_transcript(transcript)
                        
                    save_doc_to_db(video_id,vid_title,"list")
                    all_docs[video_id] = vid_title
                        
                    tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
                    chunks = split_string_with_limit(formatted_transcript, CHUNK_TOKEN_LEN,tiktoken_encoding)
                    save_doc_to_vecdb(video_id,chunks)
                    vid_title = ""
                    vid_url = ""
                    st.write("Document added successfully.")
                    st.session_state.all_docs = all_docs        

if not "chat_history" in st.session_state:
    st.session_state.chat_history = {"id":int(time.time()),"history":[]}
if not "all_chat_history" in st.session_state:
    st.session_state.all_chat_history = get_all_history_list()

 
with st.sidebar:
   
  #st.subheader("Select Your Documents")  
  #doc_options = st.multiselect('Select the documents to query',all_docs.keys(),format_func = lambda x: all_docs[x] if x in all_docs else x,)
  if not "system_prompt" in st.session_state:
    st.session_state.system_prompt = retrive_system_prompt()  
  system_prompt = st.text_area("System Prompt",st.session_state.system_prompt,key="system_prompt") 
  
  prompt_save = st.button("Save Prompt")
  if prompt_save:
    document_id = "system_prompt"
    metadata = {"doc_id":document_id,"text": system_prompt}
    data = [{ "id": document_id, "values":default_vec_embedding, "metadata": metadata}]
    add_to_index(data, "system_prompt")
     

  api_option = st.selectbox(
    'Select the API',
    ('Anthropic','OpenAI')
    )
    
  st.divider()
  st.subheader("Voice")
  for idx,doc_title in st.session_state.all_docs.items():
    checked = False
    if idx in st.session_state.selected_docs.keys():
        checked = True
    st.checkbox(doc_title,checked,idx,on_change=add_selected_docs,args=(idx,doc_title) )
    st.button("Delete",key="btn-"+idx,on_click=lambda : delete_docs(idx))
     

  add_new_doc = st.button("Add Document",key="voice")
  if add_new_doc:
    new_doc_modal.open()

  st.divider()
  st.subheader("Format & Style")
  for idx,doc_title in st.session_state.all_style_docs.items():
    checked = False
    if idx in st.session_state.selected_style_docs.keys():
        checked = True
    st.checkbox(doc_title,checked,idx,on_change=add_selected_style_docs,args=(idx,doc_title) )
    st.button("Delete",key="btn-"+idx,on_click=lambda : delete_docs(idx,"default","list_style"))

  add_new_style = st.button("Add Document",key="format-style")
  if add_new_style:
    new_doc_style_modal.open()
  st.divider()

  st.subheader("Recent")
  allhistories = st.session_state.all_chat_history
  for k in allhistories.keys():
      item = allhistories[k]
      info = (item[:30] + '..') if len(item) > 75 else item
      bt = st.button(info,key=k)
      if bt:
        load_history(k)

      st.button("Delete",key="btn-history-"+str(k),on_click=lambda : delete_single_history(k))

  st.divider()
  delete_history = st.button("Clear History")
  if delete_history:
      data_index.delete(namespace="chat_history", delete_all=True) 
      data_index.delete(namespace="chat_history_list", delete_all=True)   
      st.session_state.all_chat_history = {}       
 
your_prompt = st.chat_input ("Enter your Prompt:") 

if your_prompt:
    filter = get_filter_id([doc for doc in st.session_state.selected_docs.keys() ])

    st.session_state.chat_history["history"].append({"role": "user", "content": your_prompt})
    order = len(st.session_state.chat_history["history"])
    
    your_prompt_vec = get_embedding(your_prompt)
    
    if order == 1:
        if st.session_state.chat_history["id"] not in st.session_state.all_chat_history.keys():
            save_his = [{"id":str(st.session_state.chat_history["id"]),"values":your_prompt_vec,"metadata":{ "doc_id":st.session_state.chat_history["id"],"text":your_prompt}}]
            add_to_index(save_his, "chat_history_list")
        st.session_state.all_chat_history[st.session_state.chat_history["id"]] = your_prompt 

    save_prompt = {"id":str(st.session_state.chat_history["id"])+"_"+str(order),"values":your_prompt_vec,"metadata":{"chat_id":st.session_state.chat_history["id"],"order":order,"role":"user","text":your_prompt}}

    data = get_from_index(your_prompt_vec,filter=filter)
    data = cohere_rerank(your_prompt, data,10)
    
    if len(st.session_state.selected_style_docs.keys())>0:
        filter_style = get_filter_id([doc for doc in st.session_state.selected_style_docs.keys() ])
        format_style = get_from_index(your_prompt_vec,filter=filter_style)
    else:
        format_style = {}

    if api_option == "Anthropic" :
        response = send_llm_claude(data,format_style) 
    else:    
        response = send_llm(data,format_style)

    st.session_state.chat_history["history"].append({"role": "assistant", "content": response})

    order = len(st.session_state.chat_history["history"])
    save_res = {"id":str(st.session_state.chat_history["id"])+"_"+str(order),"values":get_embedding(response),"metadata":{"chat_id":st.session_state.chat_history["id"],"order":order,"role":"assistant","text":response}}
    add_to_index([save_prompt,save_res], "chat_history")
     
for item in st.session_state.chat_history["history"]:
    if item["role"] == "user" or item["role"] == "assistant":    
        st.chat_message(item["role"]).write(item["content"])
    
    
