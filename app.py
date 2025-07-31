import streamlit as st
from io import BytesIO
from audiorecorder import audiorecorder
from dotenv import dotenv_values
from openai import OpenAI
from hashlib import md5
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct,Distance,VectorParams

env=dotenv_values('.env')
if 'QDRANT_URL' in st.secrets:
    env['QDRANT_URL']=st.secrets["QDRANT_URL"]
if 'QDRANT_API_KEY' in st.secrets:
    env["QDRANT_API_KEY"]=st.secrets['QDRANT_API_KEY']
    
EMBEDDING_MODEL='text-embedding-3-large'

EMBEDDING_DIM=3072

QDRANT_COLLECTION_NAME='notes'

AUDIO_TRANSCRIBE_MODEL='whisper-1'

def get_open_ai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"]
    )

def transcribe_audio(audio_bytes):
    openai_client=get_open_ai_client()
    audio_file=BytesIO(audio_bytes)
    audio_file.name='audio.mp3'
    transcript=openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
        response_format='verbose_json',
    )

    return transcript.text



#
#DB
#
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url=env["QDRANT_URL"], 
    api_key=env['QDRANT_API_KEY'])

def assure_db_collection_exists():
    qdrant_client=get_qdrant_client()
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        print('Tworzę Kolekcję')
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
    else:
        print('Kolekcja już istnieje')

def get_embedding(text):
    openai_client=get_open_ai_client()
    result=openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )

    return result.data[0].embedding

def add_note_to_db(note_text):
    qdrant_client=get_qdrant_client()
    points_count=qdrant_client.count(
        collection_name=QDRANT_COLLECTION_NAME,
        exact=True,
    )
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=points_count.count + 1,
                vector=get_embedding(text=note_text),
                payload={
                    'text':note_text,
                }
            )
        ]
    )
def list_notes_from_db(query=None):
    qdrantclient=get_qdrant_client()
    if not query:
        notes=qdrantclient.scroll(collection_name=QDRANT_COLLECTION_NAME,limit=10)[0]
        result=[]
        for note in notes:
            result.append( {
                'text':note.payload['text'],
                'score':None,
            })
        return result
    else:
        notes=qdrantclient.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=get_embedding(text=query),
            limit=10,
        )   
        result=[]
        for note in notes:
            result.append({
                'text':note.payload['text'],
                'score':note.score,
            })        
        return result

#
#Main
#
st.set_page_config(page_title="Audio Notatki",layout='centered')
if not st.session_state.get('openai_api_key'):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"]=env['OPENAI_API_KEY']
    
    else:
        st.info('Dodaj swój klucz API Open AI aby móc korzystać z tej aplikacji')
        st.session_state['openai_api_key']=st.text_input('Klucz API',type='password')
        if st.session_state['openai_api_key']:
            st.rerun()

if not st.session_state.get('openai_api_key'):
    st.stop()

if 'note_audio_bytes_md5' not in st.session_state:
    st.session_state["note_audio_bytes_md5"]=None

if 'note_audio_bytes' not in st.session_state:
    st.session_state["note_audio_bytes"]=None

if "note_audio_text" not in st.session_state:
    st.session_state['note_audio_text']=""

if "note_text" not in st.session_state:
    st.session_state['note_text']=""

st.title('Audio Notatki')
assure_db_collection_exists()
add_tab, search_tab= st.tabs(['Dodaj notatkę',"Wyszukaj notatkę"])
with add_tab:
    note_audio=audiorecorder(
        start_prompt="Nagraj notatkę",
        stop_prompt="Zatrzymaj nagrywanie",
    )
    if note_audio:
        audio=BytesIO()
        note_audio.export(audio,format='mp3')
        st.session_state['note_audio_bytes']=audio.getvalue()
        current_md5=md5(st.session_state['note_audio_bytes']).hexdigest()
        if st.session_state['note_audio_bytes_md5']!= current_md5:
            st.session_state['note_audio_text']=""
            st.session_state['note_text']=""
            st.session_state['note_audio_bytes_md5']= current_md5
        st.audio(st.session_state['note_audio_bytes'],format='audio/mp3')
        
        if st.button('Transkrybuj audio'):
            st.session_state['note_audio_text']=transcribe_audio(st.session_state['note_audio_bytes'])

        if st.session_state['note_audio_text']:
            st.session_state['note_text']=st.text_area('Edytuj notatkę',value=st.session_state['note_audio_text'])
        
        if st.session_state['note_text'] and st.button('Zapisz notatkę',disabled=not st.session_state['note_text']):
            qdrantclient=get_qdrant_client()

            add_note_to_db(note_text=st.session_state['note_text'])
            st.toast('Notatka Zapisana',icon="🔥")
with search_tab:
    query=st.text_input('Wyszukaj notatkę')
    if st.button("Szukaj"):
        for note in list_notes_from_db(query):
            with st.container(border=True):
                st.markdown(note['text'])
                if note['score']:
                    st.markdown(f':violet[{note["score"]}]')
    
    