import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
import tempfile

# Importações do Google
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import io

# Importações LangChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Carregar variáveis de ambiente
load_dotenv()

# Configuração da página
st.set_page_config(
    page_title="RAG com Google Drive",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS (mesmo da versão anterior)
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1e3a8a; text-align: center; margin-bottom: 2rem; }
    .stButton>button { background-color: #1e3a8a; color: white; border-radius: 5px; padding: 0.5rem 1rem; font-weight: bold; }
    .stButton>button:hover { background-color: #2563eb; }
    .info-box { background-color: #f0f9ff; border-left: 5px solid #3b82f6; padding: 1rem; margin: 1rem 0; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# --- Lógica de Autenticação do Google Drive ---
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_google_drive():
    """Autentica com a API do Google Drive usando OAuth 2.0."""
    creds = None
    if 'gdrive_creds' in st.session_state:
        creds = st.session_state.gdrive_creds

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                st.error(f"Erro ao renovar token: {e}")
                creds = None # Força a re-autenticação
        else:
            if not os.path.exists('credentials.json'):
                st.error("Arquivo 'credentials.json' não encontrado. Siga as instruções no README_gdrive.md para configurá-lo.")
                return None
            
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            # A linha abaixo é para rodar localmente. Em um servidor, o fluxo seria diferente.
            creds = flow.run_local_server(port=0)
        
        st.session_state.gdrive_creds = creds

    try:
        service = build('drive', 'v3', credentials=creds)
        return service
    except HttpError as error:
        st.error(f"Ocorreu um erro ao criar o serviço do Drive: {error}")
        return None

# --- Classe RAGSystem Modificada ---
class RAGSystem:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.vectorstore_path = "vectorstore_gdrive"
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None
        Path(self.vectorstore_path).mkdir(exist_ok=True)

    def validate_api_key(self):
        if not self.api_key:
            st.error("⚠️ API Key da OpenAI não encontrada! Configure no arquivo .env")
            return False
        return True

    def load_documents_from_gdrive(self, service, folder_id):
        """Baixa e carrega PDFs de uma pasta do Google Drive."""
        if not service:
            st.error("Serviço do Google Drive não está autenticado.")
            return []

        documents = []
        try:
            query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
            results = service.files().list(q=query, fields="files(id, name)").execute()
            items = results.get('files', [])

            if not items:
                st.warning(f"Nenhum arquivo PDF encontrado na pasta com ID: {folder_id}")
                return []

            st.info(f"Encontrados {len(items)} arquivos PDF. Iniciando download e processamento...")

            for item in items:
                file_id = item['id']
                file_name = item['name']
                
                request = service.files().get_media(fileId=file_id)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    downloader = MediaIoBaseDownload(tmp_file, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()
                        st.write(f"Download de {file_name}: {int(status.progress() * 100)}% completo.")
                    
                    tmp_file_path = tmp_file.name

                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = file_name
                documents.extend(docs)
                os.unlink(tmp_file_path)

            return documents

        except HttpError as error:
            st.error(f"Ocorreu um erro ao acessar o Google Drive: {error}")
            return []
        except Exception as e:
            st.error(f"Um erro inesperado ocorreu: {e}")
            return []

    def create_vectorstore(self, documents):
        if not documents:
            return 0
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key=self.api_key, model="text-embedding-3-small")
        
        # Cria um novo vectorstore do zero
        self.vectorstore = FAISS.from_documents(splits, embeddings)
        self.vectorstore.save_local(self.vectorstore_path)
        return len(splits)

    def load_vectorstore(self):
        if not os.path.exists(self.vectorstore_path) or not os.listdir(self.vectorstore_path):
            return False
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=self.api_key, model="text-embedding-3-small")
            self.vectorstore = FAISS.load_local(self.vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            return True
        except Exception as e:
            st.error(f"Erro ao carregar a base de vetores: {e}")
            return False

    def setup_qa_chain(self, model_name="gpt-4o-mini", temperature=0.3):
        if self.vectorstore is None: return False
        llm = ChatOpenAI(openai_api_key=self.api_key, model_name=model_name, temperature=temperature)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        prompt_template = """Você é um assistente especializado. Use o contexto para responder à pergunta. Se não souber, diga que não sabe.
        Contexto: {context}
        Histórico: {chat_history}
        Pergunta: {question}
        Resposta:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        return True

    def query(self, question):
        if not self.qa_chain: return None, []
        result = self.qa_chain({"question": question})
        return result["answer"], result["source_documents"]

def main():
    st.markdown('<h1 class="main-header">🤖 Sistema RAG com Google Drive</h1>', unsafe_allow_html=True)

    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    rag = st.session_state.rag_system

    if not rag.validate_api_key():
        st.stop()

    with st.sidebar:
        st.header("⚙️ Configurações")

        st.subheader("📂 Conexão com Google Drive")
        gdrive_folder_id_input = st.text_input("ID ou URL da Pasta do Google Drive", help="Cole aqui o ID ou a URL completa da pasta do Google Drive que contém os PDFs.")

        if st.button("🔄 Processar Documentos do Drive", use_container_width=True):
            if gdrive_folder_id_input:
                # Extrai o ID da URL, se o usuário colar a URL inteira
                folder_id = gdrive_folder_id_input
                if "folders/" in folder_id:
                    folder_id = folder_id.split("folders/")[-1].split("?")[0]

                with st.spinner("Autenticando e buscando arquivos..."):
                    service = authenticate_google_drive()
                    if service:
                        st.success("✅ Autenticado no Google Drive!")
                        documents = rag.load_documents_from_gdrive(service, folder_id)
                        if documents:
                            num_chunks = rag.create_vectorstore(documents)
                            st.success(f"✅ {len(documents)} docs processados em {num_chunks} chunks!")
                            st.info("Base de vetores atualizada. Agora inicialize o sistema.")
            else:
                st.warning("Por favor, insira o ID ou a URL da pasta do Google Drive.")

        st.divider()

        st.subheader("🧠 Configurações do Modelo")
        model_name = st.selectbox("Modelo OpenAI", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
        temperature = st.slider("Temperatura", 0.0, 1.0, 0.3, 0.1)

        if st.button("🚀 Inicializar Sistema", use_container_width=True):
            with st.spinner("Inicializando..."):
                if rag.load_vectorstore():
                    if rag.setup_qa_chain(model_name, temperature):
                        st.success("✅ Sistema inicializado com sucesso!")
                    else:
                        st.error("❌ Erro ao configurar o sistema de QA.")
                else:
                    st.warning("⚠️ Base de dados não encontrada. Processe os documentos primeiro.")

        st.divider()

        st.subheader("ℹ️ Status do Sistema")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Base de Dados", "Carregada" if rag.vectorstore else "Não Carregada")
        with col2:
            st.metric("Sistema de QA", "Ativo" if rag.qa_chain else "Inativo")

        if st.button("🗑️ Limpar Histórico", use_container_width=True):
            if rag.memory:
                rag.memory.clear()
                st.session_state.messages = []
                st.success("Histórico limpo!")

    # Área principal do Chat
    st.header("💬 Chat com seus Documentos")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Fontes"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**Fonte {i}:** {source.metadata.get('source', 'N/A')} - Página {source.metadata.get('page', 'N/A')}")

    if prompt := st.chat_input("Digite sua pergunta aqui..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if rag.qa_chain:
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    answer, sources = rag.query(prompt)
                    if answer:
                        st.markdown(answer)
                        with st.expander("📚 Fontes consultadas"):
                            for i, source in enumerate(sources, 1):
                                st.write(f"**Fonte {i}:** {source.metadata.get('source', 'N/A')} - Página {source.metadata.get('page', 'N/A')}")
                        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                    else:
                        st.error("Não foi possível gerar uma resposta.")
        else:
            st.warning("⚠️ Por favor, inicialize o sistema na barra lateral.")

    with st.expander("❓ Como usar este sistema"):
        st.markdown("""
        <div class="info-box">
        1. **Configure as APIs**: Siga o `README_gdrive.md` para criar o `credentials.json` (Google) e o `.env` (OpenAI).
        2. **ID da Pasta**: Cole o ID da pasta do Google Drive na barra lateral.
        3. **Processe os Documentos**: Clique em "Processar Documentos do Drive".
        4. **Inicialize o Sistema**: Clique em "Inicializar Sistema".
        5. **Converse**: Faça suas perguntas no chat!
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()