import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
import tempfile
import shutil

# Importações para processamento de documentos
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from openai import OpenAI

# Carregar variáveis de ambiente
load_dotenv()

# Configuração da página
st.set_page_config(
    page_title="Sistema RAG Empresarial",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
    .info-box {
        background-color: #f0f9ff;
        border-left: 5px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

class RAGSystem:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.vectorstore_path = "vectorstore"
        self.documents_path = "documentos"
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None
        
        # Criar diretórios se não existirem
        Path(self.vectorstore_path).mkdir(exist_ok=True)
        Path(self.documents_path).mkdir(exist_ok=True)
        
    def validate_api_key(self):
        """Valida se a API Key está configurada"""
        if not self.api_key:
            st.error("⚠️ API Key da OpenAI não encontrada! Configure no arquivo .env")
            st.info("Adicione OPENAI_API_KEY=sua-chave no arquivo .env")
            return False
        return True
    
    def load_documents(self, uploaded_files):
        """Carrega e processa documentos PDF"""
        documents = []
        
        for uploaded_file in uploaded_files:
            # Salvar arquivo temporariamente
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Carregar PDF
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            
            # Adicionar metadados
            for doc in docs:
                doc.metadata['source'] = uploaded_file.name
                doc.metadata['file_type'] = 'pdf'
            
            documents.extend(docs)
            
            # Limpar arquivo temporário
            os.unlink(tmp_file_path)
            
        return documents
    
    def create_vectorstore(self, documents):
        """Cria ou atualiza o vectorstore com os documentos"""
        # Configurar text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Dividir documentos
        splits = text_splitter.split_documents(documents)
        
        # Criar embeddings
        embeddings = OpenAIEmbeddings(
            openai_api_key=self.api_key,
            model="text-embedding-3-small"
        )
        
        # Criar ou atualizar vectorstore
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(splits, embeddings)
        else:
            new_vectorstore = FAISS.from_documents(splits, embeddings)
            self.vectorstore.merge_from(new_vectorstore)
        
        # Salvar vectorstore
        self.vectorstore.save_local(self.vectorstore_path)
        
        return len(splits)
    
    def load_vectorstore(self):
        """Carrega vectorstore existente"""
        try:
            embeddings = OpenAIEmbeddings(
                openai_api_key=self.api_key,
                model="text-embedding-3-small"
            )
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            return True
        except:
            return False
    
    def get_available_models(self):
        """Retorna a lista de modelos disponíveis na API da OpenAI."""
        if not self.client:
            return []
        try:
            return self.client.models.list().data
        except Exception as e:
            st.error(f"Erro ao buscar modelos: {e}")
            return []

    def setup_qa_chain(self, model_name="gpt-4o-mini", temperature=0.3):
        """Configura a chain de perguntas e respostas"""
        if self.vectorstore is None:
            return False
        
        # Configurar o modelo
        llm = ChatOpenAI(
            openai_api_key=self.api_key,
            model_name=model_name,
            temperature=temperature
        )
        
        # Configurar memória
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Template de prompt personalizado
        prompt_template = """Você é um assistente especializado da empresa. 
        Use o contexto fornecido para responder à pergunta de forma precisa e profissional.
        Se não souber a resposta com base no contexto, diga claramente que não tem essa informação.
        
        Contexto: {context}
        
        Histórico da conversa: {chat_history}
        
        Pergunta: {question}
        
        Resposta detalhada e profissional:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Criar chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=False
        )
        
        return True
    
    def query(self, question):
        """Realiza uma consulta ao sistema"""
        if self.qa_chain is None:
            return None, []
        
        result = self.qa_chain({"question": question})
        return result["answer"], result["source_documents"]

def main():
    # Título principal
    st.markdown('<h1 class="main-header">🤖 Sistema RAG Empresarial</h1>', unsafe_allow_html=True)
    
    # Inicializar sistema
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    
    rag = st.session_state.rag_system
    
    # Validar API Key
    if not rag.validate_api_key():
        st.stop()
    
    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Upload de documentos
        st.subheader("📄 Upload de Documentos")
        uploaded_files = st.file_uploader(
            "Selecione arquivos PDF",
            type=['pdf'],
            accept_multiple_files=True,
            help="Faça upload de documentos PDF para adicionar à base de conhecimento"
        )
        
        if uploaded_files:
            if st.button("🔄 Processar Documentos", use_container_width=True):
                with st.spinner("Processando documentos..."):
                    documents = rag.load_documents(uploaded_files)
                    num_chunks = rag.create_vectorstore(documents)
                    st.success(f"✅ {len(documents)} documentos processados em {num_chunks} chunks!")
        
        st.divider()
        
        # Configurações do modelo
        st.subheader("🧠 Configurações do Modelo")
        
        model_name = st.selectbox(
            "Modelo OpenAI",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            help="Escolha o modelo da OpenAI a ser usado"
        )
        
        temperature = st.slider(
            "Temperatura",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Controla a criatividade das respostas (0 = mais conservador, 1 = mais criativo)"
        )
        
        if st.button("🚀 Inicializar Sistema", use_container_width=True):
            with st.spinner("Inicializando..."):
                if rag.load_vectorstore():
                    if rag.setup_qa_chain(model_name, temperature):
                        st.success("✅ Sistema inicializado com sucesso!")
                    else:
                        st.error("❌ Erro ao configurar o sistema de QA")
                else:
                    st.warning("⚠️ Nenhuma base de dados encontrada. Faça upload de documentos primeiro.")
        
        st.divider()
        
        # Informações do sistema
        st.subheader("ℹ️ Status do Sistema")
        
        # Status dos componentes
        col1, col2 = st.columns(2)
        
        with col1:
            if rag.vectorstore:
                st.success("✅ Base carregada")
            else:
                st.info("⏳ Base não carregada")
        
        with col2:
            if rag.qa_chain:
                st.success("✅ QA ativo")
            else:
                st.info("⏳ QA não inicializado")
        
        # Estatísticas
        if rag.client:
            if st.button("📊 Ver Estatísticas da API", use_container_width=True):
                with st.spinner("Carregando..."):
                    try:
                        models_count = len(rag.get_available_models())
                        st.metric("Modelos Disponíveis", models_count)
                    except:
                        st.info("Não foi possível carregar estatísticas")
        
        # Limpar histórico
        if st.button("🗑️ Limpar Histórico", use_container_width=True):
            if rag.memory:
                rag.memory.clear()
                st.success("Histórico limpo!")
                st.session_state.messages = []
    
    # Área principal - Chat
    st.header("💬 Chat com seus Documentos")
    
    # Inicializar histórico de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Exibir histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Fontes"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**Fonte {i}:** {source.metadata.get('source', 'Desconhecido')} - Página {source.metadata.get('page', 'N/A')}")
    
    # Input do usuário
    if prompt := st.chat_input("Digite sua pergunta aqui..."):
        # Adicionar mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Gerar resposta
        if rag.qa_chain:
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    answer, sources = rag.query(prompt)
                    
                    if answer:
                        st.markdown(answer)
                        
                        # Mostrar fontes
                        if sources:
                            with st.expander("📚 Fontes consultadas"):
                                for i, source in enumerate(sources, 1):
                                    st.write(f"**Fonte {i}:** {source.metadata.get('source', 'Desconhecido')} - Página {source.metadata.get('page', 'N/A')}")
                        
                        # Adicionar resposta ao histórico
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        st.error("Não foi possível gerar uma resposta.")
        else:
            with st.chat_message("assistant"):
                st.warning("⚠️ Por favor, inicialize o sistema primeiro usando o botão na barra lateral.")
    
    # Informações de ajuda
    with st.expander("❓ Como usar este sistema"):
        st.markdown("""
        <div class="info-box">
        
        ### 🚀 Início Rápido:
        
        1. **Configure sua API Key**: Adicione sua chave da OpenAI no arquivo `.env`
        2. **Carregue Documentos**: Use a barra lateral para fazer upload de PDFs
        3. **Processe os Documentos**: Clique em "Processar Documentos"
        4. **Inicialize o Sistema**: Clique em "Inicializar Sistema"
        5. **Faça Perguntas**: Digite suas perguntas no chat
        
        ### 💡 Dicas:
        
        - O sistema mantém o contexto da conversa
        - Você pode fazer perguntas de acompanhamento
        - As fontes são sempre citadas para transparência
        - Ajuste a temperatura para controlar a criatividade das respostas
        
        ### 🎯 Casos de Uso:
        
        - **Advocacia**: Consulta rápida a contratos e jurisprudências
        - **Desenvolvimento**: Acesso a documentação técnica e procedimentos
        - **RH**: Informações sobre políticas e benefícios
        - **Financeiro**: Análise de relatórios e demonstrativos
        
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()