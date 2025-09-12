# 🤖 Sistema RAG Empresarial com Streamlit e LangChain

Este é um sistema de **Retrieval-Augmented Generation (RAG)** construído com Streamlit, LangChain e OpenAI. A aplicação permite que os usuários façam upload de documentos PDF, processem esses documentos em uma base de conhecimento vetorial e, em seguida, conversem com um assistente de IA que responde a perguntas com base no conteúdo dos documentos fornecidos.

## ✨ Funcionalidades

- **Interface Web Intuitiva**: Interface de usuário amigável construída com Streamlit.
- **Upload de Documentos**: Suporte para upload de múltiplos arquivos PDF.
- **Processamento e Vetorização**: Os documentos são divididos em chunks, vetorizados usando os modelos de embedding da OpenAI e armazenados em uma base de dados vetorial local (FAISS).
- **Chat Interativo**: Uma interface de chat para fazer perguntas em linguagem natural.
- **Citação de Fontes**: As respostas da IA incluem referências aos documentos e páginas de origem, garantindo transparência e rastreabilidade.
- **Memória de Conversa**: O sistema mantém o contexto do diálogo, permitindo perguntas de acompanhamento.
- **Configuração Flexível**:
  - Seleção de modelos da OpenAI (ex: `gpt-4o-mini`, `gpt-4o`).
  - Ajuste da "temperatura" para controlar a criatividade das respostas.
- **Gerenciamento de Estado**: O estado da aplicação e o histórico do chat são mantidos durante a sessão do usuário.

---

## ⚙️ Tecnologias Utilizadas

- **Frontend**: Streamlit
- **Orquestração de LLM**: LangChain
- **Modelos de Linguagem (LLM)**: OpenAI (GPT-4o, GPT-3.5-Turbo, etc.)
- **Embeddings**: OpenAI (`text-embedding-3-small`)
- **Base de Dados Vetorial**: FAISS (CPU)
- **Processamento de PDF**: PyPDF
- **Gerenciamento de Ambiente**: Python-dotenv

---

## 🚀 Instalação e Execução

Siga os passos abaixo para configurar e executar o projeto em seu ambiente local.

### 1. Pré-requisitos

- Python 3.8 ou superior.
- Uma chave de API da OpenAI.

### 2. Clone o Repositório

```bash
git clone <url-do-seu-repositorio>
cd <nome-do-repositorio>
```

### 3. Crie um Ambiente Virtual (Recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Instale as Dependências

Instale todos os pacotes necessários a partir do arquivo `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 5. Configure a Chave de API

Crie um arquivo chamado `.env` na raiz do projeto e adicione sua chave de API da OpenAI:

```env
OPENAI_API_KEY="sua-chave-secreta-da-openai-aqui"
```

### 6. Execute a Aplicação

Inicie o servidor do Streamlit com o seguinte comando:

```bash
streamlit run app_new.py
```

A aplicação será aberta automaticamente no seu navegador padrão.

---

## 📖 Como Usar

1.  **Carregue os Documentos**: Na barra lateral, clique em "Procurar arquivos" e selecione um ou mais documentos PDF que você deseja adicionar à base de conhecimento.
2.  **Processe os Documentos**: Após o upload, clique no botão **"🔄 Processar Documentos"**. Isso irá ler, dividir e vetorizar o conteúdo dos arquivos.
3.  **Inicialize o Sistema**: Escolha o modelo OpenAI e a temperatura desejada. Em seguida, clique em **"🚀 Inicializar Sistema"**. Isso carrega a base de dados vetorial e prepara o assistente para responder.
4.  **Comece a Conversar**: Com o sistema inicializado, digite suas perguntas na caixa de chat na área principal e pressione Enter. O assistente responderá com base nos documentos fornecidos, citando as fontes.

---

## 📂 Estrutura do Projeto

```
├── app_new.py            # Script principal da aplicação Streamlit
├── requirements.txt      # Lista de dependências Python
├── .env                  # Arquivo para armazenar variáveis de ambiente (API Key)
├── vectorstore/          # Diretório onde a base de dados vetorial FAISS é salva
└── README.md             # Este arquivo
```