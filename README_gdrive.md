# 🤖 Sistema RAG com Integração ao Google Drive

Esta é uma versão do Sistema RAG que se conecta diretamente a uma pasta do seu Google Drive para ler documentos PDF, em vez de exigir o upload manual.

## 🌟 Funcionalidade Adicional

- **Conexão com Google Drive**: Busca e processa arquivos PDF diretamente de uma pasta especificada no Google Drive.
- **Autenticação OAuth 2.0**: Fluxo de autenticação seguro para permitir que a aplicação acesse seus arquivos em modo de apenas leitura.

---

## 🚀 Instalação e Execução

O processo é semelhante ao original, mas com um passo crucial de configuração da API do Google.

### 1. Pré-requisitos

- Python 3.8+
- Chave de API da OpenAI
- Uma conta Google

### 2. Clone o Repositório e Instale as Dependências

```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd <NOME_DO_DIRETORIO>

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows

# Instale as dependências específicas desta versão
pip install -r requirements_gdrive.txt
```

### 3. Configure a API da OpenAI

Crie um arquivo `.env` na raiz do projeto e adicione sua chave:

```env
OPENAI_API_KEY="sua-chave-secreta-da-openai-aqui"
```

### 4. Configure a API do Google Drive (Passo Crucial)

Você precisa habilitar a API do Google Drive e obter credenciais de autenticação.

1.  **Acesse o Google Cloud Console**: Vá para https://console.cloud.google.com/.
2.  **Crie um novo projeto** (ou selecione um existente).
3.  **Habilite a API do Google Drive**:
    - No menu de navegação, vá para **APIs e serviços > Biblioteca**.
    - Procure por "Google Drive API" e clique em **Ativar**.
4.  **Configure a Tela de Consentimento OAuth**:
    - Vá para **APIs e serviços > Tela de consentimento OAuth**.
    - Escolha **Externo** e clique em **Criar**.
    - Preencha as informações obrigatórias (nome do app, e-mail de suporte). Não precisa enviar para verificação para uso pessoal. Salve e continue.
    - Na tela de "Escopos", não adicione nada. Salve e continue.
    - **❗️ Passo Crítico: Adicionar Usuário de Teste**
      - Na seção "Usuários de teste", clique em **+ ADD USERS**.
      - Adicione o endereço de e-mail da conta Google que você usará para fazer login no aplicativo. Este é o passo que resolve o erro "access_denied".
      - Salve e continue.
5.  **Crie as Credenciais**:
    - Vá para **APIs e serviços > Credenciais**.
    - Clique em **+ CRIAR CREDENCIAIS** e selecione **ID do cliente OAuth**.
    - Em "Tipo de aplicativo", selecione **App para computador**.
    - Dê um nome para a credencial e clique em **Criar**.
6.  **Baixe o JSON**:
    - Uma janela pop-up aparecerá com seu ID de cliente. Clique em **FAZER O DOWNLOAD DO JSON**.
    - **Renomeie o arquivo baixado para `credentials.json`** e coloque-o na raiz do seu projeto.

### 5. Execute a Aplicação

```bash
streamlit run app_gdrive.py
```

Na primeira vez que você executar, uma aba do navegador será aberta pedindo para você fazer login com sua conta Google e autorizar o acesso. Após a autorização, a aplicação funcionará normalmente.

### 6. Como Usar

1.  **Encontre a Pasta**: Abra o Google Drive no navegador e navegue até a pasta desejada.
2.  **Copie a URL ou o ID**: Você pode copiar a URL inteira da barra de endereços (ex: `https://drive.google.com/drive/folders/ID_DA_PASTA_AQUI`) ou apenas o ID da pasta.
3.  **Cole na Aplicação**: Na barra lateral da aplicação, cole a URL ou o ID no campo "ID ou URL da Pasta do Google Drive".
3.  **Processe**: Clique em "Processar Documentos do Drive".
4.  **Inicialize e Converse**: Após o processamento, clique em "Inicializar Sistema" e comece a fazer suas perguntas.

---