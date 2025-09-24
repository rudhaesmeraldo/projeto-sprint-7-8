# Chatbot Jurídico com AWS Bedrock e LangChain - Projeto 4

### Squad 6

## Sumário
1. [Sobre o Projeto](#1-sobre-o-projeto)
2. [Arquitetura da Solução](#2-arquitetura-da-solução)
3. [Tecnologias Utilizadas](#3-tecnologias-utilizadas)
4. [Como Executar o Sistema](#4-como-executar-o-sistema)
5. [Acesso ao Chatbot](#5-acesso-ao-chatbot)
6. [Dificuldades e Soluções](#6-dificuldades-e-soluções)
7. [Autores](#7-autores)

---

### 1. Sobre o Projeto

Este projeto foi desenvolvido como parte do programa de bolsas da Compass UOL para formação em Inteligência Artificial para AWS (Sprints 7 e 8). O objetivo é construir um chatbot funcional para consulta de documentos jurídicos, utilizando uma arquitetura de **Geração Aumentada por Recuperação (RAG)**.

O chatbot é capaz de responder a perguntas em linguagem natural com base em uma coleção de documentos PDF previamente carregados. Ele utiliza os serviços de IA Generativa da AWS, orquestrados pela biblioteca LangChain, para encontrar trechos relevantes nos documentos e formular respostas coesas e precisas, evitando o uso de conhecimento externo.

---

### 2. Arquitetura da Solução

A solução foi implementada seguindo o requisito de ter todo o gerenciamento centralizado em uma instância **Amazon EC2**. A arquitetura de comunicação adotada foi a de **Polling**, onde a aplicação na EC2 é responsável por buscar ativamente novas mensagens na API do Telegram.

Os componentes principais são:

- **Interface do Usuário:**
  - **Telegram:** Plataforma de mensagens utilizada para a interação do usuário com o chatbot.

- **Computação e Lógica da Aplicação:**
  - **Amazon EC2:** Instância `t2.micro` que hospeda a aplicação principal em Python. Ela gerencia todo o fluxo, desde a busca por novas mensagens até o processamento e a resposta.
  - **LangChain:** Framework utilizado para orquestrar toda a lógica do RAG, conectando a base de conhecimento, os modelos de IA e a memória da conversa.

- **Base de Conhecimento (RAG):**
  - **Amazon S3:** Bucket utilizado para armazenar de forma permanente os documentos jurídicos originais em formato PDF.
  - **ChromaDB:** Banco de dados vetorial que armazena os *embeddings* dos documentos. É a base de conhecimento que o LangChain consulta para encontrar os trechos de texto relevantes para a pergunta do usuário.
  - **Amazon Bedrock:** Serviço de IA Generativa da AWS, utilizado para duas finalidades:
    1.  **Geração de Embeddings:** Com o modelo `amazon.titan-embed-text-v1`, para converter os textos dos documentos e as perguntas dos usuários em vetores numéricos.
    2.  **Geração de Texto:** Com o modelo `amazon.titan-text-premier-v1:0`, para formular as respostas finais com base no contexto recuperado.

- **Monitoramento:**
  - **Amazon CloudWatch:** Serviço utilizado para armazenar e visualizar os logs gerados pela aplicação, permitindo o monitoramento da atividade do chatbot.

---

### 3. Tecnologias Utilizadas

- **Linguagem:** Python 3.10
- **Cloud:** AWS (EC2, S3, Bedrock, CloudWatch, IAM)
- **Frameworks de IA:** LangChain, LangChain AWS
- **Banco de Dados Vetorial:** ChromaDB
- **Interface:** Python-Telegram-Bot
- **Bibliotecas Principais:** Boto3, PyPDF, Python-dotenv

---

### 4. Como Executar o Sistema

Siga os passos abaixo para configurar e executar o projeto.

#### Pré-requisitos
- Conta na AWS com permissões para criar e gerenciar EC2, S3, IAM Roles e Bedrock.
- Python 3.10 ou superior instalado.
- Um bot criado no Telegram para obter o Token de acesso.

#### a. Configuração do Ambiente AWS
1.  **Bucket S3:** Crie um bucket no S3 e faça o upload dos seus documentos `.pdf`.
2.  **IAM Role:** Crie uma IAM Role para a instância EC2 com as seguintes políticas gerenciadas pela AWS:
    - `AmazonS3ReadOnlyAccess`
    - `AmazonBedrockFullAccess`
    - `CloudWatchLogsFullAccess`
3.  **Instância EC2:** Lance uma instância EC2 (ex: `t2.micro` com Ubuntu Server) e anexe a IAM Role criada no passo anterior.

#### b. Configuração do Projeto na EC2
1.  Conecte-se à a instância EC2 via SSH.
    ```bash
    ssh -i "seu\caminho\para\as\keys\chatbot-keys.pem" ubuntu@54.221.3.162
    ```
2.  Clone o repositório:
    ```bash
    git clone https://github.com/Compass-pb-aws-2025-JUNHO/sprints-7-8-junho/tree/squad-6
    cd sprints-7-8-junho
    ```
3.  Crie e ative um ambiente virtual:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
4.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
5.  Crie e configure o arquivo de variáveis de ambiente. Você pode criar um arquivo `.env` com o seguinte conteúdo:
    ```bash
    nano .env
    ```
    Preencha com suas variáveis (deixe as credenciais da AWS em branco, pois a IAM Role cuidará disso):
    ```env
    TELEGRAM_BOT_API_KEY=8171216067:AAG_qlbquxmHngCmpSNZEj8nxUQnFNJPNSI
    S3_BUCKET_NAME=projeto-chatbot-squad6
    AWS_REGION_NAME=us-east-1
    ```

#### c. Execução do Chatbot
1.  **Ingestão de Dados:** Execute o script de ingestão para processar os documentos do S3 e criar a base de dados no ChromaDB. Este passo só precisa ser executado uma vez.
    ```bash
    python scripts/ingest_data.py
    ```
2.  **Iniciar o Bot:** Após a conclusão da ingestão, inicie o bot. Para mantê-lo rodando em segundo plano, mesmo após fechar a sessão SSH, use `nohup`:
    ```bash
    nohup python src/telegram_bot.py &
    ```

---

### 5. Acesso ao Chatbot

Fale com o bot através do link: **[http://t.me/rag_judicial_bot](http://t.me/rag_judicial_bot)**

---

### 6. Dificuldades e Soluções

- **Decisão Arquitetural (Webhook vs. Polling):** Inicialmente, foi explorada uma arquitetura de webhook com API Gateway por ser um padrão de mercado robusto. No entanto, para aderir estritamente ao requisito do projeto de gerenciamento total na EC2, optou-se pela implementação final com polling, que se mostrou mais simples e alinhada ao escopo.

- **Gerenciamento de Credenciais AWS:** A utilização de uma **IAM Role** anexada à instância EC2 foi uma solução de segurança e boas práticas fundamental, eliminando a necessidade de armazenar credenciais de acesso em arquivos de configuração e permitindo que a aplicação acesse os serviços da AWS de forma segura.

---

### 7. Autores

Este projeto foi desenvolvido pela **Squad 6**:

- Rudhá Esmeraldo de Sousa
- Yuri Kiev de Sousa Barreto
- Agnes Ludmila de Araújo Teixeira
- Rafaela Bezerra Rodrigues