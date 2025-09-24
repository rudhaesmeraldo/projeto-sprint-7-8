# 📖 Chatbot Jurídico com AWS Bedrock e LangChain

![Python](https://img.shields.io/badge/Python-3.10-blue) ![AWS EC2](https://img.shields.io/badge/AWS%20EC2-cloud-orange) ![AWS S3](https://img.shields.io/badge/AWS%20S3-storage-yellow) ![AWS Bedrock](https://img.shields.io/badge/AWS%20Bedrock-GenerativeAI-purple) ![LangChain](https://img.shields.io/badge/LangChain-framework-green) ![ChromaDB](https://img.shields.io/badge/ChromaDB-vectorDB-lightgrey) ![Telegram Bot](https://img.shields.io/badge/Telegram%20Bot-chat-blueviolet) ![Boto3](https://img.shields.io/badge/Boto3-AWS%20SDK-orange) ![PyPDF](https://img.shields.io/badge/PyPDF-PDF%20Processing-red)

> Chatbot jurídico que responde perguntas usando documentos PDF, AWS Bedrock, LangChain e ChromaDB.

---

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

A solução foi implementada utilizando um **API Gateway** como ponto de entrada (endpoint) para receber notificações de webhook do Telegram. Esse gateway repassa as requisições para uma aplicação servidora rodando em uma instância **Amazon EC2**, que centraliza toda a lógica de negócio.

Os componentes principais são:

- **Interface do Usuário:**
  - **Telegram:** Plataforma de mensagens utilizada para a interação do usuário com o chatbot.

- **Computação e Lógica da Aplicação:**
  - **Amazon API Gateway:** Atua como front-end da aplicação, recebendo as chamadas da API do Telegram de forma segura e gerenciável.
  - **Amazon EC2:** Instância que hospeda a aplicação principal em Python (servidor Flask/Gunicorn). Ela gerencia todo o fluxo de processamento do RAG e a lógica de conversação.
  - **LangChain:** Framework utilizado para orquestrar toda a lógica do RAG, conectando a base de conhecimento, os modelos de IA e a memória da conversa.

- **Base de Conhecimento (RAG):**
  - **Amazon S3:** Bucket utilizado para armazenar de forma permanente os documentos jurídicos originais em formato PDF.
  - **ChromaDB:** Banco de dados vetorial que armazena os *embeddings* dos documentos. É a base de conhecimento que o LangChain consulta para encontrar os trechos de texto relevantes para a pergunta do usuário.
  - **Amazon Bedrock:** Serviço de IA Generativa da AWS, utilizado para duas finalidades:
    1.  **Geração de Embeddings:** Com o modelo `amazon.titan-embed-text-v1`, para converter os textos dos documentos e as perguntas dos usuários em vetores numéricos.
    2.  **Geração de Texto:** Com o modelo `amazon.titan-text-premier-v1:0`, para formular as respostas finais com base no contexto recuperado.

- **Monitoramento:**
  - **Amazon CloudWatch:** Serviço utilizado para armazenar e visualizar os logs gerados pela aplicação, permitindo o monitoramento da atividade do chatbot.

  ### Diagrama do Fluxo do Chatbot
  
<p align="center">
  <img src="diagrama_flow.png" alt="Diagrama do Chatbot" width="600"/>
</p>
  
  > Este diagrama representa o fluxo de dados entre o usuário, EC2, ChromaDB, Bedrock e CloudWatch.
  
  ---

---

### 3. Tecnologias Utilizadas

- **Linguagem:** Python 3.10
- **Cloud:** AWS (EC2, S3, Bedrock, CloudWatch, API Gateway, IAM)
- **Frameworks de IA:** LangChain, LangChain AWS
- **Banco de Dados Vetorial:** ChromaDB
- **Servidor Web:** Flask, Gunicorn
- **Interface:** Python-Telegram-Bot
- **Bibliotecas Principais:** Boto3, PyPDF, Python-dotenv

---

### 4. Como Executar o Sistema

Siga os passos abaixo para configurar e executar o projeto.

#### Pré-requisitos
- Conta na AWS com permissões para criar e gerenciar EC2, S3, IAM Roles, API Gateway e Bedrock.
- Python 3.10 ou superior instalado.
- Um bot criado no Telegram para obter o Token de acesso.

#### a. Configuração do Ambiente AWS
1.  **Bucket S3:** Crie um bucket no S3 e faça o upload dos seus documentos `.pdf`.
2.  **IAM Role:** Crie uma IAM Role para a instância EC2 com as seguintes políticas gerenciadas pela AWS:
    - `AmazonS3ReadOnlyAccess`
    - `AmazonBedrockFullAccess`
    - `CloudWatchLogsFullAccess`
3.  **Instância EC2:** Lance uma instância EC2 (ex: `t2.micro` com Ubuntu Server), anexe a IAM Role criada e configure um Security Group para permitir tráfego de entrada na porta `5000`.
4.  **API Gateway:** Crie um API Gateway com uma rota (ex: `/webhook`, método `POST`) e configure a integração para apontar para o IP público e a porta da sua instância EC2.

#### b. Configuração do Projeto na EC2
1.  Conecte-se à sua instância EC2 via SSH.
2.  Clone o repositório:
    ```bash
    git clone [URL_DO_REPOSITORIO]
    cd [NOME_DA_PASTA_DO_REPOSITORIO]
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
5.  Crie e configure o arquivo de variáveis de ambiente (`.env`):
    ```env
    TELEGRAM_BOT_API_KEY=[TOKEN_DO_TELEGRAM]
    S3_BUCKET_NAME=[NOME_DO_BUCKET_S3]
    AWS_REGION_NAME=us-east-1
    ```

#### c. Execução do Chatbot
1.  **Ingestão de Dados:** Execute o script de ingestão para processar os documentos do S3 e criar a base de dados no ChromaDB.
    ```bash
    python scripts/ingest_data.py
    ```
2.  **Iniciar o Servidor:** Após a conclusão da ingestão, inicie o servidor Gunicorn para que a aplicação comece a receber requisições do API Gateway.
    ```bash
    gunicorn --bind 0.0.0.0:5000 src.app:app
    ```

---

### 5. Acesso ao Chatbot

Para interagir com o chatbot, acesse o link público do bot no Telegram.
**Link:** `[URL_DO_SEU_BOT_NO_TELEGRAM]`

---

### 6. Dificuldades e Soluções

- **Decisão Arquitetural:** O projeto apresentava uma aparente contradição entre o diagrama de arquitetura (que incluía o API Gateway) e o texto (que sugeria gerenciamento 100% na EC2). A solução foi seguir o diagrama como guia principal, utilizando o API Gateway como endpoint e a EC2 como o host da lógica de negócio, interpretando "gerenciamento" como o processamento da aplicação e não o ponto de entrada da rede.

- **Gerenciamento de Credenciais AWS:** A utilização de uma **IAM Role** anexada à instância EC2 foi uma solução de segurança e boas práticas fundamental, eliminando a necessidade de armazenar credenciais de acesso em arquivos de configuração e permitindo que a aplicação acesse os serviços da AWS de forma segura.

---

### 7. Autores

Este projeto foi desenvolvido pela **Squad 6** como parte do programa de bolsas da Compass UOL.

* [Agnes Ludmilla](https://github.com/agnesludmila)
* [Rafaela Bezerra](https://github.com/Rafa01B)
* [Rudhá Esmeraldo](https://github.com/rudhaesmeraldo)
* [Yuri Kiev](https://github.com/YuriKievBarreto)