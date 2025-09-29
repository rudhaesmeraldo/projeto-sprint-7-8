import os
import boto3
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from src.aws_utils import inicializar_bedrock_client

# carrega as variáveis de ambiente do arquivo .env
load_dotenv()
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
LOCAL_DATA_PATH = 'temp_dataset' # pasta temporária para onde os arquivo do s3 serão baixados
CHROMA_PATH = 'chroma_db'

def baixar_arquivos_do_s3(bucket, pasta_local):
    # baixa todos os arquivos do bucket s3 para uma pasta temporária
    print(f'⏳ Baixando arquivos do bucket S3: {bucket}...')
    s3_client = boto3.client('s3')
    
    if os.path.exists(pasta_local):
        shutil.rmtree(pasta_local)
    os.makedirs(pasta_local)

    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket)

        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                # garante que o nome do arquivo seja usado como base no destino
                caminho_destino = os.path.join(pasta_local, os.path.basename(key))
                if not key.endswith('/'):
                    s3_client.download_file(bucket, key, caminho_destino)
        print('✅ Arquivos baixados com sucesso.')
        return True
    except Exception as e:
        print(f'❌ Erro ao baixar arquivos do S3: {e}')
        return False

def processar_e_salvar_dados(bedrock_client):
    # primeiro, baixa os arquivos do s3
    if not baixar_arquivos_do_s3(S3_BUCKET_NAME, LOCAL_DATA_PATH):
        return # para a execução se o download falhar

    print(f'⏳ Carregando documentos da pasta: {LOCAL_DATA_PATH}')
    documentos = []
    for nome_arquivo in os.listdir(LOCAL_DATA_PATH):
        if nome_arquivo.endswith('.pdf'):
            caminho_completo = os.path.join(LOCAL_DATA_PATH, nome_arquivo)
            loader = PyPDFLoader(caminho_completo)
            documentos.extend(loader.load())

    if not documentos:
        print('❌ Nenhum pdf encontrado no bucket s3!')
        return

    print(f'✅ {len(documentos)} páginas de documentos carregadas com sucesso.')

    # splitter para os documentos 'pai'
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    # splitter para os 'filhos', que serão usados para a busca de similaridade
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    
    # modelo de embedding
    modelo_embedding = BedrockEmbeddings(client=bedrock_client, model_id='amazon.titan-embed-text-v1')

    # vector store que irá armazenar os embeddings dos 'filhos'
    vectorstore = Chroma(
        collection_name="split_parents", 
        embedding_function=modelo_embedding,
        persist_directory=CHROMA_PATH
    )

    # armazenador em memória para os documentos 'pai'
    store = InMemoryStore()

    print('⏳ Configurando o ParentDocumentRetriever...')
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    print(f'⏳ Adicionando documentos e gerando embeddings (isso pode levar um tempo)...')
    retriever.add_documents(documentos)

    print(f'✅ Base de dados vetorial criada com sucesso em: {CHROMA_PATH}')
    
    # limpa a pasta temporária após a conclusão
    print(f'🧹 Limpando a pasta temporária: {LOCAL_DATA_PATH}')
    shutil.rmtree(LOCAL_DATA_PATH)
    print('✅ Limpeza concluída.')

if __name__ == '__main__':
    cliente_bedrock = inicializar_bedrock_client()
    if cliente_bedrock:
        # apaga a base antiga antes de criar a nova
        if os.path.exists(CHROMA_PATH):
            print(f'🧹 Apagando base de dados antiga em {CHROMA_PATH}...')
            shutil.rmtree(CHROMA_PATH)
        processar_e_salvar_dados(cliente_bedrock)