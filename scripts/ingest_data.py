import os
import boto3
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import Chroma

# carrega as variáveis de ambiente do arquivo .env
load_dotenv()
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
LOCAL_DATA_PATH = 'temp_dataset' # pasta temporária para onde os arquivo do s3 serão baixados
CHROMA_PATH = '/app/chroma_db'

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
                caminho_destino = os.path.join(pasta_local, key)
                if not os.path.exists(os.path.dirname(caminho_destino)):
                    os.makedirs(os.path.dirname(caminho_destino))
                if not key.endswith('/'):
                    s3_client.download_file(bucket, key, caminho_destino)
        print('✅ Arquivos baixados com sucesso.')
        return True
    except Exception as e:
        print(f'❌ Erro ao baixar arquivos do S3: {e}')
        return False

def inicializar_bedrock_client():
    # inicializa e retorna o cliente do Bedrock Runtime
    print('⏳ Inicializando cliente Bedrock...')
    region_name = os.getenv('AWS_REGION_NAME')
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name=region_name,
    )
    print('✅ Cliente Bedrock inicializado com sucesso.')
    return bedrock_client

def processar_e_salvar_dados(bedrock_client):
    # primeiro, baixa os arquivos do s3
    if not baixar_arquivos_do_s3(S3_BUCKET_NAME, LOCAL_DATA_PATH):
        return # para a execução se o download falhar

    print(f'⏳ Carregando documentos da pasta: {LOCAL_DATA_PATH}')
    documentos = []
    # os.walk() para assim conseguir percorrer todas as pastas e subpastas
    for root, dirs, files in os.walk(LOCAL_DATA_PATH):
        for nome_arquivo in files:
            if nome_arquivo.endswith('.pdf'):
                caminho_completo = os.path.join(root, nome_arquivo)
                print(f'⏳ Processando o arquivo: {caminho_completo}')
                loader = PyPDFLoader(caminho_completo)
                documentos.extend(loader.load())

    if not documentos:
        print('❌ Nenhum pdf encontrado no bucket s3!')
        return

    print(f'✅ {len(documentos)} páginas de documentos carregadas com sucesso.')

    # dividir os documentos em pedaços
    print('⏳ Dividindo documentos em chunks...')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = text_splitter.split_documents(documentos)
    
    print(f'✅ Total de {len(chunks)} chunks criados!')

    # gerar os embeddings com o Amazon Bedrock
    print('⏳ Gerando embeddings com o Amazon Bedrock...')
    modelo_embedding = BedrockEmbeddings(
        client=bedrock_client,
        model_id='amazon.titan-embed-text-v1'
    )
    print('✅ Modelo de embedding criado com sucesso.')

    # salva os chunks e embeddings no ChromaDB
    print(f'⏳ Salvando chunks e embeddings no ChromaDB em: {CHROMA_PATH}...')
    # A função from_documents já calcula os embeddings para cada chunk e os salva
    db = Chroma.from_documents(
        documents=chunks,
        embedding=modelo_embedding,
        persist_directory=CHROMA_PATH
    )

    db.persist() # garante que os arquivos do Chroma sejam gravados
    print(f'✅ {len(chunks)} chunks salvos com sucesso no ChromaDB.')

    # limpa a pasta temporária após a conclusão
    print(f'🧹 A limpar a pasta temporária: {LOCAL_DATA_PATH}')
    shutil.rmtree(LOCAL_DATA_PATH)
    print('✅ Limpeza concluída.')

if __name__ == '__main__':
    cliente_bedrock = inicializar_bedrock_client()
    if cliente_bedrock:
        processar_e_salvar_dados(cliente_bedrock)