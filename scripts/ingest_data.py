import os
import boto3
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import Chroma

# carrega as variáveis de ambiente do arquivo .env
load_dotenv()

DATA_PATH = 'dataset/'
CHROMA_PATH = '/app/chroma_db' # aqui é onde o bd vetorial será salvo

def inicializar_bedrock_client():
    # inicializa e retorna o cliente do Bedrock Runtime
    print('⏳ Inicializando cliente Bedrock...')
    # pega a região a partir do arquivo .env para mais flexibilidade
    region_name = os.getenv('AWS_REGION_NAME')
    
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name=region_name,
    )
    print('✅ Cliente Bedrock inicializado com sucesso.')
    return bedrock_client

def carregar_e_dividir_documentos(bedrock_client):
    # carrega, processa os documentos e salva no ChromaDB
    
    print(f'⏳ Carregando documentos da pasta: {DATA_PATH}')

    documentos = []
    for nome_arquivo in os.listdir(DATA_PATH):
        if nome_arquivo.endswith('.pdf'):
            caminho_completo = os.path.join(DATA_PATH, nome_arquivo)
            print(f'⏳ Processando o arquivo: {nome_arquivo}')
            loader = PyPDFLoader(caminho_completo)
            documentos.extend(loader.load())

    if not documentos:
        print('❌ Nenhum pdf encontrado na pasta "dataset"!')
        return None, None

    print(f'✅ {len(documentos)} páginas de documentos carregadas com sucesso.')

    # dividir os documentos em pedaços
    print('⏳ Dividindo documentos em chunks...')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documentos)

    print(f'✅ Total de {len(chunks)} chunks criados!')

    # gerar os embeddings com o Amazon Bedrock
    print('⏳ Gerando embeddings com o Amazon Bedrock...')
    modelo_embedding = BedrockEmbeddings(
        client=bedrock_client,
        model_id='amazon.titan-embed-text-v1'
    )
    print('✅ Modelo de embedding criado com sucesso.')

    # Salva os chunks e embeddings no ChromaDB
    print(f'⏳ Salvando chunks e embeddings no ChromaDB em: {CHROMA_PATH}...')
    # A função from_documents já calcula os embeddings para cada chunk e os salva
    db = Chroma.from_documents(
        documents=chunks,
        embedding=modelo_embedding,
        persist_directory=CHROMA_PATH
    )
    print(f'✅ {len(chunks)} chunks salvos com sucesso no ChromaDB.')

if __name__ == '__main__':
    cliente_bedrock = inicializar_bedrock_client()
    if cliente_bedrock:
        carregar_e_dividir_documentos(cliente_bedrock)