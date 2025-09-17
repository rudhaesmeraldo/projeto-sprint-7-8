import os
import boto3
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings

# carrega as variáveis de ambiente do arquivo .env
load_dotenv()

DATA_PATH = 'dataset/'

def inicializar_bedrock_client():
    # inicializa e retorna o cliente do Bedrock Runtime
    print('⏳ Inicializando cliente Bedrock...')
    # pega a região a partir do arquivo .env para mais flexibilidade
    region_name = os.getenv('AWS_REGION_NAME')
    
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name=region_name
    )
    print('✅ Cliente Bedrock inicializado com sucesso.')
    return bedrock_client

def carregar_e_dividir_documentos(bedrock_client):
    # carrega os documentos pdf da pasta 'dataset', os divide em chunks e gera embeddings
    
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

    # exemplo para verificar se a conexão e a geração de embeddings funcionam
    primeiro_chunk_texto = chunks[0].page_content
    embedding_exemplo = modelo_embedding.embed_query(primeiro_chunk_texto)
    
    print('✅ Embeddings gerados com sucesso!')
    print(f'\n--- Exemplo de Vetor de Embedding (primeiros 10 de {len(embedding_exemplo)} valores) ---')
    print(embedding_exemplo[:10])
    print('-----------------------------------------------------------------')

    return chunks, modelo_embedding

# bloco para o script seja executado
if __name__ == '__main__':
    cliente_bedrock = inicializar_bedrock_client()
    if cliente_bedrock:
        carregar_e_dividir_documentos(cliente_bedrock)