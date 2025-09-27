import os
import boto3
from dotenv import load_dotenv

load_dotenv()

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