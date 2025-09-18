import sys
import os
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import Chroma

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

# Agora você pode importar o módulo
from ingest_data import inicializar_bedrock_client


CHROMA_PATH = 'chroma_db' 


def busca_por_similaridade(pergunta_do_usuario):

    bedrock_client = inicializar_bedrock_client()

    modelo_embedding = BedrockEmbeddings(
    client=bedrock_client,
    model_id='amazon.titan-embed-text-v1'
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=modelo_embedding
    )


    resultados_similares = db.similarity_search(pergunta_do_usuario, k=3)

    return resultados_similares


    



    