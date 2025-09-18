import sys
import os
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

# Agora você pode importar o módulo
from ingest_data import inicializar_bedrock_client


CHROMA_PATH = 'chroma_db' 

bedrock_client = inicializar_bedrock_client()

def busca_por_similaridade(pergunta_do_usuario):

   

    modelo_embedding = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v1"
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=modelo_embedding
    )


    resultados_similares = db.similarity_search(pergunta_do_usuario, k=3)

    return resultados_similares


def gera_resposta(resultados_similares, pergunta_do_usuario):
    contexto = "\n\n".join([doc.page_content for doc in resultados_similares])

    prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Você é um assistente útil e amigável. Use apenas o seguinte contexto para responder à pergunta.
    Se a resposta não estiver no contexto, diga que não sabe.

    Contexto:
    {context}

    ---

    Pergunta: {question}
    Resposta:
    """
)
    
    prompt_completo = prompt_template.format(context=contexto, question=pergunta_do_usuario)

    llm = ChatBedrock(
    client=bedrock_client,
    model_id="amazon.titan-text-express-v1",
    region_name="us-east-1" # Exemplo de um modelo de chat
    )
    resposta_do_bot = llm.invoke(prompt_completo) 

    return resposta_do_bot





    


    