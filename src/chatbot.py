import sys
import os
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

from ingest_data import inicializar_bedrock_client

CHROMA_PATH = '/app/chroma_db'

bedrock_client = inicializar_bedrock_client()

print('⏳ Carregando modelos e banco de dados na memória...')

# carrega o modelo de embeddings que traduz texto para vetores
modelo_embedding = BedrockEmbeddings(
    client=bedrock_client,
    model_id='amazon.titan-embed-text-v1'
)

#carrega a base de dados vetorial que já foi criada pelo script ingest
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=modelo_embedding
)

# carrega o llm que vai gerar as respostas
llm = ChatBedrock(
    client=bedrock_client,
    model_id='amazon.titan-text-express-v1'
)
print('✅ Componentes do chatbot prontos.')

# dicionário para armazenar as instâncias de memória por id do chat
chat_memories = {}

def busca_por_similaridade(pergunta_do_usuario):
    resultados_similares = db.similarity_search(pergunta_do_usuario, k=5)
    return resultados_similares


def gera_resposta(pergunta_do_usuario, chat_id):
    # verifica se o id do chat não está em memorias, cria uma nova.
    if chat_id not in chat_memories:
        chat_memories[chat_id] = ConversationBufferMemory(
            memory_key='chat_history',
            input_key='question',
            return_messages=True
        )

    prompt_template = """
    Você é um assistente especializado em analisar documentos jurídicos. Sua tarefa é responder às perguntas do usuário de forma completa e detalhada, utilizando exclusivamente as informações contidas no <contexto> e no <historico_conversa> abaixo.

    # REGRAS IMPORTANTES:
    1. Justifique sempre a sua resposta com base direta no texto do contexto.
    2. Se a informação necessária para responder à pergunta não estiver explicitamente no contexto, afirme educadamente que a informação não foi encontrada nos documentos fornecidos. NÃO tente adivinhar ou usar conhecimento externo.
    3. Formate a resposta usando parágrafos para facilitar a leitura. Destaque termos jurídicos importantes em **negrito**.

    <historico_conversa>
    {chat_history}
    </historico_conversa>

    <contexto>
    {context}
    </contexto>

    ---

    <pergunta>
    {question}
    </pergunta>

    Resposta detalhada:
    """
    
    PROMPT_DO_USUARIO = PromptTemplate(
        input_variables=["chat_history", "context", "question"], 
        template=prompt_template
    )
    
    # cria o retriever com o novo valor de 'k' para buscar mais documentos
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    # cria a cadeia de conversa com o retriever
    cadeia_conversa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=chat_memories[chat_id],
        combine_docs_chain_kwargs={"prompt": PROMPT_DO_USUARIO}
    )

    resposta = cadeia_conversa.invoke({"question": pergunta_do_usuario})
    
    return resposta['answer']

