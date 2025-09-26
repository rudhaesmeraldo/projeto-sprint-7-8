import sys
import os
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain.memory import ConversationBufferWindowMemory
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

# lista de saudações para identificar interações
SAUDACOES  = ['olá', 'oi', 'bom dia', 'boa tarde', 'boa noite', 'saudações', 'e aí', 'fala', 'Opa', 'tudo bem', 'como vai', 'tudo certo', 'tudo em paz', 'salve']

# dicionário para armazenar as instâncias de memória por id do chat
chat_memories = {}

def gera_resposta(pergunta_do_usuario, chat_id):
    # trata a mensagem do usuário para facilitar
    input_minusculo = pergunta_do_usuario.lower()

    # aqui eu estou criando um 'roteador', vou identificar se a mensagem é uma saudação
    if any(saudacao in input_minusculo for saudacao in SAUDACOES):
        return 'Olá! Sou seu assistente jurídico. Manda a boa de hoje?'

    # verifica se o id do chat não está em memorias, cria uma nova.
    if chat_id not in chat_memories:
        chat_memories[chat_id] = ConversationBufferWindowMemory(
            k=4,
            memory_key='chat_history',
            input_key='question',
            return_messages=True
        )

    prompt_template = """
    Você é um assistente de análise jurídica altamente especializado. Sua principal função é extrair informações precisas e responder perguntas com base exclusivamente no <contexto> de documentos judiciais fornecido.

    # ESTRUTURA DOS DOCUMENTOS:
    O <contexto> pode conter diferentes tipos de documentos, como Acórdãos, Votos, Petições (Recursos, Agravos) e Ementas. Esteja atento às seções como "RELATÓRIO" (descreve o caso), "VOTO" (apresenta a decisão do juiz/ministro), "EMENTA" (resume a decisão) e "DISPOSITIVO" (a conclusão final do julgamento).

    # REGRAS CRÍTICAS DE OPERAÇÃO:
    1.  **Diferencie Fatos de Decisões:** Ao responder, sempre diferencie os argumentos das partes (o que um advogado alegou) da **decisão final do tribunal** (o que o juiz ou a turma decidiu). Se a pergunta for sobre um "entendimento firmado", "decisão" ou "julgamento", sua resposta DEVE se basear nas seções "VOTO", "EMENTA" ou "DISPOSITIVO".
    2.  **Base Exclusiva no Contexto:** Justifique todas as suas respostas citando ou se baseando diretamente no texto fornecido no <contexto>. Não utilize nenhum conhecimento externo.
    3.  **Seja Preciso sobre a Fonte:** Se a informação estiver em um voto vencido, mencione isso. Exemplo: "No voto vencido, o entendimento foi...".
    4.  **Informação Ausente:** Se a resposta não puder ser encontrada no <contexto>, afirme claramente: "A informação solicitada não foi encontrada nos documentos fornecidos."
    5.  **Formatação:** Responda em parágrafos claros. Destaque em **negrito** os termos jurídicos mais importantes, nomes de recursos (ex: **Recurso Extraordinário**) ou artigos de lei.

    <contexto>
    {context}
    </contexto>

    ---

    <pergunta>
    {question}
    </pergunta>

    Resposta Jurídica Detalhada:
    """
    
    PROMPT_DO_USUARIO = PromptTemplate(
        input_variables=["context", "question"], 
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

