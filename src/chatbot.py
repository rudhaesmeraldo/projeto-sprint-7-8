import sys
import os
import re
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
    model_id="mistral.mistral-large-2402-v1:0"
)
print('✅ Componentes do chatbot prontos.')

# lista de saudações para identificar interações
SAUDACOES_KEYWORDS = ['olá', 'oi', 'bom dia', 'boa tarde', 'boa noite', 'saudações', 'e aí', 'fala', 'Opa', 'tudo bem', 'como vai', 'tudo certo', 'tudo em paz', 'salve']

# dicionário para armazenar as instâncias de memória por id do chat
chat_memories = {}

def gera_resposta(pergunta_do_usuario, chat_id):
    # trata a mensagem do usuário para facilitar
    input_minusculo = pergunta_do_usuario.lower()
    
    # Remove pontuações para isolar as palavras
    input_limpo = re.sub(r'[^\w\s]', '', input_minusculo)
    palavras = input_limpo.split()

    # Considera uma saudação se a mensagem for curta e se contiver alguma das palavras chave
    is_greeting = len(palavras) <= 4 and any(p in SAUDACOES_KEYWORDS for p in palavras)

    if is_greeting:
        return "Olá! Sou seu assistente jurídico. Manda a boa de hoje?"

    # verifica se o id do chat não está em memorias, cria uma nova.
    if chat_id not in chat_memories:
        chat_memories[chat_id] = ConversationBufferWindowMemory(
            k=4,
            memory_key='chat_history',
            input_key='question',
            return_messages=True
        )

    prompt_template = """
    Você é um jurista-analista de IA, um especialista em dissecar acórdãos e processos judiciais. Sua precisão é cirúrgica. Você NUNCA alucina ou inventa informações. Sua resposta deve ser 100% baseada no <contexto> fornecido.

    # PROCESSO MENTAL OBRIGATÓRIO (SIGA ANTES DE RESPONDER):
    1.  **Análise da Pergunta:** Leia a <pergunta> e identifique os pontos-chave que o usuário quer saber (ex: "qual foi a alegação?", "por que foi negado?", "qual o fundamento legal?").
    2.  **Busca por Alegações:** Vasculhe o <contexto> em busca dos argumentos e alegações das partes envolvidas (o que os advogados afirmaram ou pediram).
    3.  **Busca pela Decisão do Tribunal:** Em seguida, localize os trechos que contêm a **decisão final do tribunal** sobre essas alegações. Preste atenção máxima às seções "VOTO", "EMENTA" e "DISPOSITIVO", pois elas contêm a conclusão do julgador.
    4.  **Síntese e Justificativa:** Compare as alegações das partes (passo 2) com a decisão do tribunal (passo 3). Extraia o motivo exato pelo qual a alegação foi aceita ou rejeitada, citando os fundamentos legais mencionados no texto, como artigos de lei ou súmulas.
    5.  **Formulação da Resposta:** Com base na sua análise (passo 4), construa a "Resposta Jurídica Detalhada" de forma clara, objetiva e estruturada.

    # REGRAS CRÍTICAS DE RESPOSTA:
    - **Base Exclusiva no Contexto:** Se a informação para responder a qualquer parte da pergunta não estiver explicitamente no <contexto>, afirme claramente: "A informação sobre [ponto específico] não foi encontrada nos documentos fornecidos."
    - **Diferencie Fatos de Decisões:** Deixe sempre claro o que é um argumento de uma parte e o que é a decisão do tribunal.
    - **Formatação:** Responda em parágrafos claros. Destaque em **negrito** termos jurídicos, nomes de recursos ou artigos de lei.

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
    retriever = db.as_retriever(search_kwargs={"k": 8})
    
    # cria a cadeia de conversa com o retriever
    cadeia_conversa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=chat_memories[chat_id],
        combine_docs_chain_kwargs={"prompt": PROMPT_DO_USUARIO}
    )

    resposta = cadeia_conversa.invoke({"question": pergunta_do_usuario})
    
    return resposta['answer']