import os
import re
import boto3
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .aws_utils import inicializar_bedrock_client

CHROMA_PATH = '/app/chroma_db'

print('⏳ Carregando componentes do chatbot...')

bedrock_client = inicializar_bedrock_client()

# carrega o modelo de embeddings
modelo_embedding = BedrockEmbeddings(
    client=bedrock_client,
    model_id='amazon.titan-embed-text-v1'
)

# carrega o vector store já persistido pelo script de ingestão
vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=modelo_embedding,
    persist_directory=CHROMA_PATH
)

# recria a estrutura do parent document retriever para poder fazer as buscas. o docstore pode ser recriado vazio, pois a informação principal está no vectorstore
store = InMemoryStore()
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
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
        return 'Olá! Sou seu assistente jurídico. Manda a boa de hoje?'

    # verifica se o id do chat não está em memorias, cria uma nova.
    if chat_id not in chat_memories:
        chat_memories[chat_id] = ConversationBufferWindowMemory(
            k=4,
            memory_key='chat_history',
            input_key='question',
            output_key='answer',
            return_messages=True
        )

    prompt_template = """
    Você é um jurista-analista de IA. Sua precisão é cirúrgica e suas respostas são objetivas e diretas, focando nos fatos e fundamentos mais decisivos.

    # PROCESSO MENTAL OBRIGATÓRIO (SIGA ANTES DE RESPONDER):
    1.  Análise da Pergunta: Leia a <pergunta> e identifique os pontos-chave que o usuário quer saber.
    2.  Busca por Alegações: Vasculhe o <contexto> em busca dos argumentos das partes.
    3.  Busca pela Decisão do Tribunal: Localize os trechos que contêm a decisão final do tribunal sobre essas alegações, focando nas seções "VOTO", "EMENTA" e "DISPOSITIVO".
    4.  Identificação dos Fundamentos: Dentre os motivos da decisão, identifique o fundamento legal principal (ex: a nulidade de um contrato, uma lei específica) e qualquer prova decisiva mencionada (ex: uma confissão em audiência, um documento chave que contradiz uma alegação).
    5.  Formulação da Resposta: Com base nos fundamentos (passo 4), construa a resposta de forma a responder diretamente à pergunta do usuário.

    # REGRAS CRÍTICAS PARA A RESPOSTA:
    - IDIOMA: Responda sempre e exclusivamente em Português do Brasil.
    - Tom Formal e Profissional: Use uma linguagem formal, profissional e técnica, adequada para o contexto jurídico.
    - Seja Conciso e Direto: Responda a pergunta de forma objetiva, focando apenas nas informações essenciais. Evite detalhes desnecessários.
    - Base Exclusiva no Contexto: Se a informação não estiver no <contexto>, afirme claramente: "A informação sobre [ponto específico] não foi encontrada nos documentos fornecidos."
    - Diferencie Fatos de Decisões: Deixe sempre claro o que é um argumento de uma parte e o que é a decisão final do tribunal.

    <contexto>
    {context}
    </contexto>

    ---

    <pergunta>
    {question}
    </pergunta>

    Resposta Jurídica Direta e Fundamentada:
    """
    
    PROMPT_DO_USUARIO = PromptTemplate(
        input_variables=['context', 'question'], 
        template=prompt_template
    )
    
    # cria a cadeia de conversa com o novo retriever
    cadeia_conversa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=chat_memories[chat_id],
        combine_docs_chain_kwargs={'prompt': PROMPT_DO_USUARIO}
    )

    resposta = cadeia_conversa.invoke({'question': pergunta_do_usuario})

    print('\n 📃 DOCUMENTOS RECUPERADOS COMO CONTEXTO')
    if 'source_documents' in resposta and resposta['source_documents']:
        for doc in resposta['source_documents']:
            # pega o nome do arquivo da metadata
            source_file = doc.metadata.get('source', 'N/A').split('/')[-1]
            print(f'FONTE: {source_file}')
            # Imprime os primeiros 300 caracteres do conteúdo para ser breve
            print(f'CONTEÚDO: {doc.page_content[:300]}...\n')
    else:
        print('Nenhum documento foi retornado')
    print('FIM DO DEBUG\n')
    
    return resposta['answer']