import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = 'dataset/'

def carregar_e_dividir_documentos():
    # carrega os documentos pdf da pasta 'dataset' e os divide em chunks
    
    print(f"⏳ Carregando documentos da pasta: {DATA_PATH}")

    documentos = []
    for nome_arquivo in os.listdir(DATA_PATH):
        if nome_arquivo.endswith('.pdf'):
            caminho_completo = os.path.join(DATA_PATH, nome_arquivo)
            print(f"⏳ Processando o arquivo: {nome_arquivo}")
            loader = PyPDFLoader(caminho_completo)
            documentos.extend(loader.load())

    if not documentos:
        print("❌ Nenhum pdf encontrado na pasta 'dataset'!")
        return

    print(f"✅ {len(documentos)} páginas de documentos carregadas com sucesso.")

    # dividir os documentos em pedaços
    print("⏳ Dividindo documentos em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documentos)

    print(f"✅ Total de {len(chunks)} chunks criados!")

    # imprimir um exemplo para verificação
    if chunks:
        print("\n--- Exemplo de um Chunk ---")
        print(chunks[0].page_content)
        print("--------------------------")

    return chunks

# bloco para o script seja executado
if __name__ == '__main__':
    carregar_e_dividir_documentos()