# imagem base -> seguindo a recomendação do instrutor Allan.
FROM python:3.10-slim

# diretório de Trabalho
WORKDIR /app

# copiar arquivos
COPY requirements.txt .

# instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# copiar o restante do código
COPY ./src ./src
COPY ./scripts ./scripts

# comando padrão, posteriormente vou alterar este comando para iniciar o bot do Telegram
CMD ["python", "-c", "print('Ambiente para o chatbot está funcionando!!')"]