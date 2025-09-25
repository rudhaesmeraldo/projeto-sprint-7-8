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

# expõe a porta que o guinicorn vai usar dentro o container 
EXPOSE 5000

CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:5000", "src.app:app"]