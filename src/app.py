from flask import Flask, request, jsonify
import logging
from src.chatbot import busca_por_similaridade, gera_resposta

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Define a endpoint tipo POST que vai receber as mensagens
@app.route('/webhook', methods=['POST'])
def telegram_webhook():
    # Pega os dados json enviado
    update = request.get_json()
    app.logger.info('Recebido um update: %s', update)

    if 'message' in update and 'text' in update['message']:
        chat_id = update['message']['chat']['id']
        texto_do_usuario = update['message']['text']
        app.logger.info(f'Chat ID: {chat_id}, Mensagem: "{texto_do_usuario}"')

        # pega o documento similar do ChromaDB
        resultados_similares = busca_por_similaridade(texto_do_usuario)

        # gera a resposta usando o Bedrock
        resposta_bedrock = gera_resposta(resultados_similares, texto_do_usuario)
        
        # extrai o conteúdo da resposta
        reposta_do_bot = resposta_bedrock.content
        app.logger.info(f'Resposta gerada: "{reposta_do_bot}"')

        # a resposta para o webhook do telegram é um json instruindo o que fazer, então preciso dizer para o telegram enviar a minha resposta de volta para o chat de onde a mensagem veio
        response_data = {
            'method': 'sendMessage',
            'chat_id': chat_id,
            'text': reposta_do_bot
        }
        
        return jsonify(response_data)
    
    # vou garantir pelo menos que retorne 200 OK
    return 'OK', 200

# vou manter apenas para meus testes, mas vou remover posteriormente
if __name__ == '__main__':
    app.run(debug=True, port=5000)