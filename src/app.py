from flask import Flask, request, jsonify
import logging
from .chatbot import gera_resposta
import os, requests

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
TOKEN = os.getenv('TELEGRAM_BOT_API_KEY')
TELEGRAM_API_URL = f'https://api.telegram.org/bot{TOKEN}/sendMessage'

# Define a endpoint tipo POST que vai receber as mensagens
@app.route('/webhook', methods=['POST'])
def telegram_webhook():
    # Pega os dados json enviado
    update = request.get_json()
    app.logger.info('Recebido um update: %s', update)

    if 'message' in update and 'text' in update['message']:
        # ignora mensagens de bots para evitar loops
        if update['message']['from']['is_bot']:
            return 'OK', 200

        chat_id = update['message']['chat']['id']
        texto_do_usuario = update['message']['text']
        app.logger.info(f'Chat ID: {chat_id}, Mensagem: "{texto_do_usuario}"')
        
        # a função gera_resposta já cuida de tudo
        resposta_do_bot = gera_resposta(texto_do_usuario, chat_id)
        app.logger.info(f'Resposta gerada: "{resposta_do_bot}"')

        # monta o payload para a API do Telegram
        payload = {
            'chat_id': chat_id,
            'text': resposta_do_bot
        }

        # envia a mensagem fazendo uma nova requisição
        requests.post(TELEGRAM_API_URL, json=payload)

        # envia a resposta para o Telegram
        return 'OK', 200
    
        # # a resposta para o webhook do telegram é um json instruindo o que fazer, então preciso dizer para o telegram enviar a minha resposta de volta para o chat de onde a mensagem veio
        # response_data = {
        #     'method': 'sendMessage',
        #     'chat_id': chat_id,
        #     'text': resposta_do_bot
        # }
        
        # return jsonify(response_data)
    
    # vou garantir pelo menos que retorne 200 OK
    return 'OK', 200

# vou manter apenas para meus testes, mas vou remover posteriormente
if __name__ == '__main__':
    app.run(debug=True, port=5000)