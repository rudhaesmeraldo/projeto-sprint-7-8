from flask import Flask, request
import logging
from .chatbot import gera_resposta
from .cloudwatch_logger import CloudWatchLogger
import os, requests, json

logging.basicConfig(level=logging.INFO)

app = Flask('__name__')
TOKEN = os.getenv('TELEGRAM_BOT_API_KEY')
TELEGRAM_API_URL = f'https://api.telegram.org/bot{TOKEN}/sendMessage'
LOG_GROUP = os.getenv('CLOUDWATCH_LOG_GROUP', 'ChatbotJuridicoApp')
LOG_STREAM_PREFIX = os.getenv('CLOUDWATCH_LOG_STREAM_PREFIX', 'interacoes')

# instância do meu logger
logger_cw = CloudWatchLogger(LOG_GROUP, LOG_STREAM_PREFIX)

@app.route('/webhook', methods=['POST'])
def telegram_webhook():
    update = request.get_json()
    
    # registro o corpo bruto da requisição que o Telegram me enviou
    logger_cw.log(f'✅ Requisição recebida: {json.dumps(update)}')

    if 'message' in update and 'text' in update['message']:
        if update['message']['from']['is_bot']:
            return 'OK', 200

        chat_id = update['message']['chat']['id']
        texto_do_usuario = update['message']['text']
        
        # registro a pergunta do usuário
        logger_cw.log(f'🙋 Pergunta do chat_id={chat_id}: "{texto_do_usuario}"')
        
        try:
            resposta_do_bot = gera_resposta(texto_do_usuario, chat_id)
            
            # registro a resposta que meu chatbot gerou
            logger_cw.log(f'🤖 Resposta para chat_id={chat_id}: "{resposta_do_bot}"')

            payload = {'chat_id': chat_id, 'text': resposta_do_bot}
            requests.post(TELEGRAM_API_URL, json=payload)

            return 'OK', 200
        except Exception as e:
            # se der algum erro, eu registro o erro
            error_message = f'❌ ERRO no chat_id={chat_id}: {str(e)}'
            logger_cw.log(error_message)
            app.logger.error(error_message)
            
            payload = {'chat_id': chat_id, 'text': 'Desculpe, ocorreu um erro ao processar sua solicitação.'}
            requests.post(TELEGRAM_API_URL, json=payload)

            return 'OK', 200
            
    return 'OK', 200