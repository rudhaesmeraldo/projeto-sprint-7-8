from flask import Flask, request
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Define a endpoint tipo POST que vai receber as mensagens
@app.route('/webhook', methods=['POST'])
def telegram_webhook():
    # Pega os dados json enviado
    update = request.get_json()

    # logging para ver o que recebemos
    app.logger.info('Recebido um update: %s', update)

    # ainda será necessário chamar a lógica do chatbot
    
    return 'OK', 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)