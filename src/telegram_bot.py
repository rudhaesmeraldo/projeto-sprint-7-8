import logging
from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters
import os
from dotenv import load_dotenv

from chatbot import busca_por_similaridade, gera_resposta

load_dotenv()
# Configura o log para ver o que está acontecendo
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Substitua com o seu token de acesso
TOKEN = os.environ.get("TELEGRAM_BOT_API_KEY")

async def responder(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Função que será chamada para qualquer mensagem de texto."""
    # O texto que o usuário enviou
    texto_do_usuario = update.message.text

    resposta_bedrock = gera_resposta(resultados_similares=busca_por_similaridade(texto_do_usuario),
                              pergunta_do_usuario=texto_do_usuario)
    
    reposta_do_bot = resposta_bedrock.content

    
    # Simplesmente responde o que o usuário disse
    await update.message.reply_text(reposta_do_bot)

def main() -> None:
    """Inicia o bot."""
    application = Application.builder().token(TOKEN).build()

    # Adiciona um MessageHandler que filtra apenas mensagens de TEXTO
    # Ele chama a função 'responder' para qualquer mensagem que contenha texto.
    application.add_handler(MessageHandler(filters.TEXT, responder))

    # Executa o bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()