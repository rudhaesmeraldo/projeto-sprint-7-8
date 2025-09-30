import boto3
import datetime
import os
from dotenv import load_dotenv

# variáveis de ambiente do arquivo .env
load_dotenv()

class CloudWatchLogger:
    def __init__(self, log_group_name, log_stream_name_prefix):
        self.client = boto3.client('logs', region_name=os.getenv('AWS_REGION_NAME'))
        self.log_group = log_group_name

        # nome dinâmico para o stream de logs, um para cada dia
        self.log_stream = f'{log_stream_name_prefix}-{datetime.datetime.now().strftime("%Y-%m-%d")}'
        
        # aqui eu garanto que tanto o grupo quanto o stream de logs existem antes de usá-los
        self._inicializar_log_stream()

    def _inicializar_log_stream(self):
        try:
            # tento criar o grupo de log
            self.client.create_log_group(logGroupName=self.log_group)
        except self.client.exceptions.ResourceAlreadyExistsException:
            # caso ele já exista, tudo bem, eu apenas continuo
            pass
        
        try:
            # cria o stream de log dentro do grupo
            self.client.create_log_stream(logGroupName=self.log_group, logStreamName=self.log_stream)
        except self.client.exceptions.ResourceAlreadyExistsException:
            # se o stream para o dia de hoje já existir, passo
            pass

    def log(self, message):
        # eu pego o tempo atual, como a api do cloudwatch espera
        timestamp = int(datetime.datetime.now().timestamp() * 1000)
        try:
            # envio o evento de log para o stream
            self.client.put_log_events(
                logGroupName=self.log_group,
                logStreamName=self.log_stream,
                logEvents=[{'timestamp': timestamp, 'message': message}]
            )
        except Exception as e:
            # se algo der errado ao enviar o log, eu imprimo o erro no console para não quebrar a aplicação
            print(f'❌ Erro ao enviar log para o CloudWatch: {e}')