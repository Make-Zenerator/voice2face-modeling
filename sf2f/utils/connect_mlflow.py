import mlflow
import mlflow.entities
import mlflow.environment_variables
import mlflow.tracking._tracking_service
import requests
import os

class TimeoutAdapter(requests.adapters.HTTPAdapter):
    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.pop("timeout", 120)
        super().__init__(*args, **kwargs)

    def send(self, *args, **kwargs):
        kwargs["timeout"] = self.timeout
        return super().send(*args, **kwargs)


def init_mlflow(options):
    
    # ##mlflow set
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = options['MLFLOW_S3_ENDPOINT_URL']
    os.environ["MLFLOW_TRACKING_URI"] = options['MLFLOW_TRACKING_URI']
    os.environ["AWS_ACCESS_KEY_ID"] = options['AWS_ACCESS_KEY_ID']
    os.environ["AWS_SECRET_ACCESS_KEY"] = options['AWS_SECRET_ACCESS_KEY']

    # session = requests.Session()
    # adapter = TimeoutAdapter(timeout=options['timeout'])
    # session.mount("http://", adapter)
    # session.mount("https://", adapter)

    # mlflow.set_tracking_uri(options['MLFLOW_TRACKING_URI'])
    # mlflow._tracking._get_rest_client()._session = session
    mlflow.environment_variables.MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT.set(options['timeout'])
    mlflow.environment_variables.MLFLOW_HTTP_REQUEST_TIMEOUT.set(options['timeout'])


    mlflow.set_experiment(options['experiment'])
    mlflow.start_run()
