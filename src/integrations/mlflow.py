import mlflow
import os
from dotenv import load_dotenv


class MLFlowIntegration:
    @staticmethod
    def connect() -> str:
        """
        Connects to the MLflow tracking server using credentials from a .env file.
        
        :return: The tracking URI of the MLflow server.
        :rtype: str
        """
        
        if not os.path.exists('.env'):
            raise FileNotFoundError("The .env file is missing. Please create it with the required MLflow tracking credentials.")
        
        load_dotenv()

        try:
            os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")
            os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
            os.environ['MLFLOW_TRACKING_URI'] = os.getenv("MLFLOW_TRACKING_URI")
        except KeyError as e:
            raise KeyError(f"Missing required credential: {e}")
        
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        return mlflow.get_tracking_uri()