import pytest
from fastapi.testclient import TestClient
import pandas as pd
from unittest.mock import patch

from api.main import app

client = TestClient(app)

class TestRootEndpoint:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Housing Regression API is running"}

class TestHealthEndpoint:
    @patch('api.main.MODEL_PATH')
    def test_health_model_not_found(self, mock_path):
        mock_path.exists.return_value = False
        mock_path.__str__ = lambda x: "test/path"
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "Model not found" in data["error"]

    @patch('api.main.MODEL_PATH')
    @patch('api.main.TRAIN_FEATURE_COLUMNS', ['feature1', 'feature2'])
    def test_health_model_exists(self, mock_path):
        mock_path.exists.return_value = True
        mock_path.__str__ = lambda x: "test/path"
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["n_features_expected"] == 2

class TestPredictEndpoint:
    @patch('api.main.MODEL_PATH')
    def test_predict_model_not_found(self, mock_path):
        mock_path.exists.return_value = False
        mock_path.__str__ = lambda x: "test/path"
        
        response = client.post("/predict", json=[{"feature1": 1}])
        assert response.status_code == 200
        assert "Model not found" in response.json()["error"]

    @patch('api.main.MODEL_PATH')
    @patch('api.main.predict')
    def test_predict_success(self, mock_predict, mock_path):
        mock_path.exists.return_value = True
        mock_predict.return_value = pd.DataFrame({
            "predicted_price": [100000.0, 200000.0]
        })
        
        test_data = [
            {"feature1": 1, "feature2": 2},
            {"feature1": 3, "feature2": 4}
        ]
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        assert data["predictions"] == [100000.0, 200000.0]