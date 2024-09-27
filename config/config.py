import os

class Config:
    WATSONX_URL = os.getenv("WATSONX_URL")
    WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
    WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
    THRESHOLD_LIMIT = 0.75
    RECOMMENDATIONS_COUNT = 5
    FILE_PATH = "static/mobile_data.csv"