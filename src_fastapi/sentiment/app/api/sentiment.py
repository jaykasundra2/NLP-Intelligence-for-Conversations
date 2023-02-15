import json
from typing import List
from fastapi import APIRouter, HTTPException

from api.model import Input, SentimentResponse
from api.class_sentiment import ClassSentiment

sentiment = APIRouter()


# Path to provide details of the services loaded in the backend. Details are obtained from the config json file
@sentiment.get("/info")
async def get_models():
    """
    This method returns model details to the front end. Based on the service argument
    * :param service: Service can from one of the services such as: Classification, sentiment analysis etc.
    * :return:
    """
    with open("./api/config.json") as f:
        config = json.load(f)
    return config


# Path for classification service
@sentiment.post("/predict", response_model=SentimentResponse)
async def sentiment_predict(item: Input):
    """
    This is the API method for sentiment analysis based models and related task.
    * :param item: This is the payload that is sent to the server. The structure of item defined above
    * :return: Label/category of the Input text and the confidence for the prediction.
    """
    output_dict = dict()
    class_process = ClassSentiment(model=item.model.lower())
    text = item.text
    prediction = class_process.inference(input_text=text)
    output_dict['response'] = prediction
    return output_dict
