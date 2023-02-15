import json
from typing import List
from fastapi import APIRouter, HTTPException

from api.model import Input,TopicResponse
from api.class_topic import ClassTopic

topic = APIRouter()


# Path to provide details of the services loaded in the backend. Details are obtained from the config json file
@topic.get("/info")
async def get_models():
    """
    This method returns model details to the front end. Based on the service argument
    * :return:
    """
    with open("./api/config.json") as f:
        config = json.load(f)
    return config

# Path for Summarization service
@topic.post("/predict", response_model=TopicResponse)
async def topic_predict(item: Input):
    """
    This function will return the topics of the input text.
    * :param item: This is the payload that is sent to the server. The structure of item defined above
    * :return: topics of the input text and the confidence for these topics
    """
    output_dict = dict()
    text = item.text
    topics = item.topic_list

    knowledge_search_obj = ClassTopic(model=item.model.lower(), topic_list=topics)
    answer = knowledge_search_obj.inference(text = text)
    output_dict["response"] = answer
    return output_dict
