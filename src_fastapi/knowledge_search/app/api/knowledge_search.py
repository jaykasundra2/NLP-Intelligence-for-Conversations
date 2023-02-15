import json
from typing import List
from fastapi import APIRouter, HTTPException

from api.model import Input, KnowledgeSearchResponse
from api.class_knowledge_search import ClassKnowledgeSearch

knowledge_search = APIRouter()


# Path to provide details of the services loaded in the backend. Details are obtained from the config json file
@knowledge_search.get("/info")
async def get_models():
    """
    This method returns model details to the front end. Based on the service argument
    * :return:
    """
    with open("./api/config.json") as f:
        config = json.load(f)
    return config

# Path for knowledge search service
@knowledge_search.post("/predict", response_model=KnowledgeSearchResponse)
async def knowledge_search_predict(item: Input):
    """
    This function will return the answer for the query from the knowledge base. 
    * :param item: This is the payload that is sent to the server.
    * :return: Answer for the query
    """
    output_dict = dict()
    knowledge_url = item.text
    knowledge_articles = item.knowledge_articles
    query = item.query
    
    knowledge_search_obj = ClassKnowledgeSearch(model=item.model.lower(), knowledge_url=knowledge_url,
                                                knowledge_articles = knowledge_articles)
    answer = knowledge_search_obj.inference(
        query = query
    )
    output_dict["response"] = answer
    return output_dict
