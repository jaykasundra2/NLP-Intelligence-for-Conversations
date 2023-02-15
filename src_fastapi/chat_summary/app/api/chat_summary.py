import json
from typing import List
from fastapi import APIRouter, HTTPException

from api.model import Input, ChatSummaryResponse
from api.class_chat_summary import ClassChatSummary

chat_summary = APIRouter()


# Path to provide details of the services loaded in the backend. Details are obtained from the config json file
@chat_summary.get("/info")
async def get_models():
    """
    This method returns model details to the front end. 
    * :return:
    """
    with open("./api/config.json") as f:
        config = json.load(f)
    return config

# Path for Summarization service
@chat_summary.post("/predict", response_model=ChatSummaryResponse)
async def summarization(item: Input):
    """
    This function will return the summary of the input text.
    * :param item: This is the payload that is sent to the server. The structure of item defined above
    * :return: A dictionary with summary for the input text length of the new summary and length of the original input.
        {
            "summary": "Multiline summary",
            "summary_length": length of the original text,
            "original_length": length of the summary text
        }
    """
    output_dict = dict()
    text = item.text
    summarize_obj = ClassChatSummary(model=item.model.lower())
    summary, summary_length, original_length = summarize_obj.inference(input_text=text)
    output_dict["summary"] = summary
    output_dict["summary_length"] = summary_length
    output_dict["original_length"] = original_length
    return output_dict
