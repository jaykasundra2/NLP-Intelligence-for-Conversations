import torch
import json
from transformers import pipeline
import openai
        
device = torch.device("cpu")
import json
with open("../../../secrets.json", "r") as f:
    secret_keys = json.load(f)
    openai_key = secret_keys['openai']
    
class ClassChatSummary:
    def __init__(self, model: str = None, service: str = "chat_summary"):
        """
        Constructor to the class that does the Chat Summarization in the back end
        :param model: Model that will be used for Classification Task
        :param service: string to represent the service
        """
        
        self.model = model
        if self.model == 'bart':
            self.summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
        else:
            openai.api_key = openai_key

    def preprocess(self, input_text):
        # add preprocess steps here, if any
        return input_text
    
    def postprocess(self, summary):
        # add post-process steps here, if any
        return summary
    
    def inference(self, input_text: str, query: str = None):
        """
        Method to perform the inference
        :param input_text: Input text for the inference
        :param query: Input query
        :return: summary, length of summary text and length of original text
        """

        original_length = len(input_text)
        input_text = self.preprocess(input_text)
        summary = ""
        if self.model=='bart':
            summary = self.summarizer(input_text)[0].get('summary_text')    
        else:
            prompt = f"{input_text} \n Summary of the above conversation is :"
            summary = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=1000)["choices"][0]["text"]
        summary = self.postprocess(summary)
        summary_length = len(summary)
        
        return summary, summary_length, original_length