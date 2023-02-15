import torch
import json
import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import openai
import os
import json
with open("../../../secrets.json", "r") as f:
    secret_keys = json.load(f)
    openai_key = secret_keys['openai']

device = torch.device("cpu")

class ClassTopic:
    def __init__(self, model: str = None, text:str = None, topic_list:str = None, service: str = "topic"):
        self.model = model
        self.text = text

        if topic_list != "":
            self.candidate_labels = topic_list.split(",")
            self.candidate_labels = [x.strip() for x in self.candidate_labels]
        else:
            self.candidate_labels = None
        
        if self.model=="gpt-3":
            self.mode = 'topic_generation_gpt3'
        elif self.candidate_labels is not None:
            self.mode = 'zero_shot_classification'
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        else:
            self.mode = 'topic_generation_hf'
            self.tokenizer = AutoTokenizer.from_pretrained("knkarthick/TOPIC-DIALOGSUM")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("knkarthick/TOPIC-DIALOGSUM")

    def preprocess(self, query):
        # add preprocess steps here, if any
        return query
    
    def postprocess(self, answer):
        # add post-process steps here, if any
        return answer
    
    def inference(self, text: str = None):
        """
        Method to perform the inference
        :param text: Input text for the inference
        :return: topics and confidence for the topics
        """
        print(self.mode)
        self.text = text
        if self.mode =='topic_generation_gpt3':
            openai.api_key = openai_key
            print(self.candidate_labels)
            if self.candidate_labels is None:
                print("Topic Generation")
                prompt = f"Assign the all the relevant topics to below conversation:\n{self.text}\n List of topics are:"
            else:
                print("Topic Assignment")
                prompt = f"{self.text}\n  Assign one or more topics to the above conversation from the following list of topics [{','.join(self.candidate_labels)}]"
                print(prompt)
            completion = openai.Completion.create(model= "text-davinci-003",temperature=0, prompt=prompt,max_tokens=120)
            response = {"topics" : completion.choices[0]['text']}
        elif self.mode=='topic_generation_hf':
            input_ids = self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
            generated_ids = self.model.generate(input_ids=input_ids, max_length=100)
            preds = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            response = {"topics" : preds[0]}
        else: #zero_shot_classification
            response = self.classifier(self.text, self.candidate_labels, multi_label=True)
            response = { label:score for label,score in zip(response['labels'],response['scores'])}
        return response