import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import json

device = torch.device("cpu")


class ClassSentiment:
    def __init__(self, model: str = None, service: str = "sentiment"):
        """
        Constructor to the class that does the Classification in the back end
        :param model: Model that will be used for Classification Task
        :param service: string to represent the service, this will be defaulted to classification
        """

        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.label_mapping = {0:'Negative', 1: 'Neutral', 2:'Positive'}
        self.model.eval()

    def inference(self, input_text: str, query: str = None):
        """
        Method to perform the inference
        :param input_text: Input text for the inference
        :param query: Input query in case of QnA
        :return: correct category and confidence for that category
        """

        tokenized_inputs = self.tokenizer(input_text, return_tensors='pt')
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        scores = outputs[0][0].detach().numpy()
        scores = softmax(scores)
        output = { self.label_mapping[i]:float("{:.2f}".format(score)) for i, score in enumerate(scores)}
        return output