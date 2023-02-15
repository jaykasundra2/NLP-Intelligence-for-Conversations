import torch
import json
from gpt_index import GPTListIndex, SimpleWebPageReader
    
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from transformers import pipeline
with open("../../../secrets.json", "r") as f:
    secret_keys = json.load(f)
    openai_key = secret_keys['openai']

import os
os.environ["OPENAI_API_KEY"] = openai_key

import base64

device = torch.device("cpu")

class ClassKnowledgeSearch:
    def __init__(self, model: str = None, knowledge_url:str = None,knowledge_articles:dict = {},service: str = "knowledge_search"):
        
        if knowledge_url!="":
            documents = SimpleWebPageReader(html_to_text=True).load_data([knowledge_url])
        
        if model == 'gpt-3':
            self.MODE = 'OpenAI'
            self.index = GPTListIndex(documents)
        else:
            self.MODE = 'HF'
            BIENCODER_MODEL = "multi-qa-MiniLM-L6-cos-v1"
            CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            QA_MODEL = "deepset/minilm-uncased-squad2" #"deepset/roberta-base-squad2"
            self.TOP_K = 3
            
            filenames = knowledge_articles['filenames']
            filecontents = [base64.b64decode(filecontent).decode('utf-8') for filecontent in knowledge_articles['filecontents']]

            self.kb_articles_df = pd.DataFrame(zip(filenames,filecontents), columns=['filename','text'])
            self.passages = self.kb_articles_df.text.to_list()
            # pre-trained models
            self.bi_encoder = SentenceTransformer(BIENCODER_MODEL)
            self.bi_encoder.max_seq_length = 256
            self.cross_encoder = CrossEncoder(CROSSENCODER_MODEL)
            # bi-encoder embeddings
            self.corpus_embeddings = self.bi_encoder.encode(self.passages, convert_to_tensor=True)
            # question-answering model
            self.question_answerer = pipeline("question-answering", model=QA_MODEL)

    def preprocess(self, query):
        # add preprocess steps here, if any
        return query
    
    def postprocess(self, answer):
        # add post-process steps here, if any
        return answer

    def _get_context(self, query):
        # bi-encoder
        question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        # question_embedding = question_embedding.cuda()
        hits = util.semantic_search(question_embedding, self.corpus_embeddings, top_k=self.TOP_K)
        hits = hits[0]

        # cross-encoder reranking
        cross_inp = [[query, self.passages[hit["corpus_id"]]] for hit in hits]
        cross_scores = self.cross_encoder.predict(cross_inp)

        # sort results by the cross-encoder scores
        for idx in range(len(cross_scores)):
            hits[idx]["cross_score"] = cross_scores[idx]

        hits = sorted(hits, key=lambda x: x["cross_score"], reverse=True)

        # return the passage with highest score
        idx = hits[0]["corpus_id"]
        row = self.kb_articles_df.iloc[idx]
        return {
            "passage": self.passages[idx],
            "score": hits[0]["cross_score"],
        }


    def _qa_HF(self, question):
        # try:
        context_dict = self._get_context(question)
        context = context_dict["passage"]
        answer = self.question_answerer(question=question, context=context)
        return {
            "answer": answer["answer"],
            "qa_score": float(answer["score"]),
            "context_score": float(context_dict["score"]),
            "context": context_dict["passage"],
        }
        # return answer["answer"]
        # except Exception as e:
        #     # return {"answer" : "Sorry, I am unable to answer that at the moment."}
        #     return "Sorry, I am unable to answer that at the moment."
    
    def inference(self, query: str = None):
        """
        Method to perform the inference
        :param input_text: Input text for the inference
        :param query: Input query in case of QnA
        :return: correct category and confidence for that category
        """
        self.query = query
        self.query = self.preprocess(self.query)
        if self.MODE=='OpenAI':
            response = self.index.query(query, verbose=False)
        else:
            response = self._qa_HF(query)
        return response