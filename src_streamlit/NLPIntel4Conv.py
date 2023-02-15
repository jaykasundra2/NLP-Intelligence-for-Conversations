# Importing packages: streamlit for the frontend, requests to make the api calls
import streamlit as st
import requests
import json
import base64

class MakeCalls:
    def __init__(self, url: str = "http://localhost") -> None:
        """
        Constructor for the MakeCalls class. This class is used to perform API calls to the backend service.
        :param url: URL of the server. Default value is set to local host: http://localhost:8080
        """
        self.url = url
        self.headers = {"Content-Type": "application/json"}
        self.service_port_mapping = {'sentiment' : 8001,
                                     'chat_summary' : 8002,
                                     'knowledge_search' : 8003,
                                     'topic' : 8004}

    def model_list(self, service: str) -> dict:
        """
        Making an API request to backend service to get the details for each service. This function returns, list of names of trained models 
        :param service: NLP service that is being used.
        :return: List of names of trained models
        """
        service_port = self.service_port_mapping[service]
        model_info_url = self.url + f":{service_port}/" + f"api/v1/{service}/info"
        models = requests.get(url=model_info_url)
        return json.loads(models.text)

    def run_inference(
        self, service: str, model: str, text: str, knowledge_articles:dict, query: str = None, topic_list:str = None

    ) -> json:
        """
        This function is used to send the api request for the actual service for the specified model to the
        :param service: String for the actual service.
        :param model: Model that is selected from the drop down.
        :param text: Input text that is used for analysis and to run inference.
        :param query: Input query for Information extraction use case.
        :return: results from the inference done by the model.
        """
        service_port = self.service_port_mapping[service]
        inference_endpoint = self.url + f":{service_port}/" + f"api/v1/{service}/predict"
        if service=='knowledge_search':
            payload = {"model": model.lower(), "text": text,"knowledge_articles" : knowledge_articles, 
                       "query": query.lower()}
        elif service=='topic':
            payload = {"model": model.lower(), "text": text, "topic_list" : topic_list}
        else:
            payload = {"model": model.lower(), "text": text, "query": query.lower()}

        result = requests.post(
            url=inference_endpoint, headers=self.headers, data=json.dumps(payload)
        )
        return json.loads(result.text)


class Display:
    def __init__(self):
        st.title("NLP Intelligence for Conversations")
        st.sidebar.header("Select the NLP Service")
        self.service_options = st.sidebar.selectbox(
            label="",
            options=[
                "Project NI 4 Conv",
                "Sentiment Analysis",
                "Named Entity Recognition",
                "Knowledge Search",
                "Chat Summarization",
                "Chat Topic"
            ],
        )
        self.service = {
            "Project NI 4 Conv": "about",
            "Sentiment Analysis": "sentiment",
            "Named Entity Recognition": "ner",
            "Knowledge Search": "knowledge_search",
            "Chat Summarization": "chat_summary",
            "Chat Topic" : "topic"
        }

    def static_elements(self):
        return self.service[self.service_options]

    def dynamic_element(self, models_dict: dict):
        """
        This function is used to generate the page for each service.
        :param service: String of the service being selected from the side bar.
        :param models_dict: Dictionary of Model and its information. This is used to render elements of the page.
        :return: model, input_text run_button: Selected model from the drop down, input text by the user and run botton to kick off the process.
        """
        
        st.header(self.service_options)
        model_name = list()
        model_info = list()
        knowledge_articles = {}
        for i in models_dict.keys():
            model_name.append(models_dict[i]["name"])
            model_info.append(models_dict[i]["info"])
        st.sidebar.header("Model Information")
        for i in range(len(model_name)):
            st.sidebar.subheader(model_name[i])
            st.sidebar.info(model_info[i])
        model: str = st.selectbox("Select the Trained Model", model_name)
        
        if self.service_options=='Knowledge Search':
            input_text: str = st.text_area("Enter FAQs (knowledge) URL here", height=10)
            st.write("Or")
            uploaded_files = st.file_uploader("Choose Knowledge Article files", type=["csv", "txt"], accept_multiple_files=True)
            if uploaded_files is not None:
                filenames = []
                filecontents = []
                for file in uploaded_files:
                    filenames.append(file.name)
                    filecontents.append(base64.b64encode(file.read()).decode('utf-8'))
                knowledge_articles = {"filenames": filenames, "filecontents": filecontents}
        else:
            input_text: str = st.text_area("Enter Text here", height=250)
        
        if self.service_options in ["qna","Knowledge Search"]:
            query: str = st.text_input("Enter query here.")
        else:
            query: str = "None"
        
        if self.service_options == 'Chat Topic':
            topic_list:str = st.text_input("Enter the list of topics separated by comma. (optional)")
        else:
            topic_list: str = None
        
        run_button: bool = st.button("Run")
        return model, input_text, query, knowledge_articles, topic_list, run_button


def main():

    page = Display()
    service = page.static_elements()
    apicall = MakeCalls()
    if service == "about":
        # st.header("NLP Intelligence for Conversations")
        st.write(
            "The users can leverage fine-tuned language models to perform multiple downstream tasks, via GUI and API access."
        )
        st.write(
            "To use this solution, select a service from the dropdown in the side bar. Details of pre-loaded  pre-trained model will be available based on the service."
        )
    else:
        model_details = apicall.model_list(service=service)
        model, input_text, query,knowledge_articles, topic_list, run_button = page.dynamic_element(model_details)
        if run_button:
            with st.spinner(text="Getting Results.."):
                result = apicall.run_inference(
                    service=service,
                    model=model.lower(),
                    text=input_text,
                    knowledge_articles = knowledge_articles,
                    query=query.lower(),
                    topic_list = topic_list
                )
            st.write(result)


if __name__ == "__main__":
    main()
