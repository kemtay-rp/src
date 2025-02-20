import dotenv, os, requests
import google.generativeai as gemini
from google.generativeai.types.generation_types import GenerateContentResponse
import absl.logging
import json

absl.logging._initialize()

#
#print(os.environ)

class Ollama():
    '''
        Class that represents an Ollama GenAI model
    '''

    def __init__(self, model, temperature=0.18):
        dotenv.load_dotenv()
        self.api_url = os.getenv('Ollama_API_URL')
        self.model = model
        self.temperature = temperature
        print("Model:", self.model, "at", self.api_url)


    def invoke(self, prompt, temperature=0):
        
        try:

            headers = {
                "Content-Type": "application/json"
            }
           
            user_prompt = prompt["context_window"]
            if "system_message" in prompt:
                system_prompt = prompt["system_message"]
            else:
                system_prompt = "You are a helpful and cheerful professional counselor with a focus on mental health. You should be capable of detecting negative emotions of the human. Be a good listener, and don't rush to offer solutions. You should response in positive and optimistic way, in simple English and short sentence."
            print(system_prompt)

            data = {
                "model": self.model,
                "prompt": system_prompt + user_prompt,
                "stream": False
            }

            response = requests.post(self.api_url, 
                                     headers=headers, 
                                     data=json.dumps(data))

            response.raise_for_status()  # Raises an HTTPError if the response code was unsuccessful
            #print(response.json())

            response_text = response.text
            data = json.loads(response_text)
            assistant_message = data["response"]
            
            return (assistant_message)
        except requests.exceptions.RequestException as error:
            print('Error interacting with Ollama API:', error)
            return ({"error": "An error occurred while processing your request."})
        

class ChatGPT4All():
    '''
        Class that represents an GPT4All GenAI model
    '''

    def __init__(self, model, temperature=0.18):
        dotenv.load_dotenv()
        self.api_url = os.getenv('GPT4ALL_API_URL')
        self.model = model
        self.temperature = temperature
        print("Model:", self.model, "at", self.api_url)

    def invoke(self, prompt, temperature=0):
        
        try:
            if temperature == 0:
                temperature = self.temperature

            user_prompt = prompt["context_window"]
            if "system_message" in prompt:
                system_prompt = prompt["system_message"]
            else:
                system_prompt = "You are a helpful and cheerful professional counselor with a focus on mental health. You should be capable of detecting negative emotions of the human. Be a good listener, and don't rush to offer solutions. You should response in positive and optimistic way, in simple English and short sentence."
            print(system_prompt)

            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,                 
                    "prompt":  f"system\n{system_prompt}\n\n{user_prompt}",
                    "max_tokens": 500,
                    "temperature": temperature,
                }
            )

            response.raise_for_status()  # Raises an HTTPError if the response code was unsuccessful
            #print(response.json())
            assistant_message = response.json()['choices'][0]['text']
            return (assistant_message)
        except requests.exceptions.RequestException as error:
            print('Error interacting with GPT-4-All API:', error)
            return ({"error": "An error occurred while processing your request."})
        
class OpenAIGPT():
    '''
        Class that represents an OpenAI GenAI model
    '''

    def __init__(self, model, temperature=0.18):
        dotenv.load_dotenv()
        self.api_url = os.getenv('OpenAI_API_URL')
        self.api_key = os.getenv('OpenAI_API_KEY')
        self.model = model
        self.temperature = temperature
        print("Model:", self.model, "at", self.api_url)

    def invoke(self, prompt, temperature=0):
        
        try:
            if temperature == 0:
                temperature = self.temperature

            user_prompt = prompt["context_window"]
            if "system_message" in prompt:
                system_prompt = prompt["system_message"]
            else:
                system_prompt = "You are a helpful assistant with a focus on healthcare."

            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,   
                    "messages": [{ 'role': 'user', "content": user_prompt+"\n"+system_prompt }],              
                    "max_tokens": 500,
                    "temperature": temperature,
                },
                headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
                }
            )

            response.raise_for_status()  # Raises an HTTPError if the response code was unsuccessful
            print(response, type(response))
            #print(response.json())
            assistant_message = response.json()['choices'][0]['message']['content']
            #print(assistant_message)
            return (assistant_message)
        except requests.exceptions.RequestException as error:
            print('Error interacting with OpenAI API:', error)
            return ({"error": "An error occurred while processing your request."})

class GoogleGemini():
    '''
        Class that represents a Google GenAI model
    '''

    def __init__(self, model, temperature=0.18):
        dotenv.load_dotenv()
        self.api_url = os.getenv('Gemini_API_URL')
        gemini.configure(api_key=os.getenv('Gemini_API_KEY'))
        self.model = gemini.GenerativeModel(
            model,
            #system_instruction="You are an customer service professional specializing in Aviva Insurance.")
            #system_instruction="This GPT acts as a professional mental health counselor offering compassionate and practical support. It avoids using therapy jargon and long explanations, focusing on brief, empathetic responses that provide clear guidance. The goal is to listen attentively, offer simple advice, and create a safe, non-judgmental space for users. The GPT adjusts its tone to provide concise, actionable help, particularly in times of emotional need.")
            system_instruction="You are a helpful AI assistant.")
            #system_instruction="You are a helpful assistant with a focus on mental health.")
            #system_instruction="You are a helpful and cheerful professional counselor with a focus on mental health. You should be capable of detecting negative emotions of the human. You should be a good listener who don't rush to offer solutions or to tell the human what to do. You should try to guide the human to draw a conclusion themselves. Your responses should be in a positive and optimistic way, in simple English and short sentence. You should not repeat your responses given before.")
        self.temperature = temperature
        
        print("Model:", self.model, "at", self.api_url+model)

    def invoke(self, prompt, temperature=0):
        
        try:
            if temperature == 0:
                temperature = self.temperature

            user_prompt = prompt["context_window"]
            if "system_message" in prompt:
                system_prompt = prompt["system_message"]
            else:
                system_prompt = "You are a helpful assistant with a focus on healthcare."

            response = self.model.generate_content(
                user_prompt+system_prompt,
                generation_config=gemini.types.GenerationConfig(
                # Only one candidate for now.
                candidate_count=1,
                #stop_sequences=["x"],
                max_output_tokens=256,
                temperature=temperature,
                ),
            )
            #print(response, type(response))
            #print(response._result.candidates[0].content.parts[0].text)
            #response.raise_for_status()  # Raises an HTTPError if the response code was unsuccessful
            assistant_message = response._result.candidates[0].content.parts[0].text
            #print(assistant_message)
            return (assistant_message)
        except requests.exceptions.RequestException as error:
            print('Error interacting with Gemini API:', error)
            return ({"error": "An error occurred while processing your request."})
        
class XGrok2():
    '''
        Class that represents X Grok2 GenAI model
    '''

    def __init__(self, model, temperature=0.18):
        dotenv.load_dotenv()
        self.api_url = os.getenv('X_Grok2_API_URL')
        self.api_key = os.getenv('X_Grok2_API_KEY')
        self.model = model
        self.temperature = temperature
        print("Model:", self.model, "at", self.api_url)

    def invoke(self, prompt, temperature=0):
        
        try:
            if temperature == 0:
                temperature = self.temperature

            user_prompt = prompt["context_window"]
            if "system_message" in prompt:
                system_prompt = prompt["system_message"]
            else:
                system_prompt = "You are a helpful assistant with a focus on healthcare."

            response = requests.post(
                self.api_url,
                json={
                    "messages": [{ 'role': 'system', "content": system_prompt }, { 'role': 'user', "content": user_prompt }],
                    "model": self.model,          
                    "stream": False,
                    "temperature": temperature,
                },
                headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
                }
            )
            response.raise_for_status()  # Raises an HTTPError if the response code was unsuccessful
            print(response, type(response))
            #print(response.json())
            assistant_message = response.json()['choices'][0]['message']['content']
            #print(assistant_message)
            return (assistant_message)
        except requests.exceptions.RequestException as error:
            print('Error interacting with Grok2 API:', error)
            return ({"error": "An error occurred while processing your request."})

class Claude():
    '''
        Class that represents Anthropic Claude GenAI model
    '''

    def __init__(self, model, temperature=0.18):
        dotenv.load_dotenv()
        self.api_url = os.getenv('Claude_API_URL')
        self.api_key = os.getenv('Claude_API_KEY')
        self.model = model
        self.temperature = temperature
        print("Model:", self.model, "at", self.api_url, self.api_key)

    def invoke(self, prompt, temperature=0):
        
        try:
            if temperature == 0:
                temperature = self.temperature

            user_prompt = prompt["context_window"]
            if "system_message" in prompt:
                system_prompt = prompt["system_message"]
            else:
                system_prompt = "You are a helpful assistant with a focus on healthcare."

            url = self.api_url
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            data = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, world"
                    }
                ],
                "temperature": temperature
            }

            response = requests.post(url, headers=headers, json=data)

            # Print the response status and content
            print(f"Status Code: {response.status_code}")
            print("Response Body:", response.json())
            response.raise_for_status()  # Raises an HTTPError if the response code was unsuccessful
            print(response, type(response))
            #print(response.json())
            assistant_message = response.json()['choices'][0]['message']['content']
            #print(assistant_message)
            return (assistant_message)
        except requests.exceptions.RequestException as error:
            print('Error interacting with Claude API:', error)
            return ({"error": "An error occurred while processing your request."})
        
def main():
    #chat_model = ChatGPT4All(model="Meta-Llama-38B-Instruct.Q4_0")
    #chat_model = ChatGPT4All(model="Meta-Llama-3.1-8B-Instruct-128k-Q4_0")
    #chat_model = OpenAIGPT(model="gpt-3.5-turbo")
    #chat_model = GoogleGemini(model="gemini-1.5-pro-exp-0827")   #gemini-1.5-flash-exp-0827, gemini-1.5-flash, gemini-1.5-pro-exp-0827
    #chat_model = XGrok2(model="grok-beta")
    chat_model = Claude(model="claude-3-5-sonnet-20241022")


    user_messages1 = { "system_message": "You're an assistant knowledgeable about healthcare. Only answer healthcare-related questions.",
                  "context_window": "Who is GOAT?" }
    
    user_messages2 = { "system_message": "Your job is to use patient reviews to answer questions about their experience at a hospital. Use the following context to answer questions. \
                      Be as detailed as possible, but don't make up any information that's not from the context. If you don't know an answer, say you don't know.", 
                      "context_window": "Who is GOAT?" }
    
    user_messages3 = { "system_message": "You're an assistant knowledgeable about sports. Only answer sports-related questions.",
                  "context_window": "Who is GOAT?" }
    
    user_messages = [ { "context_window": "Who is the only GOAT in men single tennis?" }, 
                      { "context_window": "When and where was he born?" },
                      { "system_message": "You're an assistant knowledgeable about sports. Only answer sports-related questions.", "context_window": "Is he still active in the sport? If not when was he retired?" },]

    for i in range(len(user_messages)):
        print("Chat message", i+1,  ":", user_messages[i]["context_window"])

        if i != 0:
            user_messages[i]["context_window"] += user_messages[i-1]["context_window"] + "\n" + assistant

        assistant = chat_model.invoke(user_messages[i], temperature=1.8)
        print(assistant)

if __name__ == "__main__":
    main()