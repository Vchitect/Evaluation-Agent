from openai import OpenAI
import json
import os


class BaseAgent:
    def __init__(self, system_prompt="", use_history=True, temp=0, top_p=1):
        self.use_history = use_history
        self.client = OpenAI()
        self.system = system_prompt
        self.temp = temp
        self.top_p = top_p
        self.input_tokens_count = 0
        self.output_tokens_count = 0
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system_prompt})
    
    
    def __call__(self, message, parse=False):
        self.messages.append({"role": "user", "content": message})
        result = self.generate(message, parse)
        self.messages.append({"role": "assistant", "content": result})

        if parse:
            try:
                result = self.parse_json(result)
            except:
                raise Exception("Error content is list below:\n", result)
            
        return result
        
    
    
    def generate(self, message, json_format):
        if self.use_history:
            input_messages = self.messages
        else:
            input_messages = [
                {"role": "system", "content": self.system},
                {"role": "user", "content": message}
            ]
            
        
        if json_format:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06", # gpt-4
                messages=input_messages,
                temperature=self.temp,
                top_p=self.top_p,
                response_format = { "type": "json_object" }
                )
        else:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06", # gpt-4
                messages=input_messages,
                temperature=self.temp,
                top_p=self.top_p,
                )
        self.update_tokens_count(response)
        return response.choices[0].message.content
    
    
    def parse_json(self, response):
        return json.loads(response)

    
    def add(self, message: dict):
        self.messages.append(message)
    
    
    def update_tokens_count(self, response):
        self.input_tokens_count += response.usage.prompt_tokens
        self.output_tokens_count += response.usage.completion_tokens
    
    
    def show_usage(self):
        print(f"Total input tokens used: {self.input_tokens_count}\nTotal output tokens used: {self.output_tokens_count}")


class BaseAgentSFT:
    """Chat agent backed by an OpenAI-compatible local inference server."""

    def __init__(
        self,
        system_prompt="",
        use_history=True,
        temp=0.0,
        top_p=1.0,
        base_url="http://127.0.0.1:12333/v1",
        model_name="eval-agent",
        api_key=None,
        max_tokens=1024,
        timeout=120.0,
    ):
        self.use_history = use_history
        self.system = system_prompt
        self.temp = temp
        self.top_p = top_p
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.input_tokens_count = 0
        self.output_tokens_count = 0

        normalized_url = base_url.rstrip("/")
        suffix = "/chat/completions"
        if normalized_url.endswith(suffix):
            normalized_url = normalized_url[: -len(suffix)]

        self.client = OpenAI(
            base_url=normalized_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY",
            timeout=timeout,
            max_retries=2,
        )
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.generate(message)
        self.messages.append({"role": "assistant", "content": result})
        return result

    def generate(self, message):
        if self.use_history:
            input_messages = self.messages
        else:
            input_messages = []
            if self.system:
                input_messages.append({"role": "system", "content": self.system})
            input_messages.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=input_messages,
            max_tokens=self.max_tokens,
            temperature=self.temp,
            top_p=self.top_p,
        )
        if response.usage is not None:
            self.input_tokens_count += response.usage.prompt_tokens or 0
            self.output_tokens_count += response.usage.completion_tokens or 0
        return response.choices[0].message.content

    def add(self, message):
        self.messages.append(message)

    def show_usage(self):
        print(
            f"Total input tokens used: {self.input_tokens_count}\n"
            f"Total output tokens used: {self.output_tokens_count}"
        )


# Compatibility with the development-repository class name.
BaseAgent_SFT = BaseAgentSFT


