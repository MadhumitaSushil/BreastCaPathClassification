import os

import backoff
import openai

from dotenv import load_dotenv
from openai.error import RateLimitError, APIError


load_dotenv('.env')
API_KEY = os.environ.get('STAGE_API_KEY')
API_VERSION = os.environ.get('API_VERSION')
RESOURCE_ENDPOINT = os.environ.get('RESOURCE_ENDPOINT')


class OpenaiApiCall:
    def __init__(self, model='gpt-35-turbo',
                 temperature=0., max_tokens=512, append_prompt=False, num_completions=1):
        openai.api_type = "azure"
        openai.api_base = RESOURCE_ENDPOINT
        openai.api_version = API_VERSION
        openai.api_key = API_KEY

        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.append_prompt = append_prompt
        self.num_completions = num_completions

    @backoff.on_exception(backoff.expo, (RateLimitError, APIError))
    def get_response(self, prompt_preamble, prompt):
        if 'davinci' in self.model_name:
            response = self._get_completion(prompt_preamble, prompt)
        elif 'gpt-35-turbo' in self.model_name or 'gpt-4' in self.model_name:
            response = self._get_chat_completion(prompt_preamble, prompt)

        print(response)
        return response

    def _get_completion(self, prompt_preamble, prompt, **kwargs):
        prompt = prompt_preamble + prompt
        return openai.Completion.create(
                engine=self.model_name,
                temperature=self.temperature,
                prompt=prompt,
                max_tokens=self.max_tokens,
                echo=self.append_prompt,
                n=self.num_completions,
                **kwargs
            )

    def _get_chat_completion(self, prompt_preamble, prompt, **kwargs):
        return openai.ChatCompletion.create(
                engine=self.model_name,
                messages=[
                    {"role": "system",
                     "content": prompt_preamble,
                     },
                    {
                        "role": "user",
                        "content": prompt,
                    }],
                temperature=self.temperature,
                n=self.num_completions,
                **kwargs
            )

    def get_response_text(self, prompt, prompt_preamble=""):
        response = self.get_response(prompt_preamble, prompt)

        if response['choices'][0]['finish_reason'] == 'content_filter':
            print("Content filter")
            return None

        if 'davinci' in self.model_name:
            result = response['choices'][0]['text']
        elif 'gpt-35-turbo' in self.model_name or 'gpt-4' in self.model_name:
            result = response['choices'][0]['message']['content']
        else:
            raise ValueError("Unsupported response model: ", self.model_name)

        return result
