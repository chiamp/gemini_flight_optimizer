"""
Gemini model for transcription, translation, extracting vocabulary and making example sentences.

API and token limit:
  - https://ai.google.dev/pricing#2_0flash
  - https://ai.google.dev/gemini-api/docs/rate-limits
  - 15 RPM (requests per minute)
  - 1 million TPM (tokens per minute)
  - 1.5K RPD (requests per day)
Text tokenization rate:
  - https://ai.google.dev/gemini-api/docs/tokens?lang=python#about-tokens
  - 4 characters == 1 token
    - for example, 100 tokens is equal to about 60-80 English words.
Audio tokenization rate:
  - https://ai.google.dev/gemini-api/docs/tokens?lang=python#multimodal-tokens
  - https://ai.google.dev/gemini-api/docs/audio?lang=python#technical-details
  - 1 second of audio == 32 tokens
    - for example, 2 hours of audio is represented as 230400 tokens
  - Audio is downsampled to a 16 Kbps (2 KBps) data resolution
    - for example, a 2 hour video that's sampled at 2 KBps would be 14400 Kilobytes
Audio upload limit:
  - https://ai.google.dev/gemini-api/docs/audio?lang=python#upload-audio
  - max 20 GB per project
  - max 2 GB per file
Image understanding:
  - https://ai.google.dev/gemini-api/docs/image-understanding


Daily rate limit resets at midnight Pacific Time (PT) every day:
- https://ai.google.dev/gemini-api/docs/rate-limits#:~:text=Rate%20limits%20are%20applied%20per,only%20apply%20to%20specific%20models.

Model info (including input and output token limits):
  - https://ai.google.dev/gemini-api/docs/models

Model codes:
  - https://ai.google.dev/gemini-api/docs/models/gemini#model-variations

Thinking budget and config:
  - https://ai.google.dev/gemini-api/docs/thinking

Billing:
- https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/cost?project=language-learner-451112
- https://console.cloud.google.com/billing/011E76-6F186B-F3F5A7/reports;timeRange=CUSTOM_RANGE;from=2025-06-24;to=2025-09-24;timeGrouping=GROUP_BY_MONTH;projects=language-learner-451112;products=services%2FAEFD-7695-64FA?project=language-learner-451112

API keys:
- https://aistudio.google.com/app/apikey
- you can see what tier (free or paid) your API key is enrolled in
"""

import functools

import traceback

import dataclasses
import datetime
from zoneinfo import ZoneInfo
import time
import os

import typing
from typing import Literal

from google import genai

from utils import get_time_id


ModelName = Literal[
  'gemini-2.5-pro',
  'gemini-2.5-flash',
  'gemini-2.5-flash-lite',
  'gemini-2.0-flash',
  'gemini-2.0-flash-lite',
]


def is_valid_model_name(model_name: str) -> bool:
  return model_name in typing.get_args(ModelName)


@dataclasses.dataclass(frozen=True)
class Response:
  response: genai.types.GenerateContentResponse | None
  traceback: str | None = None

  @functools.cached_property
  def success(self) -> bool:
    return self.response is not None

  @functools.cached_property
  def text(self) -> str:
    if (self.response is None) or (self.response.text is None):
      return ''
    return self.response.text

  @functools.cached_property
  def input_tokens(self) -> int:
    if (self.response is None) or (self.response.usage_metadata is None) or (self.response.usage_metadata.prompt_token_count is None):
      return 0
    return self.response.usage_metadata.prompt_token_count

  @functools.cached_property
  def thought_tokens(self) -> int:
    if (self.response is None) or (self.response.usage_metadata is None) or (self.response.usage_metadata.thoughts_token_count is None):
      return 0
    return self.response.usage_metadata.thoughts_token_count

  @functools.cached_property
  def output_tokens(self) -> int:
    if (self.response is None) or (self.response.usage_metadata is None) or (self.response.usage_metadata.candidates_token_count is None):
      return 0
    return self.response.usage_metadata.candidates_token_count

  @functools.cached_property
  def total_tokens(self) -> int:
    return self.input_tokens + self.thought_tokens + self.output_tokens


@dataclasses.dataclass(kw_only=True)
class GeminiConfig:
  api_key: str
  model_name: ModelName = dataclasses.field(default_factory=lambda: 'gemini-2.5-flash')
  debug: bool = False  # whether to write the prompts and responses into logs

  def make(self) -> 'Gemini':
    return Gemini(config=self)


class Gemini:
  def __init__(self, config: GeminiConfig):
    self.model_name = config.model_name
    self.client = genai.Client(api_key=config.api_key)
    self.chats = self.client.chats.create(model=self.model_name)

    self.api_call_history_path = f'api_call_history/{self.model_name}.tsv'

    self.debug = config.debug

  def update_api_call_history(self, response: Response):
    """Update api_call_history file."""
    file_exists = os.path.exists(self.api_call_history_path)
    with open(self.api_call_history_path, 'a') as file:
      if not file_exists:  # first line of the file should be the column headers
        file.write('datetime\tnum_input_tokens\tnum_thought_tokens\tnum_output_tokens\ttotal_tokens\n')

      file.write('\t'.join(
        [
          str(datetime.datetime.now(tz=ZoneInfo("US/Pacific"))),  # Use US/Pacific date time since Gemini API limits reset at midnight Pacific time.
          str(response.input_tokens),
          str(response.thought_tokens),
          str(response.output_tokens),
          str(response.total_tokens),
        ]
      ) + '\n'
    )

  def generate_response(self, prompt_str: str) -> Response:
    if self.debug:
      with open(f'logs/{get_time_id()}_user_{self.model_name}.txt', 'w') as f:
        f.write(prompt_str)

    try:
      response = Response(self.chats.send_message(prompt_str))
    except:
      traceback_str = traceback.format_exc()
      print(f'[INFO] Error occurred while calling Gemini API:\n{traceback_str}')
      response = Response(None, traceback=traceback_str)

    self.update_api_call_history(response=response)

    if self.debug:
      with open(f'logs/{get_time_id()}_model_{self.model_name}.txt', 'w') as f:
        if response.success:
          f.write(response.text)
        else:
          assert response.traceback is not None
          f.write(f'Error traceback:\n\n{response.traceback}')

    return response
