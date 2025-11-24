import os

from utils import (
  CodeExecutionResult,
  CodeExecutor,
  extract_python_code_blocks,
  extract_clarification_blocks
)
from city import CityRanges
from flight_itinerary import (
  FlightItineraries,
  search_flight_itineraries
)
from model import (
  Gemini,
  GeminiConfig,
  ModelName,
  Response,
  is_valid_model_name
)
from prompts import generate_flight_query_prompt


class Orchestrator:
  '''The interface that user interacts with, accepting user input, calling Gemini API to parse prompt intent, and calling the flight solver.'''

  def __init__(self):
    self.api_key: str | None = None
    self.model_name: ModelName = 'gemini-2.5-flash'
    self.model: Gemini | None = None

    self.top_n: int = 20

    self.debug: bool = False

    self.initialize()

  def initialize(self) -> None:
    self.load_api_key()
    self.load_model()

  def load_api_key(self) -> None:
    if not os.path.exists('api_key.txt'):
      print('[INFO] API key not set. Type /api_key to set API key.')
      return
    with open('api_key.txt', 'r') as file:
      api_key = file.read()
    self.api_key = api_key

  def set_api_key(self, api_key: str) -> None:
    with open('api_key.txt', 'w') as file:
      file.write(api_key)
    self.api_key = api_key
    print('[INFO] API key set.')

  def load_model(self) -> None:
    if self.api_key is None:
      print('[INFO] Gemini model could not be loaded because API key is not set. Type /api_key to set API key.')
      return
    self.model = GeminiConfig(
      api_key=self.api_key,
      model_name=self.model_name,
      debug=self.debug,
    ).make()
    print(f'[INFO] {self.model_name} model loaded. Chat history reset.')

  def get_multi_line_user_input_prompt(self) -> str:
    user_prompt_lines: list[str] = []
    user_prompt_line = input('')
    while user_prompt_line != '/submit':
      user_prompt_lines.append(user_prompt_line)
      user_prompt_line = input('')
    return '\n'.join(user_prompt_lines)

  def generate_response(self, user_prompt: str) -> Response | None:
    '''Wrapper function that allows for user to retry prompt in case the Gemini API request fails.'''
    assert self.api_key is not None
    assert self.model is not None

    print('[INFO] Sending prompt to Gemini API...')
    response = self.model.generate_response(user_prompt)
    while not response.success:
      print(f'[INFO] Gemini API request failed. Error traceback:\n{response.traceback}')
      print('[INFO] Type /retry to retry submitting the same prompt to Gemini. Type /back to go back to the main menu.')
      user_input = input('[USER_INPUT]: ')
      while user_input not in ('/retry', '/back'):
        print('[INFO] Invalid input. Type either /retry or /back')
        user_input = input('[USER_INPUT]: ')

      if user_input == '/retry':
        response = self.model.generate_response(user_prompt)
      else:
        print('[INFO] Returned to main menu.')
        return
    return response

  def clarification_loop(self, response: Response) -> Response | None:
    '''While Gemini keeps asking for clarification, have the user input prompts to clarify their intention.
    It's expected the final response returned will be Python code to be executed to instantiate the flight query instance object.'''
    clarification_blocks = extract_clarification_blocks(response.text)
    while clarification_blocks:
      print('[INFO] Gemini is asking for clarification.')
      print(f'[GEMINI]: ' + '\n\n'.join(clarification_blocks))
      print('[INFO] Enter your clarifications prompt to Gemini. Multi-line input is accepted. When you are done entering your prompt, submit it to Gemini by typing /submit in a new line.')
      print('Clarifications:')
      user_prompt = self.get_multi_line_user_input_prompt()
      response = self.generate_response(user_prompt)  # type: ignore
      if response is None:  # give up retrying and return to main menu
        return None
      clarification_blocks = extract_clarification_blocks(response.text)
    return response

  def get_queries(self, response: Response) -> tuple[list[CityRanges], str | None]:
    '''Attempts to extract the python code block from the response text and execute the code to instantiate a variable of list[CityRanges] representing the flight queries.
    If anything fails, this function returns a tuple of an empty list and a feedback prompt string to feed back to Gemini.
    If extraction and instantiation is successful, this function returns a tuple of the list[CityRanges] and a None for the feedback prompt.'''
    python_blocks = extract_python_code_blocks(response.text)

    if len(python_blocks) == 0:
      return [], 'Expected 1 clarification block or 1 python code block but got none for either.'
    if len(python_blocks) >= 2:
      return [], f'Expected 1 python code block, but got {len(python_blocks)}.'

    python_block = python_blocks[0]
    code_executor = CodeExecutor(code=python_block)
    code_execution_result: CodeExecutionResult = code_executor.all_queries

    if not code_execution_result.success:
      return [], f'Python code block execution failed. Error traceback:\n{code_execution_result.traceback}'

    city_ranges_list = code_execution_result.result
    if not isinstance(city_ranges_list, list):
      return [], f'Expected `all_queries` to be of type list, but got {type(city_ranges_list)}.'
    if len(city_ranges_list) == 0:
      return [], 'Expected `all_queries` to have a list length greater than 0.'
    if any((not isinstance(city_ranges, CityRanges)) for city_ranges in city_ranges_list):
      return [], f'Expected `all_queries` to be a list[CityRanges], but not all elements in `all_queries` are of type `CityRanges`.'

    return city_ranges_list, None

  def modify_or_search_query_loop(self, original_response: Response) -> tuple[Response | None, list[CityRanges] | None]:
    '''Given the initial flight query prompt or modification to an existing flight query prompt, do the following:
    - run the clarification-feedback loop (i.e. if Gemini needs clarification, have the user provide it)
    - extract the Python code and execute it to generate the Python instance object containing the flight queries
    - print out the flight queries and ask the user whether to
      - submit the queries for search (in which case the queries are returned), or
      - to modify the queries (in which case the modification prompt is sent to Gemini and Gemini's response)
    '''
    assert self.model is not None

    # If both clarification and python blocks are present, put priority on clarification blocks first.
    response = self.clarification_loop(original_response)
    if response is None:  # give up retrying and return to main menu
      return None, None

    queries, feedback_prompt = self.get_queries(response)
    while feedback_prompt is not None:
      print(f'[INFO] Invalid response from Gemini. Error message:\n{feedback_prompt}')
      print(f'[INFO] Retrying Gemini query...')
      response = self.generate_response(feedback_prompt)
      if response is None:  # give up retrying and return to main menu
        return None, None
      queries, feedback_prompt = self.get_queries(response)

    CityRanges.print_proposed_flight_queries(queries)
    print('[INFO] Submit these flight queries by typing /search')
    print('[INFO] Otherwise if you want to give feedback to Gemini to modify the queries, type /modify')
    user_input = input('[USER_INPUT]: ')
    while user_input not in ('/search', '/modify'):
      print('[INFO] Invalid input. Type either /search or /modify')
      user_input = input('[USER_INPUT]: ')

    if user_input == '/search':
      return None, queries
    else:
      print('[INFO] Type your modifications prompt to Gemini. Multi-line input is accepted. When you are done entering your prompt, submit it to Gemini by typing /submit in a new line.')
      print('Modifications:')
      user_prompt = self.get_multi_line_user_input_prompt()
      response = self.generate_response(user_prompt)
      return response, None

  def main_loop(self) -> None:
    '''Main user interaction loop.'''

    print('[INFO] Welcome to Gemini Flight Optimizer. Type /help to see all main menu commands.')

    while True:
      user_input = input('[USER_INPUT]: ')

      match user_input:

        case '/help':
          print('''[INFO] Main menu commands:
/help to see all main menu commands
/api_key to set api key
/model to set model type (defaults to gemini-2.5-flash)
/top_n to set how many flight itineraries to display (defaults to 20)
/debug to toggle debug on/off (defaults to off). If on, Gemini conversations will be saved to the logs/ folder. NOTE: toggling on or off will reset the chat history.
/prompt to begin typing a prompt denoting a flight query
/exit to exit the program'''
                )

        case '/api_key':
          api_key = input('[USER_INPUT] Input API key: ')
          self.set_api_key(api_key)
          self.load_model()

        case '/model':
          print('''[INFO] Valid Gemini model names:
- gemini-2.5-pro
- gemini-2.5-flash
- gemini-2.5-flash-lite
- gemini-2.0-flash
- gemini-2.0-flash-lite''')
          model_name = input('[USER_INPUT] Input Gemini model name: ')
          while not is_valid_model_name(model_name):
            print('[INFO] Invalid gemini model name.')
            print('''[INFO] Valid Gemini model names:
- gemini-2.5-pro
- gemini-2.5-flash
- gemini-2.5-flash-lite
- gemini-2.0-flash
- gemini-2.0-flash-lite''')
            model_name = input('[USER_INPUT] Input Gemini model name: ')
          self.load_model()

        case '/top_n':
          top_n = input('[USER_INPUT] Input how many flight itineraries to display: ')
          while not top_n.isdigit():
            print('[INFO] Invalid input. Input must be a positive integer.')
            top_n = input('[USER_INPUT] Input how many flight itineraries to display: ')
          self.top_n = int(top_n)
          print(f'[INFO] Set to display the top {self.top_n} cheapest flight itineraries.')

        case '/debug':
          self.debug = not self.debug
          if self.debug:
            print('[INFO] Debugging turned on. Gemini conversations will be saved to the logs/ folder.')
          else:
            print('[INFO] Debugging turned off.')
          self.load_model()

        case '/prompt':
          if self.api_key is None:
            print('[INFO] API key not set. Type /api_key to set API key.')
            continue
          if self.model is None:
            print('[INFO] Gemini model could not be loaded because API key is not set. Type /api_key to set API key.')
            continue

          print(f'[INFO] Enter your flight query prompt to Gemini. Multi-line input is accepted. When you are done entering your prompt, submit it to Gemini by typing /submit in a new line.')
          print('Flight query:')
          user_prompt = self.get_multi_line_user_input_prompt()
          response = self.generate_response(generate_flight_query_prompt(user_prompt))
          if response is None:  # give up retrying and return to main menu
            continue

          # Query modification loop
          response, queries = self.modify_or_search_query_loop(response)
          while response is not None:
            assert queries is None
            response, queries = self.modify_or_search_query_loop(response)

          if queries is None:  # Return to main menu
            continue

          flight_itineraries: FlightItineraries = search_flight_itineraries(queries)
          flight_itineraries.save()
          print(f'[INFO] Processing complete. Printing results:\n\n{flight_itineraries.top_n(self.top_n)}')
          print(f'\n\n[INFO] Exiting program.')
          return

        case '/exit':
          print(f'[INFO] Exiting program.')
          return

        case _:
          print('[INFO] Invalid command. Type /help to see all main menu commands.')
