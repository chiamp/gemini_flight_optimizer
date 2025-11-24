# Gemini Flight Optimizer

This program is an interface that uses Gemini API to parse the user input prompt and convert it to flight queries, calls the Google Flights API on those queries and aggregates the results for you.

See the example prompts in `example_prompts/` for inspiration.


## Table of Contents
* [Main features](#main-features)
* [How it works (at a high level)](#how-it-works-at-a-high-level)
* [Video demos](#video-demos)
  * [Single query](#single-query)
  * [Single query with clarification](#single-query-with-clarification)
  * [Multi query with modification](#multi-query-with-modification)
* [Setup instructions](#setup-instructions)
* [Main menu commands](#main-menu-commands)
* [Token limit](#token-limit)
* [File structure](#file-structure)


## Main features
* turns complex, natural language flight queries into Google Flights API calls
* user can specify a range of dates / time / locations / flight constraints in natural language and the program will have Gemini parse the natural language prompt into flight queries for every possible configuration of dates / time / locations / flight constraints.
* can ask the user for clarification if the prompt denoting the flight queries is unclear or contradictory
* allows the user to modify the query parsed by Gemini before calling the Google Flights API on it


## How it works (at a high level)
The program:
* parses the user's natural language prompt into a set of queries
* each query represents one possible itinerary, which is an order of cities you want to visit for your trip
  * if you have multiple trip ideas, put it in the prompt and Gemini will parse them out for you (e.g. separate trips to different destinations, or a trip where you are okay with or without an optional stopover)
* within each query, the following information is extracted from the user prompt (if provided):
  * the preferred arrival and departure date and time for each city
  * the preferred minimum and maximum number of hours stayed in each city
  * the preferred flight constraints for each flight between each city:
    * the maximum number of stops
    * seat type (i.e. economy, business, etc.)
    * the maximum price
    * the airlines you want to fly
    * the maximum duration of the flight
    * the layover constraints:
      * the airports you want to do a layover in
      * the maximum duration of the layover
* if Gemini has trouble understanding your intent or thinks you made a contradiction in your prompt, it will ask you for clarification
* once Gemini understands your intent and parses your prompt successfully, it will return the queries it thinks it should use to search for flights
  * if you want to make modifications to the queries, you can submit another natural language prompt to tell Gemini to modify the queries
* once you are happy with the queries, the program will call the Google Flights API and aggregate all results
* the flight itineraries constructed will be sorted by cheapest price and saved in `saved_flight_itineraries/`.


## Video demos

### Single query
![single_query](video_demos/single_query.gif)

### Single query with clarification
![single_query_with_clarification](video_demos/single_query_with_clarification.gif)

### Multi query with modification
![multi_query_with_modification](video_demos/multi_query_with_modification.gif)


## Setup instructions
* get a free Gemini API key [here](https://aistudio.google.com/app/api-key) (read more in the [docs](https://ai.google.dev/gemini-api/docs/api-key))
* ensure you have a working Python environment with version at least `3.10.10`
* clone this repo and run `pip install -r requirements.txt`
* open a terminal window and run `python main.py`
* set your API key by typing `/api_key` and copy and pasting your API key here (this will be saved for future sessions so you only have to do this once)


## Main menu commands
These are the commands you can run in the main menu of the program:
* `/help`: to see all main menu commands
* `/api_key`: to set api key
* `/model`: to set model type (defaults to `gemini-2.5-flash`)
* `/top_n`: to set how many flight itineraries to display (defaults to `20`)
* `/debug`: to toggle debug on/off (defaults to off). If on, Gemini conversations will be saved to the `logs/` folder. NOTE: toggling on or off will reset the chat history.
* `/prompt`: to begin typing a prompt denoting a flight query
* `/exit`: to exit the program


## Token limit
As of November 23rd, 2025, the free Gemini API key has a maximum tokens-per-minute (TPM) of 125000 for Gemini 2.5 Pro and 250000 for Gemini 2.5 Flash and Gemini 2.5 Flash-Lite. Since the prompt used in this program is around 120000 tokens, this will allow the user to send a prompt to Gemini once per minute for Gemini 2.5 Pro and twice a minute for Gemini 2.5 Flash and Gemini 2.5 Flash-Lite. If you get a rate limit error, simply wait a minute and retry via the `/retry` command.

You could also try Gemini 2.0 which has a 1 million TPM limit, but Gemini 2.0 models do not have thinking (although perhaps not a lot of thinking is needed to just parse the user intent into flight query constraints). If you find yourself being rate limited too often, you can switch the model by executing the `/model` command in the main menu of the program.

You can see the number of tokens used by the prompt sent to Gemini in the files in `api_call_history/`.

See more details on the rate limits [here](https://ai.google.dev/gemini-api/docs/rate-limits#current-rate-limits).


## File structure
* `main.py`: calls `orchestratorpy` to run the main program loop that interacts with the user.
* `orchestrator.py`: acts as the interface between the user and the flight query / itinerary tools. Calls Gemini API to parse the intent of the user's prompt, calls the Google Flights API on the parsed queries and then calls flight itinerary tools to generate all possible flight itineraries and aggregates the results.
* `city.py`: contains dataclasses for cities and city ranges, which contain date and time constraints as well as other travel constraints for each city in the trip. These city constraints make up a flight query.
* `flight_info.py`: contains a dataclass for flight info and functions to search for flights given a flight query.
* `flight_itinerary.py`: contains dataclasses to store flight itineraries, which are comprised of flight info instances from `flight_info.py`. Also contains functions to find all possible flight itineraries and aggregate the results, given all possible flights for some trip.
* `flight_hash.py`: utility functions for hashing flight info objects.
* `model.py`: class to call the Gemini API on a user prompt.
* `prompts.py`: contains prompts that show the dataclass structure and fields of the city ranges class in `city.py` so that Gemini knows in what format to parse the user prompt into. This city ranges class is then used as a query that the Google Flights API can be called on.
* `utils.py`: Python code execution class as well as miscellaneous utility functions.
* `datetime_range.py`: dataclasses for denoting datetime, date and time ranges for city constraints.
* `iata_codes.json`: maps airport IATA codes to the corresponding city / location. This is used to figure out the name of the city / location that corresponds to the airport we are traveling to/from; i.e. the city / location name is more user-friendly to print out for them to view than the airport codes. This mapping was derived from `iata_codes.pdf`
* `iata_codes.pdf`: the original `.pdf` that maps the IATA airport codes to the corresponding city / location.
* `iata_codes_link.txt`: contains the URL where I downloaded `iata_codes.pdf` from.
* `saved_flight_itineraries/`: saved flight itineraries in the form of `.tsv` files that are a result of flight query searches
* `api_call_history/`: keeps track of Gemini API requests and their token count.
* `logs/`: logs user input into Gemini API as well as Gemini's output, for debugging purposes. To turn on logging, toggle debugging on by executing the `/debug` command in the main menu of the program.
* `example_prompts/`: example flight query prompts.
* `video_demos/`: video demos of interacting with the program, converting the user prompt to flight queries and using them to search for flights.
* `requirements.txt`: Python dependencies file.
