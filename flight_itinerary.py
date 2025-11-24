from collections import defaultdict
import functools
from typing import Sequence

import json
import dataclasses

from tqdm import tqdm
import datetime

import pandas as pd

from fli.models import FlightSearchFilters

from flight_info import (
  FlightInfo,
  FlightType,
  search_flights_parallel,
  merge_flight_queries
)
from flight_hash import hash_flight_query
from city import (
  City,
  CityRange,
  CityRanges,
)
from utils import get_time_id, minutes_to_string


FlightPath = tuple[tuple[str, ...], tuple[str, ...]]
CityHashes = tuple[int, int]


@functools.cache
def load_iata_code_to_city_mapping() -> dict[str, str]:
  with open('iata_codes.json', 'r') as file:
    iata_code_to_city_mapping: dict[str, str] = json.load(file)
  return iata_code_to_city_mapping


@dataclasses.dataclass(kw_only=True)
class FlightItinerary:
  '''Each element in `self.flight_infos` is a `FlightInfo`.
  For a given list of `FlightInfo`'s (f1, f2, f3, ...), the arrival location of `FlightInfo` f1 is assumed to be the departure location for `FlightInfo` f2,
  and the arrival location of `FlightInfo` f2 is assumed to be the departure location of `FlightInfo` f3, etc.
  For example:
  - f1 could be a flights from London (LHR) to Toronto (YYZ)
  - f2 could be a flight from Toronto (YYZ) to New York City (JFK)
  - f3 could be a flight from New York City (JFK) to San Francisco (SFO)
  '''

  flight_infos: list[FlightInfo]

  def __post_init__(self):
    assert len(self.flight_infos) > 0
    assert len(self.flight_infos) == len(self.flight_group_ids)

    assert (len(self.flight_infos) - 1) == len(self.stopover_cities)

    # The returning flights for a round-trip should only map to one parent id.
    # Otherwise there could be ambiguity when trying to group departing and returning round-trip flights together.
    assert all((len(flight_info.parent_ids)==1) for flight_info in self.flight_infos if (flight_info.flight_type==FlightType.ROUND_TRIP_RETURNING))

  @functools.cached_property
  def total_price(self) -> float:
    return sum(flight_info.price for flight_info in self.flight_infos)

  @functools.cached_property
  def total_minutes_durations(self) -> int:
    return sum(flight_info.duration for flight_info in self.flight_infos)

  @functools.cached_property
  def total_layover_minutes_duration(self) -> float:
    return sum(sum(flight_info.layover_minutes_duration) for flight_info in self.flight_infos)

  @functools.cached_property
  def total_num_stops(self) -> int:
    return sum(flight_info.stops for flight_info in self.flight_infos)

  @functools.cached_property
  def departure_datetime(self) -> datetime.datetime:
    return self.flight_infos[0].legs[0].departure_datetime

  @functools.cached_property
  def returning_datetime(self) -> datetime.datetime:
    return self.flight_infos[-1].legs[-1].arrival_datetime

  @functools.cached_property
  def flight_group_ids(self) -> list[int]:
    '''Group round trip flights together with the same group int id. This id is unrelated to `FlightInfo.id`.
    This will be used when printing out the flight itinerary so the user knows which flights correspond to a round-trip flight together.'''

    group_ids: list[int] = []
    flight_info_id_to_group_id_mapping: dict[str, int] = {}
    group_id = 1
    for flight_info in self.flight_infos:
      if flight_info.flight_type == FlightType.ROUND_TRIP_RETURNING:
        assert len(flight_info.parent_ids) == 1
        returning_flight_group_id = flight_info_id_to_group_id_mapping[flight_info.parent_ids[0]]
        group_ids.append(returning_flight_group_id)
      else:
        if flight_info.flight_type == FlightType.ROUND_TRIP_DEPARTING:
          flight_info_id_to_group_id_mapping[flight_info.id] = group_id
        group_ids.append(group_id)
        group_id += 1
    return group_ids

  @functools.cached_property
  def iata_code_to_city_mapping(self) -> dict[str, str]:
    iata_code_to_city_mapping = load_iata_code_to_city_mapping()
    return iata_code_to_city_mapping

  @functools.cached_property
  def start_city(self) -> str:
    departure_airport = self.flight_infos[0].legs[0].departure_airport
    if departure_airport.name not in self.iata_code_to_city_mapping:
      # Use the airport name as a back up if we can't find the city / location name
      return departure_airport.value
    # Otherwise use the city / location name
    return self.iata_code_to_city_mapping[departure_airport.name]

  @functools.cached_property
  def end_city(self) -> str:
    arrival_airport = self.flight_infos[-1].legs[-1].arrival_airport
    if arrival_airport.name not in self.iata_code_to_city_mapping:
      return arrival_airport.value
    # Otherwise use the city / location name
    return self.iata_code_to_city_mapping[arrival_airport.name]

  @functools.cached_property
  def stopover_cities(self) -> list[str]:
    stopover_cities: list[str] = []
    for flight_info in self.flight_infos[:-1]:
      arrival_airport = flight_info.legs[-1].arrival_airport
      if arrival_airport.name not in self.iata_code_to_city_mapping:
        stopover_cities.append(arrival_airport.value)
      else:
        stopover_cities.append(
          self.iata_code_to_city_mapping[arrival_airport.name]
        )
    return stopover_cities

  @functools.cached_property
  def minutes_spent_in_stopover_cities(self) -> list[float]:
    minutes_spent_in_stopover_cities: list[float] = []
    for i in range(len(self.flight_infos)-1):
      stopover_city_arrival_datetime = self.flight_infos[i].legs[-1].arrival_datetime
      stopover_city_departure_datetime = self.flight_infos[i+1].legs[0].departure_datetime
      minutes_spent_in_stopover_cities.append(
        (stopover_city_departure_datetime - stopover_city_arrival_datetime).total_seconds() / 60
      )
    return minutes_spent_in_stopover_cities

  def to_dict(self) -> dict[str, int | float | str | list[str]]:
    return {
      'total_price': self.total_price,
      'total_flight_hours': self.total_minutes_durations / 60,
      'total_layover_hours': self.total_layover_minutes_duration / 60,
      'total_num_stops': self.total_num_stops,
      'departure_datetime': self.departure_datetime.isoformat(),
      'returning_datetime': self.returning_datetime.isoformat(),
      'start_city': self.start_city,
      'end_city': self.end_city,
      'stopover_cities': self.stopover_cities,
      'formatted_string': str(self),
    }

  def __str__(self) -> str:
    itinerary_stats: str = ' | '.join(
      [
        f'{self.departure_datetime} - {self.returning_datetime}',
        f'${self.total_price}',
        minutes_to_string(self.total_minutes_durations),
      ]
    )

    str_components: list[str] = [
      itinerary_stats,
      f'{self.start_city} | START',
    ]
    for i, (stopover_city, minutes_spent) in enumerate(zip(self.stopover_cities, self.minutes_spent_in_stopover_cities)):
      str_components.append(f'[FLIGHT {self.flight_group_ids[i]}]: {self.flight_infos[i]}')
      str_components.append(f'{stopover_city} | {minutes_to_string(round(minutes_spent))}')
    str_components.append(f'[FLIGHT {self.flight_group_ids[-1]}]: {self.flight_infos[-1]}')
    str_components.append(f'{self.end_city} | END')
    return '\n\n'.join(str_components)


@dataclasses.dataclass(kw_only=True)
class FlightItineraries:
  '''Represents an aggregation of all possible flight itineraries given the user's flight queries.
  Used to print out and save the proposed itineraries.'''

  flight_itinerary_list: list[FlightItinerary]

  def top_n(self, n) -> 'FlightItineraries':
    # NOTE: this method assumes `self.flight_itinerary_list` is already sorted.
    return FlightItineraries(flight_itinerary_list=self.flight_itinerary_list[:n])

  def save(self) -> None:
    flight_itineraries: list[dict] = [flight_itinerary.to_dict() for flight_itinerary in self.flight_itinerary_list]
    data = pd.DataFrame(
      flight_itineraries,
      columns=[
        'total_price',
        'total_flight_hours',
        'total_layover_hours',
        'total_num_stops',
        'departure_datetime',
        'returning_datetime',
        'start_city',
        'end_city',
        'stopover_cities',
        'formatted_string',
      ]
    )

    filename = f'./saved_flight_itineraries/{get_time_id()}.tsv'
    data.to_csv(filename, sep='\t', index=False)
    print(f'[INFO] Saved flight itineraries at {filename}')

  def __str__(self) -> str:
    flight_itinerary_strings: list[str] = []
    for i, flight_itinerary in enumerate(self.flight_itinerary_list):
      flight_itinerary_strings.append('\n' + '='*20 + f' TRIP ITINERARY {i+1} ' + '='*20 + f'\n{str(flight_itinerary)}\n')
    return '\n'.join(flight_itinerary_strings)


def extract_queries(
    *,
    cities: Sequence[City],
    query_hash_to_query: dict[int, FlightSearchFilters],
) -> dict[int, tuple[CityHashes, CityHashes | None]]:
  '''Given a sequence of cities that denote a flight itinerary:
  - generate the flight query between every sequential pair of departure/arrival cities in the flight itinerary
    - in-place update `query_hash_to_query` with these generated flight queries
  - create a mapping between the query and the corresponding departure/arrival cities in the flight itinerary
    - put all of these mappings in a dictionary and return it
    - given some flight itinerary with city order [c1, c2, c3], this dictionary will allow us to map the query result to which pair of departure/arrival cities
      - e.g. we will be able to map the query result of (c1, c2) to the first flight of the trip and the query result of (c2, c3) to the second flight of the trip
      - this is because we aggregate all flight queries from all possible flight itineraries, and execute the queries in parallel
        - so we can't rely on keeping track of the ordering for only one specific flight itinerary; a dictionary mapping can help with this
  '''

  query_hash_to_city_hashes: dict[int, tuple[CityHashes, CityHashes | None]] = {}

  # Maps flight path to the corresponding query (NOTE: we assume that we don't have multiple flights departing from the same location and arriving at the same location in the same flight itinerary).
  # If we detect a round-trip flight is possible, we use this mapping to get the departing query so we can merge it with the returning query to form a round-trip query.
  flight_paths_to_query: dict[FlightPath, FlightSearchFilters] = {}
  for flight_path_index in range(len(cities)-1):
    departure_city, arrival_city = cities[flight_path_index], cities[flight_path_index+1]

    one_way_query = City.create_flight_search_query(departure_city, arrival_city)
    query_hash = hash_flight_query(one_way_query)
    if query_hash not in query_hash_to_query:
      query_hash_to_query[query_hash] = one_way_query

    # This `city_hashes` will help us map the query result back to the corresponding flight of this trip itinerary.
    city_hashes: CityHashes = (departure_city.hash, arrival_city.hash)
    if query_hash in query_hash_to_city_hashes:
      assert query_hash_to_city_hashes[query_hash] == (city_hashes, None)
    else:
      query_hash_to_city_hashes[query_hash] = (city_hashes, None)

    flight_path: FlightPath = (departure_city.id, arrival_city.id)
    # TODO: We do not handle the case where you have multiple flights departing from the same location and arriving at the same location in the same flight itinerary.
    assert flight_path not in flight_paths_to_query
    flight_paths_to_query[flight_path] = one_way_query

    # Check if this one way flight is the reverse of a one way flight we've already seen earlier in this trip.
    # If so, that means we can construct a round-trip with these two one-way flights.
    reversed_flight_path: FlightPath = (flight_path[1], flight_path[0])
    if reversed_flight_path in flight_paths_to_query:
      departing_query = flight_paths_to_query[reversed_flight_path]
      returning_query = one_way_query

      round_trip_query = merge_flight_queries(departing_query, returning_query)
      query_hash = hash_flight_query(round_trip_query)
      if query_hash not in query_hash_to_query:
        query_hash_to_query[query_hash] = round_trip_query

      departing_city_hashes, should_be_none = query_hash_to_city_hashes[hash_flight_query(departing_query)]
      assert should_be_none is None  # one-way flight should not have a returning flight
      returning_city_hashes = city_hashes
      # This round-trip flight query will return a tuple for the departing and returning flight
      # So we equivalently need to keep track of two sets of city hashes to map those flight query results to their corresponding flight in the flight itinerary.
      query_hash_to_city_hashes[query_hash] = (departing_city_hashes, returning_city_hashes)

  return query_hash_to_city_hashes


def get_queries(city_ranges_list: list[CityRanges]) -> tuple[
  list[tuple[City, ...]],
  list[list[float | None]],
  list[list[float | None]],
  dict[int, FlightSearchFilters],
  defaultdict[int, set[tuple[CityHashes, CityHashes | None]]]
]:
  '''Given a list of possible trip itineraries denoted with CityRanges, return:
  - all possible departure/arrival datetime constraints for each city in a particular itinerary, for every itinerary in list[CityRanges]
  - the corresponding min_stay_hours for each city configuration in the first bullet point (this information is extracted from the CityRanges objects themselves)
  - the corresponding max_stay_hours for each city configuration in the first bullet point (this information is extracted from the CityRanges objects themselves)
  - all queries generated from all city configurations in the first bullet point
  - a dictionary mapping the queries generated from the fourth bullet point to the corresponding departure/arrival city pair, where the results of the flight query would correspond to
  '''
  city_itineraries: list[tuple[City, ...]] = []
  min_stay_hours_list: list[list[float | None]] = []
  max_stay_hours_list: list[list[float | None]] = []

  # Get all queries from the `city_ranges`.
  query_hash_to_query: dict[int, FlightSearchFilters] = {}
  query_hash_to_set_city_hashes: defaultdict[int, set[tuple[CityHashes, CityHashes | None]]] = defaultdict(set)
  for city_ranges in city_ranges_list:
    # This represents one trip itinerary with variable departure/arrival datetimes and min/max stay hours for each city in the itinerary.
    city_range_list = city_ranges.city_range_list

    # Don't omit the start city even if it's redundant (since it doesn't matter how long we stay there), since it will help with indexing in the recursive function in `generate_flight_itineraries`.
    # Omit the end city since it doesn't matter how long we stay there.
    # Also this way the length of `min_stay_hours` and `max_stay_hours` should match the length of the `list_flight_infos`, for consistency.
    min_stay_hours: list[float | None] = [city_range.min_stay_hours for city_range in city_range_list[:-1]]
    max_stay_hours: list[float | None] = [city_range.max_stay_hours for city_range in city_range_list[:-1]]

    # Iterate through all possible constraints of departure/arrival datetimes and min/max stay hours for each city in this itinerary.
    for city_itinerary in CityRange.get_city_itineraries(city_range_list):
      # Extract all queries for this city itinerary.
      # In-place update `query_hash_to_query` and update `query_hash_to_set_city_hashes` with `query_hash_to_city_hashes`.
      query_hash_to_city_hashes: dict[int, tuple[CityHashes, CityHashes | None]] = extract_queries(
        cities=city_itinerary,
        query_hash_to_query=query_hash_to_query,
      )
      for query_hash, city_hashes in query_hash_to_city_hashes.items():
        query_hash_to_set_city_hashes[query_hash].add(city_hashes)

      city_itineraries.append(city_itinerary)
      min_stay_hours_list.append(min_stay_hours)
      max_stay_hours_list.append(max_stay_hours)
  print(f'[INFO] {len(query_hash_to_query)} flight queries created.')

  return (
    city_itineraries,
    min_stay_hours_list,
    max_stay_hours_list,
    query_hash_to_query,
    query_hash_to_set_city_hashes,
  )


def generate_flight_itineraries(list_flight_infos: list[list[FlightInfo]], min_stay_hours: list[float | None], max_stay_hours: list[float | None]) -> list[FlightItinerary]:
  '''Each element in `list_flight_infos` is a list[FlightInfo].
  Each individual list[FlightInfo] contains a list of flights that all depart from the same location and all arrive at the same location;
  e.g. list[FlightInfo] contains [possible_flight_a_to_b_1, possible_flight_a_to_b_2, possible_flight_a_to_b_3, ...],
       and the next list[FlightInfo] in `list_flight_infos` contains [possible_flight_b_to_c_1, possible_flight_b_to_c_2, possible_flight_b_to_c_3, ...]

  For a given list of list[FlightInfo]'s (f1, f2, f3, ...), the arrival location of all flights in list[FlightInfo] f1 is assumed to be the departure location for all flights in list[FlightInfo] f2,
  and the arrival location of all flights in list[FlightInfo] f2 is assumed to be the departure location of all flights in list[FlightInfo] f3, etc.
  For example:
  - f1 could be all flights from London (LHR and LGW) to Toronto (YYZ and YTZ)
  - f2 could be all flights from Toronto (YYZ and YTZ) to New York City (JFK and LGA)
  - f3 could be all flights from New York City (JFK and LGA) to San Francisco Bay Area (SFO, OAK and SJC)

  This function will map all combinations of flight itineraries (i.e. an f1 flight, followed by an f2 flight, followed by an f3 flight, ...),
  provided that the arrival datetime of the current flight precedes the departure datetime of the next flight.
  '''

  assert len(list_flight_infos) > 0

  if len(list_flight_infos) == 1:
    return [FlightItinerary(flight_infos=flight_infos) for flight_infos in list_flight_infos]

  # TODO: find a way to memoize this (probably will have to change the FlightInfo dataclass so that it doesn't inherit from `FlightResult` so that it can be hashed).
  def _recurse(prev_flight_info: FlightInfo, round_trip_departing_flight_ids: set[str], flight_infos_index: int) -> list[list[FlightInfo]]:
    '''Recursive helper function.

    `flight_infos_index` denotes what current flight in the overall trip we are trying to add to our current itinerary.

    `round_trip_departing_flight_ids` denotes the flight ids of all round-trip departing flights we have added to our current itinerary so far
    - this is used when we hit the base case of the recursive function where we add the last flight
      - if there are still any ids remaining, that means we have added round-trip departing flights with no valid returning flight to pair it with, therefore making this itinerary invalid
    - whenever we find a matching returning flight to pair one of the ids in `round_trip_departing_flight_ids`, we remove that id from `round_trip_departing_flight_ids`

    Returns a list of all possible valid flight itineraries, where each element in this list, is a list[FlightInfo] denoting one possible valid flight itinerary;
    e.g. list[FlightInfo] contains [flight_a_to_b_1, flight_b_to_c_1, flight_c_to_d_1, ...],
         and the next list[FlightInfo] contains [flight_a_to_b_2, flight_b_to_c_1, flight_c_to_d_1, ...]
    '''

    if prev_flight_info.flight_type == FlightType.ROUND_TRIP_DEPARTING:
      round_trip_departing_flight_ids = round_trip_departing_flight_ids.union([prev_flight_info.id])

    list_flight_itineraries: list[list[FlightInfo]] = []  # each element is an individual itinerary
    for next_flight_info in list_flight_infos[flight_infos_index]:
      # Make a copy since we may be removing some id's if we find a returning trip that matches with one of the departing flight ids in `round_trip_departing_flight_ids`,
      # and we don't want to edit `round_trip_departing_flight_ids` since other recursive child calls will use it.
      child_round_trip_departing_flight_ids = round_trip_departing_flight_ids.copy()

      if next_flight_info.parent_ids:  # must be a round-trip returning flight
        assert next_flight_info.flight_type == FlightType.ROUND_TRIP_RETURNING
        found_matching_departing_flight = False
        for parent_id in next_flight_info.parent_ids:
          if parent_id in child_round_trip_departing_flight_ids:
            # Set the parent id to be only this one, since we have selected this parent flight to be our departing flight for this round-trip that connects to this returning flight
            # This information will be used to link flights together in the `FlightItinerary`.
            # Make a deep copy of this returning flight so that the changes we make don't carry over for other recursive calls.
            next_flight_info = next_flight_info.copy()
            next_flight_info.parent_ids = [parent_id]
            child_round_trip_departing_flight_ids.difference_update([parent_id])
            found_matching_departing_flight = True
            # NOTE: the first matching departing flight will be used.
            # In practice this should be fine as long as we don't have multiple identical trips within the same itinerary,
            # (e.g. doing London to Toronto twice in the same itinerary).
            break
        if not found_matching_departing_flight:
          continue

      # Check if the city stay hours constraints are satisfied.
      stay_hours = (next_flight_info.legs[0].departure_datetime - prev_flight_info.legs[-1].arrival_datetime).total_seconds() / 3600
      min_stay_hours_city = min_stay_hours[flight_infos_index]
      max_stay_hours_city = max_stay_hours[flight_infos_index]
      if (
        ((min_stay_hours_city is None) or (stay_hours >= min_stay_hours_city))
        and ((max_stay_hours_city is None) or (stay_hours <= max_stay_hours_city))
        and (stay_hours > 0)
      ):
        if flight_infos_index == (len(list_flight_infos)-1):  # base case
          if child_round_trip_departing_flight_ids:
            # This means that we have some round-trip departing flights that never had a corresponding returning flight scheduled.
            # This would mean we'd have to skip the returning flight, which is possible, but probably not good practice (e.g. could get blacklisted by the airline).
            # So we consider this itinerary invalid since all round-trip departing flights must have a corresponding returning flight scheduled.
            continue
          # Otherwise we have a valid itinerary.
          list_child_flight_itineraries: list[list[FlightInfo]] = [[next_flight_info]]
        else:  # continue recursing to the next flight in the trip
          list_child_flight_itineraries: list[list[FlightInfo]] = _recurse(next_flight_info, child_round_trip_departing_flight_ids, flight_infos_index+1)

        # Aggregate all flight results and generate all possible flight itineraries.
        for child_flight_itinerary in list_child_flight_itineraries:
          list_flight_itineraries.append([prev_flight_info] + child_flight_itinerary)
    return list_flight_itineraries

  # Start the recursive calls with the first flight in the trip.
  list_flight_itineraries: list[FlightItinerary] = []
  for starting_flight_info in list_flight_infos[0]:
    list_child_flight_itineraries: list[list[FlightInfo]] = _recurse(
      starting_flight_info,
      round_trip_departing_flight_ids=set(),
      flight_infos_index=1,
    )
    for child_flight_itinerary in list_child_flight_itineraries:
      list_flight_itineraries.append(FlightItinerary(flight_infos=child_flight_itinerary))

  return list_flight_itineraries


def search_flight_itineraries(city_ranges_list: list[CityRanges]) -> FlightItineraries:
  '''Given a list of different possible trip itineraries, in which each trip itinerary has variable arrival/departure datetimes and min/max stay hours for each city,
  Do a search over all possible city configurations and aggregate all results and return them as a `FlightItineraries` instance.'''

  # Generate all queries.
  (
    city_itineraries,
    min_stay_hours_list,
    max_stay_hours_list,
    query_hash_to_query,
    query_hash_to_set_city_hashes,
  ) = get_queries(city_ranges_list)

  # Execute flight searches.
  query_hash_to_flight_infos: dict[int, tuple[list[FlightInfo], list[FlightInfo]]] = search_flights_parallel(list(query_hash_to_query.values()))
  assert query_hash_to_query.keys() == query_hash_to_flight_infos.keys()

  # Group all flight results to their corresponding `CityHashes`.
  city_hashes_to_flight_infos: defaultdict[CityHashes, list[FlightInfo]] = defaultdict(list)
  for query_hash, (departing_flight_info, returning_flight_info) in query_hash_to_flight_infos.items():
    for departing_city_hashes, returning_city_hashes in query_hash_to_set_city_hashes[query_hash]:

      # Map the query results to the corresponding departure/arrival cities.
      city_hashes_to_flight_infos[departing_city_hashes].extend(departing_flight_info)
      if returning_city_hashes is None:
        assert not returning_flight_info
      else:
        city_hashes_to_flight_infos[returning_city_hashes].extend(returning_flight_info)

  # Generate flight itineraries.
  flight_itineraries: list[FlightItinerary] = []
  print(f'[INFO] {len(city_itineraries)} city configurations created.')
  for city_itinerary, min_stay_hours, max_stay_hours in tqdm(
    zip(
      city_itineraries,
      min_stay_hours_list,
      max_stay_hours_list,
    ),
    total=len(city_itineraries),
    desc='Processing city configurations'
  ):
    # Iterate through every trip itinerary and get the corresponding flight results for every departure/arrival city pair.
    list_flight_infos: list[list[FlightInfo]] = []
    for i in range(len(city_itinerary)-1):
      city_hashes = (city_itinerary[i].hash, city_itinerary[i+1].hash)
      list_flight_infos.append(city_hashes_to_flight_infos[city_hashes])
    # Generate flight itineraries for every possible valid combination of flights between each city.
    flight_itineraries.extend(
      generate_flight_itineraries(
        list_flight_infos=list_flight_infos,
        min_stay_hours=min_stay_hours,
        max_stay_hours=max_stay_hours,
      )
    )
  print(f'[INFO] {len(flight_itineraries)} flight itineraries created.')

  print('[INFO] Sorting flight itineraries based on total price...')
  # Tie-breaking order: total price, total flight duration, total layover duration, total number of stops
  return FlightItineraries(
    flight_itinerary_list=sorted(
      flight_itineraries,
      key=lambda fi: (
        fi.total_price,
        fi.total_minutes_durations,
        fi.total_layover_minutes_duration,
        fi.total_num_stops,
      )
    )
  )
