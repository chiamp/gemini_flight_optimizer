import enum
import functools
import dataclasses

from concurrent import futures

from typing import Sequence

from tqdm import tqdm

from fli.models import (
  Airport,
  FlightLeg,
  FlightResult,
  FlightSearchFilters,
  LayoverRestrictions,
  MaxStops,
  PriceLimit,
  TripType,
)
from fli.search import SearchFlights

from flight_hash import hash_flight_query
from utils import minutes_to_string


class FlightType(enum.Enum):

  ONE_WAY = 'ONE_WAY'
  ROUND_TRIP_DEPARTING = 'ROUND_TRIP_DEPARTING'
  ROUND_TRIP_RETURNING = 'ROUND_TRIP_RETURNING'



class FlightInfo(FlightResult):

  flight_type: FlightType

  # If this is a returning flight of a round-trip, `parent_id` will be a string denoting the id of the departing flight of the round-trip, otherwise None.
  parent_ids: list[str] = dataclasses.field(default_factory=list)

  def __post_init__(self):
    if self.flight_type == FlightType.ROUND_TRIP_RETURNING:
      # A returning flight of a round-trip must be linked to a departing flight of the round-trip.
      assert self.parent_ids
    else:
      assert not self.parent_ids

  @functools.cached_property
  def id(self) -> str:
    # NOTE: we are using the string representation as a unique identifier since this class is not hashable
    return str(self)

  @functools.cached_property
  def layover_minutes_duration(self) -> list[float]:
    return [
      (self.legs[i+1].departure_datetime - self.legs[i].arrival_datetime).total_seconds() / 60
      for i in range(len(self.legs)-1)
    ]

  def copy(self) -> 'FlightInfo':
    # make a deep copy
    flight_info_copy = FlightInfo(
      legs=[
        FlightLeg(
          airline=leg.airline,
          flight_number=leg.flight_number,
          departure_airport=leg.departure_airport,
          arrival_airport=leg.arrival_airport,
          departure_datetime=leg.departure_datetime,
          arrival_datetime=leg.arrival_datetime,
          duration=leg.duration,
        ) for leg in self.legs
      ],
      price=self.price,
      duration=self.duration,
      stops=self.stops,
      flight_type=self.flight_type,
      parent_ids=self.parent_ids.copy(),
    )
    assert self.id == flight_info_copy.id
    return flight_info_copy

  @classmethod
  def from_flight_result(
    cls,
    flight_result: FlightResult,
    flight_type: FlightType,
    parent_ids: list[str] = []
  ) -> 'FlightInfo':
    return cls(
      legs=flight_result.legs,
      price=flight_result.price,
      duration=flight_result.duration,
      stops=flight_result.stops,
      flight_type=flight_type,
      parent_ids=parent_ids,
    )

  @classmethod
  def flight_leg_to_string(cls, flight_leg: FlightLeg, leg_num: int) -> str:
    return ' | '.join(
      [
        f'[LEG {leg_num}]: {flight_leg.departure_datetime} - {flight_leg.arrival_datetime}',
        f'{flight_leg.departure_airport.name}-{flight_leg.arrival_airport.name}',
        minutes_to_string(flight_leg.duration),
        f'{flight_leg.airline.name} {flight_leg.flight_number}',
        flight_leg.airline.value,
      ]
    )

  def __str__(self):
    flight_info: list[str] =  [
      ' | '.join(
        [
          f'${self.price}',
          minutes_to_string(self.duration),
          f'{self.stops} stop' if (self.stops==1) else f'{self.stops} stops',
          self.flight_type.value,
        ]
      )
    ]
    for i, flight_leg in enumerate(self.legs):
      flight_info.append(FlightInfo.flight_leg_to_string(flight_leg, leg_num=i+1))
      if i != (len(self.legs)-1):
        flight_info.append(f'[LAYOVER]: {minutes_to_string(round(self.layover_minutes_duration[i]))}')
    return '\n'.join(flight_info)


def merge_flight_queries(departing_query: FlightSearchFilters, returning_query: FlightSearchFilters) -> FlightSearchFilters:
  """Merge two one-way flight search filters together into a round-trip flight search filter. Return None if merge failed.

  Merging logic for each field type:
  - .trip_type: both trip types must be one way, and the merged trip type will be round-trip
  - .passenger_info: the passenger infos from both filters must be the same
  - .flight_segments: each filter must contain exactly one flight segment, and each segment is used in the merged filter.
    The departure and arrival airports for the departing flight(1) must exactly match the arrival and departure airports of the returning flight(2), respectively.
  - .stops: the merged filter's maximum stops is equal to the maximum between the two filters' maximum number of stops
  - .seat_type: the seat types from both filters must be the same
  - .price_limit: the sum is taken from both filters. If any of the filters' price limit is None, then the merged filter's price limit is None as well.
  - .airlines: the union is taken between both filters. If any of the filters' airlines is None, then the merged filter's airlines is None as well.
  - .max_duration: the maximum is taken between both filters. If any of the filters' max_duration is None, then the merged filter's max_duration is None as well.
  - .layover_restrictions:
    - the union of airports between the two filters' layover restrictions is taken
    - the maximum max_duration between the two filters' layover restrictions is taken
    - if any of the filters' layover_restrictions is None, then the merged filter's layover_restrictions is None as well.
  - .sort_by: the sorting mechanisms from both filters must be the same

  Essentially all merged fields try to take the max/union of the two filters.
  """
  assert departing_query.trip_type == returning_query.trip_type == TripType.ONE_WAY, (departing_query.trip_type, returning_query.trip_type)
  assert departing_query.passenger_info == returning_query.passenger_info, (departing_query.passenger_info, returning_query.passenger_info)
  assert departing_query.seat_type == returning_query.seat_type, (departing_query.seat_type, returning_query.seat_type)
  assert departing_query.sort_by == returning_query.sort_by, (departing_query.sort_by, returning_query.sort_by)

  # The departure and arrival airports for the departing flight(1) must exactly match the arrival and departure airports of the returning flight(2), respectively.
  assert len(departing_query.flight_segments) == len(returning_query.flight_segments) == 1
  flight1_departure_airport: set[tuple[Airport | int, ...]] = set(tuple(airport) for airport in departing_query.flight_segments[0].departure_airport)
  flight1_arrival_airport: set[tuple[Airport | int, ...]] = set(tuple(airport) for airport in departing_query.flight_segments[0].arrival_airport)
  flight2_departure_airport: set[tuple[Airport | int, ...]] = set(tuple(airport) for airport in returning_query.flight_segments[0].departure_airport)
  flight2_arrival_airport: set[tuple[Airport | int, ...]] = set(tuple(airport) for airport in returning_query.flight_segments[0].arrival_airport)
  assert flight1_departure_airport == flight2_arrival_airport, (flight1_departure_airport, flight2_arrival_airport)
  assert flight1_arrival_airport == flight2_departure_airport, (flight1_arrival_airport, flight2_departure_airport)

  if (departing_query.price_limit is None) or (returning_query.price_limit is None):
    price_limit = None
  else:
    price_limit = PriceLimit(max_price=departing_query.price_limit.max_price + returning_query.price_limit.max_price)

  if (departing_query.airlines is None) or (returning_query.airlines is None):
    airlines = None
  else:
    airlines = list(set(departing_query.airlines + returning_query.airlines))

  if (departing_query.max_duration is None) or (returning_query.max_duration is None):
    max_duration = None
  else:
    max_duration = max(departing_query.max_duration, returning_query.max_duration)

  if (departing_query.layover_restrictions is None) or (returning_query.layover_restrictions is None):
    layover_restrictions = None
  else:
    if (departing_query.layover_restrictions.airports is None) or (returning_query.layover_restrictions.airports is None):
      airports = None
    else:
      airports = list(set(departing_query.layover_restrictions.airports + returning_query.layover_restrictions.airports))
    if (departing_query.layover_restrictions.max_duration is None) or (returning_query.layover_restrictions.max_duration is None):
      max_layover_duration = None
    else:
      max_layover_duration = max(departing_query.layover_restrictions.max_duration, returning_query.layover_restrictions.max_duration)
    layover_restrictions = LayoverRestrictions(
      airports=airports,
      max_duration=max_layover_duration,
    )

  return FlightSearchFilters(
    trip_type=TripType.ROUND_TRIP,
    passenger_info=departing_query.passenger_info,
    flight_segments=[
      departing_query.flight_segments[0],
      returning_query.flight_segments[0],
    ],
    stops=MaxStops(max(departing_query.stops.value, returning_query.stops.value)),
    seat_type=departing_query.seat_type,
    price_limit=price_limit,
    airlines=airlines,
    max_duration=max_duration,
    layover_restrictions=layover_restrictions,
    sort_by=departing_query.sort_by,
  )


def search_flights(flight_query: FlightSearchFilters) -> tuple[tuple[list[FlightInfo], list[FlightInfo]], int]:
  search = SearchFlights()
  flight_results = search.search(
    filters=flight_query,
    top_n=1000,
  )

  if flight_results is None:
    return ([], []), hash_flight_query(flight_query)

  departing_flights: dict[str, FlightInfo] = {}
  returning_flights: dict[str, FlightInfo] = {}
  for flight_result in flight_results:

    if isinstance(flight_result, tuple):  # round-trip query
      departing_flight_result, returning_flight_result = flight_result
      departing_flight = FlightInfo.from_flight_result(
        flight_result=departing_flight_result,
        flight_type=FlightType.ROUND_TRIP_DEPARTING,
      )
      returning_flight = FlightInfo.from_flight_result(
        flight_result=returning_flight_result,
        flight_type=FlightType.ROUND_TRIP_RETURNING,
      )

      # Reset prices for round-trips before using their ids and caching them (which is dependent on price).
      returning_flight.price = max(departing_flight.price, returning_flight.price)
      departing_flight.price = 0.0

      if departing_flight.id not in departing_flights:
        departing_flights[departing_flight.id] = departing_flight
      if returning_flight.id not in returning_flights:
        returning_flights[returning_flight.id] = returning_flight
      returning_flights[returning_flight.id].parent_ids.append(departing_flight.id)

    else:  # one-way query
      flight = FlightInfo.from_flight_result(
        flight_result=flight_result,
        flight_type=FlightType.ONE_WAY,
      )
      if flight.id not in departing_flights:
        departing_flights[flight.id] = flight

  return (list(departing_flights.values()), list(returning_flights.values())), hash_flight_query(flight_query)


def search_flights_parallel(flight_queries: Sequence[FlightSearchFilters]) -> dict[int, tuple[list[FlightInfo], list[FlightInfo]]]:
  '''Performs flight search queries in parallel using a thread pool executor.
  Returns a dictionary mapping the query hash to the flight query results.'''

  executor = futures.ThreadPoolExecutor(max_workers=len(flight_queries))
  futs: list[futures.Future[tuple[tuple[list[FlightInfo], list[FlightInfo]], int]]] = []
  for flight_query in flight_queries:
    futs.append(executor.submit(search_flights, flight_query))

  query_hash_to_flight_infos: dict[int, tuple[list[FlightInfo], list[FlightInfo]]] = {}
  for fut in tqdm(futures.as_completed(futs), total=len(futs), desc='Searching flights'):
    flight_infos, query_hash = fut.result()
    query_hash_to_flight_infos[query_hash] = flight_infos
  return query_hash_to_flight_infos
