import functools
import itertools

import datetime
import dataclasses

from typing import Generator

from pydantic import PositiveInt

from fli.models import (
  Airport,
  Airline,
  FlightSearchFilters,
  FlightSegment,
  LayoverRestrictions,
  MaxStops,
  PassengerInfo,
  PriceLimit,
  SeatType,
  SortBy,
  TimeRestrictions,
  TripType,
)

from datetime_range import (
  DateTimeRange,
  DateRange,
  TimeRange
)
from flight_hash import hash_layover_restrictions
from utils import (
  convert_list_enum_to_canonical_tuple_str,
  minutes_to_string
)


@dataclasses.dataclass(kw_only=True)
class DepartureFlightFilter:
  stops: MaxStops = MaxStops.ANY
  seat_type: SeatType = SeatType.ECONOMY
  price_limit: PriceLimit | None = None
  airlines: list[Airline] | None = None
  max_duration: PositiveInt | None = None  # in minutes
  layover_restrictions: LayoverRestrictions | None = None

  @functools.cached_property
  def hash(self) -> int:
    return hash(
      (
        self.stops,
        self.seat_type,
        (self.price_limit.max_price, self.price_limit.currency) if self.price_limit else None,
        tuple(airline for airline in self.airlines) if self.airlines else None,
        self.max_duration,
        hash_layover_restrictions(self.layover_restrictions) if self.layover_restrictions else None,
      )
    )

  def __str__(self) -> str:
    string_components: list[str] = [
      f'Departing flight - maximum number of stops: {self.stops.name}',
      f'Departing flight - seat type: {self.seat_type.name}',
    ]

    if self.price_limit:
      string_components.append(f'Departing flight - maximum price: ${self.price_limit.max_price}')
    if self.airlines:
      airlines_component = '[' + ', '.join(convert_list_enum_to_canonical_tuple_str(self.airlines)) + ']'
      string_components.append(f'Departing flight - airlines constrained to: {airlines_component}')
    if self.max_duration:
      string_components.append(f'Departing flight - maximum duration of flight: {minutes_to_string(self.max_duration)}')
    if self.layover_restrictions:
      if self.layover_restrictions.airports:
        airports_component = '[' + ', '.join(convert_list_enum_to_canonical_tuple_str(self.layover_restrictions.airports)) + ']'
        string_components.append(f'Departing flight - airports constrained to: {airports_component}')
      if self.layover_restrictions.max_duration:
        string_components.append(f'Departing flight - maximum duration of layover: {minutes_to_string(self.layover_restrictions.max_duration)}')

    return '\n'.join(string_components)


@dataclasses.dataclass(kw_only=True)
class City:
  airports: list[Airport]

  arrival_date: datetime.date | None = None
  arrival_time_range: TimeRange | None = None
  departure_date: datetime.date | None = None
  departure_time_range: TimeRange | None = None

  departure_flight_filter: DepartureFlightFilter = dataclasses.field(default_factory=DepartureFlightFilter)

  @functools.cached_property
  def id(self) -> tuple[str, ...]:
    # Sort for canonical ordering. To be used as a dict key.
    return tuple(convert_list_enum_to_canonical_tuple_str(self.airports))

  @functools.cached_property
  def hash(self) -> int:
    return hash(
      (
        tuple(sorted(airport.name for airport in self.airports)),
        self.arrival_date,
        self.arrival_time_range,
        self.departure_date,
        self.departure_time_range,
        self.departure_flight_filter.hash,
      )
    )

  @functools.cached_property
  def arrival_datetime_range(self) -> DateTimeRange | None:
    if (self.arrival_date is None) or (self.arrival_time_range is None) or (self.arrival_time_range.earliest is None) or (self.arrival_time_range.latest is None):
      return None
    return DateTimeRange(
      earliest=datetime.datetime.combine(self.arrival_date, self.arrival_time_range.earliest),
      latest=datetime.datetime.combine(self.arrival_date, self.arrival_time_range.latest),
    )

  @functools.cached_property
  def departure_datetime_range(self) -> DateTimeRange | None:
    if (self.departure_date is None) or (self.departure_time_range is None) or (self.departure_time_range.earliest is None) or (self.departure_time_range.latest is None):
      return None
    return DateTimeRange(
      earliest=datetime.datetime.combine(self.departure_date, self.departure_time_range.earliest),
      latest=datetime.datetime.combine(self.departure_date, self.departure_time_range.latest),
    )

  @classmethod
  def create_flight_search_query(cls, departure_city: 'City', arrival_city: 'City') -> FlightSearchFilters:
    '''Given a departure and arrival city, construct a flight query based on their departure/arrival datetime constraints.'''

    assert departure_city.departure_date
    assert (arrival_city.arrival_date is None) or (arrival_city.arrival_date >= departure_city.departure_date)

    if departure_city.departure_time_range:
      earliest_departure, latest_departure = departure_city.departure_time_range.convert_time_range_to_hour_range()
    else:
      earliest_departure, latest_departure = None, None
    if arrival_city.arrival_time_range:
      earliest_arrival, latest_arrival = arrival_city.arrival_time_range.convert_time_range_to_hour_range()
    else:
      earliest_arrival, latest_arrival = None, None

    return FlightSearchFilters(
      trip_type=TripType.ONE_WAY,
      passenger_info=PassengerInfo(adults=1),
      flight_segments=[
        FlightSegment(
          departure_airport=[[airport, 0] for airport in departure_city.airports],
          arrival_airport=[[airport, 0] for airport in arrival_city.airports],
          travel_date=departure_city.departure_date.isoformat(),
          time_restrictions=TimeRestrictions(
            earliest_departure=earliest_departure,
            latest_departure=latest_departure,
            earliest_arrival=earliest_arrival,
            latest_arrival=latest_arrival,
          ),
        )
      ],
      stops=departure_city.departure_flight_filter.stops,
      seat_type=departure_city.departure_flight_filter.seat_type,
      price_limit=departure_city.departure_flight_filter.price_limit,
      airlines=departure_city.departure_flight_filter.airlines,
      max_duration=departure_city.departure_flight_filter.max_duration,
      layover_restrictions=departure_city.departure_flight_filter.layover_restrictions,
      sort_by=SortBy.NONE,
    )


@dataclasses.dataclass(kw_only=True)
class CityRange:
  '''Represents the arrival/departure datetime and min/max stay hour constraints for a city.
  The `get_all_possible_city_configurations` method will enumerate all possible `City` configurations for this `CityRange`.'''

  airports: list[Airport]

  min_stay_hours: float | None = None
  max_stay_hours: float | None = None

  arrival_date_range: DateRange | None = None
  arrival_time_range: TimeRange | None = None
  departure_date_range: DateRange | None = None
  departure_time_range: TimeRange | None = None

  departure_flight_filter: DepartureFlightFilter = dataclasses.field(default_factory=DepartureFlightFilter)

  def __str__(self) -> str:
    airports_component = '[' + ', '.join(convert_list_enum_to_canonical_tuple_str(self.airports)) + ']'

    string_components: list[str] = [f'Airports: {airports_component}']

    if self.min_stay_hours:
      string_components.append(f'Minimum time spent in city: {minutes_to_string(round(self.min_stay_hours * 60))}')
    if self.max_stay_hours:
      string_components.append(f'Maximum time spent in city: {minutes_to_string(round(self.max_stay_hours * 60))}')
    if self.arrival_date_range:
      if self.arrival_date_range.earliest:
        string_components.append(f'Earliest date of arrival: {self.arrival_date_range.earliest.isoformat()}')
      if self.arrival_date_range.latest:
        string_components.append(f'Latest date of arrival: {self.arrival_date_range.latest.isoformat()}')
    if self.arrival_time_range:
      if self.arrival_time_range.earliest:
        string_components.append(f'Earliest time of arrival: {self.arrival_time_range.earliest.isoformat()}')
      if self.arrival_time_range.latest:
        string_components.append(f'Latest time of arrival: {self.arrival_time_range.latest.isoformat()}')
    if self.departure_date_range:
      if self.departure_date_range.earliest:
        string_components.append(f'Earliest date of departure: {self.departure_date_range.earliest.isoformat()}')
      if self.departure_date_range.latest:
        string_components.append(f'Latest date of departure: {self.departure_date_range.latest.isoformat()}')
    if self.departure_time_range:
      if self.departure_time_range.earliest:
        string_components.append(f'Earliest time of departure: {self.departure_time_range.earliest.isoformat()}')
      if self.departure_time_range.latest:
        string_components.append(f'Latest time of departure: {self.departure_time_range.latest.isoformat()}')

    string_components.append(str(self.departure_flight_filter))

    return '\n'.join(string_components)

  def last_city_string(self) -> str:
    # Used for the last city. Omits departure information and city stay duration information.
    airports_component = '[' + ', '.join(convert_list_enum_to_canonical_tuple_str(self.airports)) + ']'

    string_components: list[str] = [f'Airports: {airports_component}']

    if self.arrival_date_range:
      if self.arrival_date_range.earliest:
        string_components.append(f'Earliest date of arrival: {self.arrival_date_range.earliest.isoformat()}')
      if self.arrival_date_range.latest:
        string_components.append(f'Latest date of arrival: {self.arrival_date_range.latest.isoformat()}')
    if self.arrival_time_range:
      if self.arrival_time_range.earliest:
        string_components.append(f'Earliest time of arrival: {self.arrival_time_range.earliest.isoformat()}')
      if self.arrival_time_range.latest:
        string_components.append(f'Latest time of arrival: {self.arrival_time_range.latest.isoformat()}')

    return '\n'.join(string_components)

  def get_all_possible_city_configurations(self) -> list[City]:
    # Get every possible combination of arrival and departure dates for this City.
    if self.arrival_date_range and self.arrival_date_range.earliest and self.arrival_date_range.latest:
      arrival_date_range = self.arrival_date_range
    else:
      arrival_date_range = [None]
    if self.departure_date_range and self.departure_date_range.earliest and self.departure_date_range.latest:
      departure_date_range = self.departure_date_range
    else:
      departure_date_range = [None]

    cities: list[City] = []
    for arrival_date in arrival_date_range:
      for departure_date in departure_date_range:
        # Filter out for arrival/departure date/time ranges that end up in a city stay that breaks the min_stay_hours or max_stay_hours constraint.
        if arrival_date and departure_date:
          if self.arrival_time_range and self.departure_time_range:
            if self.arrival_time_range.earliest and self.departure_time_range.latest:
              earliest_arrival_datetime = datetime.datetime.combine(arrival_date, self.arrival_time_range.earliest)
              latest_departure_datetime = datetime.datetime.combine(departure_date, self.departure_time_range.latest)
              max_stay_hours = (latest_departure_datetime - earliest_arrival_datetime).total_seconds() / 3600
            else:
              max_stay_hours = None
            if self.arrival_time_range.latest and self.departure_time_range.earliest:
              latest_arrival_datetime = datetime.datetime.combine(arrival_date, self.arrival_time_range.latest)
              earliest_departure_datetime = datetime.datetime.combine(departure_date, self.departure_time_range.earliest)
              min_stay_hours = (earliest_departure_datetime - latest_arrival_datetime).total_seconds() / 3600
            else:
              min_stay_hours = None
          else:
            max_stay_hours = (departure_date - arrival_date).total_seconds() / 3600
            min_stay_hours = (departure_date - arrival_date).total_seconds() / 3600

          if self.min_stay_hours and max_stay_hours and (max_stay_hours < self.min_stay_hours):
            continue
          if self.max_stay_hours and min_stay_hours and (min_stay_hours > self.max_stay_hours):
            continue

        if (arrival_date is None) or (departure_date is None) or (arrival_date <= departure_date):
          cities.append(
            City(
              airports=self.airports,
              arrival_date=arrival_date,
              arrival_time_range=self.arrival_time_range,
              departure_date=departure_date,
              departure_time_range=self.departure_time_range,
              departure_flight_filter=self.departure_flight_filter,
            )
          )
    return cities

  @classmethod
  def get_city_itineraries(cls, city_ranges: list['CityRange']) -> Generator[tuple[City, ...], None, None]:
    '''Given a list[CityRange] which represents a trip itinerary, but the city arrival/departure datetime and min/max stay hour constraints are variable,
    generate every possible trip itinerary from every possible city configuration of each CityRange in the list,
    so long as the departure city's departure date/datetime is not greater than the arrival city's arrival date/datetime.'''

    # Each element is all possible configurations for that one city.
    # Each subsequent element is all possible configurations of the next city to travel to.
    all_cities_all_possible_configurations: list[list[City]] = [city_range.get_all_possible_city_configurations() for city_range in city_ranges]
    for city_itinerary in itertools.product(*all_cities_all_possible_configurations):
      valid_itinerary = True

      for i in range(len(city_itinerary)-1):
        departure_date = city_itinerary[i].departure_date
        arrival_date = city_itinerary[i+1].arrival_date
        if departure_date and arrival_date and (departure_date > arrival_date):
          valid_itinerary = False
          break

        departure_datetime_range = city_itinerary[i].departure_datetime_range
        arrival_datetime_range = city_itinerary[i].arrival_datetime_range
        if departure_datetime_range and arrival_datetime_range and departure_datetime_range.earliest and arrival_datetime_range.latest and (departure_datetime_range.earliest > arrival_datetime_range.latest):
          valid_itinerary = False
          break

      if valid_itinerary:
        # A tuple of cities in travel order for this itinerary
        yield city_itinerary


@dataclasses.dataclass(kw_only=True)
class CityRanges:
  '''Represents one trip itinerary of `CityRanges`.
  Used as:
  - a container for printing out the proposed itinerary (to see if the user wants to modify it before submitting it as a flight query)
  - a flight query to be passed into `flight_itinerary.search_flight_itineraries`
  '''

  city_range_list: list[CityRange]

  def __str__(self) -> str:
    city_range_strings: list[str] = []
    for i, city_range in enumerate(self.city_range_list):
      if i == (len(self.city_range_list)-1):
        city_range_strings.append(f'----- CITY {i+1} -----\n{city_range.last_city_string()}')
      else:
        city_range_strings.append(f'----- CITY {i+1} -----\n{str(city_range)}')
    return '\n\n'.join(city_range_strings)

  @classmethod
  def print_proposed_flight_queries(cls, city_ranges_list: list['CityRanges']) -> None:
    proposed_queries_strings: list[str] = []
    for i, city_ranges in enumerate(city_ranges_list):
      proposed_queries_strings.append('\n' + '='*20 + f' SEARCH QUERY {i+1} ' + '='*20 + f'\n\n{city_ranges}\n')
    print(f'[GEMINI]: Proposed flight queries to search:\n\n' + '\n'.join(proposed_queries_strings))
