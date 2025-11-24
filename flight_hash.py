from fli.models import (
  Airport,
  FlightLeg,
  FlightResult,
  FlightSearchFilters,
  FlightSegment,
  LayoverRestrictions,
)


def hash_flight_leg(flight_leg: FlightLeg) -> int:
  return hash(
    (
      flight_leg.airline,
      flight_leg.flight_number,
      flight_leg.departure_airport,
      flight_leg.arrival_airport,
      flight_leg.departure_datetime,
      flight_leg.arrival_datetime,
      flight_leg.duration,
    )
  )

def hash_flight_result(flight_result: FlightResult) -> int:
  return hash(
    (
      tuple(hash_flight_leg(leg) for leg in flight_result.legs),
      flight_result.price,
      flight_result.duration,
      flight_result.stops,
    )
  )

def hash_flight_segment(flight_segment: FlightSegment) -> int:
  return hash(
    (
      tuple(
        sorted(
          tuple(a.name if isinstance(a, Airport) else a for a in airport)
          for airport in flight_segment.departure_airport
        )
      ),
      tuple(
        sorted(
          tuple(a.name if isinstance(a, Airport) else a for a in airport)
          for airport in flight_segment.arrival_airport
        )
      ),
      flight_segment.travel_date,
      (
        flight_segment.time_restrictions.earliest_departure,
        flight_segment.time_restrictions.latest_departure,
        flight_segment.time_restrictions.earliest_arrival,
        flight_segment.time_restrictions.latest_arrival,
      ) if flight_segment.time_restrictions else None,
      hash_flight_result(flight_segment.selected_flight) if flight_segment.selected_flight else None,
    )
  )

def hash_layover_restrictions(layover_restrictions: LayoverRestrictions) -> int:
  return hash(
    (
      tuple(sorted(airport.name for airport in layover_restrictions.airports)) if layover_restrictions.airports else None,
      layover_restrictions.max_duration,
    )
  )

def hash_flight_query(query: FlightSearchFilters) -> int:
  return hash(
    (
      query.trip_type,
      (
        query.passenger_info.adults,
        query.passenger_info.children,
        query.passenger_info.infants_in_seat,
        query.passenger_info.infants_on_lap,
      ),
      tuple(hash_flight_segment(flight_segment) for flight_segment in query.flight_segments),
      query.stops,
      query.seat_type,
      (query.price_limit.max_price, query.price_limit.currency) if query.price_limit else None,
      tuple(sorted(airline.name for airline in query.airlines)) if query.airlines else None,
      query.max_duration,
      hash_layover_restrictions(query.layover_restrictions) if query.layover_restrictions else None,
      query.sort_by,
    )
  )
