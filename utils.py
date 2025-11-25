import re

import time

import enum
import dataclasses

import heapq

from concurrent import futures
import traceback

from typing import Any, Callable, Sequence, TypeVar

from fli.models import Airport


AirportKey = tuple[str, ...]


_T = TypeVar('_T')


def get_time_id() -> str:
  return str(round(time.time(), 6)).replace(".", "_")


def minutes_to_string(total_minutes: int):
  assert isinstance(total_minutes, int)

  minutes_remaining = total_minutes
  days = total_minutes // (60 * 24)
  minutes_remaining -= days * 60 * 24
  hours = minutes_remaining // 60
  minutes_remaining -= hours * 60

  string = ''
  if days > 0:
    string += f'{days} day ' if (days==1) else f'{days} days '
  if hours > 0:
    string += f'{hours} hr '
  if minutes_remaining > 0:
    string += f'{minutes_remaining} min'
  return string.strip()


def get_top_n(list_elements: Sequence[_T], *, top_n: int, sort_key: Callable[[_T], Any]) -> list[_T]:
  '''Given a list of elements, get the top-n highest elements.
  NOTE: returns a list of top-n elements sorted in DESCENDING order.'''
  min_heap: list[tuple[Any, int, _T]] = []
  for i, element in enumerate(list_elements):
    comparison_value = sort_key(element)
    if len(min_heap) < top_n:
      heapq.heappush(min_heap, (comparison_value, -i, element))  # add an int for tie-breakers; prioritize earlier elements
    elif comparison_value > min_heap[0][0]:
      heapq.heappop(min_heap)
      heapq.heappush(min_heap, (comparison_value, -i, element))  # add an int for tie-breakers; prioritize earlier elements

  sorted_elements: list[_T] = []
  while min_heap:
    sorted_elements.append(heapq.heappop(min_heap)[-1])
  return list(reversed(sorted_elements))


def convert_list_enum_to_canonical_tuple_str(list_enum: Sequence[enum.Enum], *, attr_name: str = 'name') -> tuple[str, ...]:
  return tuple(sorted(getattr(e, attr_name) for e in list_enum))


def get_airport_key_from_airport_list(airport_list: list[list[Airport | int]]) -> AirportKey:
  '''Convert the airport fields in `FlightSearchFilters` to a dict key.'''
  airport_key: tuple[str, ...] = tuple(
    sorted(airport[0].name for airport in airport_list)  # type: ignore
  )
  return airport_key


def extract_python_code_blocks(text: str) -> list[str]:
  matches = re.findall(r"<python>\s*(.*?)\s*</python>", text, flags=re.DOTALL)
  if not matches:
    # Sometimes Gemini will use ```python...``` for code blocks even if you specify to use <python>...</python>
    matches = re.findall(r'```python\s*(.+?)\s*```', text, flags=re.DOTALL)
  return matches

def extract_clarification_blocks(text: str) -> list[str]:
  matches = re.findall(r"<clarification>\s*(.*?)\s*</clarification>", text, flags=re.DOTALL)
  if not matches:
    # Sometimes Gemini will use ```clarification...``` for clarification blocks even if you specify to use <clarification>...</clarification>
    matches = re.findall(r'```clarification\s*(.+?)\s*```', text, flags=re.DOTALL)
  return matches


@dataclasses.dataclass(kw_only=True, frozen=True)
class CodeExecutionResult:
  code_executor: 'CodeExecutor'
  success: bool  # code succeeded in executing without throwing an error
  result: Any | None = None  # the return value of the executed code
  traceback: str | None = None
  exception: Exception | None = None  # the Exception object thrown, if any


# TODO: make this more secure by preventing file manipulation
@dataclasses.dataclass(frozen=True)
class CodeExecutor:
  code: str
  timeout_seconds: float | None = 30.0

  def import_statements(self) -> str:
    return '''import datetime
from city import CityRanges, CityRange, DepartureFlightFilter
from datetime_range import DateRange, TimeRange
from fli.models import Airline, Airport, LayoverRestrictions, MaxStops, SeatType, PriceLimit'''

  def execute_code_and_get_variable(self, var_name: str) -> Any:
    namespace = {}
    exec(f'{self.import_statements()}\n\n{self.code}', namespace)
    return namespace[var_name]

  def __getattr__(self, name: str) -> CodeExecutionResult:
    try:
      if self.timeout_seconds is None:
        result = self.execute_code_and_get_variable(name)
      else:
        executor = futures.ThreadPoolExecutor(max_workers=1)
        result = executor.submit(self.execute_code_and_get_variable, name).result(timeout=self.timeout_seconds)
      return CodeExecutionResult(code_executor=self, success=True, result=result)
    except Exception as e:
      traceback_str = traceback.format_exc()
      return CodeExecutionResult(code_executor=self, success=False, traceback=traceback_str, exception=e)
