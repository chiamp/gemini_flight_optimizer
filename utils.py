import re

import time

import enum
import dataclasses

from concurrent import futures
import traceback

from typing import Any, Sequence


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


def convert_list_enum_to_canonical_tuple_str(list_enum: Sequence[enum.Enum]) -> tuple[str, ...]:
  return tuple(sorted(e.name for e in list_enum))


def extract_python_code_blocks(text: str) -> list[str]:
  return re.findall(r"<python>(.*?)</python>", text, flags=re.DOTALL)

def extract_clarification_blocks(text: str) -> list[str]:
  return re.findall(r"<clarification>(.*?)</clarification>", text, flags=re.DOTALL)


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
