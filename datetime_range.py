import dataclasses
import datetime

from typing import Any, Iterator


def hour_floor(dt_time: datetime.time) -> int:
  return dt_time.hour

def hour_ceil(dt_time: datetime.time) -> int:
  if dt_time.minute == dt_time.second == dt_time.microsecond == 0:
    return dt_time.hour
  return increment_hour(dt_time.hour)

def increment_hour(hour: int) -> int:
  return (hour + 1) % 24


@dataclasses.dataclass
class Range:
  # Range is inclusive between `earliest` and `latest`.
  earliest: Any
  latest: Any

  def __post_init__(self):
    if (self.earliest is not None) and (self.latest is not None):
      assert self.earliest <= self.latest, f'{self.earliest} is not less or equal to {self.latest}'

  def __contains__(self, item: Any) -> bool:
    return self.earliest <= item <= self.latest

  def __hash__(self) -> int:
    return hash(
      (
        self.earliest,
        self.latest
      )
    )

@dataclasses.dataclass
class DateTimeRange(Range):
  earliest: datetime.datetime | None
  latest: datetime.datetime | None

@dataclasses.dataclass
class DateRange(Range):
  earliest: datetime.date | None = None
  latest: datetime.date | None = None

  def __iter__(self) -> Iterator[datetime.date]:
    assert self.earliest is not None
    assert self.latest is not None

    curr_date = self.earliest
    while curr_date <= self.latest:
      yield curr_date
      curr_date += datetime.timedelta(days=1)

@dataclasses.dataclass
class TimeRange(Range):
  earliest: datetime.time | None = None
  latest: datetime.time | None = None

  def convert_time_range_to_hour_range(self) -> tuple[int | None, int | None]:
    # Hour range so that it's compatible with `fli.models.TimeRestrictions`
    if self.earliest is None:
      earliest_hour = None
    else:
      earliest_hour = hour_floor(self.earliest)
    if self.latest is None:
      latest_hour = None
    else:
      latest_hour = hour_ceil(self.latest)
      if (earliest_hour is not None) and (latest_hour <= earliest_hour):
        latest_hour = increment_hour(latest_hour)
    return earliest_hour, latest_hour
