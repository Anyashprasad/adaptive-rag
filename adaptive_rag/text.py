from __future__ import annotations

import re
from datetime import date, datetime, timedelta

TOKEN_RE = re.compile(r"[A-Za-z0-9_./:@+-]+")
DATE_PATTERNS = [
    re.compile(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b"),
    re.compile(r"\b(0?[1-9]|[12]\d|3[01])[-/](0?[1-9]|1[0-2])[-/](20\d{2})\b"),
]
MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
MONTH_DATE_RE = re.compile(
    r"\b(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+"
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"(?:\s+(20\d{2}))?\b",
    re.IGNORECASE,
)
TIME_RE = re.compile(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)(?:\s*([ap]m))?\b", re.IGNORECASE)
RELATIVE_RE = re.compile(r"\b(today|yesterday|tomorrow)\b", re.IGNORECASE)
LAST_DAYS_RE = re.compile(r"\b(?:last|past)\s+(\d{1,3})\s+days?\b", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def normalize_dates(text: str, default_year: int | None = None) -> list[str]:
    out: list[str] = []
    for pat in DATE_PATTERNS:
        for m in pat.finditer(text):
            g = m.groups()
            if len(g[0]) == 4:
                year, month, day = map(int, g)
            else:
                day, month, year = map(int, g)
            try:
                out.append(datetime(year, month, day).date().isoformat())
            except ValueError:
                pass
    for m in MONTH_DATE_RE.finditer(text):
        day = int(m.group(1))
        month_key = m.group(2).lower()
        month = MONTHS.get(month_key) or MONTHS.get(month_key[:3])
        year = int(m.group(3)) if m.group(3) else default_year
        if month and year:
            try:
                out.append(datetime(year, month, day).date().isoformat())
            except ValueError:
                pass
    return sorted(set(out))


def normalize_times(text: str) -> list[str]:
    out: list[str] = []
    for m in TIME_RE.finditer(text):
        hour, minute = int(m.group(1)), int(m.group(2))
        meridiem = (m.group(3) or "").lower()
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        out.append(f"{hour:02d}:{minute:02d}")
    return sorted(set(out))


def _iso(d: date | datetime) -> str:
    return (d.date() if isinstance(d, datetime) else d).isoformat()


def parse_temporal_constraints(text: str, reference_time: datetime | None = None) -> dict[str, object]:
    """Parse lightweight temporal constraints without external NLP dependencies.

    The parser intentionally favors deterministic, auditable rules over fuzzy date interpretation.
    It recognizes explicit dates/times, today/yesterday/tomorrow, last N days, and
    simple before/after/between/from-to operators.
    """
    ref = reference_time or datetime.now().astimezone()
    dates = normalize_dates(text, default_year=ref.year)
    lower = text.lower()

    relative_dates: list[str] = []
    for m in RELATIVE_RE.finditer(text):
        word = m.group(1).lower()
        offset = {"yesterday": -1, "today": 0, "tomorrow": 1}[word]
        relative_dates.append(_iso(ref + timedelta(days=offset)))
    dates = sorted(set(dates + relative_dates))

    times = normalize_times(text)
    date_start: str | None = None
    date_end: str | None = None
    time_start: str | None = None
    time_end: str | None = None
    operator: str | None = None

    last_days = LAST_DAYS_RE.search(text)
    if last_days:
        n = max(1, int(last_days.group(1)))
        date_end = _iso(ref)
        date_start = _iso(ref - timedelta(days=n - 1))
        operator = "range"
    elif len(dates) >= 2 and re.search(r"\b(between|from)\b", lower):
        date_start, date_end = min(dates), max(dates)
        operator = "range"
    elif dates:
        first = dates[0]
        if re.search(r"\b(after|since|newer than|later than)\b", lower):
            date_start, operator = first, "after"
        elif re.search(r"\b(before|until|through|up to|older than)\b", lower):
            date_end, operator = first, "before"
        else:
            date_start = date_end = first
            operator = "exact"

    if len(times) >= 2 and re.search(r"\b(between|from)\b", lower):
        time_start, time_end = min(times), max(times)
        operator = operator or "range"
    elif times:
        first_t = times[0]
        if re.search(r"\b(after|since|later than)\b", lower):
            time_start = first_t
        elif re.search(r"\b(before|until|through|up to|earlier than)\b", lower):
            time_end = first_t
        else:
            time_start = time_end = first_t

    return {
        "dates": dates,
        "times": times,
        "date_start": date_start,
        "date_end": date_end,
        "time_start": time_start,
        "time_end": time_end,
        "operator": operator,
        "reference_time": ref.isoformat(timespec="seconds"),
    }



DATETIME_PAIR_RE = re.compile(
    r"(?P<date>(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]20\d{2}|"
    r"\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+20\d{2})?))"
    r"[^\n]{0,28}?"
    r"(?P<time>(?:[01]?\d|2[0-3])[:.]\d{2}(?:\s*[ap]m)?)",
    re.IGNORECASE,
)


def normalize_datetimes(text: str, default_year: int | None = None) -> list[str]:
    """Extract date+time pairs that occur near one another, preserving event-level precision."""
    out: list[str] = []
    for m in DATETIME_PAIR_RE.finditer(text):
        ds = normalize_dates(m.group("date"), default_year=default_year)
        ts = normalize_times(m.group("time"))
        if ds and ts:
            out.append(f"{ds[0]}T{ts[0]}")
    return sorted(set(out))

def within_iso_range(value: str, start: str | None, end: str | None) -> bool:
    if start is not None and value < start:
        return False
    if end is not None and value > end:
        return False
    return True
