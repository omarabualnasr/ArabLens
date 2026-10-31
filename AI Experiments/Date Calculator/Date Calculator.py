"""Offline date arithmetic tool for Tiny AI. Uses only Python's standard library."""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

_MONTHS = {name.lower(): number for number, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): number for number, name in enumerate(calendar.month_abbr) if name})
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMERIC_DATE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b")
_NAMED_DATE = re.compile(
    r"\b(?:\d{1,2}\s+(?:" + "|".join(_MONTHS) + r")\s+\d{4}|"
    r"(?:" + "|".join(_MONTHS) + r")\s+\d{1,2},?\s+\d{4})\b", re.I
)


@dataclass(frozen=True)
class DateCalculatorResult:
    answer: str
    operation: str
    source: str = "tool"
    tool: str = "date_calculator"

    def to_dict(self) -> dict[str, str]:
        return {"answer": self.answer, "source": self.source, "tool": self.tool, "operation": self.operation}


class DateCalculator:
    """Recognize explicit date arithmetic requests without intercepting clock questions."""

    @staticmethod
    def _parse_date(value: str) -> date | None:
        text = value.strip().replace(",", "")
        candidates = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y")
        from datetime import datetime
        for pattern in candidates:
            try:
                return datetime.strptime(text, pattern).date()
            except ValueError:
                pass
        return None

    @classmethod
    def _dates_in(cls, text: str) -> list[date]:
        matches = []
        for pattern in (_ISO_DATE, _NUMERIC_DATE, _NAMED_DATE):
            for match in pattern.finditer(text):
                value = cls._parse_date(match.group(0))
                if value is not None:
                    matches.append((match.start(), value))
        unique: dict[tuple[int, date], date] = {(position, value): value for position, value in matches}
        return [value for _, value in sorted(unique.items(), key=lambda item: item[0][0])]

    @staticmethod
    def _format(value: date) -> str:
        return value.strftime("%A, %B %d, %Y").replace(" 0", " ")

    @staticmethod
    def _add_months(value: date, amount: int) -> date:
        absolute_month = value.year * 12 + (value.month - 1) + amount
        year, month_index = divmod(absolute_month, 12)
        month = month_index + 1
        day = min(value.day, calendar.monthrange(year, month)[1])
        return date(year, month, day)

    @staticmethod
    def _business_days(start: date, end: date) -> int:
        """Count weekdays after start through end, independent of argument order."""
        first, last = sorted((start, end))
        total_days = (last - first).days
        full_weeks, remaining_days = divmod(total_days, 7)
        count = full_weeks * 5
        current = first + timedelta(days=full_weeks * 7 + 1)
        for _ in range(remaining_days):
            if current.weekday() < 5:
                count += 1
            current += timedelta(days=1)
        return count

    def execute(self, question: str) -> DateCalculatorResult | None:
        text = re.sub(r"\s+", " ", str(question or "")).strip()
        lowered = text.lower()
        dates = self._dates_in(text)
        if not dates:
            return None

        # Add/subtract a quantity to/from a concrete date.
        amount_match = re.search(
            r"\b(add|plus|subtract|minus|in)\s+(\d{1,5})\s+(days?|weeks?|months?|years?)\s+(?:to|from|after|before)\s+", lowered
        )
        if amount_match:
            amount = int(amount_match.group(2))
            unit = amount_match.group(3).rstrip("s")
            base = dates[-1]
            direction = -1 if amount_match.group(1) in {"subtract", "minus"} or amount_match.group(0).endswith("before ") else 1
            signed = amount * direction
            try:
                if unit == "day": result = base + timedelta(days=signed)
                elif unit == "week": result = base + timedelta(days=7 * signed)
                elif unit == "month": result = self._add_months(base, signed)
                else: result = self._add_months(base, 12 * signed)
            except (OverflowError, ValueError):
                return DateCalculatorResult("That calculation falls outside the supported calendar range (years 1–9999).", "date_arithmetic")
            return DateCalculatorResult(f"{amount} {unit}{'' if amount == 1 else 's'} {'before' if signed < 0 else 'after'} {self._format(base)} is {self._format(result)}.", "date_arithmetic")

        # Date intervals and duration calculations.
        if len(dates) >= 2 and re.search(r"\b(between|from|until|through|to)\b", lowered):
            first, second = dates[0], dates[1]
            days = abs((second - first).days)
            weeks, remainder = divmod(days, 7)
            if re.search(r"\b(business|working)\s+days?\b", lowered):
                count = self._business_days(first, second)
                return DateCalculatorResult(
                    f"There are {count} weekdays between {self._format(first)} and {self._format(second)} (counting after the start date through the end date; public holidays are not excluded).",
                    "business_days_between",
                )
            if re.search(r"\bweeks?\b", lowered):
                answer = f"The dates are {weeks} week(s) and {remainder} day(s) apart ({days} days total)."
            else:
                answer = f"The dates are {days} day(s) apart ({weeks} week(s) and {remainder} day(s))."
            return DateCalculatorResult(answer, "days_between")

        # Weekday for an explicit date.
        if re.search(r"\b(weekday|day of (?:the )?week|what day)\b", lowered):
            value = dates[0]
            return DateCalculatorResult(f"{value.strftime('%B %d, %Y').replace(' 0', ' ')} fell on a {value.strftime('%A')}.", "weekday")
        return None


__all__ = ["DateCalculator", "DateCalculatorResult"]
