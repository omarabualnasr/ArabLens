"""Deterministic, offline unit conversion tool for Tiny AI.

The tool intentionally supports a finite, explicit set of physical units. It
uses Decimal arithmetic and never evaluates user-provided code or expressions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Final


@dataclass(frozen=True)
class Unit:
    dimension: str
    factor: Decimal = Decimal("1")  # factor to the dimension's base unit
    offset: Decimal = Decimal("0")  # Celsius conversion: (value + offset) * factor


@dataclass(frozen=True)
class ConversionResult:
    answer: str
    value: str
    source_unit: str
    target_unit: str
    tool: str = "unit_converter"
    source: str = "tool"

    def to_dict(self) -> dict[str, str]:
        return {
            "answer": self.answer,
            "source": self.source,
            "tool": self.tool,
            "value": self.value,
            "source_unit": self.source_unit,
            "target_unit": self.target_unit,
        }


def _unit(dimension: str, factor: str = "1", offset: str = "0") -> Unit:
    return Unit(dimension, Decimal(factor), Decimal(offset))


_UNITS: Final[dict[str, tuple[str, Unit]]] = {}


def _register(unit: Unit, canonical: str, *aliases: str) -> None:
    for alias in (canonical, *aliases):
        _UNITS[alias.casefold().replace("°", "").replace("²", "2").replace("³", "3")] = (canonical, unit)


# Length, base unit: metre.
_register(_unit("length", "0.001"), "mm", "millimeter", "millimeters", "millimetre", "millimetres")
_register(_unit("length", "0.01"), "cm", "centimeter", "centimeters", "centimetre", "centimetres")
_register(_unit("length"), "m", "meter", "meters", "metre", "metres")
_register(_unit("length", "1000"), "km", "kilometer", "kilometers", "kilometre", "kilometres")
_register(_unit("length", "0.0254"), "in", "inch", "inches")
_register(_unit("length", "0.3048"), "ft", "foot", "feet")
_register(_unit("length", "0.9144"), "yd", "yard", "yards")
_register(_unit("length", "1609.344"), "mi", "mile", "miles")

# Mass, base unit: kilogram.
_register(_unit("mass", "0.000001"), "mg", "milligram", "milligrams")
_register(_unit("mass", "0.001"), "g", "gram", "grams")
_register(_unit("mass"), "kg", "kilogram", "kilograms", "kilo", "kilos")
_register(_unit("mass", "1000"), "t", "tonne", "tonnes", "metric ton", "metric tons")
_register(_unit("mass", "0.028349523125"), "oz", "ounce", "ounces")
_register(_unit("mass", "0.45359237"), "lb", "lbs", "pound", "pounds")

# Volume, base unit: litre.
_register(_unit("volume", "0.001"), "ml", "milliliter", "milliliters", "millilitre", "millilitres")
_register(_unit("volume"), "l", "liter", "liters", "litre", "litres")
_register(_unit("volume", "0.00492892159375"), "tsp", "teaspoon", "teaspoons")
_register(_unit("volume", "0.01478676478125"), "tbsp", "tablespoon", "tablespoons")
_register(_unit("volume", "0.2365882365"), "cup", "cups")
_register(_unit("volume", "3.785411784"), "gal", "gallon", "gallons")

# Time, base unit: second.
_register(_unit("time", "0.001"), "ms", "millisecond", "milliseconds")
_register(_unit("time"), "s", "sec", "second", "seconds")
_register(_unit("time", "60"), "min", "minute", "minutes", "mins")
_register(_unit("time", "3600"), "h", "hr", "hour", "hours", "hrs")
_register(_unit("time", "86400"), "day", "days", "d")
_register(_unit("time", "604800"), "week", "weeks", "wk", "wks")

# Speed, base unit: metres per second.
_register(_unit("speed"), "m/s", "mps", "meter/second", "meters/second")
_register(_unit("speed", str(Decimal("1000") / Decimal("3600"))), "km/h", "kph", "kmh", "kilometers/hour", "kilometres/hour")
_register(_unit("speed", "0.44704"), "mph", "mile/hour", "miles/hour")

# Area, base unit: square metre.
_register(_unit("area"), "m2", "sqm", "squaremeter", "squaremeters", "squaremetre", "squaremetres")
_register(_unit("area", "1000000"), "km2", "sqkm", "squarekilometer", "squarekilometers")
_register(_unit("area", "0.09290304"), "ft2", "sqft", "squarefoot", "squarefeet")
_register(_unit("area", "4046.8564224"), "acre", "acres")
_register(_unit("area", "10000"), "hectare", "hectares", "ha")

# Digital storage, decimal SI convention (1 kB = 1,000 bytes).
_register(_unit("data"), "byte", "bytes", "b")
_register(_unit("data", "1000"), "kb", "kilobyte", "kilobytes")
_register(_unit("data", "1000000"), "mb", "megabyte", "megabytes")
_register(_unit("data", "1000000000"), "gb", "gigabyte", "gigabytes")
_register(_unit("data", "1000000000000"), "tb", "terabyte", "terabytes")

# Temperature is stored as an affine transform to Celsius.
_register(_unit("temperature"), "c", "°c", "celsius")
_register(_unit("temperature", str(Decimal(5) / Decimal(9)), "-32"), "f", "°f", "fahrenheit")
_register(_unit("temperature", str(Decimal(1)), "-273.15"), "k", "kelvin")

_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?"
_DIRECT_PATTERN = re.compile(
    rf"(?P<value>{_NUMBER})\s*(?P<source>[a-zA-Z°²³/]+)\s+(?:to|into|in)\s+(?P<target>[a-zA-Z°²³/]+)\b",
    re.IGNORECASE,
)
_REVERSED_PATTERN = re.compile(
    rf"\bhow\s+many\s+(?P<target>[a-zA-Z°²³/]+)\s+(?:is|are)\s+(?P<value>{_NUMBER})\s*(?P<source>[a-zA-Z°²³/]+)\b",
    re.IGNORECASE,
)


def _resolve_unit(text: str) -> tuple[str, Unit] | None:
    key = text.casefold().replace("°", "").replace("²", "2").replace("³", "3")
    return _UNITS.get(key)


def _format_number(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("The conversion result is not finite.")
    if value == 0:
        return "0"
    rounded = value.quantize(Decimal("0.000001")) if abs(value) < Decimal("1000000000") else value.normalize()
    text = format(rounded, "f") if rounded.adjusted() < 12 else format(rounded, "E")
    if "E" not in text:
        text = text.rstrip("0").rstrip(".") if "." in text else text
    return text


class UnitConverter:
    """Recognize and perform safe, offline conversions between supported units."""

    @staticmethod
    def classify(question: str) -> tuple[Decimal, str, str] | None:
        text = re.sub(r"\s+", " ", str(question or "").strip())
        match = _DIRECT_PATTERN.search(text) or _REVERSED_PATTERN.search(text)
        if not match:
            return None
        source = _resolve_unit(match.group("source"))
        target = _resolve_unit(match.group("target"))
        if source is None or target is None:
            return None
        if source[1].dimension != target[1].dimension:
            return None
        try:
            value = Decimal(match.group("value").replace(",", ""))
        except InvalidOperation:
            return None
        if not value.is_finite():
            return None
        return value, match.group("source"), match.group("target")

    def execute(self, question: str) -> ConversionResult | None:
        classified = self.classify(question)
        if classified is None:
            return None
        value, source_text, target_text = classified
        source = _resolve_unit(source_text)
        target = _resolve_unit(target_text)
        if source is None or target is None:
            return None
        source_name, source_unit = source
        target_name, target_unit = target
        if source_unit.dimension != target_unit.dimension:
            return None

        try:
            if source_unit.dimension == "temperature":
                # Unit transforms are expressed as (value + offset) * factor to °C.
                base_celsius = (value + source_unit.offset) * source_unit.factor
                if target_name == "c":
                    converted = base_celsius
                elif target_name == "f":
                    converted = base_celsius * Decimal(9) / Decimal(5) + Decimal(32)
                else:  # Kelvin
                    converted = base_celsius + Decimal("273.15")
            else:
                converted = value * source_unit.factor / target_unit.factor
        except (InvalidOperation, ZeroDivisionError):
            return None

        result = _format_number(converted)
        original = _format_number(value)
        answer = f"{original} {source_name} = {result} {target_name}"
        return ConversionResult(answer, result, source_name, target_name)
