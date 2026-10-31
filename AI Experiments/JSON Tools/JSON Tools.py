"""Safe JSON validation, formatting, minification, and inspection."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JSONToolsResult:
    answer: str
    operation: str
    source: str = "tool"
    tool: str = "json_tools"

    def to_dict(self) -> dict[str, str]:
        return {"answer": self.answer, "source": self.source, "tool": self.tool, "operation": self.operation}


class JSONTools:
    _COMMAND = re.compile(r"^\s*(format|pretty(?:-print)?|validate|check|minify|inspect)\s+(?:this\s+)?json\s*[:\-]?\s*", re.I | re.S)

    @staticmethod
    def _reject_constant(value: str) -> None:
        raise ValueError(f"Non-standard JSON constant '{value}' is not permitted.")

    @staticmethod
    def _shape(value: Any) -> str:
        if isinstance(value, dict):
            return f"object with {len(value)} key(s)"
        if isinstance(value, list):
            return f"array with {len(value)} item(s)"
        if value is None: return "null"
        if isinstance(value, bool): return "boolean"
        if isinstance(value, str): return "string"
        if isinstance(value, (int, float)): return "number"
        return type(value).__name__

    def execute(self, question: str) -> JSONToolsResult | None:
        match = self._COMMAND.match(str(question or ""))
        if not match:
            return None
        command = match.group(1).lower()
        operation = "pretty" if command in {"format", "pretty", "pretty-print"} else "validate" if command in {"validate", "check"} else command
        raw = str(question)[match.end():].strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
        if not raw:
            return None
        if len(raw) > 50_000:
            return JSONToolsResult("That JSON input is too large (maximum 50,000 characters).", operation)
        try:
            value = json.loads(raw, parse_constant=self._reject_constant)
        except (json.JSONDecodeError, ValueError, RecursionError) as exc:
            if isinstance(exc, json.JSONDecodeError):
                message = f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}."
            else:
                message = f"Invalid JSON: {exc}."
            return JSONToolsResult(message, operation)

        if operation == "validate":
            answer = f"Valid JSON. Top-level type: {self._shape(value)}."
        elif operation == "pretty":
            answer = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        elif operation == "minify":
            answer = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        else:
            lines = [f"Top-level: {self._shape(value)}"]
            if isinstance(value, dict):
                lines.append("Keys:")
                lines.extend(f"• {key}: {self._shape(item)}" for key, item in value.items())
            elif isinstance(value, list):
                lines.append("First items:")
                lines.extend(f"• [{index}]: {self._shape(item)}" for index, item in enumerate(value[:10]))
            lines.append(f"Serialized size: {len(json.dumps(value, ensure_ascii=False))} characters")
            answer = "\n".join(lines)
        return JSONToolsResult(answer, operation)


__all__ = ["JSONTools", "JSONToolsResult"]
