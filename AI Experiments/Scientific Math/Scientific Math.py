"""Safe scientific expression evaluator and linear/quadratic equation solver."""
from __future__ import annotations

import ast
import cmath
import math
import re
from dataclasses import dataclass
from typing import Callable


_FUNCTIONS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt, "abs": abs, "ln": math.log, "log": math.log10,
    "log10": math.log10, "log2": math.log2, "exp": math.exp,
    "floor": math.floor, "ceil": math.ceil, "round": round,
    "factorial": math.factorial,
    "sin": lambda x: math.sin(math.radians(x)),
    "cos": lambda x: math.cos(math.radians(x)),
    "tan": lambda x: math.tan(math.radians(x)),
}


@dataclass(frozen=True)
class ScientificMathResult:
    answer: str
    operation: str
    source: str = "tool"
    tool: str = "scientific_math"

    def to_dict(self) -> dict[str, str]:
        return {"answer": self.answer, "source": self.source, "tool": self.tool, "operation": self.operation}


class _Evaluator(ast.NodeVisitor):
    def __init__(self, x_value: float = 0.0) -> None:
        self.x_value = x_value
        self.nodes = 0

    def visit(self, node):  # type: ignore[override]
        self.nodes += 1
        if self.nodes > 100:
            raise ValueError("That expression is too complex.")
        return super().visit(node)

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed.")
        value = float(node.value)
        if not math.isfinite(value): raise ValueError("Numbers must be finite.")
        return value

    def visit_Name(self, node: ast.Name) -> float:
        if node.id == "x": return self.x_value
        if node.id == "pi": return math.pi
        if node.id == "e": return math.e
        raise ValueError(f"Unknown symbol: {node.id}.")

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left, right = self.visit(node.left), self.visit(node.right)
        if isinstance(node.op, ast.Add): result = left + right
        elif isinstance(node.op, ast.Sub): result = left - right
        elif isinstance(node.op, ast.Mult): result = left * right
        elif isinstance(node.op, ast.Div): result = left / right
        elif isinstance(node.op, ast.Pow):
            if abs(right) > 12: raise ValueError("Powers above 12 are not supported.")
            result = left ** right
        elif isinstance(node.op, ast.Mod): result = left % right
        else: raise ValueError("That operator is not supported.")
        if not math.isfinite(float(result)) or abs(result) > 1e100:
            raise ValueError("The result is outside the supported range.")
        return float(result)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd): return value
        if isinstance(node.op, ast.USub): return -value
        raise ValueError("That unary operator is not supported.")

    def visit_Call(self, node: ast.Call) -> float:
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS or node.keywords:
            raise ValueError("Only supported built-in math functions may be called.")
        args = [self.visit(argument) for argument in node.args]
        if not 1 <= len(args) <= 2:
            raise ValueError("Use one or two arguments for this function.")
        if node.func.id == "round" and len(args) == 2:
            digits = int(args[1])
            if digits < -12 or digits > 12:
                raise ValueError("round() precision must be between -12 and 12.")
            result = round(args[0], digits)
        elif len(args) != 1:
            raise ValueError(f"{node.func.id} accepts one argument.")
        else:
            if node.func.id == "factorial" and (args[0] < 0 or args[0] > 170 or not args[0].is_integer()):
                raise ValueError("factorial() accepts whole numbers from 0 to 170.")
            result = _FUNCTIONS[node.func.id](args[0])
        if not math.isfinite(float(result)) or abs(result) > 1e100:
            raise ValueError("The result is outside the supported range.")
        return float(result)

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError("That expression contains unsupported syntax.")


def _clean_expression(expression: str) -> str:
    cleaned = expression.strip().replace("×", "*").replace("÷", "/").replace("π", "pi").replace("^", "**")
    cleaned = re.sub(r"(?<=\d)(?=x\b)", "*", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", "", cleaned)
    if not cleaned or len(cleaned) > 500:
        raise ValueError("Enter a non-empty expression under 500 characters.")
    if not re.fullmatch(r"[0-9a-zA-Z_+*/%().,\-]+", cleaned):
        raise ValueError("The expression contains unsupported characters.")
    return cleaned


def _evaluate(expression: str, x_value: float = 0.0) -> float:
    tree = ast.parse(_clean_expression(expression), mode="eval")
    return float(_Evaluator(x_value).visit(tree))


def _format(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-10): return str(int(round(value)))
    return f"{value:.12g}"


def _polynomial(node: ast.AST) -> list[float]:
    """Return constant, x and x² coefficients; reject non-quadratic syntax."""
    zero = [0.0, 0.0, 0.0]
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        zero[0] = float(node.value)
        return zero
    if isinstance(node, ast.Name):
        if node.id == "x": return [0.0, 1.0, 0.0]
        if node.id == "pi": return [math.pi, 0.0, 0.0]
        if node.id == "e": return [math.e, 0.0, 0.0]
        raise ValueError(f"Unknown symbol: {node.id}.")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        values = _polynomial(node.operand)
        return values if isinstance(node.op, ast.UAdd) else [-value for value in values]
    if isinstance(node, ast.BinOp):
        left, right = _polynomial(node.left), _polynomial(node.right)
        if isinstance(node.op, ast.Add): return [a + b for a, b in zip(left, right)]
        if isinstance(node.op, ast.Sub): return [a - b for a, b in zip(left, right)]
        if isinstance(node.op, ast.Div):
            if not math.isclose(right[1], 0.0, abs_tol=1e-12) or not math.isclose(right[2], 0.0, abs_tol=1e-12) or math.isclose(right[0], 0.0, abs_tol=1e-12):
                raise ValueError("Equation division is supported only by a non-zero constant.")
            return [value / right[0] for value in left]
        if isinstance(node.op, ast.Mult):
            output = [0.0, 0.0, 0.0]
            for left_power, left_value in enumerate(left):
                for right_power, right_value in enumerate(right):
                    product = left_value * right_value
                    if left_power + right_power > 2 and not math.isclose(product, 0.0, abs_tol=1e-12):
                        raise ValueError("Only linear and quadratic equations are supported.")
                    if left_power + right_power <= 2:
                        output[left_power + right_power] += product
            return output
        if isinstance(node.op, ast.Pow):
            if any(not math.isclose(value, 0.0, abs_tol=1e-12) for value in right[1:]):
                raise ValueError("The equation's exponent must be a constant integer from 0 to 2.")
            exponent = right[0]
            if exponent not in (0, 1, 2):
                raise ValueError("Only linear and quadratic equations are supported.")
            if exponent == 0: return [1.0, 0.0, 0.0]
            if exponent == 1: return left
            return _polynomial(ast.BinOp(left=node.left, op=ast.Mult(), right=node.left))
        raise ValueError("Only polynomial addition, subtraction, multiplication, division by constants, and powers 0–2 are supported.")
    raise ValueError("This is not a supported polynomial equation.")


class ScientificMath:
    _SOLVE = re.compile(r"^\s*solve\s*:?\s*(.+?)\s*$", re.I | re.S)
    _CALCULATE = re.compile(r"^\s*(?:calculate|evaluate|compute)\s*:?\s*(.+?)\s*$", re.I | re.S)
    _FUNCTION_HINT = re.compile(r"\b(?:sqrt|sin|cos|tan|ln|log|log10|log2|exp|factorial|floor|ceil|round|pi|π|e)\b", re.I)

    def _solve(self, equation: str) -> ScientificMathResult | None:
        if equation.count("=") != 1 or not re.search(r"(?<![A-Za-z_])x(?![A-Za-z_])", equation, re.I):
            return None
        left, right = equation.split("=", 1)
        try:
            left_tree = ast.parse(_clean_expression(left), mode="eval").body
            right_tree = ast.parse(_clean_expression(right), mode="eval").body
            left_coefficients = _polynomial(left_tree)
            right_coefficients = _polynomial(right_tree)
            c, b, a = (left_value - right_value for left_value, right_value in zip(left_coefficients, right_coefficients))
            if math.isclose(a, 0.0, abs_tol=1e-10):
                if math.isclose(b, 0.0, abs_tol=1e-10):
                    answer = "Every value of x is a solution." if math.isclose(c, 0.0, abs_tol=1e-10) else "This equation has no solution."
                else:
                    root = -c / b
                    answer = f"x = {_format(root)}"
            else:
                discriminant = b * b - 4 * a * c
                if discriminant < -1e-10:
                    roots = ((-b + cmath.sqrt(discriminant)) / (2 * a), (-b - cmath.sqrt(discriminant)) / (2 * a))
                    answer = f"No real solutions. Complex roots: x = {roots[0]:.6g} and x = {roots[1]:.6g}."
                elif abs(discriminant) <= 1e-10:
                    answer = f"x = {_format(-b / (2 * a))} (one repeated root)"
                else:
                    root = math.sqrt(discriminant)
                    first, second = (-b + root) / (2 * a), (-b - root) / (2 * a)
                    answer = f"x = {_format(first)} or x = {_format(second)}"
            return ScientificMathResult(answer, "solve_equation")
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError, RecursionError) as exc:
            return ScientificMathResult(f"I couldn't solve that equation: {exc}", "solve_equation")

    def execute(self, question: str) -> ScientificMathResult | None:
        text = str(question or "").strip()
        solve_match = self._SOLVE.match(text)
        if solve_match:
            return self._solve(solve_match.group(1))
        match = self._CALCULATE.match(text)
        if match:
            expression = match.group(1)
            try:
                return ScientificMathResult(f"{expression.strip()} = {_format(_evaluate(expression))}", "scientific_calculation")
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError, RecursionError) as exc:
                return ScientificMathResult(f"I couldn't evaluate that expression: {exc}", "scientific_calculation")
        # Natural bare-function input is safe to intercept; ordinary prose is not.
        if self._FUNCTION_HINT.search(text) and re.fullmatch(r"[\s0-9a-zA-Z_+*/%().,\-^×÷π]+", text):
            try:
                return ScientificMathResult(f"{text} = {_format(_evaluate(text))}", "scientific_calculation")
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError, RecursionError) as exc:
                return ScientificMathResult(f"I couldn't evaluate that expression: {exc}", "scientific_calculation")
        return None


__all__ = ["ScientificMath", "ScientificMathResult"]
