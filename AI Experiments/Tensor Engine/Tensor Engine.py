"""Experiment 005: a tiny NumPy-backed tensor and reverse-mode autograd engine.

This is an educational implementation. NumPy stores and computes array values;
all operation-level gradient rules and graph traversal are implemented here.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np

ArrayLike = object


def _as_array(value: ArrayLike) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("Tensor data must be numeric") from exc
    if array.dtype == object:
        raise TypeError("Tensor data must be a regular numeric array")
    if not np.all(np.isfinite(array)):
        raise ValueError("Tensor data must contain only finite numbers")
    return array.copy()


def _unbroadcast(gradient: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Sum broadcast dimensions so a gradient matches its input's shape."""
    result = np.asarray(gradient, dtype=np.float64)
    while result.ndim > len(target_shape):
        result = result.sum(axis=0)
    for axis, target_size in enumerate(target_shape):
        if target_size == 1 and result.shape[axis] != 1:
            result = result.sum(axis=axis, keepdims=True)
    return result.reshape(target_shape)


def _normalize_axes(axis: int | Sequence[int] | None, ndim: int) -> tuple[int, ...] | None:
    if axis is None:
        return None
    raw_axes = (axis,) if isinstance(axis, int) else tuple(axis)
    normalized: list[int] = []
    for item in raw_axes:
        if not isinstance(item, int):
            raise TypeError("Axes must be integers")
        normalized_axis = item + ndim if item < 0 else item
        if normalized_axis < 0 or normalized_axis >= ndim:
            raise ValueError(f"Axis {item} is out of bounds for a tensor with {ndim} dimensions")
        if normalized_axis in normalized:
            raise ValueError("An axis cannot be reduced more than once")
        normalized.append(normalized_axis)
    return tuple(normalized)


class Tensor:
    """A multidimensional array that records operations for automatic gradients.

    ``requires_grad`` defaults to True to make small experiments convenient.
    Constants and input data can opt out by passing ``requires_grad=False``.
    Gradients are accumulated in ``.grad`` after calling ``backward()``.
    """

    __slots__ = ("data", "grad", "requires_grad", "_prev", "_op", "_backward")
    __array_priority__ = 1000

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = True,
        _children: Iterable[Tensor] = (),
        _op: str = "",
    ) -> None:
        if not isinstance(requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        self.data = _as_array(data)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._prev = tuple(parent for parent in _children if parent.requires_grad)
        self._op = _op
        self._backward = lambda: None

    @staticmethod
    def _coerce(other: Tensor | int | float | np.ndarray) -> Tensor:
        if isinstance(other, Tensor):
            return other
        if isinstance(other, (int, float, np.ndarray, list, tuple)) and not isinstance(other, bool):
            return Tensor(other, requires_grad=False)
        raise TypeError(f"Expected Tensor or numeric data; received {type(other).__name__}")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def numpy(self) -> np.ndarray:
        """Return a defensive NumPy copy of the tensor's data."""
        return self.data.copy()

    def tolist(self) -> list | float:
        return self.data.tolist()

    def item(self) -> float:
        if self.data.size != 1:
            raise ValueError("item() requires a tensor containing exactly one value")
        return float(self.data.reshape(-1)[0])

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def _result(
        self,
        data: np.ndarray | float,
        parents: tuple[Tensor, ...],
        operation: str,
        backward,
    ) -> Tensor:
        result = Tensor(
            data,
            requires_grad=any(parent.requires_grad for parent in parents),
            _children=parents,
            _op=operation,
        )
        result._backward = backward(result)
        return result

    def __add__(self, other: Tensor | int | float | np.ndarray) -> Tensor:
        rhs = self._coerce(other)
        output_data = self.data + rhs.data

        def backward(out: Tensor):
            def apply() -> None:
                if self.requires_grad:
                    self.grad += _unbroadcast(out.grad, self.shape)
                if rhs.requires_grad:
                    rhs.grad += _unbroadcast(out.grad, rhs.shape)
            return apply

        return self._result(output_data, (self, rhs), "add", backward)

    def __radd__(self, other: Tensor | int | float | np.ndarray) -> Tensor:
        return self + other

    def __neg__(self) -> Tensor:
        def backward(out: Tensor):
            def apply() -> None:
                if self.requires_grad:
                    self.grad -= out.grad
            return apply
        return self._result(-self.data, (self,), "neg", backward)

    def __sub__(self, other: Tensor | int | float | np.ndarray) -> Tensor:
        return self + (-self._coerce(other))

    def __rsub__(self, other: Tensor | int | float | np.ndarray) -> Tensor:
        return self._coerce(other) - self

    def __mul__(self, other: Tensor | int | float | np.ndarray) -> Tensor:
        rhs = self._coerce(other)
        output_data = self.data * rhs.data

        def backward(out: Tensor):
            def apply() -> None:
                if self.requires_grad:
                    self.grad += _unbroadcast(rhs.data * out.grad, self.shape)
                if rhs.requires_grad:
                    rhs.grad += _unbroadcast(self.data * out.grad, rhs.shape)
            return apply

        return self._result(output_data, (self, rhs), "mul", backward)

    def __rmul__(self, other: Tensor | int | float | np.ndarray) -> Tensor:
        return self * other

    def __truediv__(self, other: Tensor | int | float | np.ndarray) -> Tensor:
        rhs = self._coerce(other)
        if np.any(rhs.data == 0):
            raise ZeroDivisionError("Cannot divide by zero")
        output_data = self.data / rhs.data

        def backward(out: Tensor):
            def apply() -> None:
                if self.requires_grad:
                    self.grad += _unbroadcast(out.grad / rhs.data, self.shape)
                if rhs.requires_grad:
                    rhs.grad -= _unbroadcast(out.grad * self.data / (rhs.data ** 2), rhs.shape)
            return apply

        return self._result(output_data, (self, rhs), "div", backward)

    def __rtruediv__(self, other: Tensor | int | float | np.ndarray) -> Tensor:
        return self._coerce(other) / self

    def __pow__(self, exponent: int | float) -> Tensor:
        if not isinstance(exponent, (int, float)) or isinstance(exponent, bool):
            raise TypeError("Tensor powers currently require a numeric scalar exponent")
        power = float(exponent)
        if not np.isfinite(power):
            raise ValueError("Exponent must be finite")
        if power < 1.0 and np.any(self.data == 0):
            raise ValueError("This power has an undefined derivative at zero")
        output_data = self.data ** power

        def backward(out: Tensor):
            def apply() -> None:
                if self.requires_grad:
                    if power == 0:
                        local = np.zeros_like(self.data)
                    else:
                        local = power * (self.data ** (power - 1.0))
                    self.grad += local * out.grad
            return apply

        return self._result(output_data, (self,), "pow", backward)

    def __matmul__(self, other: Tensor | np.ndarray | list | tuple) -> Tensor:
        rhs = self._coerce(other)
        if self.ndim == 0 or rhs.ndim == 0:
            raise ValueError("Matrix multiplication requires tensors with at least one dimension")
        output_data = np.matmul(self.data, rhs.data)

        def backward(out: Tensor):
            def apply() -> None:
                left_was_vector = self.ndim == 1
                right_was_vector = rhs.ndim == 1
                left_matrix = np.expand_dims(self.data, axis=-2) if left_was_vector else self.data
                right_matrix = np.expand_dims(rhs.data, axis=-1) if right_was_vector else rhs.data
                upstream = out.grad
                if left_was_vector and right_was_vector:
                    upstream = np.asarray(upstream).reshape(np.asarray(upstream).shape + (1, 1))
                elif left_was_vector:
                    upstream = np.expand_dims(upstream, axis=-2)
                elif right_was_vector:
                    upstream = np.expand_dims(upstream, axis=-1)

                if self.requires_grad:
                    left_gradient = np.matmul(upstream, np.swapaxes(right_matrix, -1, -2))
                    if left_was_vector:
                        left_gradient = np.squeeze(left_gradient, axis=-2)
                    self.grad += _unbroadcast(left_gradient, self.shape)
                if rhs.requires_grad:
                    right_gradient = np.matmul(np.swapaxes(left_matrix, -1, -2), upstream)
                    if right_was_vector:
                        right_gradient = np.squeeze(right_gradient, axis=-1)
                    rhs.grad += _unbroadcast(right_gradient, rhs.shape)
            return apply

        return self._result(output_data, (self, rhs), "matmul", backward)

    def sum(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> Tensor:
        if not isinstance(keepdims, bool):
            raise TypeError("keepdims must be a bool")
        axes = _normalize_axes(axis, self.ndim)
        output_data = self.data.sum(axis=axes, keepdims=keepdims)

        def backward(out: Tensor):
            def apply() -> None:
                if not self.requires_grad:
                    return
                gradient = out.grad
                if axes is None:
                    gradient = np.broadcast_to(gradient, self.shape)
                else:
                    if not keepdims:
                        for reduced_axis in sorted(axes):
                            gradient = np.expand_dims(gradient, axis=reduced_axis)
                    gradient = np.broadcast_to(gradient, self.shape)
                self.grad += gradient
            return apply

        return self._result(output_data, (self,), "sum", backward)

    def mean(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> Tensor:
        axes = _normalize_axes(axis, self.ndim)
        if axes is None:
            divisor = self.size
        else:
            divisor = int(np.prod([self.shape[item] for item in axes], dtype=np.int64))
        if divisor == 0:
            raise ValueError("Cannot calculate the mean of an empty reduction")
        return self.sum(axis=axes, keepdims=keepdims) / divisor

    def reshape(self, *shape: int | tuple[int, ...]) -> Tensor:
        requested = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        try:
            new_shape = tuple(int(item) for item in requested)
        except (TypeError, ValueError) as exc:
            raise TypeError("reshape() requires integer dimensions") from exc
        output_data = self.data.reshape(new_shape)
        original_shape = self.shape

        def backward(out: Tensor):
            def apply() -> None:
                if self.requires_grad:
                    self.grad += out.grad.reshape(original_shape)
            return apply

        return self._result(output_data, (self,), "reshape", backward)

    def transpose(self, axes: Sequence[int] | None = None) -> Tensor:
        if axes is None:
            permutation = tuple(reversed(range(self.ndim)))
        else:
            permutation = tuple(axes)
            if sorted(permutation) != list(range(self.ndim)):
                raise ValueError("transpose axes must be a permutation of all dimensions")
        output_data = self.data.transpose(permutation)
        inverse = tuple(np.argsort(permutation).tolist())

        def backward(out: Tensor):
            def apply() -> None:
                if self.requires_grad:
                    self.grad += out.grad.transpose(inverse)
            return apply

        return self._result(output_data, (self,), "transpose", backward)

    def exp(self) -> Tensor:
        output_data = np.exp(self.data)

        def backward(out: Tensor):
            def apply() -> None:
                if self.requires_grad:
                    self.grad += out.data * out.grad
            return apply

        return self._result(output_data, (self,), "exp", backward)

    def log(self) -> Tensor:
        if np.any(self.data <= 0):
            raise ValueError("log() requires every tensor element to be positive")
        output_data = np.log(self.data)

        def backward(out: Tensor):
            def apply() -> None:
                if self.requires_grad:
                    self.grad += out.grad / self.data
            return apply

        return self._result(output_data, (self,), "log", backward)

    def tanh(self) -> Tensor:
        output_data = np.tanh(self.data)

        def backward(out: Tensor):
            def apply() -> None:
                if self.requires_grad:
                    self.grad += (1.0 - out.data**2) * out.grad
            return apply

        return self._result(output_data, (self,), "tanh", backward)

    def sigmoid(self) -> Tensor:
        # Stable sigmoid: avoid evaluating exp(large positive values).
        output_data = np.empty_like(self.data)
        positive = self.data >= 0
        output_data[positive] = 1.0 / (1.0 + np.exp(-self.data[positive]))
        exp_values = np.exp(self.data[~positive])
        output_data[~positive] = exp_values / (1.0 + exp_values)

        def backward(out: Tensor):
            def apply() -> None:
                if self.requires_grad:
                    self.grad += out.data * (1.0 - out.data) * out.grad
            return apply

        return self._result(output_data, (self,), "sigmoid", backward)

    def relu(self) -> Tensor:
        output_data = np.maximum(0.0, self.data)

        def backward(out: Tensor):
            def apply() -> None:
                if self.requires_grad:
                    self.grad += (self.data > 0.0) * out.grad
            return apply

        return self._result(output_data, (self,), "relu", backward)

    def backward(self, gradient: ArrayLike | None = None) -> None:
        """Backpropagate from this tensor through its reachable graph.

        A scalar output can be differentiated without an argument. For a
        non-scalar output, pass an upstream gradient with exactly matching shape.
        """
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on a tensor that does not require gradients")
        if gradient is None:
            if self.data.size != 1:
                raise ValueError("backward() needs an explicit gradient for non-scalar outputs")
            initial_gradient = np.ones_like(self.data)
        else:
            initial_gradient = _as_array(gradient)
            if initial_gradient.shape != self.shape:
                raise ValueError(
                    f"Upstream gradient shape {initial_gradient.shape} must match output shape {self.shape}"
                )

        topology: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            identity = id(node)
            if expanded:
                topology.append(node)
                continue
            if identity in visited:
                continue
            visited.add(identity)
            stack.append((node, True))
            for parent in reversed(node._prev):
                if id(parent) not in visited:
                    stack.append((parent, False))

        for node in topology:
            node.grad = np.zeros_like(node.data)
        self.grad = initial_gradient.copy()
        for node in reversed(topology):
            node._backward()

    def graph_summary(self) -> dict[str, int]:
        """Return the total number of graph nodes and a count by operation."""
        nodes: list[Tensor] = []
        seen: set[int] = set()
        stack = [self]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            nodes.append(node)
            stack.extend(node._prev)
        counts = Counter(node._op or "leaf" for node in nodes)
        return {"nodes": len(nodes), **dict(sorted(counts.items()))}

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        array = np.asarray(self.data, dtype=dtype)
        if copy:
            return array.copy()
        return array

    def __repr__(self) -> str:
        return (
            f"Tensor(data={self.data!r}, grad={self.grad!r}, "
            f"requires_grad={self.requires_grad})"
        )


__all__ = ["Tensor"]
