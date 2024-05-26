from typing import Iterable, SupportsIndex, TypeVar, Protocol, Generic, overload
from .types import withNone
from .quaternion import real_number
import math


class Vector:
    def __init__(self, *values: real_number) -> None:
        self.values = list((float(v) for v in values))

    @classmethod
    def from_iterable(cls, values: Iterable[real_number]) -> "Vector":
        return cls(*values)

    @property
    def length(self) -> int:
        return len(self.values)

    @property
    def is_zero_vector(self) -> bool:
        return self.length == 0 or all((v == 0 for v in self.values))

    def dot(self, other: "Vector") -> withNone[float]:
        """computes the dot prduct between self and other

        Args:
            other (Vector): other vector to multiply

        Raises:
            ValueError: when the vectors are of different lengths

        Returns:
            withNone[float]: None when the two vectors are empty, float when they are not, in such case it returns the dot product between the 2 vectors
        """

        if self.length != other.length:
            raise ValueError("The vectors must be of the same length")

        if self.length == 0:
            return None

        res: float = 0

        for i in range(self.length):
            res += self.values[i] * other.values[i]

        return res

    def abs2(self) -> float:
        return self.dot(self)

    def normalize(self) -> "Vector":
        return self / abs(self)

    def angle(self, other: "Vector") -> float:
        return math.acos(self.dot(other) / (abs(self) * abs(other)))

    def project(self, other: "Vector") -> "Vector":
        """project other onto self

        Args:
            other (Vector): other vector

        Raises:
            ValueError: when self.is_zero_vector == True

        Returns:
            Vector: the projection
        """
        if self.is_zero_vector:
            raise ValueError("cannot project onto zero vector")

        return (other.dot(self) / self.abs2()) * self

    def is_orthogonal(self, other: "Vector", /) -> bool:
        return self.dot(other) == 0

    def is_parralel(self, other: "Vector") -> bool:
        if self.is_zero_vector or other.is_zero_vector:
            return False

        if self.length != other.length:
            return False

        factor: withNone[float] = None

        for i in range(self.length):
            val = self[i]
            other_val = other[i]

            if (val == 0 and other_val != 0) or (val != 0 and other_val == 0):
                return False

            if val == 0 and other_val == 0:
                continue

            if factor == None:
                factor = val / other_val
            elif val != factor * other_val:
                return False

        return True

    def __add__(self, other: "Vector", /) -> "Vector":
        length: int = self.length
        other_length: int = other.length
        if length < other_length:
            return other.__add__(self)

        new_values_generator = (
            (self.values[i] + other.values[i] if i < other_length else self.values[i])
            for i in range(length)
        )

        return Vector.from_iterable(new_values_generator)

    def __sub__(self, other: "Vector", /) -> "Vector":
        return self + (-other)

    def __mul__(self, other: real_number, /) -> "Vector":
        return Vector.from_iterable((i * other for i in self.values))

    def __rmul__(self, other: real_number, /) -> "Vector":
        return self * other

    def __truediv__(self, other: real_number, /) -> "Vector":
        return self * (1 / other)

    def __abs__(self) -> float:
        return math.sqrt(self.abs2())

    def __len__(self) -> int:
        return self.length

    def __neg__(self) -> "Vector":
        return Vector.from_iterable((-i for i in self.values))

    def __pos__(self) -> "Vector":
        return Vector.from_iterable(self.values)

    @overload
    def __getitem__(self, i: SupportsIndex, /) -> float: ...

    @overload
    def __getitem__(self, s: slice, /) -> list[float]: ...

    def __getitem__(self, i: SupportsIndex | slice, /) -> float | list[float]:
        return self.values[i]

    @overload
    def __setitem__(self, i: SupportsIndex, value: float, /) -> None: ...

    @overload
    def __setitem__(self, s: slice, values: Iterable[float], /) -> None: ...

    def __setitem__(
        self, key: slice | SupportsIndex, value: float | Iterable[float], /
    ) -> None:
        self.values[key] = value

    def __iter__(self):
        return self.values

    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, Vector):
            return False

        if self.length != other.length:
            return False

        for i in range(self.length):
            if self.values[i] != other.values[i]:
                return False

        return True

    def __ne__(self, value: object, /) -> bool:
        return not self == value

    def __repr__(self) -> str:
        return f"Vector({", ".join(v.__repr__() for v in self.values)})"

    def __str__(self) -> str:
        return f"({", ".join(v.__str__() for v in self.values)})"
