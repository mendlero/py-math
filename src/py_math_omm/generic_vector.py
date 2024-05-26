from typing import Iterable, SupportsIndex, TypeVar, Protocol, overload
from .types import withNone

T = TypeVar("T", bound="VectorElement")


class VectorElement(Protocol):
    def __mul__(self, other: T, /) -> T: ...
    def __truediv__(self, other: T, /) -> T: ...
    def __add__(self, other: T, /) -> T: ...
    def __sub__(self, other: T, /) -> T: ...
    def __neg__(self) -> T: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class GenericVector[T]:
    def __init__(self, zero_value: T, *values: T) -> None:
        self.values = list(values)
        self.zero_value = zero_value

    @classmethod
    def from_iterable(cls, zero_value: T, values: Iterable[T]) -> "GenericVector[T]":
        return cls(zero_value, *values)

    @property
    def length(self) -> int:
        return len(self.values)

    @property
    def is_zero_vector(self) -> bool:
        return self.length == 0 or all((v == self.zero_value for v in self.values))

    def dot(self, other: "GenericVector[T]") -> withNone[T]:
        """computes the dot prduct between self and other

        Args:
            other (GenericVector[T]): other vector to multiply

        Returns:
            withNone[T]: None when the two vectors are empty or have different lengths, T when they are not, in such case it returns the dot product between the 2 vectors
        """

        if self.length != other.length:
            return None

        if self.length == 0:
            return None

        res: T = self.values[0] * other.values[0]

        for i in range(1, self.length):
            res += self.values[i] * other.values[i]

        return res

    def abs2(self) -> T:
        return self.dot(self)

    def is_orthogonal(self, other: "GenericVector[T]", /) -> bool:
        return self.dot(other) == self.zero_value

    def is_parralel(self, other: "GenericVector[T]") -> bool:
        if self.is_zero_vector or other.is_zero_vector:
            return False

        if self.length != other.length:
            return False

        factor: withNone[T] = None

        for i in range(self.length):
            val = self[i]
            other_val = other[i]

            if (val == self.zero_value and other_val != self.zero_value) or (
                val != self.zero_value and other_val == self.zero_value
            ):
                return False

            if val == self.zero_value and other_val == self.zero_value:
                continue

            if factor == None:
                factor = val / other_val
            elif val / other_val != factor:
                return False

        return True

    def __add__(self, other: "GenericVector[T]", /) -> "GenericVector[T]":
        length: int = self.length
        other_length: int = other.length
        if length < other_length:
            return other.__add__(self)

        new_values_generator = (
            (self.values[i] + other.values[i] if i < other_length else self.values[i])
            for i in range(length)
        )

        return GenericVector.from_iterable(new_values_generator)

    def __sub__(self, other: "GenericVector[T]", /) -> "GenericVector[T]":
        length: int = self.length
        other_length: int = other.length
        if length < other_length:
            new_values_generator = (
                (self.values[i] - other.values[i] if i < length else -other.values[i])
                for i in range(other_length)
            )

            return GenericVector.from_iterable(new_values_generator)

        new_values_generator = (
            (self.values[i] - other.values[i] if i < other_length else self.values[i])
            for i in range(length)
        )

        return GenericVector.from_iterable(new_values_generator)

    def __mul__(self, other: T, /) -> "GenericVector[T]":
        return GenericVector.from_iterable((i * other for i in self.values))

    def __rmul__(self, other: T, /) -> "GenericVector[T]":
        return self * other

    def __truediv__(self, other: T, /) -> "GenericVector[T]":
        return GenericVector.from_iterable((i / other for i in self.values))

    def __len__(self) -> int:
        return self.length

    def __neg__(self) -> "GenericVector[T]":
        return GenericVector.from_iterable((-i for i in self.values))

    def __pos__(self) -> "GenericVector[T]":
        return GenericVector.from_iterable(self.values)

    @overload
    def __getitem__(self, i: SupportsIndex, /) -> T: ...

    @overload
    def __getitem__(self, s: slice, /) -> list[T]: ...

    def __getitem__(self, i: SupportsIndex | slice, /) -> T | list[T]:
        return self.values[i]

    @overload
    def __setitem__(self, i: SupportsIndex, value: T, /) -> None: ...

    @overload
    def __setitem__(self, s: slice, values: Iterable[T], /) -> None: ...

    def __setitem__(
        self, key: slice | SupportsIndex, value: T | Iterable[T], /
    ) -> None:
        self.values[key] = value

    def __iter__(self):
        return self.values

    def __eq__(self, other: object, /) -> bool:
        if not isinstance(other, GenericVector):
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
        return f"GenericVector({", ".join(v.__repr__() for v in self.values)})"

    def __str__(self) -> str:
        return f"({", ".join(v.__str__() for v in self.values)})"
