from typing import Iterable, SupportsIndex, TypeVar, Protocol, Generic, overload
from some_types import withNone

S = TypeVar('S')


class VectorElement(Protocol):
    def __mul__(self, other: S, /) -> S: ...
    def __add__(self, other: S, /) -> S: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


T = TypeVar('T', bound=VectorElement)


class Vector(Generic[T]):
    def __init__(self, *values: T) -> None:
        self.values = list(values)

    @classmethod
    def from_iterable(cls, values: Iterable[T]) -> "Vector[T]":
        return cls(*values)

    @property
    def length(self) -> int:
        return len(self.values)

    def dot(self, other: "Vector[T]") -> withNone[T]:
        """computes the dot prduct between self and other

        Args:
            other (Vector[T]): other vector to multiply

        Raises:
            ValueError: when the vectors are of different lengths

        Returns:
            withNone[T]: None when the two vectors are empty, T when they are not, in such case it returns the dot product between the 2 vectors
        """

        if self.length != other.length:
            raise ValueError("The vectors must be of the same length")

        if self.length == 0:
            return None

        res: T = self.values[0] * other.values[0]

        for i in range(1, self.length):
            res += self.values[i] * other.values[i]

        return res

    def __add__(self, other: "Vector[T]") -> "Vector[T]":
        values: list[T] = []
        length: int = self.length
        other_length: int = other.length
        if length < other_length:
            return other.__add__(self)
        else:
            for i in range(length):
                if i < other_length:
                    values.append(self.values[i] + other.values[i])
                else:
                    values.append(self.values[i])
        return Vector(*values)

    @overload
    def __getitem__(self, i: SupportsIndex, /) -> T: ...

    @overload
    def __getitem__(self, s: slice, /) -> list[T]: ...

    def __getitem__(self, i: SupportsIndex | slice) -> T | list[T]:
        return self.values[i]

    def __iter__(self):
        return self.values

    def __eq__(self, other: "Vector[T]") -> bool:
        if self.length != other.length:
            return False

        for i in range(self.length):
            if self.values[i] != other.values[i]:
                return False

        return True

    def __repr__(self) -> str:
        return f'Vector({', '.join(v.__repr__() for v in self.values)})'

    def __str__(self) -> str:
        return f'({', '.join(v.__str__() for v in self.values)})'
