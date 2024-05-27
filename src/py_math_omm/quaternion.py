import math
from typing import Callable, overload
from .types import real_number, complex_number

type quaternion_number = complex_number | Quaternion


class Quaternion(object):
    @overload
    def __init__(
        self,
        r: real_number = 0,
        i: real_number = 0,
        j: real_number = 0,
        k: real_number = 0,
    ) -> None: ...

    @overload
    def __init__(self, r: complex) -> None: ...

    @overload
    def __init__(
        self, r: tuple[real_number, real_number, real_number, real_number]
    ) -> None: ...

    def __init__(
        self,
        r: (
            complex_number | tuple[real_number, real_number, real_number, real_number]
        ) = 0,
        i: real_number = 0,
        j: real_number = 0,
        k: real_number = 0,
    ) -> None:
        if isinstance(r, (int, float)):
            self.r = float(r)
            self.i = float(i)
            self.j = float(j)
            self.k = float(k)
        elif isinstance(r, complex):
            self.r = r.real
            self.i = r.imag
            self.j = 0
            self.k = 0
        else:
            self.r, self.i, self.j, self.k = r

    @property
    def real(self) -> float:
        return self.r

    @property
    def imag(self) -> tuple[float, float, float]:
        return self.i, self.j, self.k

    @property
    def is_real(self) -> bool:
        return (self.i == 0) and (self.j == 0) and (self.k == 0)

    @property
    def is_py_complex(self) -> bool:
        return (self.j == 0) and (self.k == 0)

    @property
    def is_pure_complex(self) -> bool:
        return self.r == 0

    @property
    def is_zero(self) -> bool:
        return (self.r == 0) and (self.i == 0) and (self.j == 0) and (self.k == 0)

    def normalize(self) -> "Quaternion":
        return self / self.__abs__()

    def conjugate(self) -> "Quaternion":
        return Quaternion(self.r, -self.i, -self.j, -self.k)

    def inverse(self) -> "Quaternion":
        return self.conjugate() / self.abs2()

    def abs2(self) -> float:
        return self.r * self.r + self.i * self.i + self.j * self.j + self.k * self.k

    def __create_with_transformation(
        self, transform: Callable[[float], float]
    ) -> "Quaternion":
        return Quaternion(
            transform(self.r), transform(self.i), transform(self.j), transform(self.k)
        )

    def __add__(self, other: quaternion_number) -> "Quaternion":
        if isinstance(other, (int, float)):
            return Quaternion(self.r + other, self.i, self.j, self.k)
        if isinstance(other, complex):
            return Quaternion(self.r + other.real, self.i + other.imag, self.j, self.k)
        if isinstance(other, Quaternion):
            return Quaternion(
                self.r + other.r, self.i + other.i, self.j + other.j, self.k + other.k
            )

    def __sub__(self, other: quaternion_number) -> "Quaternion":
        if isinstance(other, (int, float)):
            return Quaternion(self.r - other, self.i, self.j, self.k)
        if isinstance(other, complex):
            return Quaternion(self.r - other.real, self.i - other.imag, self.j, self.k)
        if isinstance(other, Quaternion):
            return Quaternion(
                self.r - other.r, self.i - other.i, self.j - other.j, self.k - other.k
            )

    def __mul__(self, other: quaternion_number) -> "Quaternion":
        # 1*1 = 1, 1*i = i, 1*j = j, 1*k = k
        # i*1 = i, i*i = -1, i*j = k, i*k = -j
        # j*1 = j, j*i = -k, j*j = -1, j*k = i
        # k*1 = k, k*i = j, k*j = -i, k*k = -1
        if isinstance(other, (int, float)):
            return self.__create_with_transformation(lambda f: f * other)
        if isinstance(other, complex):
            r, i = other.real, other.imag
            r_val = self.r * r - self.i * i
            i_val = self.r * i + self.i * r
            j_val = self.j * r + self.k * i
            k_val = -self.j * i + self.k * r
            return Quaternion(r_val, i_val, j_val, k_val)
        if isinstance(other, Quaternion):
            r, i, j, k = other.r, other.i, other.j, other.k
            r_val = self.r * r - self.i * i - self.j * j - self.k * k
            i_val = self.r * i + self.i * r + self.j * k - self.k * j
            j_val = self.r * j - self.i * k + self.j * r + self.k * i
            k_val = self.r * k + self.i * j - self.j * i + self.k * r
            return Quaternion(r_val, i_val, j_val, k_val)

    def __truediv__(self, other: quaternion_number) -> "Quaternion":
        # self / other
        if isinstance(other, (int, float)):
            other = float(other)
            return self.__create_with_transformation(lambda f: f / other)
        if isinstance(other, complex):
            r, i = other.real, other.imag
            square_sum = r * r + i * i
            res_r = (r * self.r + i * self.i) / square_sum
            res_i = (r * self.i - i * self.r) / square_sum
            res_j = (r * self.j + i * self.k) / square_sum
            res_k = (r * self.k - i * self.j) / square_sum
            return Quaternion(res_r, res_i, res_j, res_k)
        if isinstance(other, Quaternion):
            return self * other.inverse()

    def __floordiv__(self, other: quaternion_number) -> "Quaternion":
        return (self / other).__floor__()

    def __mod__(self, other: real_number):
        return self.__create_with_transformation(lambda f: f % other)

    def __divmod__(self, other: real_number) -> tuple["Quaternion", "Quaternion"]:
        return (self // other, self % other)

    def __radd__(self, other: quaternion_number) -> "Quaternion":
        return self + other

    def __rsub__(self, other: quaternion_number) -> "Quaternion":
        return (-self) + other

    def __rtruediv__(self, other: quaternion_number) -> "Quaternion":
        return self.inverse() * other

    def __rmul__(self, other: quaternion_number) -> "Quaternion":
        return self * other

    def __abs__(self) -> float:
        return math.sqrt(self.abs2())

    def __ceil__(self) -> "Quaternion":
        return self.__create_with_transformation(lambda f: math.ceil(f))

    def __floor__(self) -> "Quaternion":
        return self.__create_with_transformation(lambda f: math.floor(f))

    def __round__(self, n=None) -> "Quaternion":
        return self.__create_with_transformation(lambda f: round(f, n))

    def __pos__(self) -> "Quaternion":
        return self

    def __neg__(self) -> "Quaternion":
        return self.__create_with_transformation(lambda f: -f)

    def __copy__(self) -> "Quaternion":
        return Quaternion(self.r, self.i, self.j, self.k)

    def __deepcopy__(self, memodict=None) -> "Quaternion":
        return Quaternion(self.r, self.i, self.j, self.k)

    def __bool__(self) -> bool:
        return not self.is_zero

    def __eq__(self, other) -> bool:
        if isinstance(other, (int | float)):
            other = float(other)
            if self.is_real:
                return self.r == other
            return False
        if isinstance(other, complex):
            if self.is_py_complex:
                return (self.r == other.real) and (self.i == other.imag)
            return False
        if isinstance(other, Quaternion):
            return (
                (self.r == other.r)
                and (self.i == other.i)
                and (self.j == other.j)
                and (self.k == other.k)
            )
        return other.__eq__(self)

    def __ne__(self, other) -> bool:
        return not self == other

    def __str__(self) -> str:
        return f"Quaternion(r = {self.r}, i = {self.i}, j = {self.j}, k = {self.k})"

    def __repr__(self) -> str:
        return f"Quaternion({self.r}, {self.i}, {self.j}, {self.k})"
