import pytest
import math
from ...py_math_omm.quaternion import Quaternion
from ...py_math_omm.types import real_number

float_persition = 0.1**5


def floats_equal(f1: float, f2: float):
    return abs(f1 - f2) < float_persition


def values_tuple(q: Quaternion):
    return q.r, q.i, q.j, q.k


@pytest.fixture(params=[(1, 2, 3, 4), (1.3, 2.4, 3.5, 4.6)])
def quat_1(request):
    return Quaternion(request.param)


@pytest.fixture(params=[(5, 7, 10, 12), (5.9, 6.8, 7.7, 8.6)])
def quat_2(request):
    return Quaternion(request.param)


@pytest.fixture(params=[1, 2])
def float_num(request):
    return request.param


@pytest.fixture(params=[1, 2.3])
def float_num(request):
    return request.param


@pytest.fixture(params=[(5, 7), (5.9, 6.8)])
def complex_num(request):
    return complex(request.param[0], request.param[1])


constructor_paramaterize = pytest.mark.parametrize(
    "r,i,j,k",
    [(v, v, v, v) for v in [-1, -0.5, 0, 0.5, 1]],
)


def paramaterize_all_by_values(values: list[real_number], names: str = "r,i,j,k"):
    def decorator(func):
        var_names = names.split(",")
        wrapper = func

        for var_name in var_names:
            wrapper = pytest.mark.parametrize(var_name, values)(wrapper)

        return wrapper

    return decorator


@constructor_paramaterize
def test_4_param_contructor(r, i, j, k):
    res_quaternion = Quaternion(r, i, j, k)

    assert values_tuple(res_quaternion) == (r, i, j, k)


@constructor_paramaterize
def test_complex_contructor(r, i, j, k):
    res_quaternion = Quaternion(complex(r, i))

    assert values_tuple(res_quaternion) == (r, i, 0, 0)


@constructor_paramaterize
def test_iterable_contructor(r, i, j, k):
    res_quaternion = Quaternion((r, i, j, k))

    assert values_tuple(res_quaternion) == (r, i, j, k)


@paramaterize_all_by_values([0, 1])
def test_indecators(r, i, j, k):
    quat = Quaternion(r, i, j, k)

    is_real: bool = i == 0 and j == 0 and k == 0
    is_py_complex: bool = j == 0 and k == 0
    is_pure_complex: bool = r == 0
    is_zero: bool = r == 0 and i == 0 and j == 0 and k == 0

    assert quat.is_real == is_real
    assert quat.is_py_complex == is_py_complex
    assert quat.is_pure_complex == is_pure_complex
    assert quat.is_zero == is_zero
    assert bool(quat) == (not is_zero)


def test_properties(quat_1):
    imag_tuple = (quat_1.i, quat_1.j, quat_1.k)

    assert quat_1.real == quat_1.r
    assert quat_1.imag == imag_tuple


@pytest.mark.parametrize("r", [0, 4.6])
@paramaterize_all_by_values([0, 1], "i,j,k")
def test_equals_not_equals_float(r, i, j, k):
    const_float = 4.6
    quat = Quaternion(r, i, j, k)

    assert (quat == const_float) == ((r, i, j, k) == (const_float, 0, 0, 0))
    assert (quat != const_float) == ((r, i, j, k) != (const_float, 0, 0, 0))


@pytest.mark.parametrize("r", [0, 4.6])
@pytest.mark.parametrize("i", [0, 7.3])
@paramaterize_all_by_values([0, 1], "j,k")
def test_equals_not_equals_complex(r, i, j, k):
    const_real_part = 4.6
    const_imag_part = 4.6
    const_complex = complex(const_real_part, const_imag_part)

    quat = Quaternion(r, i, j, k)

    assert (quat == const_complex) == (
        (r, i, j, k) == (const_real_part, const_imag_part, 0, 0)
    )
    assert (quat != const_complex) == (
        (r, i, j, k) != (const_real_part, const_imag_part, 0, 0)
    )


@paramaterize_all_by_values([1, 2])
def test_equals_not_equals_quaternions(r, i, j, k):
    const_quat = Quaternion(1, 1, 1, 1)
    other_quat = Quaternion(r, i, j, k)

    assert (const_quat == other_quat) == (other_quat == const_quat)
    assert (const_quat == other_quat) == ((r, i, j, k) == (1, 1, 1, 1))
    assert (const_quat != other_quat) == (other_quat != const_quat)
    assert (const_quat != other_quat) == ((r, i, j, k) != (1, 1, 1, 1))


def test_conjugate(quat_1):
    conj_quat = quat_1.conjugate()

    assert quat_1.r == conj_quat.r
    assert -quat_1.i == conj_quat.i
    assert -quat_1.j == conj_quat.j
    assert -quat_1.k == conj_quat.k


def test_abs2(quat_1):
    square_sum = quat_1.r**2 + quat_1.i**2 + quat_1.j**2 + quat_1.k**2

    assert quat_1.abs2() == square_sum


def test_abs(quat_1):
    square_sum = quat_1.r**2 + quat_1.i**2 + quat_1.j**2 + quat_1.k**2

    assert abs(quat_1) == math.sqrt(square_sum)


def test_ceil(quat_1):
    ceiled_quat = quat_1.__ceil__()
    expected_quat = Quaternion(
        math.ceil(quat_1.r),
        math.ceil(quat_1.i),
        math.ceil(quat_1.j),
        math.ceil(quat_1.k),
    )

    assert ceiled_quat == expected_quat


def test_floor(quat_1):
    floored_quat = quat_1.__floor__()
    expected_quat = Quaternion(
        math.floor(quat_1.r),
        math.floor(quat_1.i),
        math.floor(quat_1.j),
        math.floor(quat_1.k),
    )

    assert floored_quat == expected_quat


def test_round(quat_1):
    rounded_quat = quat_1.__round__()
    expected_quat = Quaternion(
        round(quat_1.r),
        round(quat_1.i),
        round(quat_1.j),
        round(quat_1.k),
    )

    assert rounded_quat == expected_quat


def test_pos(quat_1):
    assert quat_1 == +quat_1


def test_neg(quat_1):
    neg_quat = -quat_1

    assert -quat_1.r == neg_quat.r
    assert -quat_1.i == neg_quat.i
    assert -quat_1.j == neg_quat.j
    assert -quat_1.k == neg_quat.k


def test_copy(quat_1):
    copy_quat = quat_1.__copy__()

    assert (quat_1 == copy_quat) and (copy_quat is not quat_1)


def test_deep_copy(quat_1):
    copy_quat = quat_1.__deepcopy__()

    assert (quat_1 == copy_quat) and (copy_quat is not quat_1)


def test_bool():
    zero_quat = Quaternion(0, 0, 0, 0)
    non_zero_quat = Quaternion(1, 2, 3, 4)

    assert bool(zero_quat) == False
    assert bool(non_zero_quat) == True


def test_string(quat_1):
    assert (
        str(quat_1)
        == f"Quaternion(r = {quat_1.r}, i = {quat_1.i}, j = {quat_1.j}, k = {quat_1.k})"
    )


def test_repr(quat_1):
    assert repr(quat_1) == f"Quaternion({quat_1.r}, {quat_1.i}, {quat_1.j}, {quat_1.k})"


def test_add_float(quat_1, float_num):
    result_quat = quat_1 + float_num

    assert result_quat == Quaternion(quat_1.r + float_num, quat_1.i, quat_1.j, quat_1.k)


def test_add_complex(quat_1, complex_num):
    result_quat = quat_1 + complex_num

    assert result_quat == Quaternion(
        quat_1.r + complex_num.real, quat_1.i + complex_num.imag, quat_1.j, quat_1.k
    )


def test_add_quaternion(quat_1, quat_2):
    result_quat = quat_1 + quat_2

    assert result_quat == Quaternion(
        quat_1.r + quat_2.r,
        quat_1.i + quat_2.i,
        quat_1.j + quat_2.j,
        quat_1.k + quat_2.k,
    )


def test_sub_float(quat_1, float_num):
    result_quat = quat_1 - float_num

    assert result_quat == Quaternion(quat_1.r - float_num, quat_1.i, quat_1.j, quat_1.k)


def test_sub_complex(quat_1, complex_num):
    result_quat = quat_1 - complex_num

    assert result_quat == Quaternion(
        quat_1.r - complex_num.real, quat_1.i - complex_num.imag, quat_1.j, quat_1.k
    )


def test_sub_quaternion(quat_1, quat_2):
    result_quat = quat_1 - quat_2

    assert result_quat == Quaternion(
        quat_1.r - quat_2.r,
        quat_1.i - quat_2.i,
        quat_1.j - quat_2.j,
        quat_1.k - quat_2.k,
    )


def test_mul_float(quat_1, float_num):
    result_quat = quat_1 * float_num

    assert result_quat == Quaternion(
        quat_1.r * float_num,
        quat_1.i * float_num,
        quat_1.j * float_num,
        quat_1.k * float_num,
    )


def test_mul_complex(quat_1, complex_num):
    result_quat = quat_1 * complex_num

    assert result_quat == Quaternion(
        quat_1.r * complex_num.real - quat_1.i * complex_num.imag,
        quat_1.i * complex_num.real + quat_1.r * complex_num.imag,
        quat_1.j * complex_num.real + quat_1.k * complex_num.imag,
        quat_1.k * complex_num.real - quat_1.j * complex_num.imag,
    )


def test_mul_quaternion(quat_1, quat_2):
    result_quat = quat_1 * quat_2

    expected_r = (
        quat_1.r * quat_2.r
        - quat_1.i * quat_2.i
        - quat_1.j * quat_2.j
        - quat_1.k * quat_2.k
    )

    expected_i = (
        quat_1.i * quat_2.r
        + quat_1.r * quat_2.i
        - quat_1.k * quat_2.j
        + quat_1.j * quat_2.k
    )

    expected_j = (
        quat_1.j * quat_2.r
        + quat_1.k * quat_2.i
        + quat_1.r * quat_2.j
        - quat_1.i * quat_2.k
    )

    expected_k = (
        quat_1.k * quat_2.r
        - quat_1.j * quat_2.i
        + quat_1.i * quat_2.j
        + quat_1.r * quat_2.k
    )

    assert floats_equal(result_quat.r, expected_r)
    assert floats_equal(result_quat.i, expected_i)
    assert floats_equal(result_quat.j, expected_j)
    assert floats_equal(result_quat.k, expected_k)


def test_rmul_float(quat_1, float_num):
    assert (quat_1.__rmul__(float_num)) == (Quaternion(float_num) * quat_1)


def test_rmul_complex(quat_1, complex_num):
    assert (quat_1.__rmul__(complex_num)) == (Quaternion(complex_num) * quat_1)


def test_rmul_quat(quat_1, quat_2):
    assert (quat_1.__rmul__(quat_2)) == (quat_2 * quat_1)


def test_div_float(quat_1, float_num):
    result_quat = quat_1 / float_num

    assert result_quat == quat_1 * (1 / float_num)


def test_inverse(quat_1):
    inv_quat = quat_1.inverse()

    assert inv_quat == (quat_1.conjugate() / quat_1.abs2())


def test_normalize(quat_1):
    norm_quat = quat_1.normalize()

    assert norm_quat == quat_1 / abs(quat_1)
    assert floats_equal(abs(norm_quat), 1)


def test_div_complex(quat_1, complex_num):
    result_quat = quat_1 / complex_num

    assert result_quat == (quat_1 * (1 / complex_num))


# @pytest.mark.skip
def test_div_quaternion(quat_1, quat_2):
    result_quat = quat_1 / quat_2

    square_sum = quat_2.abs2()

    expected_r = (
        quat_1.r * quat_2.r
        + quat_1.i * quat_2.i
        + quat_1.j * quat_2.j
        + quat_1.k * quat_2.k
    ) / square_sum

    expected_i = (
        quat_1.i * quat_2.r
        - quat_1.r * quat_2.i
        + quat_1.k * quat_2.j
        - quat_1.j * quat_2.k
    ) / square_sum

    expected_j = (
        quat_1.j * quat_2.r
        - quat_1.k * quat_2.i
        - quat_1.r * quat_2.j
        + quat_1.i * quat_2.k
    ) / square_sum

    expected_k = (
        quat_1.k * quat_2.r
        + quat_1.j * quat_2.i
        - quat_1.i * quat_2.j
        - quat_1.r * quat_2.k
    ) / square_sum

    assert floats_equal(result_quat.r, expected_r)
    assert floats_equal(result_quat.i, expected_i)
    assert floats_equal(result_quat.j, expected_j)
    assert floats_equal(result_quat.k, expected_k)
    assert result_quat == quat_1 * quat_2.inverse()


def test_rdiv_float(quat_1, float_num):
    assert (quat_1.__rtruediv__(float_num)) == (
        Quaternion(float_num) * quat_1.inverse()
    )


def test_rdiv_complex(quat_1, complex_num):
    assert (quat_1.__rtruediv__(complex_num)) == (
        Quaternion(complex_num) * quat_1.inverse()
    )


def test_rdiv_quat(quat_1, quat_2):
    assert (quat_1.__rtruediv__(quat_2)) == (quat_2 * quat_1.inverse())


def test_floordiv_float(quat_1, float_num):
    assert (quat_1 // float_num) == math.floor(quat_1 / float_num)


def test_floordiv_complex(quat_1, complex_num):
    assert (quat_1 // complex_num) == math.floor(quat_1 / complex_num)


def test_floordiv_quat(quat_1, quat_2):
    assert (quat_1 // quat_2) == math.floor(quat_1 / quat_2)


def test_mod_quat(quat_1, float_num):
    res_quat = quat_1 % float_num

    assert res_quat == Quaternion(
        quat_1.r % float_num,
        quat_1.i % float_num,
        quat_1.j % float_num,
        quat_1.k % float_num,
    )


def test_divmod(quat_1, float_num):
    assert divmod(quat_1, float_num) == (quat_1 // float_num, quat_1 % float_num)
