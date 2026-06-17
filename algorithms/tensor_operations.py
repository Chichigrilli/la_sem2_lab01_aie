# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядрами,
не восстанавливая полный тензор.

Содержит:
    - tt_add:         поэлементное сложение
    - tt_scalar_mul:  умножение на скаляр
    - tt_hadamard:    поэлементное произведение (Адамар)
    - tt_dot:         скалярное произведение <A, B>
    - tt_norm:        Фробениусова норма
    - tt_diff_norm:   ||A - B||_F без восстановления полных тензоров

Все операции через backend.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


Number = int | float


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного сложения двух TT-тензоров.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError(f"Формы тензоров не совпадают: {tt1.shape} != {tt2.shape}")

    d = tt1.order
    new_cores = []

    for k in range(d):
        core1_k = tt1.cores[k]  # shape (r1_k, n_k, r1_{k+1})
        core2_k = tt2.cores[k]  # shape (r2_k, n_k, r2_{k+1})

        r1_k, n_k, r1_kp1 = core1_k.shape
        r2_k, _, r2_kp1 = core2_k.shape

        # Новое ядро имеет блочную структуру:
        # [core1_k    0      ]
        # [0         core2_k ]

        if d == 1:
            # Единственное ядро: оба (1, n, 1), просто складываем
            data = [a + b for a, b in zip(core1_k.data, core2_k.data)]
            new_cores.append(DenseTensor((1, n_k, 1), data=data))
        elif k == 0:
            # Первое ядро: горизонтальная конкатенация по третьей оси
            # (1, n_k, r1_kp1) || (1, n_k, r2_kp1) -> (1, n_k, r1_kp1+r2_kp1)
            r_new = r1_kp1 + r2_kp1
            data = [0.0] * (n_k * r_new)
            for i in range(n_k):
                for j in range(r1_kp1):
                    data[i * r_new + j] = core1_k[0, i, j]
                for j in range(r2_kp1):
                    data[i * r_new + r1_kp1 + j] = core2_k[0, i, j]
            new_cores.append(DenseTensor((1, n_k, r_new), data=data))
        elif k == d - 1:
            # Последнее ядро: вертикальная конкатенация по первой оси
            # (r1_k, n_k, 1) || (r2_k, n_k, 1) -> (r1_k+r2_k, n_k, 1)
            r_new = r1_k + r2_k
            data = [0.0] * (r_new * n_k)
            for r in range(r1_k):
                for i in range(n_k):
                    data[r * n_k + i] = core1_k[r, i, 0]
            for r in range(r2_k):
                for i in range(n_k):
                    data[(r1_k + r) * n_k + i] = core2_k[r, i, 0]
            new_cores.append(DenseTensor((r_new, n_k, 1), data=data))
        else:
            # Внутреннее ядро: блочно-диагональная структура
            # Размер: (r1_k + r2_k, n_k, r1_{k+1} + r2_{k+1})
            r_k_new = r1_k + r2_k
            r_kp1_new = r1_kp1 + r2_kp1

            core_sum_data = [0.0] * (r_k_new * n_k * r_kp1_new)
            core_sum = DenseTensor((r_k_new, n_k, r_kp1_new), data=core_sum_data)

            # Заполняем блочную структуру
            for i in range(n_k):
                # Верхний левый блок: core1_k[:, i, :]
                for r1 in range(r1_k):
                    for r1_next in range(r1_kp1):
                        core_sum[r1, i, r1_next] = core1_k[r1, i, r1_next]

                # Нижний правый блок: core2_k[:, i, :]
                for r2 in range(r2_k):
                    for r2_next in range(r2_kp1):
                        core_sum[r1_k + r2, i, r1_kp1 + r2_next] = core2_k[r2, i, r2_next]

            new_cores.append(core_sum)

    return TTTensor(new_cores)

def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат умножения TT-тензора на скаляр.
    Модифицируем только первое ядро.

    Args:
        tt:      TTTensor
        alpha:   число
        backend: интерфейс backend
    """
    new_cores = [core.copy() for core in tt.cores]

    # Умножаем только первое ядро на скаляр
    first_core = new_cores[0]
    new_cores[0] = first_core * alpha

    return TTTensor(new_cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения (произведения Адамара).

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError(f"Формы тензоров не совпадают: {tt1.shape} != {tt2.shape}")

    d = tt1.order
    new_cores = []

    for k in range(d):
        core1_k = tt1.cores[k]  # shape (r1_k, n_k, r1_{k+1})
        core2_k = tt2.cores[k]  # shape (r2_k, n_k, r2_{k+1})

        r1_k, n_k, r1_kp1 = core1_k.shape
        r2_k, _, r2_kp1 = core2_k.shape

        # TT-ядро для произведения Адамара: G_k[i_k] = G1_k[i_k] ⊗ G2_k[i_k]
        # onde ⊗ - кронекерово произведение матриц

        # Новый ранг: r1_k * r2_k и r1_{k+1} * r2_{k+1}
        r_k_new = r1_k * r2_k
        r_kp1_new = r1_kp1 * r2_kp1

        core_hadamard_data = [0.0] * (r_k_new * n_k * r_kp1_new)
        core_hadamard = DenseTensor((r_k_new, n_k, r_kp1_new), data=core_hadamard_data)

        # Заполняем кронекеровым произведением матриц
        for i in range(n_k):
            # Кронекерово произведение: G1[i] ⊗ G2[i]
            # G1[i]: (r1_k, r1_kp1), G2[i]: (r2_k, r2_kp1)
            # (G1 ⊗ G2)[r1*r2_k+r2, r1_next*r2_kp1+r2_next] = G1[r1,r1_next]*G2[r2,r2_next]
            for r1 in range(r1_k):
                for r2 in range(r2_k):
                    for r1_next in range(r1_kp1):
                        for r2_next in range(r2_kp1):
                            r_global = r1 * r2_k + r2
                            r_next_global = r1_next * r2_kp1 + r2_next
                            core_hadamard[r_global, i, r_next_global] = (
                                core1_k[r1, i, r1_next] * core2_k[r2, i, r2_next]
                            )

        new_cores.append(core_hadamard)

    return TTTensor(new_cores)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError(f"Формы тензоров не совпадают: {tt1.shape} != {tt2.shape}")

    d = tt1.order

    # Алгоритм из теории:
    # Z_1 = sum_{i_1} G1_1[i_1]^T @ G2_1[i_1]         — матрица (rA_1, rB_1)
    # Z_k = sum_{i_k} G1_k[i_k]^T @ Z_{k-1} @ G2_k[i_k]
    # <A,B> = Z_d  (скаляр 1x1)

    # Шаг k=0: Z = sum_{i_0} G1[i_0]^T @ G2[i_0]
    # G1[0]: (1, n_0, rA), G2[0]: (1, n_0, rB)
    # G1[i_0]^T: (rA, 1), G2[i_0]: (1, rB) => произведение (rA, rB)
    core1 = tt1.cores[0]  # (1, n_0, rA)
    core2 = tt2.cores[0]  # (1, n_0, rB)
    rA = tt1.ranks[1]
    rB = tt2.ranks[1]
    n_0 = core1.shape[1]

    # Z: (rA, rB)
    Z_data = [0.0] * (rA * rB)
    for i in range(n_0):
        # G1[i]: вектор длины rA (строка), G2[i]: вектор длины rB
        for a in range(rA):
            for b in range(rB):
                Z_data[a * rB + b] += core1.data[i * rA + a] * core2.data[i * rB + b]
    Z = DenseTensor((rA, rB), data=Z_data)

    # Шаги k=1,...,d-1
    for k in range(1, d):
        core1 = tt1.cores[k]  # (rA_left, n_k, rA_right)
        core2 = tt2.cores[k]  # (rB_left, n_k, rB_right)
        rA_left, n_k, rA_right = core1.shape
        rB_left, _,   rB_right = core2.shape

        # new_Z[aR, bR] = sum_{i_k, aL, bL} G1[aL,i_k,aR] * Z[aL,bL] * G2[bL,i_k,bR]
        new_Z_data = [0.0] * (rA_right * rB_right)
        for i in range(n_k):
            for aR in range(rA_right):
                for bR in range(rB_right):
                    s = 0.0
                    for aL in range(rA_left):
                        g1 = core1.data[aL * n_k * rA_right + i * rA_right + aR]
                        for bL in range(rB_left):
                            g2 = core2.data[bL * n_k * rB_right + i * rB_right + bR]
                            s += g1 * Z.data[aL * rB_left + bL] * g2
                    new_Z_data[aR * rB_right + bR] += s
        Z = DenseTensor((rA_right, rB_right), data=new_Z_data)

    # Z — матрица 1x1
    return float(Z.data[0])


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.

    Args:
        tt:      TTTensor
        backend: интерфейс backend
    """
    return math.sqrt(tt_dot(tt, tt, backend))


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    Вычисляется без восстановления полных тензоров:

    Args:
        tt1, tt2: TTTensor
        backend:  интерфейс backend
    """
    # ||A - B||^2 = ||A||^2 + ||B||^2 - 2<A, B>
    norm_sq_tt1 = tt_dot(tt1, tt1, backend)
    norm_sq_tt2 = tt_dot(tt2, tt2, backend)
    dot_product = tt_dot(tt1, tt2, backend)

    diff_norm_sq = norm_sq_tt1 + norm_sq_tt2 - 2 * dot_product

    # Гарантируем, что это не отрицательное число (могут быть числовые ошибки)
    if diff_norm_sq < 0:
        diff_norm_sq = 0

    return math.sqrt(diff_norm_sq)
