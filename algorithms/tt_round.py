# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами

    Args:
        tt:       исходный тензор
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    # Шаг 1: Полная правая ортогонализация
    tt_right = right_canonicalize(tt, backend)

    # Шаг 2: Левый проход с SVD-усечением
    d = tt_right.order

    # Вычисляем локальный порог ошибки по норме первого ядра
    # (после правой ортогонализации вся норма сосредоточена в первом ядре)
    first_core_norm = backend.norm(tt_right.cores[0])
    if d > 1 and first_core_norm > 1e-30:
        delta = eps * first_core_norm / math.sqrt(d - 1)
    else:
        delta = 0.0

    new_cores = []

    # Проход слева направо (k = 0, ..., d-2)
    for k in range(d - 1):
        core_k = tt_right.cores[k]  # shape (r_k, n_k, r_{k+1})

        # РазворачиваемG_k в матрицу (r_k * n_k, r_{k+1})
        r_k, n_k, r_kp1 = core_k.shape
        mat_k = core_k.reshape((r_k * n_k, r_kp1))

        # SVD
        U, S, Vt = backend.svd(mat_k, full_matrices=False)

        # Выбор нового ранга
        r_new = _compute_rank(S, delta, max_rank)

        # Усечение U
        U_trunc = _truncate_columns(U, r_new, backend)

        # Усечение S и Vt
        S_trunc = _truncate_vector(S, r_new, backend)
        Vt_trunc = _truncate_rows(Vt, r_new, backend)

        # Сворачиваем U в новое ядро
        new_core_k = U_trunc.reshape((r_k, n_k, r_new))
        new_cores.append(new_core_k)

        # Поглощаем остаток в следующее ядро
        # Σ @ V^T имеет shape (r_new, r_{k+1})
        remainder = _multiply_diag_matrix(S_trunc, Vt_trunc, r_new, backend)

        # Умножаем следующее ядро слева на remainder
        core_kp1 = tt_right.cores[k + 1]  # shape (r_{k+1}, n_{k+1}, r_{k+2})
        r_kp1_old, n_kp1, r_kp2 = core_kp1.shape

        # Переформируем в матрицу (r_{k+1}, n_{k+1} * r_{k+2})
        mat_kp1 = core_kp1.reshape((r_kp1_old, n_kp1 * r_kp2))

        # Матричное произведение remainder @ mat_kp1
        # remainder: (r_new, r_{k+1}_old)
        # mat_kp1:   (r_{k+1}_old, n_{k+1} * r_{k+2})
        # результат: (r_new, n_{k+1} * r_{k+2})
        mat_new_kp1 = backend.matmul(remainder, mat_kp1)

        # Обновляем ядро для следующей итерации
        tt_right.cores[k + 1] = mat_new_kp1.reshape((r_new, n_kp1, r_kp2))

    # Последнее ядро остаётся как есть
    new_cores.append(tt_right.cores[d - 1].copy())

    return TTTensor(new_cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.

    Args:
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        delta:    абсолютный порог усечения (0 — без усечения по delta)
        max_rank: максимально допустимый ранг (None = без ограничения)
    """
    if S.size == 0:
        return 1

    # Начинаем с полного ранга и уменьшаем
    r = S.size

    if delta > 0.0:
        delta_sq = delta * delta
        tail_sq = 0.0
        while r > 1:
            new_tail = tail_sq + S.data[r - 1] ** 2
            if new_tail <= delta_sq:
                tail_sq = new_tail
                r -= 1
            else:
                break

    # Применяем ограничение на максимальный ранг
    if max_rank is not None:
        r = min(r, max_rank)

    # Гарантируем хотя бы ранг 1
    r = max(1, r)

    return r


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    m, n = matrix.shape
    if rank > n:
        rank = n

    result_data = []
    for i in range(m):
        for j in range(rank):
            result_data.append(matrix[i, j])

    return DenseTensor((m, rank), data=result_data)


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (k, n)
        rank:    число сохраняемых строк
        backend: интерфейс backend
    """
    m, n = matrix.shape
    if rank > m:
        rank = m

    result_data = []
    for i in range(rank):
        for j in range(n):
            result_data.append(matrix[i, j])

    return DenseTensor((rank, n), data=result_data)


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    if rank > vector.size:
        rank = vector.size

    return DenseTensor((rank,), data=vector.data[:rank])


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n)
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """
    m, n = matrix.shape
    if m != rank or diag_vec.size != rank:
        raise ValueError(f"Несовместимые размеры: diag_vec={diag_vec.size}, matrix={matrix.shape}, rank={rank}")

    result_data = []
    for i in range(rank):
        for j in range(n):
            result_data.append(diag_vec.data[i] * matrix[i, j])

    return DenseTensor((rank, n), data=result_data)
