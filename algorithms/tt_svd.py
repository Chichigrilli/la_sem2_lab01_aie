# algorithms/tt_svd.py

"""
TT-SVD алгоритм: разложение плотного тензора в TT-формат.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — тензор в TT-формате.

    Args:
        tensor:   DenseTensor с shape (n_0, n_1, ..., n_{d-1})
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    from core.tt_tensor import TTTensor

    d = tensor.ndim

    # Специальный случай: 1D тензор
    if d == 1:
        core = tensor.reshape((1, tensor.shape[0], 1))
        return TTTensor([core])

    # Инициализация
    cores = []
    r_prev = 1

    # Вычисляем локальный порог усечения
    tensor_norm = backend.norm(tensor)
    if tensor_norm > 1e-30:
        delta = (eps / math.sqrt(d - 1)) * tensor_norm
    else:
        delta = 0.0

    # C хранит "остаток" — сначала это весь тензор в плоском виде
    # C имеет shape (r_prev * n_k, n_{k+1} * ... * n_{d-1}) на каждом шаге
    # Начинаем: разворачиваем tensor в (n_0, n_1*...*n_{d-1})
    import math as _math
    remaining = _math.prod(tensor.shape[1:])
    C = tensor.reshape((tensor.shape[0], remaining))

    # Основной цикл: k от 0 до d-2
    for k in range(d - 1):
        n_k = tensor.shape[k]
        # C имеет shape (r_prev * n_k, n_{k+1} * ... * n_{d-1})
        # (на первом шаге r_prev=1 и C уже (n_0, prod_rest))

        # SVD
        U, S, Vt = backend.svd(C, full_matrices=False)

        # Выбор ранга
        r_k = _compute_truncated_rank(S, delta, max_rank)

        # Усечение
        U_trunc = _truncate_columns(U, r_k, backend)
        S_trunc = _truncate_vector(S, r_k, backend)
        Vt_trunc = _truncate_rows(Vt, r_k, backend)

        # Формируем ядро: (r_prev, n_k, r_k)
        core_k = U_trunc.reshape((r_prev, n_k, r_k))
        cores.append(core_k)

        # Обновляем C = diag(S) @ Vt: shape (r_k, prod_остатка)
        C = _multiply_diag_matrix(S_trunc, Vt_trunc, r_k, backend)

        # Переформируем C для следующего шага: (r_k * n_{k+1}, prod_оставшихся)
        if k < d - 2:
            n_next = tensor.shape[k + 1]
            cols = C.shape[1] // n_next
            C = C.reshape((r_k * n_next, cols))

        r_prev = r_k

    # Последнее ядро: C имеет shape (r_{d-1}, n_{d-1})
    n_final = tensor.shape[d - 1]
    core_d = C.reshape((r_prev, n_final, 1))
    cores.append(core_d)

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_truncated_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает ранг усечения по сингулярным значениям.

    Args:
        S:        DenseTensor (k,) — сингулярные значения по убыванию
        delta:    порог усечения
        max_rank: максимальный ранг (None = без ограничения)
    """
    if S.size == 0:
        return 1

    # Числовой ранг: отсекаем машинные нули
    sigma_1 = S.data[0] if S.size > 0 else 0.0
    numerical_rank = 0
    for i in range(S.size):
        if abs(S.data[i]) > max(1e-12, 1e-8 * sigma_1):
            numerical_rank = i + 1

    r = numerical_rank if numerical_rank > 0 else 1

    # Дополнительное усечение по delta
    if delta > 0.0:
        delta_sq = delta * delta
        tail_sq = sum(S.data[i] ** 2 for i in range(r, S.size))
        # Уменьшаем r, пока хвост не превысит delta^2
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

    Используется после SVD для усечения матрицы левых сингулярных векторов:
        U in R^{m x n} -> U_trunc in R^{m x rank}

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
