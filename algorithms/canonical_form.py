# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    new_cores = [core.copy() for core in tt.cores]

    # Проходим слева направо за исключением последнего ядра
    for k in range(tt.order - 1):
        core_k = new_cores[k]  # shape (r_k, n_k, r_{k+1})

        r_k, n_k, r_kp1 = core_k.shape
        mat_k = core_k.reshape((r_k * n_k, r_kp1))

        # QR-разложение (требует m >= n)
        # Если r_k*n_k < r_kp1 — широкая матрица, используем SVD
        m_qr, n_qr = mat_k.shape
        if m_qr >= n_qr:
            Q, R = backend.qr(mat_k)
            r_new = Q.shape[1]
        else:
            # Широкая матрица: QR через SVD
            U, S, Vt = backend.svd(mat_k)
            r_new = _numerical_rank(S)
            r_new = min(r_new, m_qr)
            Q = _truncate_columns(U, r_new, backend)
            SV = _multiply_diag_matrix(
                _truncate_vector(S, r_new, backend),
                _truncate_rows(Vt, r_new, backend),
                r_new, backend
            )
            R = SV

        new_cores[k] = Q.reshape((r_k, n_k, r_new))

        # Поглощаем R в следующее ядро
        # G_{k+1}[i_{k+1}] ← R @ G_{k+1}[i_{k+1}] для каждого i_{k+1}
        core_kp1 = new_cores[k + 1]  # shape (r_{k+1}, n_{k+1}, r_{k+2})
        r_kp1_old = core_kp1.shape[0]
        n_kp1 = core_kp1.shape[1]
        r_kp2 = core_kp1.shape[2]

        # Переформируем G_{k+1} в матрицу (r_{k+1}, n_{k+1} * r_{k+2})
        mat_kp1 = core_kp1.reshape((r_kp1_old, n_kp1 * r_kp2))

        mat_new = backend.matmul(R, mat_kp1)

        r_new_kp1 = mat_new.shape[0]
        new_cores[k + 1] = mat_new.reshape((r_new_kp1, n_kp1, r_kp2))

    return TTTensor(new_cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    new_cores = [core.copy() for core in tt.cores]

    # Проходим справа налево за исключением первого ядра
    for k in range(tt.order - 1, 0, -1):
        core_k = new_cores[k]  # shape (r_k, n_k, r_{k+1})

        # Развернём ядро в матрицу: (r_k, n_k * r_{k+1})
        r_k, n_k, r_kp1 = core_k.shape
        mat_k = core_k.reshape((r_k, n_k * r_kp1))

        # RQ-разложение: mat_k = R @ Q, где Q имеет ортонормированные строки.
        # Реализуем через QR транспонированной матрицы:
        #   mat_k^T = Q' @ R'  (QR)
        #   mat_k   = R'^T @ Q'^T
        # Q = Q'^T имеет ортонормированные строки, R = R'^T.
        # Требует mat_k^T tall, т.е. n_k*r_kp1 >= r_k.
        m_g = r_k
        n_g = n_k * r_kp1

        if n_g >= m_g:
            # mat_k^T — tall матрица, QR применим
            mat_k_t = backend.transpose(mat_k)  # (n_g, m_g)
            Q_t, R_t = backend.qr(mat_k_t)      # Q_t: (n_g, r_new), R_t: (r_new, m_g)
            # mat_k = R_t^T @ Q_t^T
            R = backend.transpose(R_t)   # (m_g, r_new)
            Q = backend.transpose(Q_t)   # (r_new, n_g)
            r_new = Q.shape[0]
        else:
            # mat_k — tall матрица (r_k > n_k*r_kp1): используем SVD для LQ
            # mat_k = U @ diag(S) @ Vt = (U @ diag(S)) @ Vt = R_new @ Q_new
            U, S, Vt = backend.svd(mat_k)
            r_new = _numerical_rank(S)
            r_new = min(r_new, n_g)
            U_trunc = _truncate_columns(U, r_new, backend)
            S_trunc = _truncate_vector(S, r_new, backend)
            Vt_trunc = _truncate_rows(Vt, r_new, backend)
            Q = Vt_trunc                                                      # (r_new, n_g)
            R = _multiply_columns_by_diag(U_trunc, S_trunc, backend)         # (m_g, r_new)

        # Обновляем ядро G_k: (r_new, n_k, r_kp1)
        new_cores[k] = Q.reshape((r_new, n_k, r_kp1))

        # Поглощаем R в предыдущее ядро G_{k-1}
        # G_{k-1}[i_{k-1}] ← G_{k-1}[i_{k-1}] @ R для каждого i_{k-1}
        core_km1 = new_cores[k - 1]  # shape (r_{k-1}, n_{k-1}, r_k)
        r_km1 = core_km1.shape[0]
        n_km1 = core_km1.shape[1]
        r_k_old = core_km1.shape[2]

        # Переформируем в матрицу (r_{k-1} * n_{k-1}, r_k)
        mat_km1 = core_km1.reshape((r_km1 * n_km1, r_k_old))

        # Матричное произведение mat_km1 @ R
        mat_new_km1 = backend.matmul(mat_km1, R)

        # Переформируем обратно
        r_new_right = mat_new_km1.shape[1]
        new_cores[k - 1] = mat_new_km1.reshape((r_km1, n_km1, r_new_right))

    return TTTensor(new_cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    r"""
    Возвращает числовой ранг матрицы по вектору сингулярных значений.

    Сингулярное число \sigma_i считаем ненулевым, если:
        |\sigma_i| > max(abs_tol, rel_tol * max(\sigma_1, ..., \sigma_n))

    Args:
        S:       одномерный тензор формы (k,) — сингулярные значения
                 в порядке убывания
        rel_tol: относительный допуск (по умолчанию 1e-8)
        abs_tol: абсолютный допуск (по умолчанию 1e-12)
    """
    if S.size == 0:
        return 0

    max_sv = max(S.data) if S.data else 0.0
    threshold = max(abs_tol, rel_tol * max_sv)

    rank = 0
    for sv in S.data:
        if sv > threshold:
            rank += 1

    return max(1, rank)


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
        rank:     длина диагонального вектора
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


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает результат произведения обычной матрицы на диагональную:
        matrix @ diag(diag_vec)

    Args:
        matrix:   двумерный тензор формы (m, n)
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        backend:  интерфейс backend
    """
    m, n = matrix.shape
    if n != diag_vec.size:
        raise ValueError(f"Несовместимые размеры: matrix={matrix.shape}, diag_vec={diag_vec.size}")

    result_data = []
    for i in range(m):
        for j in range(n):
            result_data.append(matrix[i, j] * diag_vec.data[j])

    return DenseTensor((m, n), data=result_data)
