# core/tt_tensor.py

"""
Тензор в TT-формате (Tensor Train).

TT-тензор порядка d с shape (n_0, n_1, ..., n_{d-1}) хранится как
список d ядер (cores), где k-е ядро — это 3D DenseTensor с shape:
    (r_k, n_k, r_{k+1})

Граничные условия: r_0 = r_d = 1.

TT-ранги: (r_0, r_1, ..., r_d) = (1, r_1, ..., r_{d-1}, 1).
"""

from __future__ import annotations

from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size


class TTTensor:
    """
    Тензор в TT-формате.

    Атрибуты:
        cores:  список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        order:  порядок тензора d (число мод)
        shape:  кортеж (n_0, n_1, ..., n_{d-1})
        ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d), r_0 = r_d = 1
    """

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(self, cores: list[DenseTensor]) -> None:
        """
        Создаёт TT-тензор из списка ядер.

        Args:
            cores: список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        """
        if len(cores) == 0:
            raise ValueError("Список ядер не может быть пустым")

        self.order = len(cores)
        self.cores = cores

        # Извлекаем shape мод
        self.shape = tuple(core.shape[1] for core in cores)

        # Извлекаем TT-ранги
        ranks = [1]  # r_0 = 1
        for core in cores:
            if core.ndim != 3:
                raise ValueError(f"Ядро должно быть 3D, получено ndim={core.ndim}")
            ranks.append(core.shape[2])

        self.ranks = tuple(ranks)

        # Проверяем консистентность
        if self.ranks[-1] != 1:
            raise ValueError(f"r_d должно быть 1, получено {self.ranks[-1]}")

        for k in range(self.order):
            expected_left_rank = self.ranks[k]
            expected_right_rank = self.ranks[k + 1]
            actual_left_rank = cores[k].shape[0]
            actual_right_rank = cores[k].shape[2]

            if actual_left_rank != expected_left_rank or actual_right_rank != expected_right_rank:
                raise ValueError(
                    f"Ядро {k}: ожидается ({expected_left_rank}, n_{k}, {expected_right_rank}), "
                    f"получено {cores[k].shape}"
                )


    @staticmethod
    def random(shape, ranks, seed=None):
        """
        Создаёт случайный TT-тензор с заданными рангами.

        Args:
            shape:  кортеж размеров мод (n_0, ..., n_{d-1})
            ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d)
                    или список внутренних рангов (r_1, ..., r_{d-1})
            seed:   seed для воспроизводимости

        NB: это отладочная функция, она не проверяется тестами
        """
        import random
        if seed is not None:
            random.seed(seed)

        shape = validate_shape(shape)
        d = len(shape)

        # Преобразуем ranks
        if isinstance(ranks, (list, tuple)):
            if len(ranks) == d:
                # Это полные ранги (r_0, r_1, ..., r_d)
                tt_ranks = ranks
            elif len(ranks) == d - 1:
                # Это внутренние ранги (r_1, ..., r_{d-1})
                tt_ranks = (1,) + tuple(ranks) + (1,)
            else:
                raise ValueError(f"ranks должна иметь {d} или {d-1} элементов, получено {len(ranks)}")
        else:
            raise ValueError("ranks должна быть list или tuple")

        cores = []
        for k in range(d):
            core_shape = (tt_ranks[k], shape[k], tt_ranks[k + 1])
            core_data = [random.uniform(-1, 1) for _ in range(core_shape[0] * core_shape[1] * core_shape[2])]
            core = DenseTensor(core_shape, data=core_data)
            cores.append(core)

        return TTTensor(cores)

    # ────────────────────────────────────────────
    # Доступ к элементам
    # ────────────────────────────────────────────

    def get_element(
        self,
        indices: tuple[int, ...] | list[int]
    ) -> float:
        """
        Возвращает элемент TT-тензора по его мультииндексу.

        Args:
            indices: кортеж/список длины d
        """
        indices = tuple(indices) if isinstance(indices, list) else indices

        if len(indices) != self.order:
            raise ValueError(f"Ожидается {self.order} индексов, получено {len(indices)}")

        # result — текущий вектор-строка длины r_k
        # Начинаем: G_0[i_0] — строка (1, r_1), берём как вектор длины r_1
        core = self.cores[0]
        i_0 = indices[0]
        r_1 = self.ranks[1]
        n_0 = core.shape[1]
        # core.data: (1, n_0, r_1) => G_0[0, i_0, :] = data[i_0 * r_1 : (i_0+1) * r_1]
        result = [core.data[i_0 * r_1 + alpha] for alpha in range(r_1)]

        # Проходим по оставшимся ядрам
        for k in range(1, self.order):
            core = self.cores[k]  # shape (r_k, n_k, r_{k+1})
            i_k = indices[k]
            r_left = self.ranks[k]    # = len(result)
            r_right = self.ranks[k + 1]
            n_k = core.shape[1]

            # G_k[i_k] — матрица (r_left, r_right)
            # new_result[alpha_right] = sum_{alpha_left} result[alpha_left] * G_k[alpha_left, i_k, alpha_right]
            new_result = [0.0] * r_right
            for alpha_left in range(r_left):
                v = result[alpha_left]
                # смещение в данных ядра: alpha_left * n_k * r_right + i_k * r_right
                base = alpha_left * n_k * r_right + i_k * r_right
                for alpha_right in range(r_right):
                    new_result[alpha_right] += v * core.data[base + alpha_right]
            result = new_result

        # result — список длины 1
        return float(result[0])

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        from core.utils import flat_to_multi_index, compute_strides

        full_size = compute_size(self.shape)
        data = [0.0] * full_size

        for flat_idx in range(full_size):
            multi_idx = flat_to_multi_index(flat_idx, self.shape)
            data[flat_idx] = self.get_element(multi_idx)

        return DenseTensor(self.shape, data=data)

    # ────────────────────────────────────────────
    # Информация и отладка
    # ────────────────────────────────────────────

    def core_sizes(self) -> list[tuple[int, ...]]:
        """Возвращает размеры всех ядер."""
        return [core.shape for core in self.cores]

    def total_storage(self) -> int:
        """
        Возвращает общее число элементов во всех ядрах.
        Это то, сколько памяти реально занимает TT-тензор.
        """
        return sum(core.size for core in self.cores)

    def compression_ratio(self) -> float:
        """
        Возвращает отношение числа элементов полного тензора к числу
        элементов TT-тензора. Показывает, насколько TT-формат компактнее.
        """
        full_size = compute_size(self.shape)
        tt_size = self.total_storage()
        if tt_size == 0:
            return 0.0
        return full_size / tt_size

    def copy(self) -> TTTensor:
        """Возвращает глубокую копию TT-тензора."""
        cores_copy = [core.copy() for core in self.cores]
        return TTTensor(cores_copy)

    def __repr__(self) -> str:
        """
        Возвращает строковое представление TT-тензора для отладки.

        Формирует многострочную строку с основной служебной информацией
        об объекте:
            - порядок тензора (order),
            - исходная форма (shape),
            - TT-ранги (ranks),
            - размеры TT-ядер (cores),
            - суммарный объём хранения в элементах.

        NB: это отладочная функция, которая не покрывается тестами
        """
        core_sizes_str = ", ".join(str(size) for size in self.core_sizes())
        return (f"TTTensor(\n"
                f"  order={self.order},\n"
                f"  shape={self.shape},\n"
                f"  ranks={self.ranks},\n"
                f"  core_sizes=[{core_sizes_str}],\n"
                f"  total_storage={self.total_storage()}\n"
                f")")

    def __str__(self) -> str:
        """
        Возвращает строковое представление TT-тензора.

        Делегирует работу методу __repr__, обеспечивая единый формат
        отображения при вызове.
        """
        return self.__repr__()
