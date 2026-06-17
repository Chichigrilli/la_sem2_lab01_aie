# core/dense_tensor.py

"""Функции для работы с тензорами в стандартной плотной форме."""


from __future__ import annotations

import random
import math

from core.utils import (
    validate_shape,
    compute_size,
    compute_strides,
    multi_index_to_flat,
    flat_to_multi_index,
check_shapes_match,
)


class DenseTensor:
    """
    Плотный тензор произвольного порядка.

    Атрибуты:
        shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        ndim:    порядок тензора (число мод)
        size:    общее число элементов
        data:    плоский список значений (row-major / C-order)
        strides: шаги для перевода мультииндекса в плоский индекс
    """

    __slots__ = ('shape', 'ndim', 'size', 'data', 'strides')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(
        self,
        shape: tuple[int, ...] | list[int],
        data: list[float] | None = None,
        fill: float = 0.0
    ) -> None:
        """
        Создаёт тензор заданной формы.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            data:  плоский список значений (если None — заполняется fill)
            fill:  значение для заполнения (по умолчанию 0.0)
        """
        self.shape = validate_shape(shape)
        self.ndim = len(self.shape)
        self.size = compute_size(self.shape)
        self.strides = compute_strides(self.shape)

        if data is None:
            self.data = [fill] * self.size
        else:
            if len(data) != self.size:
                raise ValueError(f"Размер data {len(data)} не совпадает с требуемым {self.size}")
            self.data = list(data)

    @staticmethod
    def zeros(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный нулями.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=0.0)

    @staticmethod
    def ones(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный единицами.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=1.0)

    @staticmethod
    def random(
        shape: tuple[int, ...] | list[int],
        low: int = -5,
        high: int = 5,
        integer: bool = True,
        seed: int | None = None
    ) -> DenseTensor:
        """
        Возвращает тензор со случайными значениями.

        Args:
            shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            low:     нижняя граница значений тензора
            high:    верхняя граница значений тензора
            integer: True — целые числа, False — вещественные
            seed:    seed для воспроизводимости (None — без фиксации)

        NB: эта функция не тестируется, ее можно использовать для отладки
        """
        if seed is not None:
            random.seed(seed)

        size = compute_size(validate_shape(shape))
        data = []

        if integer:
            for _ in range(size):
                data.append(float(random.randint(low, high)))
        else:
            for _ in range(size):
                data.append(random.uniform(low, high))

        return DenseTensor(shape, data=data)

    @staticmethod
    def from_nested_list(nested: list) -> DenseTensor:
        """
        Создаёт тензор из вложенного списка Python.
        Автоматически определяет shape.

        Args:
            nested: список
        """
        def get_shape(lst: list) -> tuple[int, ...]:
            if not isinstance(lst, list):
                return ()
            if len(lst) == 0:
                return (0,)
            return (len(lst),) + get_shape(lst[0])
        
        def flatten(lst: list) -> list[float]:
            result = []
            for item in lst:
                if isinstance(item, list):
                    result.extend(flatten(item))
                else:
                    result.append(float(item))
            return result
        
        shape = get_shape(nested)
        if 0 in shape:
            raise ValueError("Не может быть размера 0 в shape")
        
        data = flatten(nested)
        return DenseTensor(shape, data=data)

    # ────────────────────────────────────────────
    # Индексация
    # ────────────────────────────────────────────

    def _validate_index(
        self,
        multi_index: tuple[int, ...] | int
    ) -> tuple[int, ...]:
        """
        Возвращает нормализованный мультииндекс в виде кортежа.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        if isinstance(multi_index, int):
            multi_index = (multi_index,)

        if not isinstance(multi_index, tuple):
            multi_index = tuple(multi_index)

        if len(multi_index) != self.ndim:
            raise ValueError(f"Индекс должен иметь {self.ndim} измерений, получено {len(multi_index)}")

        for i, idx in enumerate(multi_index):
            if not isinstance(idx, int) or idx < 0 or idx >= self.shape[i]:
                raise IndexError(f"Индекс {idx} выходит за границы размера {self.shape[i]} для моды {i}")

        return multi_index

    def __getitem__(self, multi_index: tuple[int, ...] | int) -> float:
        """
        Возвращает значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        multi_index = self._validate_index(multi_index)
        flat_index = multi_index_to_flat(multi_index, self.strides)
        return self.data[flat_index]

    def __setitem__(
        self,
        multi_index: tuple[int, ...] | int,
        value: float
    ) -> None:
        """
        Устанавливает новое значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
            value:       новое значение (число)
        """
        multi_index = self._validate_index(multi_index)
        flat_index = multi_index_to_flat(multi_index, self.strides)
        self.data[flat_index] = float(value)

    # ────────────────────────────────────────────
    # Преобразования формы
    # ────────────────────────────────────────────

    def reshape(self, new_shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает новый объект тензора с новой формой и скопированными данными.

        Args:
            new_shape: кортеж новых размеров (n'_0, n'_1, ..., n'_{k-1})
        """
        new_shape = validate_shape(new_shape)
        new_size = compute_size(new_shape)

        if new_size != self.size:
            raise ValueError(f"Размер нового тензора {new_size} не совпадает с исходным {self.size}")

        return DenseTensor(new_shape, data=list(self.data))

    def unfolding(self, mode: int) -> DenseTensor:
        """
        Возвращает матрицу — развертку тензора по моде n.

        Args:
            mode: номер моды (0 ≤ mode < ndim), которая становится индексом строк
        """
        if mode < 0 or mode >= self.ndim:
            raise ValueError(f"Mode {mode} выходит за границы [0, {self.ndim})")
        
        # Размеры: строки = shape[mode], столбцы = произведение остальных
        n_rows = self.shape[mode]
        n_cols = self.size // n_rows
        
        result = DenseTensor.zeros((n_rows, n_cols))
        
        # Перебираем все элементы исходного тензора
        for flat_idx in range(self.size):
            multi_idx = flat_to_multi_index(flat_idx, self.shape)
            
            # Переупорядочиваем индексы
            i = multi_idx[mode]
            j_indices = multi_idx[:mode] + multi_idx[mode+1:]
            
            # Вычисляем позицию j в матрице
            j_strides = compute_strides(self.shape[:mode] + self.shape[mode+1:])
            j = multi_index_to_flat(j_indices, j_strides)
            
            result[i, j] = self.data[flat_idx]
        
        return result

    def left_unfolding(self, k: int) -> DenseTensor:
        """
        Возвращает матрицу — "левую развертку" тензора для TT-SVD.

        Args:
            k: номер границы разбиения (0 ≤ k < ndim - 1)
        """
        if k < 0 or k >= self.ndim - 1:
            raise ValueError(f"k должно быть в диапазоне [0, {self.ndim - 2}], получено {k}")

        # Левая часть: первые k+1 мод
        left_shape = self.shape[:k+1]
        right_shape = self.shape[k+1:]

        n_rows = compute_size(left_shape)
        n_cols = compute_size(right_shape)

        result = DenseTensor.zeros((n_rows, n_cols))

        for flat_idx in range(self.size):
            multi_idx = flat_to_multi_index(flat_idx, self.shape)

            # Левые индексы
            left_indices = multi_idx[:k+1]
            left_strides = compute_strides(left_shape)
            i = multi_index_to_flat(left_indices, left_strides)

            # Правые индексы
            right_indices = multi_idx[k+1:]
            right_strides = compute_strides(right_shape)
            j = multi_index_to_flat(right_indices, right_strides)

            result[i, j] = self.data[flat_idx]

        return result

    # ────────────────────────────────────────────
    # Копирование
    # ────────────────────────────────────────────

    def copy(self) -> DenseTensor:
        """Возвращает глубокую копию тензора."""
        return DenseTensor(self.shape, data=list(self.data))

    # ────────────────────────────────────────────
    # Арифметика
    # ────────────────────────────────────────────

    def norm(self) -> float:
        """Возвращает Фробениусову норму тензора."""
        sum_sq = sum(x * x for x in self.data)
        return math.sqrt(sum_sq)

    def __add__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного сложения: t1 + t2.

        Args:
            other: t2
        """
        check_shapes_match(self.shape, other.shape)
        result_data = [a + b for a, b in zip(self.data, other.data)]
        return DenseTensor(self.shape, data=result_data)

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного вычитания: t1 - t2.

        Args:
            other: t2
        """
        check_shapes_match(self.shape, other.shape)
        result_data = [a - b for a, b in zip(self.data, other.data)]
        return DenseTensor(self.shape, data=result_data)

    def __mul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: t1 * scalar.

        Args:
            scalar: число
        """
        result_data = [x * scalar for x in self.data]
        return DenseTensor(self.shape, data=result_data)

    def __rmul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: scalar * t1.

        Args:
            scalar: число, на которое умножаем
        """
        return self * scalar

    def __neg__(self) -> DenseTensor:
        """Возвращает тензор — результат умножения тензора на -1."""
        return self * (-1.0)

    # ────────────────────────────────────────────
    # Сравнение и отладка
    # ────────────────────────────────────────────

    def allclose(
        self,
        other: DenseTensor,
        atol: float = 1e-8,
        rtol: float = 1e-5
    ) -> bool:
        """
        Возвращает True, если тензоры равны с заданной точностью.

        Условие равенства: shape равны и для каждой пары элементов
        тензоров с равными индексами выполняется:
            |a - b| <= atol + rtol * max(|a|, |b|)


        Args:
            other: DenseTensor для сравнения
            atol:  абсолютная погрешность (по умолчанию 1e-8)
            rtol:  относительная погрешность (по умолчанию 1e-5)
        """
        if self.shape != other.shape:
            return False

        for a, b in zip(self.data, other.data):
            tolerance = atol + rtol * max(abs(a), abs(b))
            if abs(a - b) > tolerance:
                return False

        return True

    def to_nested_list(self) -> list:
        """Возвращает тензор в формате вложенного списка."""
        if self.ndim == 0:
            return self.data[0]
        if self.ndim == 1:
            return self.data[:]
        
        # Для многомерного тензора
        def build_nested(data_flat: list[float], shape: tuple[int, ...], start_idx: int) -> tuple[list, int]:
            if len(shape) == 1:
                end_idx = start_idx + shape[0]
                return data_flat[start_idx:end_idx], end_idx
            
            result = []
            idx = start_idx
            for _ in range(shape[0]):
                sub_list, idx = build_nested(data_flat, shape[1:], idx)
                result.append(sub_list)
            return result, idx
        
        result, _ = build_nested(self.data, self.shape, 0)
        return result

    def __repr__(self) -> str:
        """
        Возвращает строковое представление тензора для отладки.

        NB: эта функция не проверяется тестами, ее реализация может быть произвольной
        """
        return f"DenseTensor(shape={self.shape}, ndim={self.ndim}, size={self.size}, data={self.data[:10]}...)"

    def __str__(self) -> str:
        """Возвращает строковое представление тензора для отладки."""
        return self.__repr__()