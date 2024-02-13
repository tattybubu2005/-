#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[39]:


# Создаем двумерный массив размерности (4, 7) 
arr = np.random.uniform(0, 20, (4, 7))

# Находим минимальное и максимальное значение в массиве
min_val = arr.min()      
max_val = arr.max()

# Нормализуем значения массива по формуле ax + b
a = 1 / (max_val - min_val)
b = -a * min_val
normalized_arr = a * arr + b

print("Исходный массив:")
print(arr)
print("\nНормализованный массив:")
print(normalized_arr)


# In[40]:


# Создание матрицы 8x10 из случайных целых чисел 
matrix = np.random.randint(0, 11, size=(8, 10))

# Поиск индекса строки с минимальной суммой значений
row_index = np.sum(matrix, axis=1).argmin()


print("Матрица:")
print(matrix)
print("\nСтрока с минимальной суммой значений:")
print(matrix[row_index])
print("Индекс строки:", row_index)


# In[19]:


import numpy as np

# Задаем два одномерных вектора одинаковой размерности
vector1 = np.array([1, 2, 3, 4, 5, 6, 7])
vector2 = np.array([5, 4, 3, 2, 1, 7, 6])

# Вычисляем евклидово расстояние между векторами
distance = np.sqrt(np.sum((vector1 - vector2)**2))

print(f"Евклидово расстояние между вектором 1 и вектором 2: {distance}")


# In[29]:


import numpy as np

A = np.array([[-1, 2, 4],
              [-3, 1, 2],
              [-3, 0, 1]])
B = np.array([[3, -1], 
              [2, 1]])
C = np.array([[7, 21], 
              [11, 8],
              [8, 4]])

X = np.matmul(np.matmul(np.linalg.inv(A), -C), np.linalg.inv(B))

print(X)


# In[30]:


import numpy as np

A = np.array([[-1, 2, 4], [-3, 1, 2], [-3, 0, 1]])
B = np.array([[3, -1], [2, 1]])
C = np.array([[7, 21], [11, 8], [8, 4]])

# Найдем обратные матрицы A и B
A_inv = np.linalg.inv(A)
B_inv = np.linalg.inv(B)

# Найдем матрицу X
X = np.dot(A_inv, np.dot(-C, B_inv))

print(X)


# In[ ]:


import numpy as np
array = np.array([[]])


# In[ ]:




