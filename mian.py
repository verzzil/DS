""" Методики предварительной обработки данных """
import numpy as np
from sklearn import preprocessing

input_data = np.array([[5.1, -2.9, 3.3],
                      [-1.2, 7.8, -6.1],
                      [3.9, 0.4, 2.1],
                      [7.3, -9.9, -4.5]])

"""
    1. Метод бинаризации данных, он применяется, когда мы хотим преобразовать
        числовые значения в булевы
"""
# Укаываем пороговое значение 2,1
date_binarized = preprocessing.Binarizer(threshold = 2.1).transform(input_data)

print("First method:")
print(date_binarized)


"""
    2. Метод исключения среднего. Из векторов признаков целесообразно
        исключать средние значения, чтобы каждый признак центрировался на нуле.
        Это делается с той целью, чтобы исключить из рассмотрения смещение
        значеий в векторах признаков
"""
# Вывод среднего значения и стандартного отклонения
print("\n\nSecond method:\nBEFORE:")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

# Исключение среднего
data_scaled = preprocessing.scale(input_data)
print("\nAFTER:")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))


"""
    3. Метод масштабирования данных. Тк каждое значение признаков может меняться
        важно масштабировать признаки, чтобы они представляли собой ровное
        игровое поле для тренировки алгоритма. Мы не хотим, чтобы любой из
        признаков мог принимать большое или малое значение в силу природы
        измерений
"""
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\n\nThird method:\nMin max scaled data:\n", data_scaled_minmax)


"""
    4. Метод нормализации заключается в изменении значений в векторе признаков
        таким образом, чтобы для их измерения можно было использовать одну общ
        шкалу. L1-нормализация(сумма абсолютных значений в каждом ряду = 1)
               L2-нормализация(сумма квадратов значений в каждом ряду = 1)
"""
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\n\nForth method:\n L1:", data_normalized_l1)
print("L2:",data_normalized_l2)
