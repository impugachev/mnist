from tensorflow import keras

# скачиваем датасет с цифрами
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
# нормализуем данные
train_images = train_images / 255.0
test_images = test_images / 255.0
"""
создаем нейронную сеть
первый слой входной, он разворачивает матрицу 28х28 пикселей в одномерный массив
второй слой скрытый из 128 нейронов, функция активации max(x, 0)
третий слой выходной из 10 нейронов(вероятность каждой цифры), функция активации softmax 
(логистическая, т.е. такая же, как сигмоида, но для многомерного случая)
"""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
"""
компилируем модель
Оптимизатор — показывает каким образом обновляется модель на основе входных данных и функции потерь
адам - стохастический (случайный) оптимизатор
Функция потерь — измеряет точность модели во время обучения
здесь задана функция для задачи классификации (сумма логарифмов вероятностей)
Метрика - используются для мониторинга тренировки и тестирования модели.
почти всегда используется accuracy - доля правильной классификации
"""
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
"""
тренируем модель
число эпох задаем 10
"""
model.fit(train_images, train_labels, epochs=10)
# сохраняем модель в файл, чтобы не обучать ее каждый раз
model.save('my_model.h5')
