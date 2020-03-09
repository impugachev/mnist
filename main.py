from tensorflow import keras
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import messagebox
from PIL import Image
import tempfile
import io
import os

def recognize_digit(img_file):
    # opencv для питона не понимает unicode пути, пришлось переводить сначала в массив numpy, а потом декодить
    image = cv.imdecode(np.fromfile(img_file), cv.IMREAD_UNCHANGED)
    # делаем нашу картинку черно-белой
    grey = cv.cvtColor(image.copy(), cv.COLOR_BGR2GRAY)
    # применяем пороговое преобразование
    ret, thresh = cv.threshold(grey.copy(), 75, 255, cv.THRESH_BINARY_INV)
    # просто берем и ищем контуры
    contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    preprocessed_digits = []
    # параметры текущего контура
    x, y, w, h = cv.boundingRect(contours[0])
    max_size = max(w, h)
    x -= int((max_size - w) / 2)
    y -= int((max_size - h) / 2)
    w = max_size
    h = max_size
    # извлекаем по контуру цифру из ч/б изображения
    digit = thresh[y:y + h, x:x + w]
    # изменяем размер до 18х18
    resized_digit = cv.resize(digit, (18, 18))
    # добавляем рамку в 5 пикселей
    padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
    preprocessed_digits.append(padded_digit)

    # cv.namedWindow('digit')
    # cv.imshow('digit', padded_digit)
    # cv.waitKey(0)

    preprocessed_digits = np.array(preprocessed_digits)
    # загружаем модель из файла и делаем предсказания
    predictions = keras.models.load_model('my_model.h5').predict(preprocessed_digits)
    # возвращаем предполагаемую цифру
    return np.argmax(predictions[0])

class Paint(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.setUI()

    def draw(self, event):
        self.canv.create_oval(event.x - 2,
                              event.y - 2,
                              event.x + 2,
                              event.y + 2,
                              fill='black', outline='black')

    def recognize(self):
        _, temp_file = tempfile.mkstemp()
        temp_file = f'{temp_file}.jpg'
        ps = self.canv.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img.save(temp_file, 'jpeg')
        messagebox.showinfo('Recognize result', f'Result is {recognize_digit(temp_file)}')
        self.clear()
        os.remove(temp_file)

    def clear(self):
        self.canv.delete('all')

    def setUI(self):
        self.parent.title("Recognize digit")
        self.pack(fill=tk.BOTH, expand=1)

        f_top = tk.Frame(self.parent)
        clear_button = tk.Button(f_top, text="Clear", width=10, command=self.clear)
        clear_button.pack(side=tk.LEFT)

        recognize_button = tk.Button(f_top, text="Recognize", width=10, command=self.recognize)
        recognize_button.pack(side=tk.LEFT)
        f_top.pack()

        f_bottom = tk.Frame(self.parent)
        self.canv = tk.Canvas(f_bottom, bg="white")
        self.canv.pack()
        self.canv.bind("<B1-Motion>", self.draw)
        f_bottom.pack()



def main():
    root = tk.Tk()
    root.geometry("160x150+300+300")
    Paint(root)
    root.mainloop()

if __name__ == '__main__':
    main()