import tkinter as tk
from tkinter import filedialog
import os
from PIL import ImageTk, Image
import numpy as np
from tensorflow.keras.models import load_model

# 加载模型
model = load_model('cats_and_dogs.h5')

# 创建GUI界面
root = tk.Tk()
root.title('猫狗识别')
root.geometry('500x600')

# 创建标签和按钮
upload_label = tk.Label(root, text='请上传或拍摄一张照片')
upload_label.pack(pady=20)

image_frame = tk.Frame(root)
image_frame.pack()

image_label = tk.Label(image_frame)
image_label.pack(pady=20)

result_label = tk.Label(root, text='')
result_label.pack()

upload_button = tk.Button(root, text='上传照片', command=lambda: upload_image())
upload_button.pack(pady=20)

# 识别按钮放在底部
predict_button = tk.Button(root, text='识别', command=lambda: predict_image())
predict_button.pack(side=tk.BOTTOM, pady=20)


# 上传照片函数
def upload_image():
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # 显示上传的图像
    image_tk = ImageTk.PhotoImage(Image.open(file_path).resize((250, 250)))
    image_label.config(image=image_tk)
    image_label.image = image_tk

    # 隐藏之前的结果
    result_label.config(text='')

    # 保存图像
    global uploaded_image
    uploaded_image = image


# 图像分类函数
def predict_image():
    result = model.predict(uploaded_image)
    if result < 0.5:
        result_label.config(text='这是一只猫')
        print("这是一只猫")
        result_label.pack(pady=20)
    else:
        result_label.config(text='这是一只狗')
        result_label.pack(pady=20)
        print("这是一只狗")

root.mainloop()
