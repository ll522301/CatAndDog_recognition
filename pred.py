import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# 加载模型
model = load_model('cats_and_dogs.h5')

# 测试集路径
test_dir = './cats_and_dogs_small/test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

# 对测试集进行预测
y_pred = model.predict(test_generator)
y_pred = np.where(y_pred > 0.5, 1, 0)
y_pred = np.round(y_pred)

# 输出预测结果到控制台
class_names = ['cat', 'dog']
correct_count = {'cat': 0, 'dog': 0}
wrong_count = {'cat': 0, 'dog': 0}
for i in range(len(test_generator)):
    batch_X, batch_y = test_generator[i]
    y_pred_batch = model.predict(batch_X)
    y_pred_batch = np.where(y_pred_batch > 0.5, 1, 0)
    for j in range(len(batch_y)):
        true_label = 'cat' if batch_y[j] == 0 else 'dog'
        pred_label = 'cat' if y_pred_batch[j] == 0 else 'dog'
        if true_label == pred_label:
            correct_count[true_label] += 1
            print('Sample {} - True: {} Predict: {} Result: Correct'.format(i * 32 + j + 1, true_label, pred_label))
        else:
            wrong_count[true_label] += 1
            print('Sample {} - True: {} Predict: {} Result: Wrong'.format(i * 32 + j + 1, true_label, pred_label))

cat_correct = correct_count['cat']
cat_wrong = wrong_count['cat']
dog_correct = correct_count['dog']
dog_wrong = wrong_count['dog']

# print('Cat Correct: {}, Cat Wrong: {}'.format(cat_correct, cat_wrong))
# print('Dog Correct: {}, Dog Wrong: {}'.format(dog_correct, dog_wrong))

accuracy = (cat_correct + dog_correct) / len(test_generator.filenames)
print('模型的测试正确率: {:.2%}'.format(accuracy))

matrix = np.array([[cat_correct,cat_wrong],[dog_wrong,dog_correct]])

print('混淆矩阵：')

print(matrix)

# 生成分类报告
predictions = model.predict(test_generator)
y_pred = np.round(predictions).flatten()  # 进行二元分类预测
y_true = test_generator.classes #获取真实标签
# 生成分类报告
report = classification_report(y_true, y_pred, target_names=test_generator.class_indices)
print(report)

