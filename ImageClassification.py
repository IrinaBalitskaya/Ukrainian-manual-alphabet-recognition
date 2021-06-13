import cv2
import numpy as np
import torch.nn.functional as F
import torch

model = cv2.dnn.readNetFromONNX("Net_100.onnx")
classes = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Є', 'Ж',
           'З', 'И', 'I', 'К', 'Л', 'М', 'Н', 'О',
           'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц',
           'Ч', 'Ш', 'Ь', 'Ю', 'Я', 'Фон']


def imagePreparation(img, img_size, key=False):
    if key:
        img = cv2.imread(img)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.divide(img, 255)
    img = np.subtract(img, 0.5)
    img = np.divide(img, 0.5)
    return img


def imageClassification(image, img_size):
    blob = cv2.dnn.blobFromImage(np.float32(image), 1, (img_size, img_size))
    model.setInput(blob)

    res = model.forward()
    predicted_classes = classes[np.array(res)[0].argmax()]

    prob = F.softmax(torch.tensor(res), dim=1)
    prob = np.array(prob)[0].max()

    return predicted_classes, prob


def letterOutput(pred_letter, letter_list, time):
    if pred_letter != 'Фон':
        letter_list.append(pred_letter)
        if len(letter_list) > 1:
            if letter_list[len(letter_list) - 1] != letter_list[len(letter_list) - 2]:
                letter_list = []
            else:
                if len(letter_list) == time * 30 + 1:
                    letter_list = []
    else:
        letter_list = []
    return letter_list


