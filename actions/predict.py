# Predict
from PIL import Image
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt 


def predict(model, image, device, encoder, transforms=None, inv_normalize=None):
    # model = torch.load('./model.h5')
    model.eval()
    if (isinstance(image, np.ndarray)):
        image = Image.fromarray(image)
    if (transforms != None):
        image = transforms(image)
    data = image.expand(1, -1, -1, -1)
    data = data.type(torch.FloatTensor).to(device)
    sm = nn.Softmax(dim=1)
    output = model(data)
    output = sm(output)
    _, preds = torch.max(output, 1)
    img_plot(image, inv_normalize)
    prediction_bar(output, encoder)
    return preds


def prediction_bar(output, encoder):
    output = output.cpu().detach().numpy()
    a = output.argsort()
    a = a[0]

    size = len(a)
    if (size > 5):
        a = np.flip(a[-5:])
    else:
        a = np.flip(a[-1 * size:])
    prediction = list()
    clas = list()
    for i in a:
        prediction.append(float(output[:, i] * 100))
        clas.append(str(i))
    for i in a:
        print('Class: {} , confidence: {}'.format(encoder[int(i)], float(output[:, i] * 100)))
    plt.bar(clas, prediction)
    plt.title("Confidence score bar graph")
    plt.xlabel("Confidence score")
    plt.ylabel("Class number")


def img_plot(image, inv_normalize=None):
    if (inv_normalize != None):
        image = inv_normalize(image)
    image = image.cpu().numpy().transpose(1, 2, 0)
    plt.imshow(image)
    plt.show()