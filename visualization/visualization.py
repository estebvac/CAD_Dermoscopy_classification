import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn import metrics


# plotting rondom images from dataset
def class_plot(data, encoder, inv_normalize=None, n_figures=12):
    n_row = int(n_figures / 4)
    fig, axes = plt.subplots(figsize=(14, 10), nrows=n_row, ncols=4)
    imgs_show = np.arange(12)
    for i, ax in zip(imgs_show, axes.flatten()):
        a = random.randint(0, len(data))
        # (image,label) = data[i]
        (image, label) = data[a]
        # print(type(image), np.mean(image.data.numpy()), np.std(image.data.numpy()))
        label = int(label)
        l = encoder[label]
        if (inv_normalize != None):
            image = inv_normalize(image)

        # print(type(image), np.mean(image.data.numpy()), np.std(image.data.numpy()))
        image = image.numpy().transpose(1, 2, 0)
        im = ax.imshow(image)
        ax.set_title(l)
        ax.axis('off')
    plt.show()


def error_plot(train_loss, valid_loss):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')


def acc_plot(train_acc, valid_acc):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy')
    plt.plot(range(1, len(valid_acc) + 1), valid_acc, label='Validation Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim(0.5, 1)  # consistent scale
    plt.xlim(0, len(train_acc) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')


# To plot the wrong predictions given by model
def wrong_plot(n_figures, true, ima, pred, encoder, inv_normalize):
    print('Classes in order Actual and Predicted')
    n_row = int(n_figures / 3)
    fig, axes = plt.subplots(figsize=(14, 10), nrows=n_row, ncols=3)
    for ax in axes.flatten():
        a = random.randint(0, len(true) - 1)

        image, correct, wrong = ima[a], true[a], pred[a]
        image = torch.from_numpy(image)
        correct = int(correct)
        c = encoder[correct]
        wrong = int(wrong)
        w = encoder[wrong]
        f = 'A:' + c + ',' + 'P:' + w
        if inv_normalize != None:
            image = inv_normalize(image)
        image = image.numpy().transpose(1, 2, 0)
        im = ax.imshow(image)
        ax.set_title(f)
        ax.axis('off')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def performance_matrix(true, pred):
    precision = metrics.precision_score(true, pred, average='macro')
    recall = metrics.recall_score(true, pred, average='macro')
    accuracy = metrics.accuracy_score(true, pred)
    f1_score = metrics.f1_score(true, pred, average='macro')
    print('Precision: {} Recall: {}, Accuracy: {}: ,f1_score: {}'.format(precision * 100, recall * 100, accuracy * 100,
                                                                         f1_score * 100))
