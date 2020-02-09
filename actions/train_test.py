from torch.autograd import Variable
import torch
import numpy as np
import pickle

from actions.EarlyStopping import EarlyStopping
from visualization.visualization import *

def train(model, dataloaders, criterion, device, num_epochs=10, lr=0.00001,
          batch_size=8, patience=None, cycle=10,
          model_path='./checkpoint.pt', history_path='./historial.pickle'):
    # Load the previous checkpoint:
    try:
        model.load_state_dict(torch.load(model_path))
        print('Previous weights loaded')
    except:
        print('Previous weights not loaded')

    # Load the previous historial:
    try:
        with open(history_path, 'rb') as handle:
            historial = pickle.load(handle)
        print('Previous Historial loaded')
        min_val_loss = min(historial['loss_val'])
    except:
        # Initialize the variables to store the training losses and accuracy
        historial = {}
        historial['loss_train'] = list()
        historial['acc_train'] = list()
        historial['loss_val'] = list()
        historial['acc_val'] = list()
        min_val_loss = np.Inf

    # Get phases validation and train
    all_phases = dataloaders.keys()
    # Move model to GPU
    model.to(device)

    # Enable the Early Stopping:
    if (patience != None):
        earlystop = EarlyStopping(patience=patience, val_loss_min=min_val_loss, verbose=True)
    else:
        earlystop = EarlyStopping(patience=num_epochs, val_loss_min=min_val_loss, verbose=True)

    # Set up the optimizer:
    # passing only those parameters that explicitly requires grad
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # Set up the LR scheduler:
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                                              verbose=True, threshold=0.0001, threshold_mode='rel',
                                                              cooldown=0, min_lr=0, eps=1e-08)

    if num_epochs == 0:
        return historial

    # Start the training process:
    for epoch in range(num_epochs):

        # SET UP THE OPTIMIZATION
        for param_group in optimizer.param_groups:
            lr_new = param_group['lr']
        print('Epoch:', epoch + 1, ' learning rate: ', lr_new)

        for phase in all_phases:
            # Set up model according phase:
            if phase == ' train':
                model.train()
            else:
                model.eval()

            # Metrics inintialization:
            running_loss = 0.0
            running_corrects = 0

            # Evaluate for each batch:
            for batch_idx, (data, target) in enumerate(dataloaders[phase]):
                # Get the data of each batch:
                data, target = Variable(data), Variable(target)
                data = data.type(torch.cuda.FloatTensor)
                target = target.type(torch.cuda.LongTensor)

                # Set the gradient to zero:
                optimizer.zero_grad()

                # Execute the model:
                output = model(data)

                # Evaluate the model:
                loss = criterion(output, target)
                _, preds = torch.max(output, 1)
                running_corrects = running_corrects + torch.sum(preds == target.data)
                running_loss += loss.item() * data.size(0)
                batch_num = batch_idx + 1

                # Perform back propagation:
                if (phase == 'train'):
                    loss.backward()
                    optimizer.step()

                # Print every 300 batches:
                if batch_num % 50 == 0:
                    print('{} Epoch: {}  [{}'.format(phase, epoch + 1, batch_num * len(data)),
                          '/{} ({:.0f}%)]'.format(len(dataloaders[phase].dataset),
                                                  100. * batch_num / len(dataloaders[phase])),
                          '\tLoss: {:.6f} \tAcc: {:.6f}'.format(running_loss / (batch_num * batch_size),
                                                                running_corrects.double() / (batch_num * batch_size)))
            # Get the final value of the epoch:
            epoch_acc = running_corrects.double() / (len(dataloaders[phase]) * batch_size)
            epoch_loss = running_loss / (len(dataloaders[phase]) * batch_size)

            print('{} Epoch: {}  [{}'.format(phase, epoch + 1, batch_num * len(data)),
                  '/{} ({:.0f}%)]'.format(len(dataloaders[phase].dataset),
                                          100. * batch_num / len(dataloaders[phase])),
                  '\tLoss: {:.6f} \tAcc: {:.6f}'.format(epoch_loss, epoch_acc))

            # Check the Early Stopping Conditions
            if (phase == 'val'):
                earlystop(epoch_loss, model, model_path)
                lr_scheduler.step(epoch_loss)
                print('Early Stopping Flag: ', earlystop.early_stop)
                loss_val = epoch_loss
                historial['loss_val'].append(epoch_loss)
                historial['acc_val'].append(epoch_acc)

            # Store the training results
            if (phase == 'train'):
                historial['loss_train'].append(epoch_loss)
                historial['acc_train'].append(epoch_acc)

        # Save the historial:
        with open(history_path, 'wb') as handle:
            pickle.dump(historial, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # If the early stopping condition is reached finish the training and restore the weights
        if (earlystop.early_stop):
            print("Early stopping")
            model.load_state_dict(torch.load(model_path))
            break
        print('{} Accuracy: '.format(phase), epoch_acc.item())

    # Restore the best parameters if the last one is not the best:
    if loss_val > earlystop.best_score:
        print('Best parameters restored')
        model.load_state_dict(torch.load(model_path))

    return historial

def test(dataloader, classifier, criterion, batch_size):
    # Initialize variables:
    running_corrects = 0
    running_loss = 0
    pred = []
    true = []
    pred_wrong = []
    true_wrong = []
    image = []

    # Run for all the batches:
    for batch_idx, (data, target) in enumerate(dataloader):
        # Evaluate the classifier in the batch
        data, target = Variable(data), Variable(target)
        data = data.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.LongTensor)
        classifier.eval()
        output = classifier(data)

        # Evaluate the output loss
        loss = criterion(output, target)
        _, preds = torch.max(output, 1)
        running_corrects = running_corrects + torch.sum(preds == target.data)
        running_loss += loss.item() * data.size(0)

        # Convert the preduction and the target to numpy
        preds = preds.cpu().numpy()
        target = target.cpu().numpy()
        preds = np.reshape(preds, (len(preds), 1))
        target = np.reshape(target, (len(preds), 1))
        data = data.cpu().numpy()

        # Concatenate all the data and find the wrong matches
        for i in range(len(preds)):
            pred.append(preds[i])
            true.append(target[i])
            if (preds[i] != target[i]):
                pred_wrong.append(preds[i])
                true_wrong.append(target[i])
                image.append(data[i])

    # Calculate the accuracy and the loss.
    epoch_acc = running_corrects.double() / (len(dataloader) * batch_size)
    epoch_loss = running_loss / (len(dataloader) * batch_size)
    print(epoch_acc, epoch_loss)
    # Return the TP, FP, TN and FN.
    return true, pred, image, true_wrong, pred_wrong


def train_model(model, dataloaders, criterion, num_epochs=10, lr=0.0001,
                batch_size=8, patience=None, classes=None, cycle=10,
                model_path='./checkpoint.pt', history_path='./historial.pickle'):
    dataloader_train = {}
    key = dataloaders.keys()
    # Select data acording the phase
    for phase in key:
        if (phase == 'test'):
            perform_test = True
        else:
            dataloader_train.update([(phase, dataloaders[phase])])
    historial = train(model, dataloader_train, criterion, num_epochs, lr,
                      batch_size, patience, cycle, model_path, history_path)

    # plot the training results:
    error_plot(historial['loss_train'], historial['loss_val'])
    acc_plot(historial['acc_train'], historial['acc_val'])

    # Evaluate the model
    if (perform_test == True):
        true, pred, image, true_wrong, pred_wrong = test(dataloaders['test'])
        wrong_plot(12, true_wrong, image, pred_wrong, encoder, inv_normalize)
        performance_matrix(true, pred)
        if (classes != None):
            plot_confusion_matrix(true, pred, classes=classes, title='Confusion matrix, without normalization')

    return historial