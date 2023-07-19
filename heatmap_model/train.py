import torch
#from torch.cuda.amp import autocast, GradScaler
from heatmap_model.utils import *
from heatmap_model.interaction_dataset import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#scaler = GradScaler()

def train_one_epoch(model, optimizer, loss_2, training_loader, scheduler):
    running_loss = 0.
    last_loss = 0.
    
    for j, data in enumerate(training_loader):
        traj, splines, lanefeature, adj, af, c_mask, timestamp, y, gtxy, gttime = data
        #print(gttime.size())
        optimizer.zero_grad()
        heatmap, tslice, hreg = model(traj, splines, lanefeature, adj, af, c_mask, timestamp, gtxy)
        loss = loss_2([heatmap, tslice, hreg], [y, gttime])
        print('loss:'+str(loss.item()),end='\r')
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if j % 800 == 799:
            scheduler.step()
            last_loss = running_loss / 800
            print('  batch {} loss: {}'.format(j + 1, last_loss))
            running_loss = 0.
                
    return last_loss
        
def train_model(epochs, batch_size, trainset, model, optimizer, validation_loader, loss_2, scheduler):
    training_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    epoch_number = 0

    EPOCHS = epochs

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        #model.half()
        avg_loss = train_one_epoch(model, optimizer, loss_2, training_loader, scheduler)
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            traj, splines, lanefeature, adj, af, c_mask, timestamp, y, gtxy, gttime = vdata
            heatmap, tslice, hreg = model(traj, splines, lanefeature, adj, af, c_mask, timestamp, gtxy)
            heatmaploss = loss_2([heatmap, tslice, hreg], [y, gttime])
            running_vloss += float(heatmaploss)

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        scheduler.step()
        epoch_number += 1