import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from gan_util import show_images

def pic_train(d_model,g_model,g_loss,d_loss,trainloader,testloader,d_optimizer,g_optimizer,num_epochs,input_channels = 3):

    iters = 0
    for epoch in range(num_epochs):
    
        # Training example
        print('EPOCH: ', (epoch+1))
        for x,y in trainloader:
            g_optimizer.zero_grad()
            y_hat = g_model(x)

            # Evaluate generator loss
            gl = g_loss(d_model(y_hat))
            gl.backward()
            g_optimizer.step()

            # Evaluate discriminator loss
            d_optimizer.zero_grad()
            dl = d_loss(d_model(y),d_model(y_hat.detach()))
            dl.backward()
            d_optimizer.step()
    
        # Test examples
        with torch.no_grad():
            for x,y in testloader:
                y_hat = g_model(x)
                gl = g_loss(d_model(y_hat))
                dl = d_loss(d_model(y),d_model(y_hat))
        
        # Logging and output visualization
            if (iter_count % 200 == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,dl.item(),gl.item()))
                imgs_numpy = (y_hat.data).cpu().numpy()
                show_images(imgs_numpy[0:16], color=input_channels!=1)
                plt.show()
                print()
            iter_count += 1