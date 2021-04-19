from torch import nn, optim, Tensor, cuda, device, no_grad
from utils_code import util_zhang_new as util_zhang, tensor_board_utils as tbutils
import numpy as np
from tqdm import tqdm
import os
from U_CapsNets.U_CapsNet.py import CapsNet_MR
from Colour_utils.training_layers import PriorBoostLayer, NNEncLayer, NonGrayMaskLayer
from PIL import Image
from utils_code.utils import save_checkpoint, rfolder, isnan, reconstruction_loss, resume_model
import matplotlib.pyplot as plt
import time
from apex import amp    #<==AMP


file_model_name = "UCapsNet"
db_used = 'Imagenet'
CUDA =  'cuda:3'
device = device(CUDA)
batch_size, n_epochs, epoch_start, epoch_val, lr_G = 32, 100, 0, 5, 2e-5

if db_used == 'Imagenet': dataloaders = util_zhang.load_Imagenet_dataset('/media/TBData3/Datasets/ImageNetOriginale', batch_size)
if db_used == 'TinyImagenet':  dataloaders = util_zhang.load_dataset('/media/TBData3/Datasets/tiny-imagenet-200/train',
                                                                     '/media/TBData3/Datasets/tiny-imagenet-200/val', batch_size)

folder_results = rfolder("Results_/"+file_model_name+"_"+db_used, n_epochs)
cuda.set_device(device)
generator = CapsNet_MR(128).to(device)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

generator, g_optimizer = amp.initialize(generator, g_optimizer, opt_level="O1")

criterion = nn.CrossEntropyLoss().to(device)
criterionAB = nn.MSELoss()
encode_layer = NNEncLayer()
boost_layer = PriorBoostLayer()
nongray_mask = NonGrayMaskLayer()

(image_batch_val, target) = next(iter(dataloaders['val']))
img_l_rs_v = image_batch_val[1].to(device)
if tb:
    writer.add_image('Colourisation',
                      tbutils.plot_colourisation(generator, img_l_rs_v, batch_size),
                      global_step=0 * len(dataloaders['train']) + 0)
tot_batch = len(dataloaders['train'])
tic = time.time()
log_loss_G = []
with tqdm(total=n_epochs-epoch_start) as pbar:
    for epoch in range(epoch_start,n_epochs):
        log_loss = 0
        generator.train()
        with cuda.amp.autocast(): #<==AMP
            for batch_id, (image_batch,target) in enumerate(dataloaders['train']):
                img_ab_rs = image_batch[0]
                img_l_rs = image_batch[1].to(device)
                img_rxs_ab = image_batch[2]

                g_optimizer.zero_grad()
                img_ab_pred,img_q =generator(img_l_rs)

                encode, max_encode = encode_layer.forward(img_rxs_ab)
                targets = Tensor(max_encode).long().to(img_q.device)
                boost = Tensor(boost_layer.forward(encode)).float().to(img_q.device)
                mask = Tensor(nongray_mask.forward(img_rxs_ab)).float().to(img_q.device)
                boost_nongray = boost*mask

                img_ab_rs = img_ab_rs.to(img_ab_pred.device)
                lossAB = reconstruction_loss(img_ab_rs, img_ab_pred,criterionAB)
                lossQ = (criterion(img_q, targets) * boost_nongray.squeeze(1)).mean()
                loss = lossAB + lossQ
                with amp.scale_loss(loss, g_optimizer) as scaled_loss: #<==AMP
                    scaled_loss.backward() #<==AMP

                g_optimizer.step()
                log_loss += loss.item()
                if batch_id % 1000 == 0:
                    toc = time.time()
                    print(f'batch: {batch_id} of {tot_batch}: time  {int((toc - tic) / 60)}m,{int((toc - tic) % 60)}s')
                del loss,img_ab_pred, img_q, encode, max_encode, targets, boost, mask, boost_nongray,\
                    image_batch, target, lossAB, lossQ, \
                    img_ab_rs, img_l_rs, img_rxs_ab
            log_loss_G.append(log_loss/batch_id)
            #VALIDATION
            if epoch%epoch_val == 0:
                generator.eval()
                with no_grad():
                    img_l_rs_v = image_batch_val[1].to(device)
                    img_ab_pred, _ = generator(img_l_rs_v)
                    for j in range(batch_size):
                        img_l_rs_v = img_l_rs_v.to(img_ab_pred.device)
                        img_rgb = util_zhang.postprocess_tens(img_l_rs_v, img_ab_pred, j, mode='bilinear')
                        im = Image.fromarray((img_rgb * 255).astype(np.uint8))
                        if not os.path.exists(os.path.join(folder_results,str(epoch))):
                            os.mkdir(os.path.join(folder_results,str(epoch)))
                        im.save(os.path.join(folder_results,str(epoch), "val_" + str(j) + ".jpeg"))
                    del img_ab_pred, img_rgb
                save_checkpoint({
                    'epoch': epoch + 1,
                    'loss_type': 'MSE',
                    'arch': file_model_name,
                    'state_dict': generator.state_dict(),
                    'optimizer': g_optimizer.state_dict(),
                }, os.path.join(folder_results,
                                "BACKUP_model_log/checkpoint_" + file_model_name + "_" + str(epoch) + ".pth.tar"))
                del img_l_rs_v
            toc = time.time()
            pbar.set_description(("{:.1f}s - loss: {:.3f}".format((toc - tic), np.mean(log_loss_G))))
            pbar.update(1)

save_checkpoint({
    'epoch': epoch + 1,
    'loss_type': 'MSE',
    'arch': file_model_name,
    'state_dict': generator.state_dict(),
    'optimizer': g_optimizer.state_dict(),
},os.path.join(folder_results,"model_log/checkpoint_"+file_model_name+"_"+str(epoch)+".pth.tar"))
epochs_G = np.arange(1, len(log_loss_G)+1)
plt.plot(epochs_G, log_loss_G, color='g', label='loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training phase')
plt.savefig(os.path.join(folder_results,"log_loss_G.png"))
plt.clf()




