import torch
import torchvision.transforms as tfms

#should be more careful but importing everything
from model import *
from utils import *
from data import *
from train import *
from export import *

n_outputs = 3 #how many classes to predict
batch_size = 64
n_epochs = 0
print_freq = 1

torch.manual_seed(1234) #fix seed for reproducibility
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

net, transform = get_resnet()
add_head(net, n_outputs)

criterion, optimizer = get_crit_opt(net)
ds_train, ds_test, dl_train, dl_test = get_ds_dl(transform, batch_size)

net, optimizer = train(n_epochs,
                       net, 
                       dl_train,
                       dl_test,
                       criterion, 
                       optimizer, 
                       print_freq,
                       device)

'''
Export to ONNX

Note that:
- The input to net (the output of train) is a transformed image
- The input to model (defined below) is an untransformed image and the transformations
are done in the forward function.
'''

#unwrapped model - expects transformed tensor
dummy_input = torch.randn(ds_train[0][0].unsqueeze(0).shape, requires_grad=True).to(device)
export_to_onnx(net, dummy_input, device, 'resnet_raw.onnx')


#wrapped model - expects tensor scaled by 1/255.
inference_transform = CustomTransformList(inference_transform_list)
model = Model(net, inference_transform)

dummy_input = Image.open('test.jpg').convert('RGB') #hard-coded for now
dummy_input = torch.from_numpy(np.array(dummy_input)).permute(2,0,1).float().unsqueeze(0).to(device)
dummy_input = dummy_input / 255.

export_to_onnx(model, dummy_input, device, 'resnet_wrapped.onnx')
