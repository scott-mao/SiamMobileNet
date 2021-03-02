from pytorch.pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.autograd import Variable

net = ptcv_get_model("mobilenet_w1", pretrained=True)
m0 = list(net.modules())
print(m0)
x = Variable(torch.randn(1, 3, 224, 224))
y = net(x)
print(y.shape)