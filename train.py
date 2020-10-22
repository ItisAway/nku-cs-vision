# training
import torch
from train_utils import training
from binary_model import BinaryLeNetCn, get_c2c


paras = {}
paras["batch_size"] = 100
paras["pre_epoch"] = 0
paras["epoch"] = 100
paras["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paras["pth_folder"] = 'binary_c2_zoo'
paras["cost"] = torch.nn.CrossEntropyLoss()
paras["optim"] = "adam"
paras["lr"] = 1e-3
paras["c2c"] = get_c2c(2)
#lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False

net = BinaryLeNetCn(2)

training(net, paras)


    