import torch

avail = torch.cuda.is_available()
print(avail)

## Get Id of default device
print(torch.cuda.current_device())
# 0

print(torch.cuda.get_device_name(0) )