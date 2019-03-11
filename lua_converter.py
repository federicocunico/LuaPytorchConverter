import torch
import CPU_converted
import os

net_path = os.path.join(os.getcwd(), 'CPU_converted.t7')

model = CPU_converted.CPU_converted
model.load_state_dict(torch.load(os.path.join('CPU_converted.pth')))
model.eval()

print(model)


torch.save(model.state_dict(), 'exported_trained_net.net')
