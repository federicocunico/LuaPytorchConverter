from torch.utils.serialization import load_lua
import torch.legacy.nn

net_name = 'score_cnn'
extension = '.t7'

# use long_size=8 to avoid TypeError exception in trained net coming from unix systems
model = load_lua(net_name + extension, long_size=8)

print(model)
print(model.__class__)

torch.save(model, net_name + '_converted.net')
