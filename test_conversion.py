import torch
import CPU_converted
import cv2 as cv
import os
from torch.autograd import Variable


net_path = os.path.join(os.getcwd(), 'CPU_converted.t7')
img_path = os.path.join(os.getcwd(), 'Test_Images', 'frame-000000.color.png')

test = cv.imread(img_path)
test_img = cv.resize(test, (480, 640))

model = CPU_converted.CPU_converted
model.load_state_dict(torch.load(os.path.join('CPU_converted.pth')))
# Remember that you must call model.eval() to set dropout and batch normalization
# layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.
model.eval()


print(model)

torch_test_img =torch.from_numpy(test_img)
data = torch_test_img.unsqueeze(0)
real_test = Variable(data).unsqueeze(0)


print(torch_test_img)
print(data)
print(real_test)


results = model(torch_test_img)
_, predicted = torch.max(results, 1)

print(results)
print(predicted)