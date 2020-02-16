import numpy as np
import cv2

from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms
from torch.nn import functional as F


features_blobs = []

def hook_feature(module, input, output):
  features_blobs.append(output.data.cpu().numpy())


def returnCAM(feature_conv, weight_softmax, class_idx):
  # generate the class activation maps upsample to 256x256
  size_upsample = (256, 256)
  bz, nc, h, w = feature_conv.shape
  output_cam = []
  for idx in class_idx:
    cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
  return output_cam


def generateCAM(net,image_np,layerName='layer3'):
  net._modules.get(layerName).register_forward_hook(hook_feature)
  params = list(net.parameters())
  weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
  normalize = transforms.Normalize(
    mean=[0.485],
    std=[0.229]
  )
  preprocess = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    normalize
  ])

  img_pil = Image.fromarray(image_np.astype('uint8')).convert('L')
  img_pil.save('test.png')

  img_tensor = preprocess(img_pil)
  img_variable = Variable(img_tensor.unsqueeze(0))
  img_variable = img_variable.cuda()
  logit = net(img_variable)
  h_x = F.softmax(logit, dim=1).data.squeeze()
  probs, idx = h_x.sort(0, True)
  idx = idx.cpu().numpy()
  CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
  img = cv2.imread('test.png')
  height, width, _ = img.shape
  heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
  result = heatmap * 0.3 + img * 0.5
  return result

