#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os, json, cv2, numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from utils import collate_fn
from torchvision.transforms import functional as F
from utilities import train_transform,ClassDataset,visualize,get_model
import time


n_keypoints = 20
keypoints_classes_ids2names = {k:str(k) for k in range(n_keypoints)}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_keypoints = n_keypoints,weights_path='keypoints_foot_20.pth')
model.to(device)
model.eval()


KEYPOINTS_FOLDER_TRAIN = '../train'
KEYPOINTS_FOLDER_TEST = '../test'

dataset_test = ClassDataset(KEYPOINTS_FOLDER_TEST,n_keypoints, transform=None, demo=False)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True, collate_fn=collate_fn)

iterator = iter(data_loader_test)

images, targets = next(iterator)
images = list(image.to(device) for image in images)

with torch.no_grad():
    model.to(device)
    model.eval()
    start_time = time.time()
    output = model(images)
    print("--- inference %s seconds ---" % (time.time() - start_time))

image = (images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
scores = output[0]['scores'].detach().cpu().numpy()

high_scores_idxs = np.where(scores > 0.7)[0].tolist() # Indexes of boxes with scores > 0.7
post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

keypoints = []
for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
    keypoints.append([list(map(int, kp[:2])) for kp in kps])

bboxes = []
for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
    bboxes.append(list(map(int, bbox.tolist())))
    

visualize(image, bboxes, keypoints,keypoints_classes_ids2names)


