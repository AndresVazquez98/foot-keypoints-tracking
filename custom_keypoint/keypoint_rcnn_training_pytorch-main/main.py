#matplotlib.use("TkAgg")
from turtle import color
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
import glob, os, json, cv2, numpy as np
from pyrsistent import v
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from utils import collate_fn
from torchvision.transforms import functional as F
from utilities import train_transform,ClassDataset,visualize,get_model
import time
import pandas as pd
import itertools
from random import randint



def read_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = F.to_tensor(img)
    return img
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        print("create folder")
        os.makedirs(folder_path)
        


class Results:
    def __init__(self,output_path,n_keypoints=20,pad = 1):
        self.n_foots = 2
        self.pad = pad
        self.n_keypoints = n_keypoints
        column_names = ["frame"]
        for n in range(self.n_foots):
            for i in range(self.n_keypoints): # number of keypoints
                column_names.append('x_'+str(n)+'_'+str(i))
                column_names.append('y_'+str(n)+'_'+str(i))
                column_names.append('v_'+str(n)+'_'+str(i))
        self.df = pd.DataFrame(columns = column_names)
        self.output_path = output_path
        self.output_path_csv = os.path.join(output_path, "results.csv")
        self.output_path_csv_ma = os.path.join(output_path, "results_ma.csv")
        
    def add_item(self,image_path,image,bboxes_list,keypoints_list):
        h,w,_ = image.shape
        new_row = {}
        for bi,(bbox, keypoints) in enumerate(zip(bboxes_list,keypoints_list)):
            for i, (key_x, key_y) in enumerate(keypoints):
                #key_x, key_y = int(key_x),int(key_y)
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # normalize
                image_gray = image_gray.astype(float) / 255
                # +1 debido al indice abierto que usa python
                kernel_y_1 = self.clamp(key_y-self.pad, maxn=h)
                kernel_y_2 = self.clamp(key_y+self.pad+1, maxn=h)
                kernel_x_1 = self.clamp(key_x-self.pad, maxn=w)
                kernel_x_2 = self.clamp(key_x+self.pad+1, maxn=w)
                
                kernel = image_gray[kernel_y_1:kernel_y_2,kernel_x_1:kernel_x_2]
                kernel_value = kernel.mean()
                new_row['frame'] = image_path
                new_row['x_'+str(bi)+'_'+str(i)] = key_x
                new_row['y_'+str(bi)+'_'+str(i)] = key_y
                new_row['v_'+str(bi)+'_'+str(i)] = kernel_value
        new_row = pd.DataFrame.from_dict(new_row,orient='index').T
        self.df = pd.concat([self.df, new_row])

    def clamp(self,n, minn=0, maxn=1):
        return max(min(maxn, n), minn)

    def save(self):
        self.df.to_csv(self.output_path_csv,index=False)
        pixel_val_cols_list = []
        for n_foot in range(self.n_foots):
            pixel_val_cols_list.append(self.df.filter(like='v_'+str(n_foot)).columns.to_list())

        colors = ["#%06X" % randint(0, 0xFFFFFF) for i in range(self.n_keypoints)]
        markers = itertools.cycle(( '+', 'o','^','v','<','>','s','x','D','h','1','2','3','4')) 
        for pi, pixel_val_cols in enumerate(pixel_val_cols_list):
            values_list = []
            plt_value = False
            for pj,pixel_val_col in enumerate(pixel_val_cols):
                pixel_val = self.df[pixel_val_col]
                if pixel_val.isna().all() == True:
                    continue
                plt_value  = True
                marker_selected = next(markers)
                plt.figure(pi,figsize=(25,15))
                values_list.append( Line2D(range(len(pixel_val)),pixel_val, marker= marker_selected, linestyle='None',
                          markersize=10, label=pixel_val_col, color=colors[pj]) )
                plt.plot(range(len(pixel_val)),pixel_val,ls="-",linewidth=2.0,color=colors[pj])
                plt.scatter(range(len(pixel_val)),pixel_val,marker = marker_selected,color=colors[pj])

                plt.figure(pi+2,figsize=(25,15))
                plt.plot(range(len(pixel_val)),pixel_val,ls="-",linewidth=2.0,color=colors[pj])
               
                           
            if plt_value:
                plt.figure(pi,figsize=(25,15))
                plt.legend(handles=values_list)
                plt.savefig(os.path.join(self.output_path, "results_foot_{}.png".format(pi)),dpi=400)
                plt.figure(pi+2,figsize=(25,15))
                plt.legend(handles=values_list)
                plt.savefig(os.path.join(self.output_path, "results_foot_no_scatter{}.png".format(pi)),dpi=400)
                plt.show()

        ## apply moving average
        df_ma = self.df.copy()
        n_frames = df_ma.shape[0]
        window_size = int(n_frames/10)
        pixel_val_cols_list_total = pixel_val_cols_list[0] +pixel_val_cols_list[1]

        df_ma_cols = df_ma[pixel_val_cols_list_total].rolling(window=window_size).mean()
        df_ma.update(df_ma_cols)
        df_ma.to_csv(self.output_path_csv_ma,index=False)

        for pi, pixel_val_cols in enumerate(pixel_val_cols_list):
            values_list = []
            plt_value = False
            for pj,pixel_val_col in enumerate(pixel_val_cols):
                pixel_val = df_ma[pixel_val_col]
                if pixel_val.isna().all() == True:
                    continue
                plt_value  = True
                marker_selected = next(markers)
                plt.figure(pi,figsize=(25,15))
                values_list.append( Line2D(range(len(pixel_val)),pixel_val, marker= marker_selected, linestyle='None',
                          markersize=10, label=pixel_val_col, color=colors[pj]) )
                plt.plot(range(len(pixel_val)),pixel_val,ls="-",linewidth=2.0,color=colors[pj])
                plt.scatter(range(len(pixel_val)),pixel_val,marker = marker_selected,color=colors[pj])
            
                plt.figure(pi+2,figsize=(25,15))
                plt.plot(range(len(pixel_val)),pixel_val,ls="-",linewidth=2.0,color=colors[pj])
               
            if plt_value:
                plt.figure(pi,figsize=(25,15))
                plt.legend(handles=values_list)
                plt.savefig(os.path.join(self.output_path, "results_foot_ma_{}.png".format(pi)),dpi=400)
                plt.figure(pi+2,figsize=(25,15))
                plt.legend(handles=values_list)
                plt.savefig(os.path.join(self.output_path, "results_foot_ma_no_scatter{}.png".format(pi)),dpi=400)
                plt.show()





folder_path = 'D:/python_scripts/repo/foot-keypoints-tracking/data/img'
folder_path = 'D:/python_scripts/repo/foot-keypoints-tracking/custom_keypoint/test/images'
folder_path = 'D:/python_scripts/repo/foot-keypoints-tracking/data/img (3)'
folder_path = 'D:/python_scripts/repo/foot-keypoints-tracking/data/img (4)'
folder_path = 'D:/python_scripts/repo/foot-keypoints-tracking/data/Roman/Dentro'

mode_show = 'cv2'
n_keypoints = 20
keypoints_classes_ids2names = {k:str(k) for k in range(n_keypoints)}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_keypoints = n_keypoints,weights_path='keypoints_foot_20.pth')
model.to(device)
model.eval()
image_paths = glob.glob(folder_path+'/*.bmp') + glob.glob(folder_path+'/*.jpg') + glob.glob(folder_path+'/*.png')
output_folder = os.path.join('./results',os.path.basename(folder_path))

create_folder(output_folder)
video_path = os.path.join(output_folder, "results.mp4")

results = Results(output_folder) 
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
image_sample = read_image(image_paths[0])
_,image_h,image_w = image_sample.shape
out = cv2.VideoWriter(video_path, fourcc, 20.0, (image_w,image_h))


with torch.no_grad():
    for image_path in image_paths:
        image = read_image(image_path).to(device)

        start_time = time.time()
        output = model([image])
        print("--- inference %s seconds ---" % (time.time() - start_time))

        image = (image.permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
        scores = output[0]['scores'].detach().cpu().numpy()

        high_scores_idxs = np.where(scores > 0.7)[0].tolist() # Indexes of boxes with scores > 0.7
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

        keypoints_list = []
        for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            keypoints_list.append([list(map(int, kp[:2])) for kp in kps])

        bboxes_list = []
        for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            bboxes_list.append(list(map(int, bbox.tolist())))
            
        image_result = visualize(image, bboxes_list, keypoints_list,keypoints_classes_ids2names)        
        results.add_item(image_path,image,bboxes_list,keypoints_list)

        if mode_show == 'colab':
            plt.imshow(image_result)
            plt.title('prediction')
            plt.show()
        elif mode_show == 'cv2':
            cv2.imshow('prediction',image_result)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        out.write(image_result)
        #time.sleep(1)

print('Done')        
cv2.destroyAllWindows()
out.release()
results.save()


