import cv2
import numpy as np
from tkinter import ttk
import matplotlib.pylab as plt
from pathlib import Path
import glob,os,json 
import pandas as pd
from shapely.geometry import MultiPoint
from datetime import datetime
import shutil

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    "kcf": cv2.legacy.TrackerKCF_create,
    "boosting": cv2.legacy.TrackerBoosting_create,
    "mil": cv2.legacy.TrackerMIL_create,
    "tld": cv2.legacy.TrackerTLD_create,
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
    "mosse": cv2.legacy.TrackerMOSSE_create}

'''
I recommend using either CSRT, KCF, or MOSSE for most object tracking applications:
Use CSRT when you need higher object tracking accuracy and can tolerate slower FPS throughput
Use KCF when you need faster FPS throughput but can handle slightly lower object tracking accuracy
Use MOSSE when you need pure speed
'''
def add_tracker(box):
    tracker = OPENCV_OBJECT_TRACKERS[tracker_type]()
    trackers.add(tracker, frame, box)

def draw_bb_s(frame, bb_s,font = cv2.FONT_HERSHEY_SIMPLEX,fontScale = 0.5, color = (255,255, 255),thickness = 1):
    cent_points = []
    x_min, y_min, x_max, y_max = None,None,None,None
    for bi, bb in enumerate(bb_s):
        x1, y1, w, h = bb
        x1, y1, w, h = int(x1),int(y1), int(w), int(h)
        cent_x, cent_y = int(x1 + (w / 2)), int(y1 + (h / 2))
        cent_points.append((cent_x, cent_y))
        box_color = values_color[comboboxvalue_list[bi]]
        #cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), box_color, 2)
        cv2.circle(frame, (cent_x, cent_y), 5, box_color, 3)
        cv2.putText(frame,str(bi), (cent_x+5, cent_y-5), font, fontScale, color, thickness, cv2.LINE_AA)
    if len(cent_points)>0:
        cent_points = MultiPoint(cent_points)
        cent_points_xy = cent_points.bounds
        x_min, y_min, x_max, y_max = cent_points_xy
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,0,255), 2)
    return x_min, y_min, x_max, y_max,frame

def on_mouse_event(event, x, y, flags, params):
    global top_left_corner, bottom_right_corner, new_tracker, lista_window, RoI, comboboxvalue
    if event == cv2.EVENT_LBUTTONDOWN:
        # new_tracker = True
        top_left_corner = (np.clip(x - (bb_x_size/2), a_min=0, a_max=image_w), np.clip(y - (bb_y_size/2), a_min=0, a_max=image_h))
        bottom_right_corner = (np.clip(x + (bb_x_size/2), a_min=0, a_max=image_w), np.clip(y + (bb_y_size/2), a_min=0, a_max=image_h))

        x1, y1 = top_left_corner
        w = bottom_right_corner[0] - top_left_corner[0]
        h = bottom_right_corner[1] - top_left_corner[1]
        box = (x1, y1, w, h)
        comboboxvalue += 1
        print("Punto #: ", comboboxvalue)
        #print('add tracker')
        add_tracker(box)        
        comboboxvalue_list.append(str(comboboxvalue))
        bb_s.append(box)

class Button:
    def __init__(self, text1, text2):
        self.text1 = text1
        self.text2 = text2
        self.state = 1
    def pressed(self):
        if self.state == 0:
            text = self.text1
            self.state = 1
        elif self.state == 1:
            text = self.text2
            self.state = 0
        return self.state, text

tracker_type = "csrt"
values = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
              "16", "17", "18", "19","20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
              "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40"]

pad = 1 # pad of kernel
bb_size_per = 25
images_path_folder = 'data/img (3)' 
border_pix = 100

result_folder = 'results'
output_folder = os.path.join(result_folder,os.path.basename(images_path_folder) + '_processed')

shutil.rmtree(output_folder, ignore_errors=True)

video_path = os.path.join(output_folder,'video.mp4')
keypoints_path = os.path.join(output_folder,'keypoints.csv')
images_folder = os.path.join(output_folder,'images')
annotations_folder = os.path.join(output_folder,'annotations')

# create output folder
Path(output_folder).mkdir(parents=True, exist_ok=True)
Path(images_folder).mkdir(parents=True, exist_ok=True)
Path(annotations_folder).mkdir(parents=True, exist_ok=True)

images_paths = sorted(glob.glob(images_path_folder + '/*.bmp'))
image_sample = cv2.imread(images_paths[0])
image_h,image_w,_ = image_sample.shape
image_h,image_w = image_h+2*border_pix,image_w+2*border_pix

# initialize OpenCV's special multi-object tracker
trackers = cv2.legacy.MultiTracker_create()

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(video_path, fourcc, 20.0, (image_w,image_h))

bb_x_size = int(image_w * (bb_size_per / 100))
bb_y_size = int(image_h * (bb_size_per / 100))
bb_s = []

### GUI
root_wind = 'Pie tracking'
cv2.namedWindow(root_wind)
cv2.setMouseCallback(root_wind, on_mouse_event)
new_tracker = False
top_left_corner = []
bottom_right_corner = []

colors = plt.cm.jet(np.linspace(0, 1, len(values)))
values_color = {value: color[1:] * 255 for value, color in zip(values, colors)}
comboboxvalue =0
comboboxvalue_list = []

btn_stop_start = Button('Stop', 'Start')
btn_stop_start_text = 'Stop'

RoI = None
# font
text_font = cv2.FONT_HERSHEY_SIMPLEX
# org
text_org = (900, 30)
# fontScale
text_fontScale = 1
# Blue color in BGR
test_color = (0, 255, 0)
# Line thickness of 2 px
text_thickness = 2
border_pix = 100

# loop over frames from the video stream
boxes_past = []
image_ind = 0

column_names = ["frame"]
for i in range(20): # number of keypoints
    column_names.append('x_'+str(i))
    column_names.append('y_'+str(i))
    column_names.append('v_'+str(i))
df = pd.DataFrame(columns = column_names)


while True:
    try:
        image_ori = cv2.imread(images_paths[image_ind])
    except:
        break
    frame = image_ori.copy()

    frame = cv2.copyMakeBorder(frame, border_pix, border_pix, border_pix, border_pix, cv2.BORDER_CONSTANT, None, value=0)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    if key == ord("s"):  # Stop/Play
        _, btn_stop_start_text = btn_stop_start.pressed()
        print(btn_stop_start_text)

    if btn_stop_start_text != 'Stop':
        (success, boxes) = trackers.update(frame)
        boxes = boxes.astype(int)
        boxes_past.append(boxes)
        boxes_past_array = np.stack(boxes_past[-2:], axis=2)
        boxes = boxes_past_array.mean( axis=2)
        if len(boxes) > 1:
            x1, y1, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1, y1, w, h = x1.reshape(-1,1), y1.reshape(-1,1), w.reshape(-1,1), h.reshape(-1,1)
            cent_x, cent_y = x1 + (w / 2), y1 + (h / 2)
            cent_x, cent_y = cent_x.astype(int), cent_y.astype(int)
            new_row = {}
            for i, (c_x, c_y) in enumerate(zip(cent_x, cent_y)):
                x, y = c_x[0],c_y[0]

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                kernel = frame_gray[y-pad:y+pad+1,x-pad:x+pad+1]
                kernel_value = np.round_(kernel.mean())
                new_row['frame'] = int(image_ind)
                new_row['x_'+str(i)] = int(x)
                new_row['y_'+str(i)] = int(y)
                new_row['v_'+str(i)] = kernel_value
            new_row = pd.DataFrame.from_dict(new_row,orient='index').T
            df = pd.concat([df, new_row])
            x_min, y_min, x_max, y_max,frame = draw_bb_s(frame, boxes)
            
            out.write(frame)
            # remove border  
            x_min, y_min, x_max, y_max = x_min- border_pix, y_min- border_pix, x_max- border_pix, y_max- border_pix
            cent_x, cent_y = cent_x - border_pix, cent_y - border_pix
            # get current time
            now = datetime.now()
            now = now.strftime("%d%m%Y%H%M%S")
            # generate image and json names
            image_name = os.path.join(images_folder,now+'_'+str(image_ind)+'.jpg')
            annotation_name = os.path.join(annotations_folder,now+'_'+str(image_ind)+'.json')
            # save image
            cv2.imwrite(image_name, image_ori)
            # generate json annotation
            keypoints = [[int(c_x[0]),int(cy[0]),1] for c_x,cy in zip(cent_x,cent_y)]
            annotation_dic= {"bboxes": [[ x_min, y_min, x_max, y_max]],
                            "keypoints": [keypoints]}
            annotation_json = json.dumps(annotation_dic)
            with open(annotation_name, "w") as file_json:
                file_json.write(annotation_json)
        image_ind += 1
    if btn_stop_start_text == 'Stop':
        x_min, y_min, x_max, y_max,frame = draw_bb_s(frame, bb_s)
        frame = cv2.putText(frame, btn_stop_start_text, text_org, text_font, text_fontScale, test_color, text_thickness,cv2.LINE_AA)
    # show the output frame
    cv2.imshow(root_wind, frame)
out.release()
df.to_csv(keypoints_path)
# close all windows
cv2.destroyAllWindows()