import glob,os,json
import cv2
import time





def draw_keypoints(frame,annotation):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2
    bbox = annotation['bboxes'][0]
    keypoints = annotation['keypoints'][0]
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,0,255), 2)
 
    for i,(x,y,_) in enumerate(keypoints):
        cv2.circle(frame, (x,y), 5, (0,255,0), 3)
        cv2.putText(frame, str(i), (x,y), font, fontScale, (0,0,255), thickness, cv2.LINE_AA)

    return frame

folder_path = './custom_keypoint/20_keypoints/train'
images_folder = os.path.join(folder_path,'images')
annotations_folder = os.path.join(folder_path,'annotations')
images = glob.glob(images_folder+'/*.jpg')
annotations = glob.glob(annotations_folder+'/*.json')

for image, annotation in zip(images,annotations):
    frame = cv2.imread(image)
    with open(annotation) as json_file:
        annotation = json.load(json_file)
    frame = draw_keypoints(frame,annotation)
    cv2.imshow('annotations', frame)
    time.sleep(0.5)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()