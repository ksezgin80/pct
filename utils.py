import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw


def create_folders(folder_paths):
    '''
    Creates folders
    
    Parameters
    ----------
    folder_paths: 
        list of folder paths
    
    Output
    ------
    none
    '''
    for folder in folder_paths:
        if not os.path.isdir(folder):
            os.makedirs(folder)
    return None


def annotate_image(image, bounding_boxes, object_labeles, id_values, colorBGR=(0,255,0)):
    '''
    Place annotation labels on image
    
    Parameters
    ----------
    image: 
        image to be annotated
    
    bounding_boxes: 
        list of rectangle points [x1,y1,x2,y2]
    
    object_labels: 
        list of object class names, e.g. person, plant, bench...

    id_values: 
        list of object ids used for tracking
    
    colorBGR:
        color to use for annotations
    
    Output
    ------
    image: 
        annotated image
    '''
    for i, pid in enumerate(id_values):
        cv2.rectangle(image, 
                      (bounding_boxes[i][0],bounding_boxes[i][1]), 
                      (bounding_boxes[i][2],bounding_boxes[i][3]), 
                      colorBGR, 2)
        
        cv2.putText(image, object_labeles[i], 
                    (bounding_boxes[i][0]+5, bounding_boxes[i][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colorBGR, 2)
        
        cv2.putText(image, str(int(pid)), 
                    (bounding_boxes[i][0]+5, bounding_boxes[i][3]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colorBGR, 2)
    return image


def get_frames(input_file, save_dir, first_N=None, resize=1.0):
    '''
    Extracts the frames from an input video file
    and saves them as separate frames in an output directory
    
    Parameters
    ----------
    input_file: 
        input video file
    
    save_dir:
        output directory
    
    first_N: 
        read only the first N frames

    resize:
        scale frames by interpolating [0, 1]
    
    Output
    ------
    frame_count: 
        number of frames in video
    '''

    video = cv2.VideoCapture()
    video.open(input_file)

    if not video.isOpened():
        print("Failed to open input video")
        video.release()
        return

    if first_N:
        frame_count = first_N
    else:
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_idx = 0

    while frame_idx < frame_count:
        ret, frame = video.read()

        if not ret:
            print ("Failed to get the frame {frame_idx:d}")
            continue
        
        # resize frame (if required) and save
        newdim = (int(frame.shape[1]*resize), int(frame.shape[0]*resize))
        frame = cv2.resize(frame, newdim, cv2.INTER_AREA)
        out_name = os.path.join(save_dir, f'f{(frame_idx+1):06d}.jpg')
        ret = cv2.imwrite(out_name, frame)
        if not ret:
            print(f"Failed to write the frame {frame_idx:d}")
            continue

        frame_idx += 1
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    return int(frame_count)


def animate_frames(input_path, save_dir, fps=25.0):
    '''
    Extracts the frames from an input video file
    and saves them as separate frames in an output directory.
    
    Parameters
    ----------
    input_path: 
        input video file
    
    save_dir:
        output directory
    
    fps: 
        frames per second (default: 25)
    
    Output
    ------
    none
    '''

    dir_frames = input_path
    files_info = os.scandir(dir_frames)

    file_names = [f.path for f in files_info if f.name.endswith(".jpg")]
    file_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    frame_Height, frame_Width = cv2.imread(file_names[0]).shape[:2]
    resolution = (frame_Width, frame_Height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(save_dir, fourcc, fps, resolution)

    frame_count = len(file_names)
    frame_idx = 0

    while frame_idx < frame_count:

        frame_i = cv2.imread(file_names[frame_idx])
        video_writer.write(frame_i)
        frame_idx += 1

    video_writer.release()
    
    return None


def set_counter_line(img):
    '''
    Defines the counter line based on the line drawn by user
    
    Parameters
    ----------
    img: 
        image to draw the counter line on
    
    Output
    ------
    clicked: 
        coordinates of clicked points
    
    line_coef:
        coefficients of counter line equation (a x + b y + c)
    '''
    print("If it doesn't get you to the drawing mode, then rerun this function again.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('double') / 255.0 
    fig = plt.figure()
    fig.set_label('Draw line to set the counter')
    plt.axis('on')
    plt.imshow(img, cmap='gray')
    xs = []
    ys = []
    xf = []
    yf = []
    line_coef = []
    clicked = []

    def on_mouse_pressed(event):
        if len(xs) < 2:
            x = event.xdata
            y = event.ydata
            xs.append(x)
            ys.append(y)
            plt.plot(x, y, 'ro')
            
            # extend the input points to hit the image borders
            if len(xs) == 2:
                x1, x2 = xs
                y1, y2 = ys
                H, W, C = img.shape
                
                x0, xH = None, None
                y0, yW = None, None
                
                # line equation
                # aa*x + bb*y + c = 0
                aa = y1 - y2
                bb = x2 - x1
                cc = (x1 - x2) * y1 + (y2 - y1) * x1
                line_coef = [aa, bb, cc]
                
                # x=0 line
                # x=W line
                y0 = (-cc - aa * 0) / bb
                yW = (-cc - aa * W) / bb
                
                if (y0 < 0) or (y0 >= H):
                    y0 = None
                if (yW < 0) or (yW >= H):
                    yW = None       
                
                # y=0 line
                # y=H line
                x0 = (-cc - bb * 0) / aa
                xH = (-cc - bb * H) / aa

                if (x0 < 0) or (x0 >= W):
                    x0 = None
                if (xH < 0) or (xH >= W):
                    xH = None 

                if not x0 == None:
                    xf.append(x0)
                    yf.append(0)
                if not xH == None:
                    xf.append(xH)
                    yf.append(H-1)
                if not y0 == None:
                    xf.append(0)
                    yf.append(y0)
                if not yW == None:
                    xf.append(W-1)
                    yf.append(yW)

                plt.plot(xf, yf, '-', color=(1,1,0))
                plt.text(sum(xf)/len(xf), sum(yf)/len(yf), "Counter", size=6, 
                         ha="center", va="center",
                         bbox=dict(boxstyle="round", ec=(1, 1, 0), fc=(1, 1, 0)))
    
    # Create an hard reference to the callback not to be cleared by the garbage collector
    fig.canvas.mpl_connect('button_press_event', on_mouse_pressed)
    
    clicked.append(xf)
    clicked.append(yf)
    return clicked, line_coef


def find_region(bounding_box, counter_line_coords):
    '''
    Finds on which side of the counter line the object centroid is
    
    Parameters
    ----------
    bounding_box: 
        coordinates of the bounding box rectangle [top, left, bottom, right]
    
    counter_line_coords:
        coordinates of two points defining the counter line [[x1, x2], [y1, y2]]
    
    Output
    ------
    region: 
        which side of the counter line the object centroid is
    '''

    # get coordinates of counter line
    x1, x2 = counter_line_coords[0]
    y1, y2 = counter_line_coords[1]
    
    # calculate centroid of bounding box
    xc = (bounding_box[0] + bounding_box[2]) / 2
    yc = (bounding_box[1] + bounding_box[3]) / 2

    # decide on what side the object is
    v1 = (x2-x1, y2-y1)                 # vector 1
    v2 = (x2-xc, y2-yc)                 # vector 2
    xp = v1[0] * v2[1] - v1[1] * v2[0]  # cross product

    if xp > 0:
        # region A
        region = 1  
    else:
        # region B
        region = 2     
    
    return int(region)


def annotate_counting(image, counted_objects, counter_line_coords, 
                      counter_line_coefs, colorBGR=(0,255,0)):
    '''
    Place annotation labels for counting the moving objects
    
    Parameters
    ----------
    image: 
        image to be annotated
    
    counted_objects: 
        [frame id, object id, rectangle points [x1,y1,x2,y2], regions, countA, countB]
    
    counter_line_coords:
        coordinates of two points defining the counter line [[x1, x2], [y1, y2]]

    counter_line_coefs:
        coefficients [a,b,c] defining the counter line (ax + by + c = 0)

    colorBGR:
        color to use for annotations

    Output
    ------
    image: 
        annotated image
    '''

    # draw counter line
    counter_color = (0,255,255)
    cv2.line(image, 
            (int(counter_line_coords[0][0]), int(counter_line_coords[1][0])), 
            (int(counter_line_coords[0][1]), int(counter_line_coords[1][1])), 
            counter_color, 3) 

    # draw counter label
    xc = int((counter_line_coords[0][0] + counter_line_coords[0][1]) / 2)
    yc = int((counter_line_coords[1][0] + counter_line_coords[1][1]) / 2)
    
    label_size = cv2.getTextSize('Counter', cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    w = int(label_size[0][0])
    h = int(label_size[0][1])
    xl = xc - int(w / 2)
    yl = yc + int(h / 2)
    
    cv2.rectangle(image, (xl-10, yl-h-5), (xl+w+10, yl+5), counter_color, -1) 
    cv2.putText(image, 'Counter', (xl, yl), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    # draw region labels on four sides
    H, W, C = image.shape

    label_size = cv2.getTextSize('B', cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    w = int(label_size[0][0])
    h = int(label_size[0][1])
    
    topleft = 'A' if find_region([0, 0, 10, 10], counter_line_coords) == 1 else 'B'
    bottomleft = 'A' if find_region([0, H-10, 10, H-1], counter_line_coords) == 1 else 'B'
    topright = 'A' if find_region([W-10, 0, W-1, 10], counter_line_coords) == 1 else 'B'
    bottomright = 'A' if find_region([W-10, H-10, W-1, H-1], counter_line_coords) == 1 else 'B'

    cv2.putText(image, str(topleft), (0, 0+h), cv2.FONT_HERSHEY_SIMPLEX, 2, counter_color, 3)
    cv2.putText(image, str(bottomleft), (0, H-1), cv2.FONT_HERSHEY_SIMPLEX, 2, counter_color, 3)
    cv2.putText(image, str(topright), (W-1-w, 0+h), cv2.FONT_HERSHEY_SIMPLEX, 2, counter_color, 3)
    cv2.putText(image, str(bottomright), (W-1-w, H-1), cv2.FONT_HERSHEY_SIMPLEX, 2, counter_color, 3)

    # draw counting related info
    for i, obj in enumerate(counted_objects):

        bbox = obj[2:6]
        region = 'A' if obj[6] == 1 else 'B'
        countA = obj[7]
        countB = obj[8]

        cv2.rectangle(image, (bbox[0],bbox[1]), (bbox[2],bbox[3]), colorBGR, 2)
        
        cv2.putText(image, region, (bbox[0]+10, bbox[1]+30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorBGR, 3)
    
    label_size = cv2.getTextSize('From A to B', cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    w = int(label_size[0][0])
    h = int(label_size[0][1])
    
    cv2.putText(image, 'From A to B: ' + str(countB), (int(W/2-w/2), H-60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, counter_color, 3)
    cv2.putText(image, 'From B to A: ' + str(countA), (int(W/2-w/2), H-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, counter_color, 3)

    return image