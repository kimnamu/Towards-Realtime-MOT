import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import random

np.random.seed(1)
##### covid19
covid19_threshld = 40
covid19_new_time = 50
covid19_rate = 0.2

covid19 = np.array([])
for i in range(50):
    elem = np.arange(20)
    np.random.shuffle(elem)
    covid19 = np.append(covid19, elem)

infection = 0
for i, c in enumerate(covid19):
    if c+0.9 < 20 * covid19_rate: 
        covid19[i] = covid19_new_time + 1
    else:
        covid19[i] = 0

###### mask
mask_shield = np.zeros(1000)
mask = np.array([])
mask_rate = 0.8
for i in range(50):
    elem = np.arange(20)
    np.random.shuffle(elem)
    mask = np.append(mask, elem)


for i, c in enumerate(mask):
    if c + 0.9 < 20 * mask_rate: 
        mask[i] = 1
    else:
        mask[i] = 0


def tlwhs_to_tlbrs(tlwhs):
    tlbrs = np.copy(tlwhs)
    if len(tlbrs) == 0:
        return tlbrs
    tlbrs[:, 2] += tlwhs[:, 0]
    tlbrs[:, 3] += tlwhs[:, 1]
    return tlbrs


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def resize_image(image, max_size=800):
    if max(image.shape[:2]) > max_size:
        scale = float(max_size) / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return image


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 500.))

    radius = max(5, int(im_w/140.))
    # cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
    #             (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)
       
    covid19_map = {}
    for i, c in enumerate(covid19):
        if c == 0:
            continue
        else:
            covid19[i] = covid19[i] + 1
    
    for i, m in enumerate(mask_shield):
        if m == 0:
            continue
        else:
            mask_shield[i] = mask_shield[i] + 1
    
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        if covid19[int(obj_id)] != 0:
            covid19_map[obj_id] = tuple(map(int, (x1, y1, x1 + w, y1 + h)))

    covid19_news = np.empty((int(0)))

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        
        if covid19[int(obj_id)]!=0:
            continue

        for covid19_key in covid19_map:
            covid19_box = covid19_map[covid19_key]
            covid19_x, covid19_y = (covid19_box[0]+covid19_box[2])//2, (covid19_box[1]+covid19_box[3])//2
            new_x, new_y = (intbox[0]+intbox[2])//2, (intbox[1]+intbox[3])//2
            if (covid19_x-new_x)**2+(covid19_y-new_y)**2 < covid19_threshld**2:
                if mask[covid19_key] == 1 or mask[int(obj_id)] == 1:
                    mask_shield[int(obj_id)] = 1
                    continue
                covid19_news = np.append(covid19_news, int(obj_id))
    
    global infection
    for covid19_new in covid19_news:
        mask_shield[int(covid19_new)] = 0
        covid19[int(covid19_new)] = 1
        infection = infection + 1

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        
        # cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        # cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
        #             thickness=text_thickness)

        color_green = (0, 255, 0)
        color_red = (0, 0, 255)

        
        if mask_shield[int(obj_id)] != 0:
            color_green2 = (0, min(255, 100+mask_shield[int(obj_id)]), 0)
            line_score = int(max(0, (60-mask_shield[int(obj_id)])//20))
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color_green2, thickness=line_thickness+line_score)
        elif covid19[int(obj_id)]==0:
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color_green, thickness=line_thickness)
        elif covid19[int(obj_id)] <= covid19_new_time:
            line_score = int((covid19_new_time-covid19[int(obj_id)])//10)
            color_score = int(covid19[int(obj_id)] * (128/covid19_new_time))
            color_purple = (max(0, 128-color_score), 0, min(255, 128+color_score))
            # cv2.rectangle(im, intbox[0:2], intbox[2:4], color=(128, 0, 128), thickness=line_thickness + line_score)
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color_purple, thickness=line_thickness+line_score)
        else:
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color_red, thickness=line_thickness)

        img_pil = Image.fromarray(im)
        d = ImageDraw.Draw(img_pil)
        
        if mask[int(obj_id)] == 1:
            font_size = 20
            font_color = (255, 255, 255)
            font_bg_color = (0, 100, 0)

            box_mask = [(intbox[0], intbox[1]-font_size), (intbox[0]+54, intbox[1])]
            font = ImageFont.truetype('./utils/fonts/FreeMonoBold.ttf', font_size)
            d.rectangle(box_mask, fill =font_bg_color, outline ="black") 
            d.text((box_mask[0][0]+4,box_mask[0][1]+2), "Mask", fill=font_color, font=font)

            # cv2.putText(im, "Mask", (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 200, 0),
            #             thickness=text_thickness*3)
            # cv2.putText(im, "Mask", (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 0),
            #             thickness=text_thickness)
        else:
            font_size = 20
            font_color = (255, 255, 255)
            font_bg_color = (0, 0, 140)

            box_mask = [(intbox[0], intbox[1]-font_size), (intbox[0]+90, intbox[1])]
            font = ImageFont.truetype('./utils/fonts/FreeMonoBold.ttf', font_size)
            d.rectangle(box_mask, fill =font_bg_color, outline ="black") 
            d.text((box_mask[0][0]+4,box_mask[0][1]+2), "No mask", fill=font_color, font=font)

            # cv2.putText(im, "No mask", (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 200),
            #             thickness=text_thickness*3)
            # cv2.putText(im, "No mask", (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 0),
            #             thickness=text_thickness)
        im = np.array(img_pil)


    overlay = im.copy()
    cv2.rectangle(overlay, (0, 0), (im.shape[1], 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, im, 1 - 0.4, 0, im)
    img_pil = Image.fromarray(im)
    d = ImageDraw.Draw(img_pil)

    font_size = 30
    font_color = (255, 255, 255)
    # font_bg_color = (255, 255, 255, 100)

    box_mask = [(0, 0), (2000, 40)]
    font = ImageFont.truetype('./utils/fonts/FreeMonoBold.ttf', font_size)
    # d.rectangle(box_mask, fill =font_bg_color, ) 
    d.text((box_mask[0][0]+4,box_mask[0][1]+2), '# Potential infection : {} people'.format(infection), fill=font_color, font=font)
    im = np.array(img_pil)

    # cv2.putText(im, '# Potential infection : {} people'.format(infection),
    #             (10, int(30 * text_scale)), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), thickness=4)
    # cv2.putText(im, '# Potential infection : {} people'.format(infection),
    #             (10, int(30 * text_scale)), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), thickness=1)
    return im


def plot_trajectory(image, tlwhs, track_ids):
    image = image.copy()
    for one_tlwhs, track_id in zip(tlwhs, track_ids):
        color = get_color(int(track_id))
        for tlwh in one_tlwhs:
            x1, y1, w, h = tuple(map(int, tlwh))
            cv2.circle(image, (int(x1 + 0.5 * w), int(y1 + h)), 2, color, thickness=2)

    return image


def plot_detections(image, tlbrs, scores=None, color=(255, 0, 0), ids=None):
    im = np.copy(image)
    text_scale = max(1, image.shape[1] / 800.)
    thickness = 2 if text_scale > 1.3 else 1
    for i, det in enumerate(tlbrs):
        x1, y1, x2, y2 = np.asarray(det[:4], dtype=np.int)
        if len(det) >= 7:
            label = 'det' if det[5] > 0 else 'trk'
            if ids is not None:
                text = '{}# {:.2f}: {:d}'.format(label, det[6], ids[i])
                cv2.putText(im, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                            thickness=thickness)
            else:
                text = '{}# {:.2f}'.format(label, det[6])

        if scores is not None:
            text = '{:.2f}'.format(scores[i])
            cv2.putText(im, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                        thickness=thickness)

        # cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)

    return im
