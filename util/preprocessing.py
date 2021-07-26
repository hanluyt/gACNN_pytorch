import shutil
import numpy as np
import imutils
import dlib
import cv2

import os
import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain

def transform_24(pts_68):
    point = [18, 21, 22, 25, 38, 36, 43, 45, 27, 29, 48, 50, 52, 54, 58, 56]
    indx_6 = np.array([[23, 25], [42, 45], [18, 20], [36, 39], [17, 58], [26, 56]])
    po_8 = np.zeros((8, 2), dtype="int")
    for i, indx in enumerate(indx_6):
        po_8[i] = [(pts_68[indx[0], 0] + pts_68[indx[1], 0]) / 2, (pts_68[indx[0], 1] + pts_68[indx[1], 1]) / 2]
    po_8[6] = [pts_68[48, 0] - 16, pts_68[48, 1] - 16]
    po_8[7] = [pts_68[54, 0] + 16, pts_68[54, 1] - 16]
    pts_24 = np.concatenate((pts_68[point, :], po_8))
    return pts_24


def landmark_to_24(root_path, shape_predictor):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    result = []
    label = 0

    for cls_name in os.listdir(root_path):
        for im_name in os.listdir(os.path.join(root_path, cls_name)):
            image = cv2.imread(os.path.join(root_path, cls_name, im_name))
            image = imutils.resize(image, width=224)
            rect = detector(image, 1)
            if len(rect) < 1:
                continue

            shape = predictor(image, rect[0])
            po_68 = np.asarray([[p.x, p.y] for p in shape.parts()])
            po_24 = transform_24(po_68)
            sample = po_24.flatten().tolist()
            sample.insert(0, im_name)
            sample.insert(1, label)
            result.append(sample)

        label += 1
    column = [[f'point_{i}_x', f'point_{i}_y'] for i in range(1, 25)]
    column = list(chain(*column))
    column.insert(0, 'img_name')
    column.insert(1, 'label')
    df_result = pd.DataFrame(result, columns=column)
    return df_result


df_result = landmark_to_24('../RAF-DB_test_label', shape_predictor='shape_predictor_68_face_landmarks.dat')
df_result.to_csv('RAF-DB_test_24.csv', index=False)

def extract_test(old_file, new_file):
    with open(old_file) as fp:
        test_res = []
        for line in fp:
            res = line.strip(' ').strip('\n').strip('\r')
            if res[0:4] == 'test':
                test_res.append(res)

    total = len(test_res)
    write_res = []
    for line in test_res:
        tmp = line.strip(' ').split(' ')
        write_res.append(tmp)

    with open(new_file, 'w') as fp:
        for idx in range(0, total):
            sample = write_res[idx]
            for ele in sample:
                fp.write(ele)
                fp.write(' ')
            fp.write('\n')

def split_file(original, destination, annotation):
    map = {'1': 'surprise', '2': 'fear', '3': 'disgust', '4': 'happy', '5': 'sad', '6': 'anger', '7': 'neutral'}
    with open(annotation) as fp:
        test_res = []
        for line in fp:
            test_res.append(line.strip(' ').strip('\n').strip('\r'))
    write_res = []
    for line in test_res:
        tmp = line.strip(' ').split(' ')
        write_res.append(tmp)

    for im_name, label in zip(os.listdir(original), write_res):
        if im_name == label[0]:
            oldname = original + "\\" + im_name
            newname = destination + '\\' + map[label[1]] + '\\' + im_name
            shutil.copyfile(oldname, newname)

