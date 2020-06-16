import cv2
import numpy as np 
import os
import glob
from typing import Optional, Any, Dict, List, Text, Tuple, Union, Type
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, help='Path to data directory', required=True)
parser.add_argument('--imagerate', type=float, help='Scale of Image show and real Image', default=0.8)
parser.add_argument('--imageext', type=str, help='Image extension (".jpg" or something)', default='.jpg')

args = parser.parse_args()

class CrowdCountingLabelTool:
    def __init__(
        self,
        images_path: str,
        image_ext: Optional[str] = '.jpg',
        label_ext: Optional[str] = '.txt',
        image_show_rate: Optional[float] = 0.5,
        label_type: Optional[List[str]] = ['label', 'x_rate', 'y_rate', 'w_rate', 'h_rate']
    ) -> None:
        self.images_path = images_path
        self.current_image = None
        self.current_mouseX = None
        self.current_mouseY = None
        self.image_ext = image_ext
        self.label_ext = label_ext
        self.image_show_rate = image_show_rate
        self.label_type = label_type
        self.list_label = []

    def draw_circle(
        self, 
        event: Any, 
        x: int, 
        y: int, 
        flags: Any, 
        param: Any
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            x = int(x / self.image_show_rate)
            y = int(y / self.image_show_rate)
            self.current_image = cv2.rectangle(self.current_image, (x, y), (x + 2, y + 2), (0, 0, 255), 5)
            self.list_label.append(
                [
                    x / self.current_image.shape[1], 
                    y / self.current_image.shape[0]
                ]
            )
            self.current_mouseX, self.current_mouseY = x, y

    def label_data(self) -> None:
        list_file_image = glob.glob(os.path.join(self.images_path, '*' + self.image_ext))
        list_file_image.sort()
        if os.path.exists('cache_{}.txt'.format(self.images_path)):
            with open('cache_{}.txt'.format(self.images_path)) as cache_file:
                used_file = cache_file.readlines()
                used_file = map(lambda file_name: file_name.replace('\n', ''), used_file)
                counter = len(used_file)
        counter = 0
        for file_image_name in list_file_image:
            counter += 1
            print('Processing: {}'.format(file_image_name))
            print('Number of image processed: {}'.format(counter))
            self.list_label = []
            file_label_name = file_image_name.replace(self.image_ext, self.label_ext)
            self.current_image = cv2.imread(file_image_name)
            shape = self.current_image.shape
            if os.path.exists(file_label_name):
                with open(file_label_name) as file_label:
                    lines = file_label.readlines()
                    obj_coordinate = list()
                    for line in lines:
                        line = line.replace('\n', '')
                        prm = line.split(' ')
                        if len(prm) == 2:
                            x, y = prm
                        else:
                            c, x, y, w, h = prm
                        self.list_label.append([x, y])
                        obj_coordinate.append([int(float(x) * shape[1]), int(float(y) * shape[0])])
                for obj in obj_coordinate:
                    self.current_image = cv2.rectangle(self.current_image, tuple(obj), (obj[0] + 2, obj[1] + 2), (0, 0, 255), 5)

            cv2.namedWindow('image')
            cv2.setMouseCallback('image', self.draw_circle)

            while(1):
                cv2.imshow('image', cv2.resize(
                    self.current_image, 
                    (int(shape[1] * self.image_show_rate), int(shape[0] * self.image_show_rate))
                ))
                k = cv2.waitKey(20) & 0xFF
                if k == 27:
                    break
                elif k == 13:
                    with open(file_label_name, 'w') as file_label:
                        for label in self.list_label:
                            file_label.write(str(label[0]) + ' ' + str(label[1]) + '\n')
                        file_label.close()
                    break
            

if __name__ == '__main__':
    cclt = CrowdCountingLabelTool(args.dir, image_show_rate=args.imagerate, image_ext=args.imageext)
    cclt.label_data()
