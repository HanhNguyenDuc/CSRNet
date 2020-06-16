import numpy as np 
import cv2
import scipy.spatial
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter 
from typing import Optional
import glob
import os


class ImagePreprocessing:
    def __init__(
        self, 
        scale:Optinal(int)=0.8, 
        image_ext:Optional(str)='.jpg', 
        label_ext:Optional(str)='.txt'
    ) -> None:
        self.scale = scale
        self.image_ext = image_ext
        self.label_ext = label_ext

    def gaussian_filter_density(self, gt: np.ndarray) -> np.array:
        # print(gt.shape)
        density = np.zeros(gt.shape, dtype=np.float32)
        gt_count = np.count_nonzero(gt)
        if gt_count == 0:
            return(density)
        pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
        # print(np.nonzero(gt))
        leafsize = 2048
        # build kdtree
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(pts, k=4)
        for i, pt in enumerate(pts):
            pt2d = np.zeros(gt.shape, dtype=np.float32)
            pt2d[pt[1],pt[0]] = 1.
            if gt_count > 1:
                sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            else:
                sigma = np.average(np.array(gt.shape)) /2. /2. #case: 1 point
            density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        print('Amount: {}'.format(np.sum(density)))
        density *= 255 * 255
        return density

    def preprocess_image(self, image: np.ndarray, labels: list) -> np.ndarray:
        image_ground_truth = np.zeros(image.shape[:2])
        obj_coordinate = []
        for label in labels:
            x, y = label
            if x >= 1 or y >= 1:
                continue
            x_int, y_int = int(x * image.shape[1]), int(y * image.shape[0])
            image_ground_truth[y_int][x_int] = 1

        density = self.gaussian_filter_density(image_ground_truth)
        return density

    def read_image_and_label(self, test_img_path: str) -> List[Any]:
        # test_img_path = 'data/obj/DJI_0527_opencv_frame_0.jpg'
        test_label_path = test_img_path.replace('.jpg', '.txt')

        # Get image
        image = cv2.imread(test_img_path)
        image = cv2.resize(image, (int(image.shape[1] * self.scale), int(image.shape[0] * self.scale)))
        # image = cv2.resize(image, (64, 64))
        # Get labels
        labels = []
        with open(test_label_path, 'r') as file_label:
            lines = file_label.readlines()
            for line in lines:
                line.replace('\n', '')
                x, y = map(float, line.split(' '))
                labels.append([x, y])
        
        return image, labels
    
    def preprocess_image_from_dir(
        self, 
        dir_path: str, 
        des_dir: Optional(str)='train_data'
    ) -> None:
        list_image_path = glob.glob(dir_path + '/*' + self.image_ext)
        list_image_path.sort()
        src_image_folder = os.path.join(des_dir, 'inputs')
        des_image_folder = os.path.join(des_dir, 'outputs')
        for image_path in list_image_path:
            image_name = image_path.split('/')[-1].replace('.jpg', '.png')
            print('Image name: {}'.format(image_name))
            image, labels = self.read_image_and_label(image_path)
            shape = image.shape
            density = self.preprocess_image(image, labels)
            cv2.imwrite(os.path.join(src_image_folder, image_name), image[shape[0] // 2:, :, :])
            cv2.imwrite(os.path.join(des_image_folder, image_name), density[shape[0] // 2:, :])


if __name__ == '__main__':
    ip = ImagePreprocessing()
    ip.preprocess_image_from_dir('obj', des_dir='train_data')