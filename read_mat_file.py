import scipy.io as io
import cv2
import glob

label_path = 'ShanghaiTech_Crowd_Counting_Dataset/part_B_final/test_data/ground_truth'
mat_file_path_lib = glob.glob(label_path + '/*.mat')

for mat_file_path in mat_file_path_lib:
    print('processing {}'.format(mat_file_path))
    mat = io.loadmat(mat_file_path)
    gt = mat.get('image_info')[0, 0][0, 0][0]
    txt_file_path = mat_file_path.replace('.mat', '.txt').replace('GT_', '').replace('ground_truth', 'images')
    image_file_path = mat_file_path.replace('ground_truth', 'images').replace('.mat', '.jpg').replace('GT_', '')
    image = cv2.imread(image_file_path)
    with open(txt_file_path, 'w') as file_label:
        for line in gt:
            file_label.write(str(line[0] / image.shape[1]) + ' ' + str(line[1] / image.shape[0]) + '\n')

