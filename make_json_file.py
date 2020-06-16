import json
import glob

def make_json():
    img_list = glob.glob('train_data/inputs/*')
    with open('train.json', 'w') as f:
        json.dump(img_list, f)

make_json()