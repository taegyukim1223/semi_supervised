from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
pylab.rcParams['figure.figsize'] = (8.0, 10.0) # Import Libraries
import os
import sys
import seaborn as sns
from matplotlib import colors
from tensorboard.backend.event_processing import event_accumulator as ea
from PIL import Image

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.10")
torch.cuda.empty_cache()
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import skimage.io as io

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from detectron2.data.datasets import register_coco_instances
# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import skimage.io as io
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from tqdm import tqdm
import seaborn as sns
from matplotlib import colors
from tensorboard.backend.event_processing import event_accumulator as ea
from PIL import Image
import glob

os.chdir('/home/mts/taegyu/semi_thyroid/123/supervised_test')
 # 원하는 GPU 번호 입력
GPU_NUM = 4
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print ('Current cuda device ', torch.cuda.current_device()) # check

test_dir = "./"

register_coco_instances("test_anno1", {}, os.path.join(test_dir,"thyroid-supervised-test.json"), os.path.join(test_dir))
val_dataset_dicts = DatasetCatalog.get("test_anno1")
val_metadata_dicts = MetadataCatalog.get("test_anno1")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "./model_final copy 3.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.DATASETS.TEST = ("test_anno1", )


predictor = DefaultPredictor(cfg)

from skimage.io import imread
from skimage import data 
from skimage.measure import find_contours
import numpy as np
from skimage import io
import glob
targetPattern = r"./*.jpg"
file_list = glob.glob(targetPattern)
print('image_num :',len(file_list))

with open('/home/mts/taegyu/semi_thyroid/123/supervised_test/thyroid-supervised-train.json', 'r+', encoding="UTF-8") as f :
	json_data = json.load(f)
 
	for d in tqdm(random.sample(val_dataset_dicts, len(file_list))):   
		im = io.imread(d["file_name"])
		outputs = predictor(im)
		image_id = d["image_id"]
		h = d["height"]
		w = d["width"]
		f_n = d["file_name"].split('/')
		f_n = f_n[-1]
		num = json_data['images'][-1]['id']
		num += 1
		json_image = {
			"id": num,
			"width": w,
			"height": h,
			"file_name": f_n,
			"license": 0,
			"flickr_url": None,
			"coco_url": None,
			"date_captured": None
		}
		json_data["images"].append(json_image)
  	
		n = outputs["instances"] 
		n = len(n._fields['scores'])
		for i1 in range(n) :
			i2 = outputs["instances"][i1]

			pred_masks = i2._fields['pred_masks'].squeeze().cpu().numpy()
			contours_list = find_contours(pred_masks)
			len1 = len(contours_list[0])
			coordinate_item = []
			for idx in range(len1):
				coordinate_item.append(contours_list[0][idx][0].item())
				coordinate_item.append(contours_list[0][idx][1].item())
				# coordinate_item.append(contours_list[i1][0].item())
			pred_boxes = i2._fields['pred_boxes'].tensor.squeeze()
			box_x = pred_boxes[0].item()
			box_y = pred_boxes[1].item()
			box_w = pred_boxes[2].item()
			box_h = pred_boxes[3].item()  
			pred_classes = i2._fields['pred_classes'][0].item()
			num_ann = json_data['annotations'][-1]['id']
			num_ann += 1
			json_ann = {
				"id": num_ann,
				"image_id": num,
				"category_id": pred_classes + 1,
				"segmentation": [
					coordinate_item 
				],
				"area": box_w*box_h,
				"bbox": [
					box_x,
					box_y,
					box_w,
					box_h
				],
				"iscrowd": False
			}
			json_data["annotations"].append(json_ann)
	f.seek(0)
	json.dump(json_data, f)

    