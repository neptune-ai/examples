import json
import os
import random

import cv2
import detectron2
import neptune

# import some common libraries
import numpy as np
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from neptune.types import File
from neptune_detectron2 import NeptuneHook

setup_logger()

# (Neptune) Initialize a new run
run = neptune.init_run(
    api_token=neptune.ANONYMOUS_API_TOKEN,
    project="common/detectron2-integration",
)

# ### Transform dataset for training


def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("balloon_train")


# Configure the model for training
device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = get_cfg()
cfg.MODEL.DEVICE = device
cfg.DATALOADER.NUM_WORKERS = 0
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.DATASETS.TRAIN = ("balloon_train",)
cfg.DATASETS.TEST = ()
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 30
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
# only has one class (balloon)
# see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1


# (Neptune) Create a hook using Neptune integration
# NOTE: You can also log checkpoints by passing `log_checkpoints=True`.
hook = NeptuneHook(run=run, log_model=True, metrics_update_freq=10)


# (Neptune) Train the model with hook
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.register_hooks([hook])
trainer.train()


# Prepare model for prediction
cfg.MODEL.WEIGHTS = os.path.join(
    cfg.OUTPUT_DIR, "model_final.pth"
)  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
predictor = DefaultPredictor(cfg)


# (Neptune) Log model predictions to neptune
dataset_dicts = get_balloon_dicts("balloon/val")
for idx, d in enumerate(random.sample(dataset_dicts, 3)):
    im = cv2.imread(d["file_name"])
    outputs = predictor(
        im
    )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(
        im[:, :, ::-1],
        metadata=balloon_metadata,
        scale=0.5,
        instance_mode=ColorMode.IMAGE,  # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image = out.get_image()[:, :, ::-1]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    run[f"training/prediction_visualization/{idx}"].upload(File.as_image(img_rgb / 255.0))

# (Neptune) Once you are done logging, stop tracking the run.
run.stop()
