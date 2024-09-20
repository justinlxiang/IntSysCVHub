"""
End-to-end Training Script.

This script is similar to the training script in detectron2/tools.

It is an example of how a user might use detectron2 for a new project.
"""

import os

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, CustomEvaluator, DatasetEvaluators, verify_results
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.logger import setup_logger
import torch

from detectron_data import get_detectron_dicts
from dataset_mapper import DatasetMapper

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        coco_eval = CustomEvaluator(dataset_name, cfg, False, output_folder)
        custom_eval = COCOEvaluator(dataset_name, cfg, False, output_folder)
        evaluators = [custom_eval, coco_eval]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    DatasetCatalog.register("full_image_train", get_detectron_dicts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="full_image")
    return cfg


def main(args):
    cfg = setup(args)
    if not torch.cuda.is_available():
        cfg.merge_from_list(['MODEL.DEVICE', 'cpu'])

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
