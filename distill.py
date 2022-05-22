import os
import datetime

from datasets.kitti import KITTIDataset
from config import cfg
from engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from utils.backup_files import sync_root

import tensorflow as tf
from distiller.distiller import create_student, Distiller


def setup(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = args.num_work
    cfg.TEST.EVAL_DIS_IOUS = args.eval_iou
    cfg.TEST.EVAL_DEPTH = args.eval_depth 
    
    if args.vis_thre > 0:
        cfg.TEST.VISUALIZE_THRESHOLD = args.vis_thre 
    
    if args.output is not None:
        cfg.OUTPUT_DIR = args.output

    if args.test:
        cfg.DATASETS.TEST_SPLIT = 'test'
        cfg.DATASETS.TEST = ("kitti_test",)

    cfg.START_TIME = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d %H:%M:%S')
    default_setup(cfg, args)

    return cfg

def main(args):
    cfg = setup(args)

    # Create x_train and y_train
    dataloader = KITTIDataset(cfg, "/home/oj10529w/Documents/DLCV/Project/Distill/training/", True, None, True)
    """ for i in range (len(dataloader)):
        img, target, original_idx = dataloader.__getitem__(i)
        calib = target.get_field("calib")

        # Camera intrinsics and extrinsics
        c_u = calib.c_u
        c_v = calib.c_v
        f_u = calib.f_u
        f_v = calib.f_v
        b_x = calib.b_x   # relative
        b_y = calib.b_y
        if(i%500 == 0): print(i)
    """

    # Creating teacher and student model
    teacher = create_student(cfg, 10)   # Change this to Monoflex pytorch Model

    student = create_student(cfg, 10)   # 
    student.summary()
    
    # Initialize and compile distiller
    distiller = Distiller(student=student, teacher=teacher)

    distiller.compile(
        optimizer = tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn= tf.keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )

    # Distill teacher to student
    # distiller.train(X_train, Y_train)

    # Evaluate student on test dataset
    # distiller.test(X_test, Y_test)

    print(len(dataloader))


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    
    print("Command Line Args:", args)

    # backup all python files when training
    if not args.eval_only and args.output is not None:
        sync_root('.', os.path.join(args.output, 'backup'))
        import shutil
        shutil.copy2(args.config_file, os.path.join(args.output, 'backup', os.path.basename(args.config_file)))

        print("Finish backup all files")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )