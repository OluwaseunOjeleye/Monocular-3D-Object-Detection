import os
import datetime
import torch
from PIL import Image
import numpy as np

from datasets.kitti import KITTIDataset
from datasets.kitti_utils import read_label
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

def load_images(dataloader, trainingset_path, i_h, i_w, i_c):
    m = 2 # len(dataloader)
    # Define X and Y as number of images along with shape of one image
    X = np.zeros((m, i_h, i_w, i_c), dtype=np.float32)
    Y = []
    Ground_truth = []

    # Resize images and masks
    for i in range(m):
        file = dataloader.image_files[i][0:6]
        
        # convert image into an array of desired shape (3 channels)
        path = os.path.join(trainingset_path + "image_2/", file + '.png')
        single_img = Image.open(path).convert('RGB')
        single_img = single_img.resize((i_h, i_w))
        single_img = np.reshape(single_img,(i_h,i_w,i_c)) 
        single_img = single_img/256.
        X[i] = single_img

        path = os.path.join(trainingset_path + "label_2/", file + '.txt')
        Ground_truth.append(read_label(path))

        # Load Teacher's predictions
        # Y.append(torch.load(dataloader.image_files[i][0:6] + '_raw_cls.txt'))
        
    return X, Y, Ground_truth

def main(args):
    cfg = setup(args)

    # Create x_train and y_train
    dataloader = KITTIDataset(cfg, "/home/oj10529w/Documents/DLCV/Project/Distill/training/", True, None, True)
    X_train, Y_train, Ground_truth = load_images(dataloader, "/home/oj10529w/Documents/DLCV/Project/Distill/training/", cfg.INPUT.HEIGHT_TRAIN, cfg.INPUT.WIDTH_TRAIN, 3)

    print(Ground_truth[0][0].print_object()+ "......")

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
    cls_files = []

    #for img_name in dataloader.image_files:
    #    cls = torch.load(img_name[0:6] + '_raw_cls.txt')
    #    cls_files.append(cls)

    """ 
    #print(cls_files)
    teacher = create_student(cfg, 3)   # Change this to Monoflex pytorch Model

    student = create_student(cfg, 3)   # 
    student.summary()
    print(X_train.shape)

    student.compile(optimizer=tf.keras.optimizers.Adam(), 
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

              
    # Run the model in a mini-batch fashion and compute the progress for each epoch
    # results = student.fit(X_train, batch_size=32, epochs=20)
    results = student(X_train, training=True)
    print(results.shape)

    # Initialize and compile distiller
    distiller = Distiller(cfg, student=student, teacher=teacher)

    distiller.compile(
        optimizer = tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn= tf.keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    ) """

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