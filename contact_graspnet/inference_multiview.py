import os
import sys
import argparse
import numpy as np
import time
import glob
import cv2

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# else:
#   print("No GPU device found")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

def inference(global_config, checkpoint_dir, input_paths, K=None, local_regions=True, skip_border_objects=False, filter_grasps=True, segmap_id=None, z_range=[0.2,1.8], forward_passes=1):
    """
    Predict 6-DoF grasp distribution for given model and input data
    
    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments. 
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """
    
    # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')
    
    # os.makedirs('results', exist_ok=True)

    # # get npy from np.save('./multiview_data/pcd_full_combine_augmented.npy', new_full_depths) and np.save('./multiview_data/pcd_combine_augmented.npy', new_depths)
    # pc_full = np.load('./multiview_data/pc_full.npy')
    # print(pc_full.shape)
    # pc_full_dict = {}
    # pc_full_dict[1] = pc_full

    # pc_segments = np.load('./multiview_data/pc_segments.npy')
    # print(pc_segments.shape)
    # # create a pc_segments dict 
    # pc_segments_dict = {}
    # # insert the pc_segments into the dict
    # pc_segments_dict[1] = pc_segments
    # print(pc_segments_dict[1].shape)

    # load the  pkl file

    import pickle
    
    pc_full = np.load('./multiview_data/pc_full_combine.npy', allow_pickle=True)
    pc_full_colors = np.load('./multiview_data/pc_full_combine_colors.npy', allow_pickle=True)
    pc_segments = np.load('./multiview_data/pc_segments_combine.npy', allow_pickle=True)
    pc_segments_dict = {}
    # # insert the pc_segments into the dict
    pc_segments_dict[1] = pc_segments
    print("dddddddd", pc_segments_dict)
    print(pc_segments_dict[1].shape)

    #Henry 要將座標移回相機座標系!！!！!！!！
    print('Generating Grasps...')
    pred_grasps_cam, scores, contact_pts, gripper_openings = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments_dict,
                                                                                        local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)  
    # print(pred_grasps_cam)
    # print(scores)
    # Save results
    # save predictions
    print("keys", pred_grasps_cam.keys())
    save_key = 1
    np.save('./results/pred_grasps_cam.npy', pred_grasps_cam[save_key])
    np.save('./results/scores.npy', scores[save_key])
    np.save('./results/contact_pts.npy', contact_pts[save_key])
    np.save('./results/gripper_openings.npy', gripper_openings[save_key])
    np.savez('./results/predictions_multiview_data', 
                pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts, gripper_openings=gripper_openings)
    # Visualize results          
    print(len(pred_grasps_cam))
    print(pred_grasps_cam[save_key].shape)
    print("1111111111", type(pred_grasps_cam))
    print(type(scores))
    visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_full_colors*255)

    if not glob.glob(input_paths):
        print('No files found: ', input_paths)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--np_path', default='test_data/7.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    
    print(str(global_config))
    print('pid: %s'%(str(os.getpid())))

    inference(global_config, FLAGS.ckpt_dir, FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path, z_range=eval(str(FLAGS.z_range)),
                K=FLAGS.K, local_regions=FLAGS.local_regions, filter_grasps=FLAGS.filter_grasps, segmap_id=FLAGS.segmap_id, 
                forward_passes=FLAGS.forward_passes, skip_border_objects=FLAGS.skip_border_objects)

    print(FLAGS.local_regions, filter_grasps=FLAGS.filter_grasps)