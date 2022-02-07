import os
import cv2
import open3d as o3d
import pyrealsense2 as rs
import numpy as np
import threading
import socket
import struct
import keyboard
# import pytransform3d.rotations
# import matplotlib_inline

from RS_and_3D_OD_common_functions import *

PROJECT_FOLDER_ABS_PATH = os.path.dirname(os.path.abspath("__file__")).replace(os.sep, "/")



class ThreadWithReturnValue(threading.Thread):

    def __init__(self, target=None, args=[]):
        threading.Thread.__init__(self, target=target, args=args, daemon=True)
        self.return_value = None
    
    def run(self):
        if self._target is not None:
            self.return_value = self._target(*self._args)
    
    def join(self, *args):
        threading.Thread.join(self, *args)



class CommonThreadWithReturnValue():

    def __init__(self):
        pass
    
    def start_thread(self, threading_function_handle, variables_list):
        self.threading_updater = ThreadWithReturnValue(target=threading_function_handle, args=variables_list)
        self.threading_updater.start()

    def end_thread_if_alive_and_get_return_value(self):
        if self.threading_updater.is_alive():
            self.threading_updater.join()
        
        return self.threading_updater.return_value



class RealSenseStreaming():
    
    def __init__(self, constants, use_already_processed_data):
        self.const = constants

        self.exit_loop_flag = False
        self.continue_loop_flag = False

        # configure depth and color streams
        self.streams_pipeline = rs.pipeline()
        self.streams_configuration = rs.config()

        # start streaming
        try:
            if use_already_processed_data == True:
                self.streams_configuration.enable_device_from_file(self.const.WARMUP_VIDEO_FILE_ABS_PATH, repeat_playback=True)
            else:
                self.streams_configuration.enable_stream(rs.stream.depth, self.const.CAM_FRAME_SIZE[0], self.const.CAM_FRAME_SIZE[1], rs.format.z16, \
                    self.const.NUM_OF_STREAMING_FPS)
                self.streams_configuration.enable_stream(rs.stream.color, self.const.CAM_FRAME_SIZE[0], self.const.CAM_FRAME_SIZE[1], rs.format.bgr8, \
                    self.const.NUM_OF_STREAMING_FPS)
            
            self.pipeline_handle = self.streams_pipeline.start(self.streams_configuration)

            self.alignment_handle = rs.align(rs.stream.color)
            self.colorizing_handle = rs.colorizer()
            self.device_handle = self.pipeline_handle.get_device()

            self.thresholding_handle = rs.threshold_filter(min_dist=self.const.DEPTH_MIN_THRESHOLD_DIST_IN_METERS, max_dist=self.const.DEPTH_MAX_THRESHOLD_DIST_IN_METERS)
            self.depth_to_disparity_handle = rs.disparity_transform(True)
            self.spatialization_handle = rs.spatial_filter()
            self.disparity_to_depth_handle = rs.disparity_transform(False)

            self.spatialization_handle.set_option(rs.option.filter_magnitude, self.const.SPATIALIZATION_MAGNITUDE)
            self.spatialization_handle.set_option(rs.option.filter_smooth_alpha, self.const.SPATIALIZATION_ALPHA)
            self.spatialization_handle.set_option(rs.option.filter_smooth_delta, self.const.SPATIALIZATION_DELTA)

            self.intrinsics_handle = self.pipeline_handle.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

            self.depth_resolution_in_meters = self.pipeline_handle.get_device().first_depth_sensor().get_depth_scale()
            # self.depth_resolution_in_meters is 0.001, just for information if it is needed for the Open3D function create_from_rgbd_image(...)

            self.streams_pipeline.stop()
        
        except:
            print("Realsense camera is not connected or there is no streaming video!\n")
            raise ValueError
    

    def get_camera_intrinsics_matrix(self):
        camera_intrinsics_matrix = np.array([[self.intrinsics_handle.fx, 0, self.intrinsics_handle.ppx], [0, self.intrinsics_handle.fy, self.intrinsics_handle.ppy], \
            [0, 0, 1]], np.float32)

        return camera_intrinsics_matrix
    

    def get_camera_distortion_coeffs(self):
        camera_distortion_coeffs = np.array([[self.intrinsics_handle.coeffs[0], self.intrinsics_handle.coeffs[1], self.intrinsics_handle.coeffs[2], \
            self.intrinsics_handle.coeffs[3], self.intrinsics_handle.coeffs[4]]], np.float32)

        return camera_distortion_coeffs


    def get_depth_resolution_in_meters(self):
        return self.depth_resolution_in_meters


    def set_exit_loop_flag(self, exit_loop_flag):
        self.exit_loop_flag = exit_loop_flag
    

    def get_exit_loop_flag(self):
        return self.exit_loop_flag


    def set_continue_loop_flag(self, continue_loop_flag):
        self.continue_loop_flag = continue_loop_flag


    def get_continue_loop_flag(self):
        return self.continue_loop_flag


    def start_streaming_images(self):
        try:
            self.pipeline_handle = self.streams_pipeline.start(self.streams_configuration)
        except:
            pass

    def stop_streaming_images(self):
        try:
            cv2.destroyWindow(self.const.STREAMING_WINDOW_NAME)
        except:
            pass

        try:
            self.streams_pipeline.stop()
        except:
            pass


    def process_streamed_images(self):
        # wait for coherent pair of depth and color frames
        streams_frames = self.streams_pipeline.wait_for_frames()
        aligned_streams_frames = self.alignment_handle.process(streams_frames)

        depth_frame = aligned_streams_frames.get_depth_frame()
        color_frame = aligned_streams_frames.get_color_frame()

        self.set_continue_loop_flag(False)
        
        if not depth_frame or not color_frame:
            self.set_continue_loop_flag(True)
            
            return

        # apply filtering
        depth_frame = self.thresholding_handle.process(depth_frame)
        depth_frame = self.depth_to_disparity_handle.process(depth_frame)
        depth_frame = self.spatialization_handle.process(depth_frame)
        depth_frame = self.disparity_to_depth_handle.process(depth_frame)

        # convert frames to images that are numpy arrays
        self.depth_image = np.asanyarray(depth_frame.get_data())
        self.color_image = np.asanyarray(color_frame.get_data())

        if self.depth_image.shape != self.color_image.shape:
            self.color_image = cv2.resize(self.color_image, (self.depth_image.shape[1], self.depth_image.shape[0]), interpolation=cv2.INTER_AREA)

        # apply colormap on depth image
        depth_colormap = np.asanyarray(self.colorizing_handle.colorize(depth_frame).get_data())

        if self.const.VISUALIZE_STREAMED_IMAGES == True:
            stacked_images = np.hstack((self.color_image, depth_colormap))

            cv2.imshow(self.const.STREAMING_WINDOW_NAME, stacked_images)
            cv2.waitKey(1)


    def get_last_processed_data_as_color_image(self): return self.color_image


    def save_processed_data_as_color_image(self, timestamp_with_unique_id_string):
        create_folder_if_does_not_exists(self.const.COLOR_IMAGES_FOLDER_ABS_PATH)
        COLOR_IMAGE_FILE_NAME = timestamp_with_unique_id_string + "_" + self.const.COMMON_PART_OF_COLOR_IMAGE_FILE_NAME + ".jpg"
        COLOR_IMAGE_FILE_ABS_PATH = self.const.COLOR_IMAGES_FOLDER_ABS_PATH + "/" + COLOR_IMAGE_FILE_NAME

        cv2.imwrite(COLOR_IMAGE_FILE_ABS_PATH, self.color_image)


    def save_processed_data_as_depth_array(self, timestamp_with_unique_id_string):
        create_folder_if_does_not_exists(self.const.DEPTH_ARRAYS_FOLDER_ABS_PATH)
        DEPTH_ARRAY_FILE_NAME = timestamp_with_unique_id_string + "_" + self.const.COMMON_PART_OF_DEPTH_ARRAY_FILE_NAME + ".npy"
        DEPTH_ARRAY_FILE_ABS_PATH = self.const.DEPTH_ARRAYS_FOLDER_ABS_PATH + "/" + DEPTH_ARRAY_FILE_NAME

        depth_array = self.depth_image

        np.save(DEPTH_ARRAY_FILE_ABS_PATH, depth_array)


    def save_processed_data_as_pointcloud(self, timestamp_with_unique_id_string):
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

        # save pointcloud with colors
        colored_depth_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(self.color_image), o3d.geometry.Image(self.depth_image), \
            depth_scale=1/self.depth_resolution_in_meters, convert_rgb_to_intensity=False)

        camera_intrinsics_object = o3d.camera.PinholeCameraIntrinsic()
        camera_intrinsics_object.set_intrinsics(width=self.intrinsics_handle.width, height=self.intrinsics_handle.height, fx=self.intrinsics_handle.fx, \
            fy=self.intrinsics_handle.fy, cx=self.intrinsics_handle.ppx, cy=self.intrinsics_handle.ppy)

        pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(colored_depth_image, camera_intrinsics_object)

        create_folder_if_does_not_exists(self.const.POINTCLOUDS_FOLDER_ABS_PATH)
        POINTCLOUD_FILE_NAME = timestamp_with_unique_id_string + "_" + self.const.COMMON_PART_OF_POINTCLOUD_FILE_NAME + ".ply"
        POINTCLOUD_FILE_ABS_PATH = self.const.POINTCLOUDS_FOLDER_ABS_PATH + "/" + POINTCLOUD_FILE_NAME

        o3d.io.write_point_cloud(POINTCLOUD_FILE_ABS_PATH, pointcloud)

        if self.const.VISUALIZE_SAVED_POINTCLOUD == True:
            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()
            visualizer.add_geometry(pointcloud)        
            visualizer.get_view_control().set_up((0, -1, 0))
            visualizer.get_view_control().set_front((0, 0, -1))
            visualizer.run()
            visualizer.destroy_window()



class RealSenseCalibration():
    
    def __init__(self, constants):
        self.const = constants

        self.accept_save_flag = False
        
        self.chessboard_intersections_size = (self.const.CHESSBOARD_INTERSECTIONS_PER_ROW, self.const.CHESSBOARD_INTERSECTIONS_PER_COLUMN)

        self.world_plane_corners = np.zeros((np.prod(self.chessboard_intersections_size), 3), np.float32)
        self.world_plane_corners[:, 0:2] = np.indices(self.chessboard_intersections_size).T.reshape(-1, 2)
        self.world_plane_corners = self.world_plane_corners * self.const.CHESSBOARD_CELL_SIDE_IN_METERS

        self.is_key_for_accepting_save_of_calibration_data_hold = False
        self.is_key_for_exiting_pose_visualization_hold = False
        
        (self.camera_translation_vector, self.camera_rotation_vector) = (None, None)


    def set_accept_save_flag(self, accept_save_flag): self.accept_save_flag = accept_save_flag

    def get_accept_save_flag(self): return self.accept_save_flag


    def visualize_processed_chessboard_poses_and_save_visualized_data(self, camera_intrinsics_matrix, camera_distortion_coeffs, num_of_wanted_chessboard_images):
        create_folder_if_does_not_exists(self.const.COLOR_IMAGES_FOLDER_ABS_PATH)
        create_folder_if_does_not_exists(self.const.DETECTED_COLOR_IMAGES_FOLDER_ABS_PATH)
        
        if len(sorted(os.listdir(self.const.COLOR_IMAGES_FOLDER_ABS_PATH))) <= num_of_wanted_chessboard_images:
            num_of_saved_chessboard_images = len(sorted(os.listdir(self.const.COLOR_IMAGES_FOLDER_ABS_PATH)))
        else:
            num_of_saved_chessboard_images = num_of_wanted_chessboard_images
        
        num_of_valid_chessboard_images = 0

        if num_of_saved_chessboard_images > 0:
            for chessboard_image_file_name in sorted(os.listdir(self.const.COLOR_IMAGES_FOLDER_ABS_PATH))[0:num_of_saved_chessboard_images]:
                chessboard_image = cv2.imread(self.const.COLOR_IMAGES_FOLDER_ABS_PATH + "/" + chessboard_image_file_name)

                try:
                    (ret, image_corners) = cv2.findChessboardCorners(chessboard_image, self.chessboard_intersections_size)

                    if ret == True:
                        gray_chessboard_image = cv2.cvtColor(chessboard_image, cv2.COLOR_BGR2GRAY)
                        image_corners = cv2.cornerSubPix(gray_chessboard_image, image_corners, (7, 7), (-1, -1), self.const.CORNER_SUBPIX_CRITERIA)

                        (_, proc_camera_rotation_vector, proc_camera_translation_vector, _) = cv2.solvePnPRansac(self.world_plane_corners, image_corners, \
                            camera_intrinsics_matrix, camera_distortion_coeffs)
                        (proc_camera_translation_vector, proc_camera_rotation_vector) = (proc_camera_translation_vector.reshape(-1), \
                            proc_camera_rotation_vector.reshape(-1))

                        cv2.drawChessboardCorners(chessboard_image, self.chessboard_intersections_size, image_corners, ret)
                        visualized_chessboard_image = cv2.drawFrameAxes(chessboard_image, camera_intrinsics_matrix, camera_distortion_coeffs, proc_camera_rotation_vector, \
                            proc_camera_translation_vector, self.const.CHESSBOARD_CELL_SIDE_IN_METERS, 3)
                        num_of_valid_chessboard_images = num_of_valid_chessboard_images + 1

                        PROCESSED_CHESSBOARD_WINDOW_NAME = self.const.COMMON_PART_OF_PROCESSED_CHESSBOARD_WINDOW_NAME + "'" + chessboard_image_file_name + "'"

                        print("Processed RealSense camera's translation vector: {}".format(proc_camera_translation_vector.T))
                        print("Processed RealSense camera's rotation vector: {}\n".format(proc_camera_rotation_vector.T))

                        cv2.imwrite(self.const.DETECTED_COLOR_IMAGES_FOLDER_ABS_PATH + "/" + chessboard_image_file_name, visualized_chessboard_image)

                        if self.const.VISUALIZE_SAVED_CHESSBOARD_IMAGES == True:
                            cv2.imshow(PROCESSED_CHESSBOARD_WINDOW_NAME, visualized_chessboard_image)

                            while True:
                                cv2.waitKey(1)
        
                                if not keyboard.is_pressed(self.const.KEY_FOR_EXITING_POSE_VISUALIZATION):
                                    self.is_key_for_exiting_pose_visualization_hold = False

                                if keyboard.is_pressed(self.const.KEY_FOR_EXITING_POSE_VISUALIZATION) and self.is_key_for_exiting_pose_visualization_hold == False:
                                    self.is_key_for_exiting_pose_visualization_hold = True
                                
                                    break

                            cv2.destroyWindow(PROCESSED_CHESSBOARD_WINDOW_NAME)
                except:
                    print("The chessboard image '" + chessboard_image_file_name + "' is not valid!\n")

            if num_of_valid_chessboard_images > 0:
                print("Number of valid processed chessboard images: {} / {}\n".format(num_of_valid_chessboard_images, num_of_saved_chessboard_images))
            else:
                print("There is no any valid processed chessboard image!\n")
                raise ValueError
        
        else:
            print("There is no any processed chessboard image!\n")
            raise ValueError


    def process_chessboard_pose(self, camera_intrinsics_matrix, camera_distortion_coeffs, chessboard_image):
        try:
            chessboard_image = chessboard_image.copy()
            
            (ret, image_corners) = cv2.findChessboardCorners(chessboard_image, self.chessboard_intersections_size)

            if ret == True:
                gray_chessboard_image = cv2.cvtColor(chessboard_image, cv2.COLOR_BGR2GRAY)
                image_corners = cv2.cornerSubPix(gray_chessboard_image, image_corners, (7, 7), (-1, -1), self.const.CORNER_SUBPIX_CRITERIA)

                (_, self.camera_rotation_vector, self.camera_translation_vector, _) = cv2.solvePnPRansac(self.world_plane_corners, image_corners, \
                    camera_intrinsics_matrix, camera_distortion_coeffs)
                (self.camera_translation_vector, self.camera_rotation_vector) = (self.camera_translation_vector.reshape(-1), self.camera_rotation_vector.reshape(-1))

                print("RealSense camera's translation vector: {}".format(self.camera_translation_vector.T))
                print("RealSense camera's rotation vector: {}\n".format(self.camera_rotation_vector.T))

                cv2.drawChessboardCorners(chessboard_image, self.chessboard_intersections_size, image_corners, ret)
                chessboard_image = cv2.drawFrameAxes(chessboard_image, camera_intrinsics_matrix, camera_distortion_coeffs, self.camera_rotation_vector, \
                    self.camera_translation_vector, self.const.CHESSBOARD_CELL_SIDE_IN_METERS, 3)

                cv2.imshow(self.const.CHESSBOARD_WINDOW_NAME, chessboard_image)                    
                
                while True:
                    cv2.waitKey(1)

                    if not keyboard.is_pressed(self.const.KEY_FOR_ACCEPTING_SAVE_OF_CALIBRATION_DATA):
                        self.is_key_for_accepting_save_of_calibration_data_hold = False

                    if keyboard.is_pressed(self.const.KEY_FOR_ACCEPTING_SAVE_OF_CALIBRATION_DATA) and self.is_key_for_accepting_save_of_calibration_data_hold == False:
                        self.is_key_for_accepting_save_of_calibration_data_hold = True
                        
                        self.set_accept_save_flag(True)
                        break

                    if not keyboard.is_pressed(self.const.KEY_FOR_EXITING_POSE_VISUALIZATION):
                        self.is_key_for_exiting_pose_visualization_hold = False
                        
                    if keyboard.is_pressed(self.const.KEY_FOR_EXITING_POSE_VISUALIZATION) and self.is_key_for_exiting_pose_visualization_hold == False:
                        self.is_key_for_exiting_pose_visualization_hold = True

                        self.set_accept_save_flag(False)
                        break

                cv2.destroyWindow(self.const.CHESSBOARD_WINDOW_NAME)

            else:
                self.set_accept_save_flag(False)
                print("The chessboard image is not valid!\n")
                
        except:
            print("Chessboard intersections' size is wrong!\n")
            raise ValueError
    

    def save_processed_chessboard_pose_data(self, timestamp_with_unique_id_string):
            create_folder_if_does_not_exists(self.const.CAMERA_TRANSLATION_VECTORS_FOLDER_ABS_PATH)
            CAMERA_TRANSLATION_VECTOR_FILE_NAME = timestamp_with_unique_id_string + "_" + self.const.COMMON_PART_OF_CAMERA_TRANS_VECTOR_FILE_NAME + ".txt"
            CAMERA_TRANSLATION_VECTOR_FILE_ABS_PATH = self.const.CAMERA_TRANSLATION_VECTORS_FOLDER_ABS_PATH + "/" + CAMERA_TRANSLATION_VECTOR_FILE_NAME

            create_folder_if_does_not_exists(self.const.CAMERA_ROTATION_VECTORS_FOLDER_ABS_PATH)
            CAMERA_ROTATION_VECTOR_FILE_NAME = timestamp_with_unique_id_string + "_" + self.const.COMMON_PART_OF_CAMERA_ROT_VECTOR_FILE_NAME + ".txt"
            CAMERA_ROTATION_VECTOR_FILE_ABS_PATH = self.const.CAMERA_ROTATION_VECTORS_FOLDER_ABS_PATH + "/" + CAMERA_ROTATION_VECTOR_FILE_NAME

            with open(CAMERA_TRANSLATION_VECTOR_FILE_ABS_PATH, "w") as camera_translation_vector_file:
                np.savetxt(camera_translation_vector_file, self.camera_translation_vector)

            with open(CAMERA_ROTATION_VECTOR_FILE_ABS_PATH, "w") as camera_rotation_vector_file:
                np.savetxt(camera_rotation_vector_file, self.camera_rotation_vector)


    def load_processed_chessboard_pose_data(self, camera_translation_vector_file_name, camera_rotation_vector_file_name):
        proc_camera_translation_vector = np.loadtxt(self.const.CAMERA_TRANSLATION_VECTORS_FOLDER_ABS_PATH + "/" + camera_translation_vector_file_name, np.float32)
        proc_camera_rotation_vector = np.loadtxt(self.const.CAMERA_ROTATION_VECTORS_FOLDER_ABS_PATH + "/" + camera_rotation_vector_file_name, np.float32)

        return (proc_camera_translation_vector, proc_camera_rotation_vector)



class UniversalRobotCalibration():
    
    def __init__(self, constants):
        self.const = constants

        (self.robot_translation_vector, self.robot_rotation_vector) = (None, None)
        

    def process_robot_pose(self):
        try:
            client_socket = socket.socket()
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            client_socket.settimeout(self.const.COMMUNICATION_CONNECTION_TIMEOUT_TIME)
            client_socket.connect((self.const.COMMUNICATION_ROBOT_IP_ADDRESS, self.const.COMMUNICATION_ROBOT_PORT))
            
            received_robot_data = client_socket.recv(self.const.COMMUNICATION_MAX_RECIEVING_NUM_OF_BYTES)

            while len(received_robot_data) != self.const.COMMUNICATION_MAX_RECIEVING_NUM_OF_BYTES:
                received_robot_data = client_socket.recv(self.const.COMMUNICATION_MAX_RECIEVING_NUM_OF_BYTES)
            
            self.robot_translation_vector = np.array([round(i, 5) for i in struct.unpack("!3d", received_robot_data[444:468])], np.float32)
            self.robot_rotation_vector = np.array([round(i, 3) for i in struct.unpack("!3d", received_robot_data[468:492])], np.float32)

            client_socket.close()

            print("Universal Robot's TCP's translation vector: {}".format(self.robot_translation_vector.T))
            print("Universal Robot's TCP's rotation vector: {}\n".format(self.robot_rotation_vector.T))
        
        except:
            print("Universal Robot server is not started!\n")
            raise ValueError


    def get_last_processed_robot_pose(self): return (self.robot_translation_vector, self.robot_rotation_vector)


    def save_processed_robot_pose_data(self, timestamp_with_unique_id_string):
            create_folder_if_does_not_exists(self.const.ROBOT_TRANSLATION_VECTORS_FOLDER_ABS_PATH)
            robot_translation_vector_file_name = timestamp_with_unique_id_string + "_" + self.const.COMMON_PART_OF_ROBOT_TRANS_VECTOR_FILE_NAME + ".txt"
            robot_translation_vector_file_abs_path = self.const.ROBOT_TRANSLATION_VECTORS_FOLDER_ABS_PATH + "/" + robot_translation_vector_file_name

            create_folder_if_does_not_exists(self.const.ROBOT_ROTATION_VECTORS_FOLDER_ABS_PATH)
            robot_rotation_vector_file_name = timestamp_with_unique_id_string + "_" + self.const.COMMON_PART_OF_ROBOT_ROT_VECTOR_FILE_NAME + ".txt"
            robot_rotation_vector_file_abs_path = self.const.ROBOT_ROTATION_VECTORS_FOLDER_ABS_PATH + "/" + robot_rotation_vector_file_name

            with open(robot_translation_vector_file_abs_path, "w") as robot_translation_vector_file:
                np.savetxt(robot_translation_vector_file, self.robot_translation_vector)

            with open(robot_rotation_vector_file_abs_path, "w") as robot_rotation_vector_file:
                np.savetxt(robot_rotation_vector_file, self.robot_rotation_vector)


    def load_processed_robot_pose_data(self, robot_translation_vector_file_name, robot_rotation_vector_file_name):
        proc_robot_translation_vector = np.loadtxt(self.const.ROBOT_TRANSLATION_VECTORS_FOLDER_ABS_PATH + "/" + robot_translation_vector_file_name, np.float32)
        proc_robot_rotation_vector = np.loadtxt(self.const.ROBOT_ROTATION_VECTORS_FOLDER_ABS_PATH + "/" + robot_rotation_vector_file_name, np.float32)

        return (proc_robot_translation_vector, proc_robot_rotation_vector)



class HandEyeCalibration():
    
    def __init__(self, constants):
        self.const = constants

        (self.hand_eye_translation_vector, self.hand_eye_rotation_vector) = (None, None)


    def process_hand_eye_calibration(self, rs_camera_translation_vectors_list, rs_camera_rotation_matrices_list, ur_translation_vectors_list, \
        ur_rotation_matrices_list):
        try:
            (self.hand_eye_rotation_matrix, self.hand_eye_translation_vector) = cv2.calibrateHandEye(ur_rotation_matrices_list, ur_translation_vectors_list, \
                rs_camera_rotation_matrices_list, rs_camera_translation_vectors_list, method=cv2.CALIB_HAND_EYE_TSAI)
            (self.hand_eye_rotation_vector, _) = cv2.Rodrigues(self.hand_eye_rotation_matrix)
    
            print("Translation vector from RealSense camera to Universal Robot's TCP: {}".format(self.hand_eye_translation_vector.T))
            print("Rotation vector from RealSense camera to Universal Robot's TCP: {}\n".format(self.hand_eye_rotation_vector.T))
        
        except:
            print("There are not enough valid chessboard images to do hand-eye calibration, at least three are needed!\n")
            raise ValueError


    def save_processed_hand_eye_calibration_data(self):
            with open(self.const.HAND_EYE_TRANSLATION_VECTOR_FILE_ABS_PATH, "w") as hand_eye_translation_vector_file:
                np.savetxt(hand_eye_translation_vector_file, self.hand_eye_translation_vector)

            with open(self.const.HAND_EYE_ROTATION_VECTOR_FILE_ABS_PATH, "w") as hand_eye_rotation_vector_file:
                np.savetxt(hand_eye_rotation_vector_file, self.hand_eye_rotation_vector)


    def load_processed_hand_eye_calibration_data(self):
        proc_hand_eye_translation_vector = np.loadtxt(self.const.HAND_EYE_TRANSLATION_VECTOR_FILE_ABS_PATH, np.float32)
        proc_hand_eye_rotation_vector = np.loadtxt(self.const.HAND_EYE_ROTATION_VECTOR_FILE_ABS_PATH, np.float32)

        return (proc_hand_eye_translation_vector, proc_hand_eye_rotation_vector)



class TransformationMatrix():
    
    def __init__(self, constants):
        self.const = constants

    def convert_translation_and_rotation_vector_to_transformation_matrix(self, translation_vector, rotation_vector):
        (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)

        transformation_matrix = np.identity(4, np.float32)
        transformation_matrix[0:3, 0:3] = rotation_matrix
        transformation_matrix[0:3, 3] = translation_vector

        return transformation_matrix


    def convert_transformation_matrix_to_translation_and_rotation_vector(self, transformation_matrix):
        rotation_matrix = np.array(transformation_matrix[0:3, 0:3], np.float32)
        (rotation_vector, _) = cv2.Rodrigues(rotation_matrix)
        rotation_vector = rotation_vector[:, 0]

        translation_vector = np.array(transformation_matrix[0:3, 3], np.float32)

        return (translation_vector, rotation_vector)


    def get_pointcloud_transformation_matrix(self, ur_calibration_object, hand_eye_calibration_object, file_name_index):
        robot_translation_vector_file_names_list = sorted(os.listdir(self.const.ROBOT_TRANSLATION_VECTORS_FOLDER_ABS_PATH))
        robot_rotation_vector_file_names_list = sorted(os.listdir(self.const.ROBOT_ROTATION_VECTORS_FOLDER_ABS_PATH))

        (hand_eye_translation_vector, hand_eye_rotation_vector) = hand_eye_calibration_object.load_processed_hand_eye_calibration_data()
        hand_eye_transformation_matrix = self.convert_translation_and_rotation_vector_to_transformation_matrix(hand_eye_translation_vector, hand_eye_rotation_vector)
        # print("Transformation matrix from Universal Robot's TCP to RealSense camera: \n{}".format(hand_eye_transformation_matrix))

        (ur_translation_vector, ur_rotation_vector) = ur_calibration_object.load_processed_robot_pose_data(robot_translation_vector_file_names_list[file_name_index], \
            robot_rotation_vector_file_names_list[file_name_index])
        (ur_rotation_matrix, _) = cv2.Rodrigues(ur_rotation_vector)

        ur_transformation_matrix = self.convert_translation_and_rotation_vector_to_transformation_matrix(ur_translation_vector, ur_rotation_vector)
        # print("Transformation matrix from Universal Robot's base to it's TCP: \n{}".format(ur_transformation_matrix))

        pointcloud_transformation_matrix = np.dot(ur_transformation_matrix, hand_eye_transformation_matrix)
        # print("Transformation matrix from Universal Robot's base to RealSense camera: \n{}".format(pointcloud_transformation_matrix))

        (pointcloud_translation_vector, pointcloud_rotation_vector) = self.convert_transformation_matrix_to_translation_and_rotation_vector(pointcloud_transformation_matrix)
        (pointcloud_rotation_matrix, _) = cv2.Rodrigues(pointcloud_rotation_vector)

        # pytransform3d.rotations.plot_basis(linestyle="--")
        # pytransform3d.rotations.plot_basis(R=ur_rotation_matrix, p=ur_translation_vector.T, linestyle="-.")
        # pytransform3d.rotations.plot_basis(R=pointcloud_rotation_matrix, p=pointcloud_translation_vector.T, linestyle="-")
        # matplotlib_inline.backend_inline.show()

        return pointcloud_transformation_matrix



class UniversalRobotCommunication():

    def __init__(self, constants, connection_staying_opened_bool=False):
        self.const = constants
        self.connection_currently_opened_bool = False
        self.connection_staying_opened_bool = connection_staying_opened_bool
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.settimeout(self.const.COMMUNICATION_CONNECTION_TIMEOUT_TIME)


    def get_connection_currently_opened_bool(self): return self.connection_currently_opened_bool


    def get_connection_staying_opened_bool(self): return self.connection_staying_opened_bool


    def start_pc_server_with_accepting_robot_connection_and_connect_pc_with_robot_server(self, server_port=None):
        if self.connection_currently_opened_bool == False:
            try:
                if server_port == None:
                    self.server_socket.bind((self.const.COMMUNICATION_PC_IP_ADDRESS, self.const.COMMUNICATION_PC_PORT))
                else:
                    self.server_socket.bind((self.const.COMMUNICATION_PC_IP_ADDRESS, server_port))
                self.server_socket.listen(self.const.COMMUNICATION_MAX_NUM_OF_WAITING_CONNECTIONS)
                print("PC server is started and waits for Universal Robot client to connect.")
                (self.accepted_connection, self.accepted_connection_address_and_port) = self.server_socket.accept()
                self.accepted_connection.settimeout(self.const.COMMUNICATION_LISTENING_TIMEOUT_TIME)
                print("Connection from Universal Robot's server address {} at port {} is accepted as client.".format(*self.accepted_connection_address_and_port))
                self.connection_currently_opened_bool = True
            
            except:
                print("PC and Universal Robot don't have the same network part of address and so are not connectable!\n")
                raise ValueError


    def send_string_data(self, sending_string_data):
        self.accepted_connection.send(sending_string_data.encode())


    def send_float_data(self, sending_float_data):
        sending_float_data = "({})".format(sending_float_data)
        self.accepted_connection.send(sending_float_data.encode())


    def send_pose_data(self, sending_translation_data, sending_rotation_data):
        sending_pose_data = "({},{},{},{},{},{})".format(*sending_translation_data, *sending_rotation_data)
        self.accepted_connection.send(sending_pose_data.encode())


    def receive_string_data(self):
        received_string_data = self.accepted_connection.recv(self.const.COMMUNICATION_MAX_RECIEVING_NUM_OF_BYTES).decode()

        return received_string_data


    def shutdown_pc_server_and_close_pc_connection_with_robot_server(self):
        if self.connection_staying_opened_bool == False:
            self.server_socket.close()
            self.connection_currently_opened_bool = False