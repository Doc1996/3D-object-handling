import os
import cv2
import keyboard
import pytransform3d.visualizer
# import pytransform3d.rotations
# import matplotlib_inline

from CNN_OD_detect_objects import *
from RS_and_3D_OD_common_functions import *
from RS_and_3D_OD_common_classes import *

PROJECT_FOLDER_ABS_PATH = os.path.dirname(os.path.abspath("__file__")).replace(os.sep, "/")



def warmup_object_detection(const):
    if not os.path.exists(const.WARMUP_IMAGE_FILE_ABS_PATH):
        warmup_image = 255 * np.ones([const.CAM_FRAME_SIZE[1], const.CAM_FRAME_SIZE[0], 3], np.uint8)
        cv2.imwrite(const.WARMUP_IMAGE_FILE_ABS_PATH, warmup_image)
    
    detect_objects(source_abs_path_or_zero=const.WARMUP_IMAGE_FILE_ABS_PATH, weights_file_abs_path=const.SELECTED_WEIGHTS_FILE_ABS_PATH, selected_class_indices_list=\
        const.SELECTED_CLASS_INDICES_LIST, image_size=const.DETECTION_CAM_FRAME_WIDTH, confidence_threshold=const.DETECTION_MIN_PERCENTAGE_OF_CONFIDENCE, nms_iou_threshold=\
        const.DETECTION_MAX_PERCENTAGE_OF_NMS_IOU)

    os.remove(const.WARMUP_IMAGE_FILE_ABS_PATH)



def calibrate_realsense_camera_and_universal_robot_with_hand_eye_method_and_get_matrix(const, calibrate_with_already_processed_data=False, num_of_wanted_chessboard_images=\
    None):
    # camera vectors define transformation from camera to upper left point of chessboard
    # robot vectors define transformation from robot's base to it's gripper
    # hand-eye vectors define transformation from robot's gripper to camera
    # hand-eye transformation is visualized in the plot as transformation from coordinate system with dash-dotted lines to one with solid lines
    
    rs_streaming_object = RealSenseStreaming(const, calibrate_with_already_processed_data)
    rs_calibration_object = RealSenseCalibration(const)
    ur_calibration_object = UniversalRobotCalibration(const)
    hand_eye_calibration_object = HandEyeCalibration(const)
    transformation_matrix_object = TransformationMatrix(const)


    try:
        if calibrate_with_already_processed_data == False:
            timestamp_with_unique_id_string = get_timestamp_with_unique_id_string()

            backup_and_empty_folder_if_exists(const.CALIBRATION_OR_RECONSTRUCTION_FOLDER_ABS_PATH, timestamp_with_unique_id_string)
        

        create_folder_if_does_not_exists(const.COLOR_IMAGES_FOLDER_ABS_PATH)

        num_of_saved_chessboard_images = len(sorted(os.listdir(const.COLOR_IMAGES_FOLDER_ABS_PATH)))

        rs_camera_intrinsics_matrix = rs_streaming_object.get_camera_intrinsics_matrix()
        rs_camera_distortion_coeffs = rs_streaming_object.get_camera_distortion_coeffs()

        if num_of_wanted_chessboard_images == None:
            if num_of_saved_chessboard_images >= const.MIN_NUM_OF_WANTED_COLOR_IMAGES:
                num_of_wanted_chessboard_images = num_of_saved_chessboard_images
            else:
                num_of_wanted_chessboard_images = const.MIN_NUM_OF_WANTED_COLOR_IMAGES


        if num_of_saved_chessboard_images < num_of_wanted_chessboard_images:
            rs_streaming_object.start_streaming_images()

            is_key_for_proposing_save_of_calibration_data_hold = False

            while num_of_saved_chessboard_images < num_of_wanted_chessboard_images:
                rs_streaming_object.process_streamed_images()

                if rs_streaming_object.get_continue_loop_flag() == True:
                    continue

                if not keyboard.is_pressed(const.KEY_FOR_PROPOSING_SAVE_OF_CALIBRATION_DATA):
                    is_key_for_proposing_save_of_calibration_data_hold = False

                if keyboard.is_pressed(const.KEY_FOR_PROPOSING_SAVE_OF_CALIBRATION_DATA) and is_key_for_proposing_save_of_calibration_data_hold == False:
                    is_key_for_proposing_save_of_calibration_data_hold = True

                    chessboard_image = rs_streaming_object.get_last_processed_data_as_color_image()
                    rs_calibration_object.process_chessboard_pose(rs_camera_intrinsics_matrix, rs_camera_distortion_coeffs, chessboard_image)

                    if rs_calibration_object.get_accept_save_flag() == True:
                        timestamp_with_unique_id_string = get_timestamp_with_unique_id_string()

                        rs_streaming_object.save_processed_data_as_color_image(timestamp_with_unique_id_string)
                        rs_calibration_object.save_processed_chessboard_pose_data(timestamp_with_unique_id_string)
                        ur_calibration_object.process_robot_pose()
                        ur_calibration_object.save_processed_robot_pose_data(timestamp_with_unique_id_string)

                num_of_saved_chessboard_images = len(sorted(os.listdir(const.COLOR_IMAGES_FOLDER_ABS_PATH)))

                if keyboard.is_pressed(const.KEY_FOR_EXITING_LOOP):
                    break

            rs_streaming_object.stop_streaming_images()

        rs_calibration_object.visualize_processed_chessboard_poses_and_save_visualized_data(rs_camera_intrinsics_matrix, rs_camera_distortion_coeffs, \
            num_of_wanted_chessboard_images)


        camera_translation_vector_file_names_list = sorted(os.listdir(const.CAMERA_TRANSLATION_VECTORS_FOLDER_ABS_PATH))
        camera_rotation_vector_file_names_list = sorted(os.listdir(const.CAMERA_ROTATION_VECTORS_FOLDER_ABS_PATH))

        robot_translation_vector_file_names_list = sorted(os.listdir(const.ROBOT_TRANSLATION_VECTORS_FOLDER_ABS_PATH))
        robot_rotation_vector_file_names_list = sorted(os.listdir(const.ROBOT_ROTATION_VECTORS_FOLDER_ABS_PATH))

        rs_camera_translation_vectors_list = []
        rs_camera_rotation_matrices_list = []

        ur_translation_vectors_list = []
        ur_rotation_matrices_list = []

        for file_name_index in range(0, num_of_saved_chessboard_images):
            (rs_camera_translation_vector, rs_camera_rotation_vector) = rs_calibration_object.load_processed_chessboard_pose_data(\
                camera_translation_vector_file_names_list[file_name_index], camera_rotation_vector_file_names_list[file_name_index])
            (rs_camera_rotation_matrix, _) = cv2.Rodrigues(rs_camera_rotation_vector)

            (ur_translation_vector, ur_rotation_vector) = ur_calibration_object.load_processed_robot_pose_data(\
                robot_translation_vector_file_names_list[file_name_index], robot_rotation_vector_file_names_list[file_name_index])
            (ur_rotation_matrix, _) = cv2.Rodrigues(ur_rotation_vector)

            rs_camera_translation_vectors_list.append(rs_camera_translation_vector)
            rs_camera_rotation_matrices_list.append(rs_camera_rotation_matrix)

            ur_translation_vectors_list.append(ur_translation_vector)
            ur_rotation_matrices_list.append(ur_rotation_matrix)


        hand_eye_calibration_object.process_hand_eye_calibration(rs_camera_translation_vectors_list, rs_camera_rotation_matrices_list, ur_translation_vectors_list, \
            ur_rotation_matrices_list)
        hand_eye_calibration_object.save_processed_hand_eye_calibration_data()

        (hand_eye_translation_vector, hand_eye_rotation_vector) = hand_eye_calibration_object.load_processed_hand_eye_calibration_data()
        hand_eye_transformation_matrix = transformation_matrix_object.convert_translation_and_rotation_vector_to_transformation_matrix(hand_eye_translation_vector, \
            hand_eye_rotation_vector)
        (hand_eye_rotation_matrix, _) = cv2.Rodrigues(hand_eye_rotation_vector)
        print("Transformation matrix from Universal Robot's TCP to RealSense camera: \n{}".format(hand_eye_transformation_matrix))
                    
        hand_eye_transformation_figure = pytransform3d.visualizer.Figure(width=const.CAM_FRAME_SIZE[0], height=const.CAM_FRAME_SIZE[1])
        hand_eye_transformation_figure.plot_basis()
        hand_eye_transformation_figure.plot_basis(R=hand_eye_rotation_matrix, p=hand_eye_translation_vector.T)
        hand_eye_transformation_figure.save_image(const.HAND_EYE_TRANSFORMATION_FILE_ABS_PATH)
        
        # pytransform3d.rotations.plot_basis(linestyle="-.")
        # pytransform3d.rotations.plot_basis(R=hand_eye_rotation_matrix, p=hand_eye_translation_vector.T, linestyle="-")
        # matplotlib_inline.backend_inline.show()


        return hand_eye_transformation_matrix


    except:
        print("There are not enough chessboard images to do hand-eye calibration, at least three valid are needed!\n")
        raise ValueError



def detect_objects_on_data_from_realsense_camera_and_universal_robot(const, ur_communication_object, detect_with_already_processed_data=False, server_port=None):
    # robot transformation is visualized in the plot as transformation from coordinate system with dashed lines to one with dash-dotted lines
    # pointcloud transformation is visualized in the plot as transformation from coordinate system with dashed lines to one with solid lines

    rs_streaming_object = RealSenseStreaming(const, detect_with_already_processed_data)
    ur_calibration_object = UniversalRobotCalibration(const)
    

    if detect_with_already_processed_data == False:
        try:
            ur_communication_object.start_pc_server_with_accepting_robot_connection_and_connect_pc_with_robot_server(server_port)
            ur_communication_object.send_string_data(const.COMMUNICATION_RECONSTRUCT_STRING)

            timestamp_with_unique_id_string = get_timestamp_with_unique_id_string()

            backup_and_empty_folder_if_exists(const.CALIBRATION_OR_RECONSTRUCTION_FOLDER_ABS_PATH, timestamp_with_unique_id_string)

            rs_streaming_object.start_streaming_images()

            while True:
                rs_streaming_object.process_streamed_images()

                if rs_streaming_object.get_continue_loop_flag() == True:
                    continue

                try:
                    received_string = ur_communication_object.receive_string_data()

                    if received_string == const.COMMUNICATION_ROBOT_READY_STRING:
                        timestamp_with_unique_id_string = get_timestamp_with_unique_id_string()

                        time.sleep(const.COMMUNICATION_ROBOT_SETTLING_TIME)

                        rs_streaming_object.save_processed_data_as_color_image(timestamp_with_unique_id_string)
                        rs_streaming_object.save_processed_data_as_depth_array(timestamp_with_unique_id_string)
                        rs_streaming_object.save_processed_data_as_pointcloud(timestamp_with_unique_id_string)
                        ur_calibration_object.process_robot_pose()
                        ur_calibration_object.save_processed_robot_pose_data(timestamp_with_unique_id_string)

                        ur_communication_object.send_string_data(const.COMMUNICATION_PC_READY_STRING)

                    elif received_string == const.COMMUNICATION_FINISHED_STRING:
                        break
                
                except:
                    pass
                
            rs_streaming_object.stop_streaming_images()

            ur_communication_object.shutdown_pc_server_and_close_pc_connection_with_robot_server()
        
        except:
            print("Universal Robot client is not connected!\n")
            raise ValueError
    

    create_folder_if_does_not_exists_or_empty_it_if_exists(const.DETECTED_COLOR_IMAGES_FOLDER_ABS_PATH)
    create_folder_if_does_not_exists_or_empty_it_if_exists(const.DETECTED_LABELS_FOR_COLOR_IMAGES_FOLDER_ABS_PATH)
        
    detect_objects(source_abs_path_or_zero=const.COLOR_IMAGES_FOLDER_ABS_PATH, detected_images_or_videos_folder_abs_path=const.DETECTED_COLOR_IMAGES_FOLDER_ABS_PATH, \
        detected_labels_folder_abs_path=const.DETECTED_LABELS_FOR_COLOR_IMAGES_FOLDER_ABS_PATH, weights_file_abs_path=const.SELECTED_WEIGHTS_FILE_ABS_PATH, \
        selected_class_indices_list=const.SELECTED_CLASS_INDICES_LIST, image_size=const.DETECTION_CAM_FRAME_WIDTH, confidence_threshold=\
        const.DETECTION_MIN_PERCENTAGE_OF_CONFIDENCE, nms_iou_threshold=const.DETECTION_MAX_PERCENTAGE_OF_NMS_IOU)