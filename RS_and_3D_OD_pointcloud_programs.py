import os
import cv2
import numpy as np
import math

from RS_and_3D_OD_common_classes import *
from RS_and_3D_OD_pointcloud_functions import *

PROJECT_FOLDER_ABS_PATH = os.path.dirname(os.path.abspath("__file__")).replace(os.sep, "/")



def reconstruct_and_register_pointclouds_and_get_combined_pointcloud(const, use_registration_for_pointclouds_reconstruction=True):
    ur_calibration_object = UniversalRobotCalibration(const)
    hand_eye_calibration_object = HandEyeCalibration(const)
    transformation_matrix_object = TransformationMatrix(const)


    try:
        pointclouds_list = []
        pointcloud_file_names_list = sorted(os.listdir(const.POINTCLOUDS_FOLDER_ABS_PATH))

        for file_name_index in range(0, len(pointcloud_file_names_list)):
            pointcloud_file_name = pointcloud_file_names_list[file_name_index]
            pointcloud_file_abs_path = const.POINTCLOUDS_FOLDER_ABS_PATH + "/" + pointcloud_file_name

            pointcloud_transformation_matrix = transformation_matrix_object.get_pointcloud_transformation_matrix(ur_calibration_object, hand_eye_calibration_object, file_name_index)

            pointcloud = load_pointcloud(pointcloud_file_abs_path)
            pointcloud = get_prepared_pointcloud(pointcloud, pointcloud_transformation_matrix)

            pointclouds_list.append(pointcloud)

        print("The pointclouds are combined.\n")


        fixed_pointcloud = pointclouds_list[0]
        combined_pointcloud = get_copy_of_pointcloud(fixed_pointcloud)

        for floating_file_name_index in range(1, len(pointcloud_file_names_list)):
            floating_pointcloud = pointclouds_list[floating_file_name_index]

            if use_registration_for_pointclouds_reconstruction == True:
                (correspondence_fitness, correspondence_inlier_rmse) = get_pointclouds_correspondence_result(fixed_pointcloud, floating_pointcloud)

                try:
                    registered_floating_pointcloud = get_floating_pointcloud_registered_by_generalized_method(fixed_pointcloud, floating_pointcloud)
                    (registered_correspondence_fitness, registered_correspondence_inlier_rmse) = get_pointclouds_correspondence_result(fixed_pointcloud, \
                        registered_floating_pointcloud)

                    if (registered_correspondence_fitness >= correspondence_fitness) and (registered_correspondence_inlier_rmse <= correspondence_inlier_rmse):
                        floating_pointcloud = registered_floating_pointcloud
                        print("Original pointcloud {} is replaced with registered: correspondence fitness has grew for {} and correspondence inlier RMSE has fell for {}.".\
                            format(floating_file_name_index, registered_correspondence_fitness - correspondence_fitness, - registered_correspondence_inlier_rmse + \
                            correspondence_inlier_rmse))
                    else:
                        print("Original pointcloud {} isn't replaced with registered: correspondence fitness and correspondence inlier RMSE haven't improved.".format(\
                            floating_file_name_index))
                except:
                    print("The pointclouds are not correspondent enough to be registered!")

            combined_pointcloud = combined_pointcloud + floating_pointcloud
        
        if use_registration_for_pointclouds_reconstruction == True:
            print("")


        combined_pointcloud = get_everything_but_largest_plane_of_pointcloud(combined_pointcloud)
        combined_pointcloud = get_everything_over_ground_level_of_pointcloud(combined_pointcloud)
        combined_pointcloud = get_prepared_combined_pointcloud(combined_pointcloud)
        combined_pointcloud = get_clustered_pointcloud(combined_pointcloud, filter_clustered_pointcloud_by_noise=True, filter_clustered_pointcloud_by_diag_dist=True, \
            paint_clustered_pointcloud=False)


        return combined_pointcloud

    except:
        print("There is no any pointcloud!\n")
        raise ValueError



def run_detection_on_combined_pointcloud_for_selected_class_index_and_get_detected_pointcloud_and_bounding_boxes_list(const, combined_pointcloud, selected_class_index):
    rs_streaming_object = RealSenseStreaming(const, use_already_processed_data=True)
    ur_calibration_object = UniversalRobotCalibration(const)
    hand_eye_calibration_object = HandEyeCalibration(const)
    transformation_matrix_object = TransformationMatrix(const)


    detected_pointcloud = get_copy_of_pointcloud(combined_pointcloud)

    detected_oriented_bounding_boxes_for_cropping_list = []

    if len(np.asarray(detected_pointcloud.points)) > 0:
        filtered_detected_data = get_filtered_detected_data_for_selected_class_index(selected_class_index, rs_streaming_object, ur_calibration_object, \
            hand_eye_calibration_object, transformation_matrix_object)

        for detected_data_row in filtered_detected_data:
            detected_points_array_for_cropping = detected_data_row[2]

            detected_oriented_bounding_box_for_cropping = get_oriented_bounding_box_from_points_array(detected_points_array_for_cropping)
            detected_pointcloud = get_pointcloud_cropped_with_bounding_box(detected_pointcloud, detected_oriented_bounding_box_for_cropping)

            detected_oriented_bounding_boxes_for_cropping_list.append(detected_oriented_bounding_box_for_cropping)

        if len(detected_oriented_bounding_boxes_for_cropping_list) > 0:
            detected_pointcloud = get_largest_severely_filtered_pointcloud_containing_noise(detected_pointcloud, paint_clustered_pointcloud=False)
            detected_pointcloud = get_largest_faintly_filtered_pointcloud(detected_pointcloud, filter_clustered_pointcloud_by_noise=True, \
                paint_clustered_pointcloud=False)
        else:
            detected_pointcloud = get_empty_pointcloud()


    return (detected_pointcloud, detected_oriented_bounding_boxes_for_cropping_list)



def run_PCA_on_detected_pointcloud_and_get_sides_center_points_with_normals(const, detected_pointcloud):
    hull_pointcloud = get_convex_hull_pointcloud(detected_pointcloud)
    hull_pointcloud = get_painted_pointcloud(hull_pointcloud)
    hull_center_point = get_pointcloud_center_point_array(hull_pointcloud)
    hull_center_pointcloud = get_pointcloud_from_points_array([hull_center_point])

    (hull_PCA_eigenvalues, hull_PCA_eigenvectors) = get_pointcloud_principal_components(hull_pointcloud)
    hull_PCA_eigenvalues = normalize_vector(hull_PCA_eigenvalues)

    hull_oriented_bounding_box = get_oriented_bounding_box_from_pointcloud(hull_pointcloud)
    dimensions_of_hull_bounding_box = get_dimensions_of_oriented_bounding_box(hull_oriented_bounding_box)
    dimensions_sorting_list = dimensions_of_hull_bounding_box.argsort()[::-1]
    dimensions_of_hull_bounding_box = dimensions_of_hull_bounding_box[dimensions_sorting_list]


    hull_PCA_vectors_list = []

    for hull_PCA_eigenvector_index in range(0, len(hull_PCA_eigenvectors)):
        hull_positive_PCA_vector = hull_PCA_eigenvectors[:, hull_PCA_eigenvector_index] * hull_PCA_eigenvalues[hull_PCA_eigenvector_index]
        hull_positive_PCA_vector = normalize_vector(hull_positive_PCA_vector) * dimensions_of_hull_bounding_box[hull_PCA_eigenvector_index] / 2 * \
            const.PRINCIPAL_COMPONENTS_DIMENSION_FACTOR
        hull_negative_PCA_vector = - hull_positive_PCA_vector

        hull_PCA_vectors_list.append(hull_positive_PCA_vector)
        hull_PCA_vectors_list.append(hull_negative_PCA_vector)

    hull_neighbouring_points_list = hull_center_point + hull_PCA_vectors_list


    hull_sides_center_pointcloud = get_empty_pointcloud()

    for hull_neighbouring_point_index in range(0, len(hull_neighbouring_points_list)):
        (_, hull_some_KD_points_indices, _) = o3d.geometry.KDTreeFlann(hull_pointcloud).search_knn_vector_3d(hull_neighbouring_points_list[hull_neighbouring_point_index], 1)
        hull_some_side_center_pointcloud = hull_pointcloud.select_by_index(hull_some_KD_points_indices, invert=False)    
        hull_PCA_vector = hull_PCA_vectors_list[hull_neighbouring_point_index]
        hull_some_side_center_pointcloud.normals = o3d.utility.Vector3dVector([normalize_vector(hull_PCA_vector)])

        hull_sides_center_pointcloud = hull_sides_center_pointcloud + hull_some_side_center_pointcloud

    hull_sides_center_pointcloud.normalize_normals
    hull_sides_center_points_list = np.asarray(hull_sides_center_pointcloud.points)


    hull_sides_neighbouring_points_list = hull_sides_center_points_list + hull_PCA_vectors_list
    hull_sides_neighbouring_pointcloud = get_pointcloud_from_points_array(hull_sides_neighbouring_points_list)

    hull_sides_points_correspondences_list = [(0, hull_sides_neighbouring_point_index) for hull_sides_neighbouring_point_index in range(0, len(hull_sides_neighbouring_points_list))]
    hull_sides_points_lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(hull_center_pointcloud, hull_sides_neighbouring_pointcloud, \
        hull_sides_points_correspondences_list)


    return (hull_pointcloud, hull_oriented_bounding_box, hull_sides_points_lineset, hull_center_point, hull_sides_center_points_list, hull_PCA_vectors_list)



def process_sides_center_points_with_normals_and_get_handling_data(const, hull_center_point, hull_sides_center_points_list, hull_PCA_vectors_list):
    min_angle_from_hull_PCA_to_upwards_vertical_vector = np.pi / 2
    hull_sides_upwards_center_point_index = None

    for hull_PCA_vector_index in range(0, len(hull_PCA_vectors_list)):
        hull_PCA_vector = hull_PCA_vectors_list[hull_PCA_vector_index]
        upwards_vertical_vector = np.array([0, 0, 1])
        angle_from_hull_PCA_to_upwards_vertical_vector = np.arccos(np.dot(normalize_vector(hull_PCA_vector), normalize_vector(upwards_vertical_vector)))
        angle_from_hull_PCA_to_undesireable_horizontal_vector = np.arccos(np.dot(normalize_vector(hull_PCA_vector), \
            normalize_vector(const.HANDLING_TOOL_UNDESIREABLE_HORIZONTAL_VECTOR)))

        if (angle_from_hull_PCA_to_upwards_vertical_vector < min_angle_from_hull_PCA_to_upwards_vertical_vector) and \
            (angle_from_hull_PCA_to_undesireable_horizontal_vector >= const.HANDLING_TOOL_MIN_ALLOWED_UNDESIREABLE_HORIZONTAL_ANGLE):
            hull_sides_upwards_center_point_index = hull_PCA_vector_index
            min_angle_from_hull_PCA_to_upwards_vertical_vector = angle_from_hull_PCA_to_upwards_vertical_vector


    hull_PCA_upwards_vector = hull_PCA_vectors_list[hull_sides_upwards_center_point_index]
    hull_sides_upwards_center_point = hull_sides_center_points_list[hull_sides_upwards_center_point_index]

    hull_sides_upwards_center_dist = np.linalg.norm(hull_sides_upwards_center_point - hull_center_point)

    hull_PCA_downwards_vector = - hull_PCA_upwards_vector
    hull_sides_downwards_center_point_index = np.where((hull_PCA_vectors_list == hull_PCA_downwards_vector).all(axis=1))[0][0]

    hull_PCA_sidewards_vectors_list = np.delete(hull_PCA_vectors_list, np.append(hull_sides_upwards_center_point_index, hull_sides_downwards_center_point_index), axis=0)
    hull_sides_sidewards_center_points_list = np.delete(hull_sides_center_points_list, np.append(hull_sides_upwards_center_point_index, \
        hull_sides_downwards_center_point_index), axis=0)


    hull_sides_sidewards_center_points_first_pair = (hull_sides_sidewards_center_points_list[0], hull_sides_sidewards_center_points_list[1])
    hull_sides_sidewards_center_points_second_pair = (hull_sides_sidewards_center_points_list[2], hull_sides_sidewards_center_points_list[3])

    hull_sides_sidewards_first_pair_dist = np.linalg.norm(hull_sides_sidewards_center_points_first_pair[1] - hull_sides_sidewards_center_points_first_pair[0])
    hull_sides_sidewards_second_pair_dist = np.linalg.norm(hull_sides_sidewards_center_points_second_pair[1] - hull_sides_sidewards_center_points_second_pair[0])

    if hull_sides_sidewards_first_pair_dist < hull_sides_sidewards_second_pair_dist:
        hull_sides_sidewards_center_points_primary_pair = hull_sides_sidewards_center_points_first_pair
        hull_sides_sidewards_center_points_secondary_pair = hull_sides_sidewards_center_points_second_pair

        hull_sides_sidewards_primary_pair_dist = hull_sides_sidewards_first_pair_dist
        hull_sides_sidewards_secondary_pair_dist = hull_sides_sidewards_second_pair_dist
    else:
        hull_sides_sidewards_center_points_primary_pair = hull_sides_sidewards_center_points_second_pair
        hull_sides_sidewards_center_points_secondary_pair = hull_sides_sidewards_center_points_first_pair
        
        hull_sides_sidewards_primary_pair_dist = hull_sides_sidewards_second_pair_dist
        hull_sides_sidewards_secondary_pair_dist = hull_sides_sidewards_first_pair_dist


    hull_PCA_sidewards_original_primary_vector = hull_PCA_sidewards_vectors_list[np.where((hull_sides_sidewards_center_points_list == \
        hull_sides_sidewards_center_points_primary_pair[0]).all(axis=1))[0][0]]
    hull_PCA_sidewards_original_secondary_vector = hull_PCA_sidewards_vectors_list[np.where((hull_sides_sidewards_center_points_list == \
        hull_sides_sidewards_center_points_secondary_pair[0]).all(axis=1))[0][0]]

    hull_PCA_sidewards_alternative_primary_vector = - hull_PCA_sidewards_original_primary_vector
    hull_PCA_sidewards_alternative_secondary_vector = - hull_PCA_sidewards_original_secondary_vector

    tool_compensation_rotation_matrix = np.array([[math.cos(const.HANDLING_TOOL_COMPENSATION_ROTATION_ANGLE), - math.sin(const.HANDLING_TOOL_COMPENSATION_ROTATION_ANGLE), 0], \
        [math.sin(const.HANDLING_TOOL_COMPENSATION_ROTATION_ANGLE), math.cos(const.HANDLING_TOOL_COMPENSATION_ROTATION_ANGLE), 0], [0, 0, 1]], np.float32)
    robot_tool_horizontal_vector = np.dot(np.linalg.inv(tool_compensation_rotation_matrix), const.HANDLING_ROBOT_VECTOR_FOR_TOOL_X_VECTOR.T)

    if np.arccos(np.dot(normalize_vector(hull_PCA_sidewards_original_primary_vector), normalize_vector(robot_tool_horizontal_vector))) <= np.arccos(np.dot(normalize_vector(\
        hull_PCA_sidewards_alternative_primary_vector), normalize_vector(robot_tool_horizontal_vector))):
        hull_PCA_sidewards_primary_vector = hull_PCA_sidewards_original_primary_vector
    else:
        hull_PCA_sidewards_primary_vector = hull_PCA_sidewards_alternative_primary_vector

    if np.arccos(np.dot(normalize_vector(hull_PCA_sidewards_original_secondary_vector), normalize_vector(robot_tool_horizontal_vector))) <= np.arccos(np.dot(normalize_vector(\
        hull_PCA_sidewards_alternative_secondary_vector), normalize_vector(robot_tool_horizontal_vector))):
        hull_PCA_sidewards_secondary_vector = hull_PCA_sidewards_original_secondary_vector
    else:
        hull_PCA_sidewards_secondary_vector = hull_PCA_sidewards_alternative_secondary_vector


    tool_needed_primary_opened_gripper_width = hull_sides_sidewards_primary_pair_dist + const.HANDLING_OPENED_GRIPPER_WIDTH_OFFSET
    tool_needed_secondary_opened_gripper_width = hull_sides_sidewards_secondary_pair_dist + const.HANDLING_OPENED_GRIPPER_WIDTH_OFFSET

    if (tool_needed_primary_opened_gripper_width <= tool_needed_secondary_opened_gripper_width) and (tool_needed_primary_opened_gripper_width <= \
        const.HANDLING_OPENED_GRIPPER_TOTAL_WIDTH):
        hull_PCA_sidewards_optimal_vector = hull_PCA_sidewards_primary_vector
        tool_needed_opened_gripper_width = tool_needed_primary_opened_gripper_width
    elif (tool_needed_secondary_opened_gripper_width < tool_needed_primary_opened_gripper_width) and (tool_needed_secondary_opened_gripper_width <= \
        const.HANDLING_OPENED_GRIPPER_TOTAL_WIDTH):
        hull_PCA_sidewards_optimal_vector = hull_PCA_sidewards_secondary_vector
        tool_needed_opened_gripper_width = tool_needed_secondary_opened_gripper_width
    else:
        print("There is no any feasible handling orientation!\n")
        raise ValueError


    tool_approach_vector = normalize_vector(hull_PCA_upwards_vector)
    tool_normal_vector = normalize_vector(hull_PCA_sidewards_optimal_vector)
    tool_orientation_vector = normalize_vector(np.cross(tool_approach_vector, tool_normal_vector))

    tool_rotation_matrix = np.array([tool_normal_vector, tool_orientation_vector, tool_approach_vector]).T

    tool_partially_opened_gripper_height_offset = tool_needed_opened_gripper_width / const.HANDLING_OPENED_GRIPPER_TOTAL_WIDTH * const.HANDLING_OPENED_GRIPPER_HEIGHT_DELTA
    tool_partially_opened_gripper_height = const.HANDLING_CLOSED_GRIPPER_TOTAL_HEIGHT - tool_partially_opened_gripper_height_offset
    tool_needed_gripper_height = hull_sides_upwards_center_dist + const.HANDLING_CLOSED_GRIPPER_HEIGHT_OFFSET

    if tool_needed_gripper_height >= tool_partially_opened_gripper_height:
        tool_target_point = hull_center_point + tool_approach_vector * (hull_sides_upwards_center_dist * const.HANDLING_UPWARDS_DIMENSION_SCALE_OFFSET + \
            tool_needed_gripper_height - tool_partially_opened_gripper_height)
    else:
        tool_target_point = hull_center_point + tool_approach_vector * hull_sides_upwards_center_dist * const.HANDLING_UPWARDS_DIMENSION_SCALE_OFFSET

    tool_target_approach_point = tool_target_point + tool_approach_vector * const.HANDLING_TOOL_APPROACH_OFFSET

    compensated_tool_rotation_matrix = np.dot(tool_rotation_matrix, tool_compensation_rotation_matrix)

    (tool_rotation_rotvec, _) = cv2.Rodrigues(compensated_tool_rotation_matrix)
    tool_rotation_rotvec = tool_rotation_rotvec[:, 0]


    tool_target_approach_coordinate_system_arrows = get_translated_and_rotated_smaller_coordinate_system_arrows(translation_vector=\
        tool_target_approach_point, rotation_matrix=tool_rotation_matrix)
    tool_target_coordinate_system_arrows = get_translated_and_rotated_smaller_coordinate_system_arrows(translation_vector=tool_target_point, rotation_matrix=\
        tool_rotation_matrix)


    return (tool_target_approach_point, tool_target_point, tool_needed_opened_gripper_width, tool_partially_opened_gripper_height, tool_rotation_matrix, \
        tool_rotation_rotvec, tool_target_coordinate_system_arrows, tool_target_approach_coordinate_system_arrows)



def create_tool_and_ground_level_visualization_and_get_their_geometry(const, tool_rotation_matrix, tool_target_point, tool_needed_opened_gripper_width, \
    tool_partially_opened_gripper_height):
    tool_left_jaw_mesh = get_translated_box_mesh(const.VISUALIZATION_JAW_WIDTH, const.VISUALIZATION_JAW_DEPTH, tool_partially_opened_gripper_height, \
        [-tool_needed_opened_gripper_width/2-const.VISUALIZATION_JAW_WIDTH, -const.VISUALIZATION_JAW_DEPTH/2, 0], const.VISUALIZATION_BLACK_COLOR_TUPLE)
    tool_right_jaw_mesh = get_translated_box_mesh(const.VISUALIZATION_JAW_WIDTH, const.VISUALIZATION_JAW_DEPTH, tool_partially_opened_gripper_height, \
        [tool_needed_opened_gripper_width/2, -const.VISUALIZATION_JAW_DEPTH/2, 0], const.VISUALIZATION_BLACK_COLOR_TUPLE)
    tool_jaws_base_mesh = get_translated_box_mesh(const.VISUALIZATION_JAWS_BASE_WIDTH, const.VISUALIZATION_JAWS_BASE_DEPTH, \
        const.VISUALIZATION_JAWS_BASE_HEIGHT, [-const.VISUALIZATION_JAWS_BASE_WIDTH/2, -const.VISUALIZATION_JAWS_BASE_DEPTH/2, \
        tool_partially_opened_gripper_height], const.VISUALIZATION_BLACK_COLOR_TUPLE)
    tool_support_mesh = get_translated_cylinder_mesh(const.VISUALIZATION_SUPPORT_RADIUS, const.VISUALIZATION_SUPPORT_HEIGHT, [0, 0, \
        tool_partially_opened_gripper_height+const.VISUALIZATION_JAWS_BASE_HEIGHT+const.VISUALIZATION_SUPPORT_HEIGHT/2], const.VISUALIZATION_BLACK_COLOR_TUPLE)
    tool_camera_base_mesh = get_translated_box_mesh(const.VISUALIZATION_CAMERA_BASE_WIDTH, const.VISUALIZATION_CAMERA_BASE_DEPTH, \
        const.VISUALIZATION_CAMERA_BASE_HEIGHT, [-const.VISUALIZATION_CAMERA_BASE_WIDTH/2, const.VISUALIZATION_SUPPORT_RADIUS, tool_partially_opened_gripper_height+\
        const.VISUALIZATION_JAWS_BASE_HEIGHT+const.VISUALIZATION_SUPPORT_HEIGHT/4], const.VISUALIZATION_BLACK_COLOR_TUPLE)
    tool_camera_mesh = get_translated_box_mesh(const.VISUALIZATION_CAMERA_WIDTH, const.VISUALIZATION_CAMERA_DEPTH, const.VISUALIZATION_CAMERA_HEIGHT, \
        [-const.VISUALIZATION_CAMERA_WIDTH/2, const.VISUALIZATION_SUPPORT_RADIUS+const.VISUALIZATION_CAMERA_BASE_DEPTH, tool_partially_opened_gripper_height+\
        const.VISUALIZATION_JAWS_BASE_HEIGHT+const.VISUALIZATION_SUPPORT_HEIGHT/4], const.VISUALIZATION_BLUE_COLOR_TUPLE)

    tool_mesh = tool_left_jaw_mesh + tool_right_jaw_mesh + tool_jaws_base_mesh + tool_support_mesh + tool_camera_base_mesh + tool_camera_mesh


    ground_level_mesh = get_translated_box_mesh(const.VISUALIZATION_GROUND_LEVEL_WIDTH, const.VISUALIZATION_GROUND_LEVEL_DEPTH, const.VISUALIZATION_GROUND_LEVEL_HEIGHT, \
        [-const.VISUALIZATION_GROUND_LEVEL_WIDTH/2, -const.VISUALIZATION_GROUND_LEVEL_DEPTH/4, -const.VISUALIZATION_GROUND_LEVEL_HEIGHT + const.GROUND_LEVEL_MIN_HEIGHT], \
            const.VISUALIZATION_GRAY_COLOR_TUPLE)
    

    tool_mesh_transformation_matrix = np.identity(4)
    tool_mesh_transformation_matrix[0:3, 0:3] = tool_rotation_matrix
    tool_mesh_transformation_matrix[0:3, 3] = tool_target_point
    tool_mesh.transform(tool_mesh_transformation_matrix)


    return (tool_mesh, ground_level_mesh)



def send_handling_data_to_robot(ur_communication_object, tool_target_approach_point, tool_target_point, tool_rotation_rotvec, tool_needed_opened_gripper_width, server_port=\
    None):
    try:
        ur_communication_object.start_pc_server_with_accepting_robot_connection_and_connect_pc_with_robot_server(server_port)
        ur_communication_object.send_string_data(const.COMMUNICATION_HANDLE_STRING)

        while True:
            try:
                time.sleep(const.COMMUNICATION_RECEIVING_LOOP_TIME)
                received_string = ur_communication_object.receive_string_data()

                if received_string == const.COMMUNICATION_ROBOT_READY_STRING:
                    ur_communication_object.send_pose_data(tool_target_approach_point, tool_rotation_rotvec)
                    break
            except:
                pass

        while True:
            try:
                time.sleep(const.COMMUNICATION_RECEIVING_LOOP_TIME)
                received_string = ur_communication_object.receive_string_data()

                if received_string == const.COMMUNICATION_ROBOT_READY_STRING:
                    ur_communication_object.send_pose_data(tool_target_point, tool_rotation_rotvec)
                    break
            except:
                pass

        while True:
            try:
                time.sleep(const.COMMUNICATION_RECEIVING_LOOP_TIME)
                received_string = ur_communication_object.receive_string_data()

                if received_string == const.COMMUNICATION_ROBOT_READY_STRING:
                    ur_communication_object.send_float_data(tool_needed_opened_gripper_width)
                    break
            except:
                pass

        while True:
            try:
                time.sleep(const.COMMUNICATION_RECEIVING_LOOP_TIME)
                received_string = ur_communication_object.receive_string_data()

                if received_string == const.COMMUNICATION_FINISHED_STRING:
                    break
            except:
                pass
            
        ur_communication_object.shutdown_pc_server_and_close_pc_connection_with_robot_server()

    except:
        print("Universal Robot client is not connected!\n")
        raise ValueError