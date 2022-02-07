import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import copy
import csv

from RS_and_3D_OD_constants import Constants

PROJECT_FOLDER_ABS_PATH = os.path.dirname(os.path.abspath("__file__")).replace(os.sep, "/")

const = Constants(project_folder_abs_path=PROJECT_FOLDER_ABS_PATH, reconstruct_pointclouds_instead_of_calibrating_hand_eye=True)



def get_depth_at_image_coordinates_from_depth_array(depth_array, image_coordinate_x, image_coordinate_y, depth_resolution_in_meters):
    # depth_array is indexed first with image_coordinate_y and then with image_coordinate_x
    
    depth_at_coordinates_in_meters = depth_array[image_coordinate_y, image_coordinate_x] * depth_resolution_in_meters

    return depth_at_coordinates_in_meters



def normalize_vector(vector):
    return vector / np.linalg.norm(vector)



def get_translated_and_rotated_smaller_coordinate_system_arrows(translation_vector=np.array([0, 0, 0], np.float32), rotation_matrix=np.array([[1, 0, 0], [0, 1, 0], \
    [0, 0, 1]], np.float32)):
    coordinate_system_arrows = o3d.geometry.TriangleMesh.create_coordinate_frame(size=const.VISUALIZATION_COORD_SYSTEM_ARROWS_SMALLER_SIZE, origin=translation_vector)
    coordinate_system_arrows.rotate(rotation_matrix)

    return coordinate_system_arrows



def get_translated_and_rotated_larger_coordinate_system_arrows(translation_vector=np.array([0, 0, 0], np.float32), rotation_matrix=np.array([[1, 0, 0], [0, 1, 0], \
    [0, 0, 1]], np.float32)):
    coordinate_system_arrows = o3d.geometry.TriangleMesh.create_coordinate_frame(size=const.VISUALIZATION_COORD_SYSTEM_ARROWS_LARGER_SIZE, origin=translation_vector)
    coordinate_system_arrows.rotate(rotation_matrix)

    return coordinate_system_arrows



def visualize_geometries(geometries_list, visualize_coordinate_system=False, view_from_camera_instead_of_robot_base=False):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(visible=True, window_name=const.VISUALIZATION_WINDOW_NAME, width=const.VISUALIZATION_WINDOW_WIDTH, height=const.VISUALIZATION_WINDOW_HEIGHT, \
        left=const.VISUALIZATION_WINDOW_STARTING_LEFT, top=const.VISUALIZATION_WINDOW_STARTING_TOP)

    for geometry in geometries_list:
        visualizer.add_geometry(geometry)

    if visualize_coordinate_system == True:
        coordinate_system_arrows = get_translated_and_rotated_larger_coordinate_system_arrows()
        visualizer.add_geometry(coordinate_system_arrows)

    if view_from_camera_instead_of_robot_base == True:
        visualizer.get_view_control().set_up((0, -1, 0))  # defines which axis is directed upwards
        visualizer.get_view_control().set_front((0, 0, -1))  # defines which axis is directed outwards
    else:
        visualizer.get_view_control().set_up((0, 0, 1))  # defines which axis is directed upwards
        visualizer.get_view_control().set_front((1, 1, 1))  # defines which axis is directed outwards

    visualizer.run()
    visualizer.destroy_window()



def save_geometries_image(geometries_list, saving_image_abs_path, zoom_factor, up_view_vector, front_view_vector, translation_vector):
    if ".jpg" not in saving_image_abs_path:
        saving_image_abs_path = saving_image_abs_path + ".jpg"
    
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(visible=False, window_name=const.VISUALIZATION_WINDOW_NAME, width=const.VISUALIZATION_WINDOW_WIDTH, height=const.VISUALIZATION_WINDOW_HEIGHT, \
        left=const.VISUALIZATION_WINDOW_STARTING_LEFT, top=const.VISUALIZATION_WINDOW_STARTING_TOP)

    for geometry in geometries_list:
        visualizer.add_geometry(geometry)

    visualizer.get_view_control().set_zoom(zoom_factor)
    visualizer.get_view_control().set_up(up_view_vector)  # defines which axis is directed upwards
    visualizer.get_view_control().set_front(front_view_vector)  # defines which axis is directed outwards
    visualizer.get_view_control().camera_local_translate(*translation_vector)
    visualizer.poll_events()
    visualizer.update_renderer()
    visualizer.capture_screen_image(saving_image_abs_path)
    visualizer.destroy_window()



def get_random_color_tuple():
    return (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))



def save_pointcloud(pointcloud, pointcloud_file_abs_path):
    if ".ply" not in pointcloud_file_abs_path:
        pointcloud_file_abs_path = pointcloud_file_abs_path + ".ply"
    
    o3d.io.write_point_cloud(pointcloud_file_abs_path, pointcloud)



def load_pointcloud(pointcloud_file_abs_path):
    if ".ply" not in pointcloud_file_abs_path:
        pointcloud_file_abs_path = pointcloud_file_abs_path + ".ply"
    
    pointcloud = o3d.io.read_point_cloud(pointcloud_file_abs_path)

    return pointcloud



def save_mesh(mesh, mesh_file_abs_path):
    if ".ply" not in mesh_file_abs_path:
        mesh_file_abs_path = mesh_file_abs_path + ".ply"
    
    o3d.io.write_triangle_mesh(mesh_file_abs_path, mesh)



def load_mesh(mesh_file_abs_path):
    if ".ply" not in mesh_file_abs_path:
        mesh_file_abs_path = mesh_file_abs_path + ".ply"
    
    mesh = o3d.io.read_triangle_mesh(mesh_file_abs_path)

    return mesh



def print_pointcloud_specs_and_data(pointcloud):
    print("{}\n".format(pointcloud))
    print("Pointcloud points data: \n{}\n".format(np.asarray(pointcloud.points)))
    print("Pointcloud normals data: \n{}\n".format(np.asarray(pointcloud.normals)))
    print("Pointcloud colors data: \n{}\n".format(np.asarray(pointcloud.colors)))



def get_empty_pointcloud():
    return o3d.geometry.PointCloud()



def get_copy_of_pointcloud(pointcloud):
    return copy.deepcopy(pointcloud)



def get_pointcloud_center_point_array(pointcloud):
    return pointcloud.get_center()



def get_pointcloud_from_points_array(pointcloud_points_array):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pointcloud_points_array)

    return pointcloud



def get_translated_cylinder_mesh(radius, height, translation_vector, color_tuple):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    mesh.compute_vertex_normals()
    mesh.translate(np.array(translation_vector, np.float32))
    mesh = get_painted_pointcloud(mesh, color_tuple)

    return mesh



def get_translated_box_mesh(width, depth, height, translation_vector, color_tuple):
    mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=depth, depth=height)
    mesh.compute_vertex_normals()
    mesh.translate(np.array(translation_vector, np.float32))
    mesh = get_painted_pointcloud(mesh, color_tuple)
    
    return mesh



def get_prepared_pointcloud(pointcloud, pointcloud_transformation_matrix):
    pointcloud = pointcloud.voxel_down_sample(voxel_size=const.DOWNSAMPLE_VOXEL_SIZE)
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=const.NORMALS_ESTIMATION_SEARCH_RADIUS, max_nn=\
        const.NORMALS_ESTIMATION_MAX_NUM_OF_NEIGHBOURS))
    pointcloud.normalize_normals()
    pointcloud.orient_normals_towards_camera_location()
    pointcloud.transform(pointcloud_transformation_matrix)
    (pointcloud, _) = pointcloud.remove_statistical_outlier(nb_neighbors=const.OUTLIERS_REMOVAL_NUM_OF_NEIGHBOURS, std_ratio=const.OUTLIERS_REMOVAL_STD_DEV_RATIO)

    return pointcloud



def get_noisy_pointcloud(pointcloud, noise_mean, noise_std_dev):
    pointcloud = get_copy_of_pointcloud(pointcloud)
    pointcloud_points_array = np.asarray(pointcloud.points)
    pointcloud_points_array = pointcloud_points_array + np.random.normal(noise_mean, noise_std_dev, size=pointcloud_points_array.shape)
    pointcloud.points = o3d.utility.Vector3dVector(pointcloud_points_array)
    
    return pointcloud



def get_painted_pointcloud(pointcloud, tuple_of_rgb_value_in_percentage=(None, None, None)):
    pointcloud = get_copy_of_pointcloud(pointcloud)
    if (tuple_of_rgb_value_in_percentage) != (None, None, None):
        pointcloud.paint_uniform_color(tuple_of_rgb_value_in_percentage)
    else:
        pointcloud.paint_uniform_color(get_random_color_tuple())

    return pointcloud



def get_convex_hull_pointcloud(pointcloud):
    pointcloud = get_copy_of_pointcloud(pointcloud)
    (triangle_mesh, _) = pointcloud.compute_convex_hull()
    triangle_mesh.orient_triangles()
    pointcloud = triangle_mesh.sample_points_poisson_disk(number_of_points=const.POISSON_SAMPLE_NUM_OF_POINTS, use_triangle_normal=True)
    
    return pointcloud



def get_pointcloud_cropped_with_bounding_box(pointcloud, oriented_bounding_box):
    pointcloud = pointcloud.crop(oriented_bounding_box)

    return pointcloud



def get_oriented_bounding_box_from_pointcloud(pointcloud, tuple_of_rgb_value_in_percentage=(None, None, None)):
    oriented_bounding_box = pointcloud.get_oriented_bounding_box()
    if (tuple_of_rgb_value_in_percentage) != (None, None, None):
        oriented_bounding_box.color = tuple_of_rgb_value_in_percentage
    else:
        oriented_bounding_box.color = (0, 0, 0)

    return oriented_bounding_box



def get_oriented_bounding_box_from_points_array(pointcloud_points_array):
    pointcloud_points = o3d.utility.Vector3dVector(pointcloud_points_array)
    oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(pointcloud_points)
    oriented_bounding_box.color = (0, 0, 0)
    
    return oriented_bounding_box



def get_dimensions_of_oriented_bounding_box(bounding_box):
    return bounding_box.extent



def get_diagonal_distance_of_bounding_box(bounding_box):
    bounding_box_diagonal_distance = 0
    bounding_box_array = np.asarray(bounding_box.get_box_points())
    
    for first_index in range(0, len(bounding_box_array)):
        for second_index in range(first_index, len(bounding_box_array)):
            bounding_box_some_distance = math.sqrt((bounding_box_array[first_index][0] - bounding_box_array[second_index][0]) ** 2 + \
                (bounding_box_array[first_index][1] - bounding_box_array[second_index][1]) ** 2 + \
                (bounding_box_array[first_index][2] - bounding_box_array[second_index][2]) ** 2)
            if bounding_box_some_distance > bounding_box_diagonal_distance:
                bounding_box_diagonal_distance = bounding_box_some_distance

    return bounding_box_diagonal_distance



def get_segment_of_pointcloud_close_to_other(pointcloud, other_pointcloud):
    distance_between_pointclouds = np.asarray(pointcloud.compute_point_cloud_distance(other_pointcloud))
    pointcloud = pointcloud.select_by_index(np.where(distance_between_pointclouds <= const.MAX_DIST_BETWEEN_POINTCLOUDS_FOR_CLOSENESS)[0])

    return pointcloud



def get_segment_of_pointcloud_far_from_other(pointcloud, other_pointcloud):
    distance_between_pointclouds = np.asarray(pointcloud.compute_point_cloud_distance(other_pointcloud))
    pointcloud = pointcloud.select_by_index(np.where(distance_between_pointclouds >= const.MIN_DIST_BETWEEN_POINTCLOUDS_FOR_FARNESS)[0])

    return pointcloud



def get_largest_plane_of_pointcloud(pointcloud):
    (_, points_of_largest_plane_indices) = pointcloud.segment_plane(distance_threshold=const.RANSAC_MAX_DISTANCE_FROM_PLANE, ransac_n=\
        const.RANSAC_NUM_OF_SAMPLED_POINTS_IN_PLANE, num_iterations=const.RANSAC_NUM_OF_ITERATIONS)
    pointcloud = pointcloud.select_by_index(points_of_largest_plane_indices, invert=False)

    return pointcloud



def get_everything_but_largest_plane_of_pointcloud(pointcloud):
    (_, points_of_largest_plane_indices) = pointcloud.segment_plane(distance_threshold=const.RANSAC_MAX_DISTANCE_FROM_PLANE, ransac_n=\
        const.RANSAC_NUM_OF_SAMPLED_POINTS_IN_PLANE, num_iterations=const.RANSAC_NUM_OF_ITERATIONS)
    pointcloud = pointcloud.select_by_index(points_of_largest_plane_indices, invert=True)

    return pointcloud



def get_everything_over_ground_level_of_pointcloud(pointcloud):
    pointcloud_points = np.asarray(pointcloud.points)
    pointcloud = pointcloud.select_by_index(np.where(pointcloud_points[:, 2] >= const.GROUND_LEVEL_MIN_HEIGHT)[0], invert=False)

    return pointcloud



def get_clustered_pointcloud(pointcloud, filter_clustered_pointcloud_by_noise=True, filter_clustered_pointcloud_by_diag_dist=False, paint_clustered_pointcloud=False):
    cluster_labels_of_pointcloud = np.array(pointcloud.cluster_dbscan(eps=const.CLUSTERING_DISTANCE_TO_NEIGHBOURS, min_points=const.CLUSTERING_NUM_OF_MIN_POINTS))

    num_of_cluster_labels = len(np.unique(cluster_labels_of_pointcloud))
    print("Original pointcloud has {} clusters with noise.".format(num_of_cluster_labels))

    if filter_clustered_pointcloud_by_noise == True:
        pointcloud = pointcloud.select_by_index(np.where(cluster_labels_of_pointcloud != -1)[0], invert=False)
        cluster_labels_of_pointcloud = np.delete(cluster_labels_of_pointcloud, np.where(cluster_labels_of_pointcloud == -1)[0])

    if filter_clustered_pointcloud_by_diag_dist == True:
        num_of_cluster_labels = len(np.unique(cluster_labels_of_pointcloud))

        for cluster_label in range(0, num_of_cluster_labels):
            pointcloud_with_some_label = pointcloud.select_by_index(np.where(cluster_labels_of_pointcloud == cluster_label)[0])

            try:
                if np.asarray(pointcloud_with_some_label.points).shape[0] > 4:  # four points are needed for constructing the bounding box
                    oriented_bounding_box_with_some_label = pointcloud_with_some_label.get_oriented_bounding_box()
                    oriented_bounding_box_with_some_label_diag_dist = get_diagonal_distance_of_bounding_box(oriented_bounding_box_with_some_label)

                    if (oriented_bounding_box_with_some_label_diag_dist < const.CLUSTERING_MIN_DIAG_DIST_FOR_FILTERING) or \
                        (oriented_bounding_box_with_some_label_diag_dist > const.CLUSTERING_MAX_DIAG_DIST_FOR_FILTERING):
                        pointcloud = pointcloud.select_by_index(np.where(cluster_labels_of_pointcloud != cluster_label)[0], invert=False)
                        cluster_labels_of_pointcloud = np.delete(cluster_labels_of_pointcloud, np.where(cluster_labels_of_pointcloud == cluster_label)[0])
            except:
                pass

        num_of_cluster_labels = len(np.unique(cluster_labels_of_pointcloud))
        print("Filtered pointcloud has {} clusters.\n".format(num_of_cluster_labels))
    else:
        print("Original pointcloud is not filtered.\n")

    if paint_clustered_pointcloud == True:
        num_of_cluster_labels = len(np.unique(cluster_labels_of_pointcloud))

        cluster_colors_of_pointcloud = plt.get_cmap("gist_rainbow")(cluster_labels_of_pointcloud / (num_of_cluster_labels if num_of_cluster_labels > 0 else 1))
        cluster_colors_of_pointcloud[cluster_labels_of_pointcloud == -1] = 0
        pointcloud.colors = o3d.utility.Vector3dVector(cluster_colors_of_pointcloud[:, 0:3])
    
    return pointcloud



def get_pointcloud_of_keypoints(pointcloud):
    pointcloud = o3d.geometry.keypoint.compute_iss_keypoints(pointcloud)
    
    return pointcloud



def get_pointclouds_correspondence_result(fixed_pointcloud, floating_pointcloud):
    correspondence_result = o3d.pipelines.registration.evaluate_registration(floating_pointcloud, fixed_pointcloud, const.REGISTRATION_MAX_THRESHOLD_DIST, np.identity(4))
    (correspondence_fitness, correspondence_inlier_rmse) = (correspondence_result.fitness, correspondence_result.inlier_rmse)

    return (correspondence_fitness, correspondence_inlier_rmse)



def get_floating_pointcloud_registered_by_generalized_method(fixed_pointcloud, floating_pointcloud):
    fixed_pointcloud_of_keypoints = get_pointcloud_of_keypoints(fixed_pointcloud)
    floating_pointcloud_of_keypoints = get_pointcloud_of_keypoints(floating_pointcloud)

    point_to_plane_registration = o3d.pipelines.registration.registration_generalized_icp(floating_pointcloud_of_keypoints, fixed_pointcloud_of_keypoints, \
        const.REGISTRATION_MAX_THRESHOLD_DIST, np.identity(4), o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(), \
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=const.REGISTRATION_MAX_NUM_OF_ITERATIONS))
    registration_transformation_matrix = point_to_plane_registration.transformation

    floating_pointcloud = get_copy_of_pointcloud(floating_pointcloud)
    floating_pointcloud.transform(registration_transformation_matrix)

    return floating_pointcloud



def get_floating_pointcloud_registered_by_color_method(fixed_pointcloud, floating_pointcloud):
    # pointclouds registration by color can't be used because it doesn't work if registration max threshold distance is small (it generally uses only some corresponding
    # points) and it sometimes randomly gives wrong results, although it is usually more precise tran pointclouds registration by point to plane

    fixed_pointcloud_of_keypoints = get_pointcloud_of_keypoints(fixed_pointcloud)
    floating_pointcloud_of_keypoints = get_pointcloud_of_keypoints(floating_pointcloud)
    
    color_registration = o3d.pipelines.registration.registration_colored_icp(floating_pointcloud_of_keypoints, fixed_pointcloud_of_keypoints, \
        const.REGISTRATION_MAX_THRESHOLD_DIST, np.identity(4), o3d.pipelines.registration.TransformationEstimationForColoredICP(), \
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=const.REGISTRATION_MAX_NUM_OF_ITERATIONS))
    registration_transformation_matrix = color_registration.transformation

    floating_pointcloud = get_copy_of_pointcloud(floating_pointcloud)
    floating_pointcloud.transform(registration_transformation_matrix)

    return floating_pointcloud



def get_prepared_combined_pointcloud(pointcloud):
    # function get_prepared_pointcloud(pointcloud) can't be performed on combined pointcloud because it would mess it's normals
    
    pointcloud = pointcloud.voxel_down_sample(voxel_size=const.DOWNSAMPLE_VOXEL_SIZE)
    (pointcloud, _) = pointcloud.remove_statistical_outlier(nb_neighbors=const.OUTLIERS_REMOVAL_NUM_OF_NEIGHBOURS, std_ratio=const.OUTLIERS_REMOVAL_STD_DEV_RATIO)

    return pointcloud



def get_pointcloud_principal_components(pointcloud):
    pointcloud_points = np.asarray(pointcloud.points)
    pointcloud_points_mean = np.mean(pointcloud_points, axis=0)
    pointcloud_points_offset = pointcloud_points - pointcloud_points_mean

    pointcloud_points_covariance_matrix = np.cov(pointcloud_points_offset.T)
    (pointcloud_points_eigenvalues, pointcloud_points_eigenvectors) = np.linalg.eig(pointcloud_points_covariance_matrix)

    eigenvalues_sorting_list = pointcloud_points_eigenvalues.argsort()[::-1]
    pointcloud_points_eigenvalues = pointcloud_points_eigenvalues[eigenvalues_sorting_list]
    pointcloud_points_eigenvectors = pointcloud_points_eigenvectors[:, eigenvalues_sorting_list]

    return (pointcloud_points_eigenvalues, pointcloud_points_eigenvectors)



def get_transformed_points_array(points_array, pointcloud_transformation_matrix):
    transformed_points_array = np.zeros_like(points_array)

    for (point_index, point) in enumerate(points_array):
        point = np.append(point, 1)
        transformed_point = np.dot(pointcloud_transformation_matrix, point)
        transformed_points_array[point_index] = transformed_point[0:3]

    return transformed_points_array



def get_world_coordinates_from_image_and_world_depth_coordinates(image_x, image_y, world_z, depth_offset, rs_camera_focal_length_x, rs_camera_focal_length_y):
    (image_width, image_height) = (const.CAM_FRAME_SIZE[0], const.CAM_FRAME_SIZE[1])
    
    image_x_from_center = image_x - image_width // 2
    image_y_from_center = image_y - image_height // 2

    world_x = image_x_from_center * world_z / rs_camera_focal_length_x
    world_y = image_y_from_center * world_z / rs_camera_focal_length_y
    world_z = world_z + depth_offset

    return [world_x, world_y, world_z]



def get_median_depth_at_image_region_from_depth_array(depth_array, image_center_x, image_center_y, image_width, image_height, image_scale_factor, default_world_z, depth_resolution_in_meters):
    (image_starting_x, image_starting_y) = (image_center_x - int(image_width // 2 * image_scale_factor), image_center_y - int(image_height // 2 * image_scale_factor))
    (image_ending_x, image_ending_y) = (image_center_x + int(image_width // 2 * image_scale_factor), image_center_y + int(image_height // 2 * image_scale_factor))
    
    (step_for_image_x, step_for_image_y) = ((image_ending_x - image_starting_x) // const.CROPPING_REGION_NUM_OF_INTERVALS, (image_ending_y - image_starting_y) // \
        const.CROPPING_REGION_NUM_OF_INTERVALS)

    world_z_list = []

    for image_x in range(image_starting_x, image_ending_x + step_for_image_x, step_for_image_x):
            for image_y in range(image_starting_y, image_ending_y + step_for_image_y, step_for_image_y):
                world_z = get_depth_at_image_coordinates_from_depth_array(depth_array, image_x, image_y, depth_resolution_in_meters)
                if world_z > 0:
                    world_z_list.append(world_z)
    
    if len(world_z_list) > 0:
        world_z_median = float(np.median(np.array(world_z_list)))
        if world_z_median >= const.DEPTH_MIN_THRESHOLD_DIST_IN_METERS:
            world_z = world_z_median
        else:
            world_z = const.DEPTH_MIN_THRESHOLD_DIST_IN_METERS
    else:
        world_z = default_world_z

    return world_z



def get_detected_data_from_detected_labels_and_depth_array(detected_labels, depth_array, rs_depth_resolution_in_meters, rs_camera_focal_length_x, rs_camera_focal_length_y, \
    ur_calibration_object, hand_eye_calibration_object, transformation_matrix_object, file_name_index):
    detected_data = []

    for detected_labels_row in detected_labels:
        detected_class_index = int(detected_labels_row[0])
        (image_width, image_height) = (const.CAM_FRAME_SIZE[1], const.CAM_FRAME_SIZE[0])
        (detected_image_center_x, detected_image_center_y) = (int(detected_labels_row[1]), int(detected_labels_row[2]))
        (detected_image_width, detected_image_height) = (int(detected_labels_row[3]), int(detected_labels_row[4]))

        if (detected_image_width >= const.CROPPING_MIN_PERCENTAGE_OF_DETECTED_DIMENSION * image_width) and (detected_image_height >= \
            const.CROPPING_MIN_PERCENTAGE_OF_DETECTED_DIMENSION * image_height):
            foreground_world_z = get_median_depth_at_image_region_from_depth_array(depth_array, detected_image_center_x, detected_image_center_y, detected_image_width, \
                detected_image_height, const.CROPPING_FOREGROUND_SCALE_FACTOR, const.DEPTH_MIN_THRESHOLD_DIST_IN_METERS, rs_depth_resolution_in_meters)

            (detected_image_starting_x, detected_image_starting_y) = (detected_image_center_x - int(detected_image_width // 2 * const.CROPPING_BORDER_SCALE_FACTOR), \
                detected_image_center_y - int(detected_image_height // 2 * const.CROPPING_BORDER_SCALE_FACTOR))
            (detected_image_ending_x, detected_image_ending_y) = (detected_image_center_x + int(detected_image_width // 2 * const.CROPPING_BORDER_SCALE_FACTOR), \
                detected_image_center_y + int(detected_image_height // 2 * const.CROPPING_BORDER_SCALE_FACTOR))

            (lower_left_detected_image_x, lower_left_detected_image_y) = (detected_image_starting_x, detected_image_starting_y)
            (lower_right_detected_image_x, lower_right_detected_image_y) = (detected_image_ending_x, detected_image_starting_y)
            (upper_left_detected_image_x, upper_left_detected_image_y) = (detected_image_starting_x, detected_image_ending_y)
            (upper_right_detected_image_x, upper_right_detected_image_y) = (detected_image_ending_x, detected_image_ending_y)

            lower_left_foreground_point = get_world_coordinates_from_image_and_world_depth_coordinates(lower_left_detected_image_x, lower_left_detected_image_y, \
                foreground_world_z, - const.CROPPING_OFFSET_FROM_DETECTED_DEPTH, rs_camera_focal_length_x, rs_camera_focal_length_y)
            lower_right_foreground_point = get_world_coordinates_from_image_and_world_depth_coordinates(lower_right_detected_image_x, lower_right_detected_image_y, \
                foreground_world_z, - const.CROPPING_OFFSET_FROM_DETECTED_DEPTH, rs_camera_focal_length_x, rs_camera_focal_length_y)
            upper_left_foreground_point = get_world_coordinates_from_image_and_world_depth_coordinates(upper_left_detected_image_x, upper_left_detected_image_y, \
                foreground_world_z, - const.CROPPING_OFFSET_FROM_DETECTED_DEPTH, rs_camera_focal_length_x, rs_camera_focal_length_y)
            upper_right_foreground_point = get_world_coordinates_from_image_and_world_depth_coordinates(upper_right_detected_image_x, upper_right_detected_image_y, \
                foreground_world_z, - const.CROPPING_OFFSET_FROM_DETECTED_DEPTH, rs_camera_focal_length_x, rs_camera_focal_length_y)

            pointcloud_transformation_matrix = transformation_matrix_object.get_pointcloud_transformation_matrix(ur_calibration_object, hand_eye_calibration_object, \
                file_name_index)

            foreground_points_array = np.array([lower_left_foreground_point, lower_right_foreground_point, upper_left_foreground_point, upper_right_foreground_point])
            transformed_foreground_points_array = get_transformed_points_array(foreground_points_array, pointcloud_transformation_matrix)

            transformed_foreground_pointcloud = get_pointcloud_from_points_array(transformed_foreground_points_array)

            (transformed_foreground_plane_model, transformed_foreground_points_of_largest_plane_indices) = transformed_foreground_pointcloud.segment_plane(\
                distance_threshold=const.RANSAC_MAX_DISTANCE_FROM_PLANE, ransac_n=const.RANSAC_NUM_OF_SAMPLED_POINTS_IN_PLANE, num_iterations=const.RANSAC_NUM_OF_ITERATIONS)
            transformed_foreground_pointcloud = transformed_foreground_pointcloud.select_by_index(transformed_foreground_points_of_largest_plane_indices, invert=False)
            [transformed_foreground_normal_x, transformed_foreground_normal_y, transformed_foreground_normal_z, _] = transformed_foreground_plane_model

            transformed_foreground_normals_array = np.array([[transformed_foreground_normal_x, transformed_foreground_normal_y, transformed_foreground_normal_z]])

            transformed_background_points_array = transformed_foreground_points_array - transformed_foreground_normals_array * (const.CROPPING_MAX_DETECTED_DIMENSION + \
                const.CROPPING_OFFSET_FROM_DETECTED_DEPTH)

            transformed_detected_points_array = np.concatenate([transformed_foreground_points_array, transformed_background_points_array])
            transformed_center_detected_point_array = np.mean(transformed_detected_points_array, axis=0, keepdims=True)

            detected_data_row = [detected_class_index, transformed_center_detected_point_array, transformed_detected_points_array]
            detected_data.append(detected_data_row)

    return detected_data



def get_filtered_detected_data_for_selected_class_index(selected_class_index, rs_streaming_object, ur_calibration_object, hand_eye_calibration_object, \
    transformation_matrix_object):
    rs_depth_resolution_in_meters = rs_streaming_object.get_depth_resolution_in_meters()
    
    rs_camera_intrinsics_matrix = rs_streaming_object.get_camera_intrinsics_matrix()
    rs_camera_focal_length_x = rs_camera_intrinsics_matrix[0, 0]
    rs_camera_focal_length_y = rs_camera_intrinsics_matrix[1, 1]

    total_detected_data = []

    for (file_name_index, color_image_file_name) in enumerate(sorted(os.listdir(const.COLOR_IMAGES_FOLDER_ABS_PATH))):
        detected_labels_file_abs_path = (const.DETECTED_LABELS_FOR_COLOR_IMAGES_FOLDER_ABS_PATH + "/" + color_image_file_name).replace(".jpg", ".txt")

        if os.path.isfile(detected_labels_file_abs_path):
            with open(detected_labels_file_abs_path, newline="") as detected_labels_file:
                detected_labels_reader = csv.reader(detected_labels_file, quoting=csv.QUOTE_NONNUMERIC)
                detected_labels = list(detected_labels_reader)

            depth_file_name = sorted(os.listdir(const.DEPTH_ARRAYS_FOLDER_ABS_PATH))[file_name_index]
            depth_array = np.load(const.DEPTH_ARRAYS_FOLDER_ABS_PATH + "/" + depth_file_name)

            detected_data = get_detected_data_from_detected_labels_and_depth_array(detected_labels, depth_array, rs_depth_resolution_in_meters, \
                rs_camera_focal_length_x, rs_camera_focal_length_y, ur_calibration_object, hand_eye_calibration_object, transformation_matrix_object, \
                file_name_index)

            total_detected_data = total_detected_data + detected_data

    selected_transformed_center_points_arrays_list = []
    filtered_detected_data = []

    for detected_data_row in total_detected_data:
        if detected_data_row[0] == selected_class_index:
            selected_transformed_center_points_array = detected_data_row[1]
            selected_transformed_center_points_arrays_list.append(selected_transformed_center_points_array)

    if len(selected_transformed_center_points_arrays_list) > 0:
        selected_transformed_center_points_median = np.median(np.stack(selected_transformed_center_points_arrays_list, axis=0), axis=0)

        for detected_data_row in total_detected_data:
            if (detected_data_row[0] == selected_class_index) and (np.linalg.norm(detected_data_row[1] - selected_transformed_center_points_median) <= \
                const.CROPPING_MAX_DIAG_DIST_FROM_DETECTED_CENTERS):
                filtered_detected_data.append(detected_data_row)

    return filtered_detected_data



def get_largest_severely_filtered_pointcloud_containing_noise(pointcloud, paint_clustered_pointcloud=False):
    cluster_labels_of_pointcloud = np.array(pointcloud.cluster_dbscan(eps=const.CLUSTERING_DISTANCE_TO_NEIGHBOURS, min_points=\
        const.CLUSTERING_SEVERE_NUM_OF_MIN_POINTS))

    noise_pointcloud = pointcloud.select_by_index(np.where(cluster_labels_of_pointcloud == -1)[0], invert=False)

    pointcloud = pointcloud.select_by_index(np.where(cluster_labels_of_pointcloud != -1)[0], invert=False)
    cluster_labels_of_pointcloud = np.delete(cluster_labels_of_pointcloud, np.where(cluster_labels_of_pointcloud == -1)[0])

    max_num_of_points_for_some_label = 0
    cluster_label_for_max_num_of_points = None

    num_of_cluster_labels = len(np.unique(cluster_labels_of_pointcloud))

    for cluster_label in range(0, num_of_cluster_labels):
        pointcloud_with_some_label = pointcloud.select_by_index(np.where(cluster_labels_of_pointcloud == cluster_label)[0])
        num_of_points_for_some_label = len(np.asarray(pointcloud_with_some_label.points))

        try:
            if (num_of_points_for_some_label > 0) and (num_of_points_for_some_label >= max_num_of_points_for_some_label):
                    max_num_of_points_for_some_label = num_of_points_for_some_label
                    cluster_label_for_max_num_of_points = cluster_label
        except:
            pass
    
    if paint_clustered_pointcloud == True:
        cluster_colors_of_pointcloud = plt.get_cmap("gist_rainbow")(cluster_labels_of_pointcloud / (num_of_cluster_labels if num_of_cluster_labels > 0 else 1))
        pointcloud.colors = o3d.utility.Vector3dVector(cluster_colors_of_pointcloud[:, 0:3])
        noise_pointcloud = get_painted_pointcloud(noise_pointcloud, (0, 0, 0))

    pointcloud = pointcloud.select_by_index(np.where(cluster_labels_of_pointcloud == cluster_label_for_max_num_of_points)[0], invert=False)
    cluster_labels_of_pointcloud = np.delete(cluster_labels_of_pointcloud, np.where(cluster_labels_of_pointcloud != cluster_label_for_max_num_of_points)[0])

    pointcloud = pointcloud + noise_pointcloud

    return pointcloud



def get_largest_faintly_filtered_pointcloud(pointcloud, filter_clustered_pointcloud_by_noise=True, paint_clustered_pointcloud=False):
    cluster_labels_of_pointcloud = np.array(pointcloud.cluster_dbscan(eps=const.CLUSTERING_DISTANCE_TO_NEIGHBOURS, min_points=\
        const.CLUSTERING_FAINT_NUM_OF_MIN_POINTS))

    if filter_clustered_pointcloud_by_noise == True:
        pointcloud = pointcloud.select_by_index(np.where(cluster_labels_of_pointcloud != -1)[0], invert=False)
        cluster_labels_of_pointcloud = np.delete(cluster_labels_of_pointcloud, np.where(cluster_labels_of_pointcloud == -1)[0])

    max_num_of_points_for_some_label = 0
    cluster_label_for_max_num_of_points = None

    num_of_cluster_labels = len(np.unique(cluster_labels_of_pointcloud))

    for cluster_label in range(0, num_of_cluster_labels):
        pointcloud_with_some_label = pointcloud.select_by_index(np.where(cluster_labels_of_pointcloud == cluster_label)[0])
        num_of_points_for_some_label = len(np.asarray(pointcloud_with_some_label.points))

        try:
            if (num_of_points_for_some_label > 0) and (num_of_points_for_some_label >= max_num_of_points_for_some_label):
                    max_num_of_points_for_some_label = num_of_points_for_some_label
                    cluster_label_for_max_num_of_points = cluster_label
        except:
            pass
    
    pointcloud = pointcloud.select_by_index(np.where(cluster_labels_of_pointcloud == cluster_label_for_max_num_of_points)[0], invert=False)
    cluster_labels_of_pointcloud = np.delete(cluster_labels_of_pointcloud, np.where(cluster_labels_of_pointcloud != cluster_label_for_max_num_of_points)[0])

    if paint_clustered_pointcloud == True:
        num_of_cluster_labels = len(np.unique(cluster_labels_of_pointcloud))
        
        cluster_colors_of_pointcloud = plt.get_cmap("gist_rainbow")(cluster_labels_of_pointcloud / (num_of_cluster_labels if num_of_cluster_labels > 0 else 1))
        cluster_colors_of_pointcloud[cluster_labels_of_pointcloud == -1] = 0
        pointcloud.colors = o3d.utility.Vector3dVector(cluster_colors_of_pointcloud[:, 0:3])
    
    return pointcloud