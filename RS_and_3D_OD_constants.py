import os
import cv2
import numpy as np


class Constants:
    def __init__(self, project_folder_abs_path, reconstruct_pointclouds_instead_of_calibrating_hand_eye=True):

        # all distances are in meters
        # chessboard must have constant num of cell intersections per row and per column so it must not be rotated too much
        # pointcloud cluster with label equal to -1 represents noise
        # press N in visualizer to show or hide the normals and +/- to control their length
        # radius specifies the search radius in meters and max_nn specifies the maximum of nearest neighbours
        # nb_neighbors specifies how many neighbors are taken into account in order to calculate the average distance for a given point and std_ratio specifies the 
        # threshold level based on the standard deviation of the average distances across the point cloud (lower std_ratio corresponds to the more aggressive filter)
        # distance_threshold defines the maximum distance a point can have to an estimated plane to be considered an inlier, ransac_n defines the number of points that are 
        # randomly sampled to estimate a plane and num_iterations defines how often a random plane is sampled and verified
        # eps specifies the distance to neighbours in a cluster and min_points specifies the minimum number of points required to form a cluster
        # if eps is too large, the calculation can be slow due to greater resource demand


        # CHANGEABLE VARIABLES:

        self.WARMUP_VIDEO_FILE_NAME = "warmup_video.bag"

        self.VISUALIZE_STREAMED_IMAGES = True
        self.VISUALIZE_SAVED_POINTCLOUD = False
        self.VISUALIZE_SAVED_CHESSBOARD_IMAGES = False

        self.COMMUNICATION_PC_IP_ADDRESS = "192.168.40.191"
        self.COMMUNICATION_PC_PORT = 34500
        self.COMMUNICATION_ROBOT_IP_ADDRESS = "192.168.40.14"
        self.COMMUNICATION_ROBOT_PORT = 30003
        self.COMMUNICATION_CONNECTION_TIMEOUT_TIME = 20
        self.COMMUNICATION_LISTENING_TIMEOUT_TIME = 0.005
        self.COMMUNICATION_RECEIVING_LOOP_TIME = 0.005

        self.SELECTED_TRAINED_YOLO_MODEL_FOLDER_REL_PATH = "household_items_model_640DIM_0,005LR_100EP"

        self.DEPTH_MIN_THRESHOLD_DIST_IN_METERS = 0.1
        self.DEPTH_MAX_THRESHOLD_DIST_IN_METERS = 1

        self.CHESSBOARD_INTERSECTIONS_PER_ROW = 7
        self.CHESSBOARD_INTERSECTIONS_PER_COLUMN = 9
        self.CHESSBOARD_CELL_SIDE_IN_METERS = 0.02

        self.DETECTION_CAM_FRAME_WIDTH = 640
        self.DETECTION_MIN_PERCENTAGE_OF_CONFIDENCE = 0.5
        self.DETECTION_MAX_PERCENTAGE_OF_NMS_IOU = 0.4
        
        self.MAX_DIST_BETWEEN_POINTCLOUDS_FOR_CLOSENESS = 0.02
        self.MIN_DIST_BETWEEN_POINTCLOUDS_FOR_FARNESS = 0.1
        
        self.CLUSTERING_MIN_DIAG_DIST_FOR_FILTERING = 0.08
        self.CLUSTERING_MAX_DIAG_DIST_FOR_FILTERING = 2

        self.CROPPING_FOREGROUND_SCALE_FACTOR = 0.2
        self.CROPPING_SEGMENT_OFFSET_FACTOR = 0.3
        self.CROPPING_SEMGENT_SCALE_FACTOR = 0.2
        self.CROPPING_BORDER_SCALE_FACTOR = 1.3
        self.CROPPING_REGION_NUM_OF_INTERVALS = 5
        self.CROPPING_MIN_PERCENTAGE_OF_DETECTED_DIMENSION = 0.05
        self.CROPPING_MAX_DETECTED_DIMENSION = 0.15
        self.CROPPING_OFFSET_FROM_DETECTED_DEPTH = 0.03

        self.PRINCIPAL_COMPONENTS_DIMENSION_FACTOR = 1.15

        self.HANDLING_TOOL_APPROACH_OFFSET = 0.2
        self.HANDLING_TOOL_UNDESIREABLE_HORIZONTAL_VECTOR_DICT_KEY = "-y"
        self.HANDLING_TOOL_MIN_ALLOWED_UNDESIREABLE_HORIZONTAL_ANGLE = np.pi / 3
        self.HANDLING_ROBOT_VECTOR_FOR_TOOL_X_VECTOR_DICT_KEY = "x"

        self.APPLICATION_ICON_FILE_NAME = "robot_icon.png"


        # FIXED VARIABLES:

        self.CAM_FRAME_SIZE = (640, 480)
        self.NUM_OF_STREAMING_FPS = 30

        self.SPATIALIZATION_MAGNITUDE = 3
        self.SPATIALIZATION_ALPHA = 0.5
        self.SPATIALIZATION_DELTA = 20

        self.COMMUNICATION_MIN_PORT_NUMBER = 1024
        self.COMMUNICATION_MAX_PORT_NUMBER = 65535
        self.COMMUNICATION_MAX_NUM_OF_WAITING_CONNECTIONS = 1
        self.COMMUNICATION_MAX_RECIEVING_NUM_OF_BYTES = 1140
        self.COMMUNICATION_ROBOT_READY_STRING = "ROBOT READY"
        self.COMMUNICATION_PC_READY_STRING = "PC READY"
        self.COMMUNICATION_RECONSTRUCT_STRING = "RECONSTRUCT"
        self.COMMUNICATION_HANDLE_STRING = "HANDLE"
        self.COMMUNICATION_FINISHED_STRING = "FINISHED"
        self.COMMUNICATION_ROBOT_SETTLING_TIME = 0.1

        self.MIN_NUM_OF_WANTED_COLOR_IMAGES = 3
        self.MAX_NUM_OF_WANTED_COLOR_IMAGES = 21
        self.NUM_OF_WANTED_COLOR_IMAGES_LIST = [str(num_of_wanted_color_images) for num_of_wanted_color_images in range(self.MIN_NUM_OF_WANTED_COLOR_IMAGES, \
            self.MAX_NUM_OF_WANTED_COLOR_IMAGES + 1)]

        self.CORNER_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.DOWNSAMPLE_VOXEL_SIZE = 0.0025
        
        self.NORMALS_ESTIMATION_SEARCH_RADIUS = 0.1
        self.NORMALS_ESTIMATION_MAX_NUM_OF_NEIGHBOURS = 30

        self.OUTLIERS_REMOVAL_NUM_OF_NEIGHBOURS = 100
        self.OUTLIERS_REMOVAL_STD_DEV_RATIO = 3

        self.RANSAC_MAX_DISTANCE_FROM_PLANE = 0.012
        self.RANSAC_NUM_OF_SAMPLED_POINTS_IN_PLANE = 3
        self.RANSAC_NUM_OF_ITERATIONS = 1000

        self.GROUND_LEVEL_MIN_HEIGHT = -0.02

        self.CLUSTERING_DISTANCE_TO_NEIGHBOURS = 0.01
        self.CLUSTERING_NUM_OF_MIN_POINTS = 30
        self.CLUSTERING_SEVERE_NUM_OF_MIN_POINTS = self.CLUSTERING_NUM_OF_MIN_POINTS * 5
        self.CLUSTERING_FAINT_NUM_OF_MIN_POINTS = self.CLUSTERING_NUM_OF_MIN_POINTS

        self.REGISTRATION_MAX_THRESHOLD_DIST = self.DOWNSAMPLE_VOXEL_SIZE / 2
        self.REGISTRATION_MAX_NUM_OF_ITERATIONS = 30

        self.CROPPING_MAX_DIAG_DIST_FROM_DETECTED_CENTERS = self.CROPPING_MAX_DETECTED_DIMENSION / 3

        self.POISSON_SAMPLE_NUM_OF_POINTS = 2000

        self.HANDLING_TOOL_HORIZONTAL_VECTORS_DICT = {"-x": np.array([-1, 0, 0]), "-y": np.array([0, -1, 0]), "x": np.array([1, 0, 0]), "y": np.array([0, 1, \
            0]), "-x -y": np.array([-1, -1, 0]), "-x y": np.array([-1, 1, 0]), "x y": np.array([1, 1, 0]), "x -y": np.array([1, -1, 0])}
        self.HANDLING_TOOL_UNDESIREABLE_HORIZONTAL_VECTOR = self.HANDLING_TOOL_HORIZONTAL_VECTORS_DICT[self.HANDLING_TOOL_UNDESIREABLE_HORIZONTAL_VECTOR_DICT_KEY]
        self.HANDLING_ROBOT_VECTOR_FOR_TOOL_X_VECTOR = self.HANDLING_TOOL_HORIZONTAL_VECTORS_DICT[self.HANDLING_ROBOT_VECTOR_FOR_TOOL_X_VECTOR_DICT_KEY]
        self.HANDLING_TOOL_COMPENSATION_ROTATION_ANGLE = np.pi / 3
        self.HANDLING_OPENED_GRIPPER_TOTAL_WIDTH = 0.14
        self.HANDLING_OPENED_GRIPPER_WIDTH_OFFSET = 0.02
        self.HANDLING_CLOSED_GRIPPER_TOTAL_HEIGHT = 0.125
        self.HANDLING_CLOSED_GRIPPER_HEIGHT_OFFSET = 0.02
        self.HANDLING_OPENED_GRIPPER_HEIGHT_DELTA = 0.025
        self.HANDLING_UPWARDS_DIMENSION_SCALE_OFFSET = -0.05

        self.VISUALIZATION_WINDOW_NAME = "Open3D"
        self.VISUALIZATION_WINDOW_WIDTH = 1366
        self.VISUALIZATION_WINDOW_HEIGHT = 768
        self.VISUALIZATION_WINDOW_STARTING_LEFT = 50
        self.VISUALIZATION_WINDOW_STARTING_TOP = 100
        self.VISUALIZATION_BLACK_COLOR_TUPLE = (0.2, 0.2, 0.2)
        self.VISUALIZATION_GRAY_COLOR_TUPLE = (0.8, 0.8, 0.8)
        self.VISUALIZATION_BLUE_COLOR_TUPLE = (0.2, 0.2, 0.8)
        self.VISUALIZATION_JAW_WIDTH = 0.01
        self.VISUALIZATION_JAW_DEPTH = 0.04
        self.VISUALIZATION_JAWS_BASE_WIDTH = self.HANDLING_OPENED_GRIPPER_TOTAL_WIDTH + 2 * self.VISUALIZATION_JAW_WIDTH
        self.VISUALIZATION_JAWS_BASE_DEPTH = self.VISUALIZATION_JAW_DEPTH
        self.VISUALIZATION_JAWS_BASE_HEIGHT = 0.02
        self.VISUALIZATION_SUPPORT_RADIUS = self.VISUALIZATION_JAW_DEPTH / 2
        self.VISUALIZATION_SUPPORT_HEIGHT = 0.1
        self.VISUALIZATION_CAMERA_BASE_WIDTH = 0.12
        self.VISUALIZATION_CAMERA_BASE_DEPTH = 0.02
        self.VISUALIZATION_CAMERA_BASE_HEIGHT = 0.03
        self.VISUALIZATION_CAMERA_WIDTH = self.VISUALIZATION_CAMERA_BASE_WIDTH
        self.VISUALIZATION_CAMERA_DEPTH = 0.01
        self.VISUALIZATION_CAMERA_HEIGHT = self.VISUALIZATION_CAMERA_BASE_HEIGHT
        self.VISUALIZATION_GROUND_LEVEL_WIDTH = 0.7
        self.VISUALIZATION_GROUND_LEVEL_DEPTH = 0.7
        self.VISUALIZATION_GROUND_LEVEL_HEIGHT = 0.01
        self.VISUALIZATION_COORD_SYSTEM_ARROWS_SMALLER_SIZE = 0.025
        self.VISUALIZATION_COORD_SYSTEM_ARROWS_LARGER_SIZE = 0.05

        self.VISUALIZED_GEOMETRIES_IMAGES_SLEEPING_TIME = 0.1
        self.VISUALIZED_GEOMETRIES_IMAGES_ZOOM_FACTOR = 0.7
        self.VISUALIZED_GEOMETRIES_IMAGES_UP_VIEW_VECTOR = [0, 0, 1]
        self.VISUALIZED_GEOMETRIES_IMAGES_FRONT_VIEW_VECTORS_LIST = [[1, 0, 1], [1, 1, 1], [0, 1, 1], [-1, 1, 1], [-1, 0, 1], [0, 1, 0.5]]
        self.VISUALIZED_GEOMETRIES_IMAGES_TRANSLATION_VECTOR = [0, 0, -0.1]

        self.APPLICATION_LABEL_FONT_SIZE = 13
        self.APPLICATION_ORIGINAL_WIDTH = 960
        self.APPLICATION_ORIGINAL_HEIGHT = 540
        self.APPLICATION_MARGINS_SIZE = 40
        self.APPLICATION_GENERAL_SPACING_SIZE = 40
        self.APPLICATION_IMAGES_SPACING_SIZE = 20
        self.APPLICATION_CALIBRATION_LIGHT_GRAY_COLOR_TUPLE = (192, 192, 208)
        self.APPLICATION_RECONSTRUCTION_LIGHT_GRAY_COLOR_TUPLE = (192, 208, 192)
        self.APPLICATION_ERROR_LIGHT_GRAY_COLOR_TUPLE = (208, 192, 192)
        self.APPLICATION_DARK_GRAY_COLOR_TUPLE = (160, 160, 160)
        self.APPLICATION_NUM_OF_SMALLER_IMAGES_ROWS = 4
        self.APPLICATION_NUM_OF_SMALLER_IMAGES_COLUMNS = 6
        self.APPLICATION_MAX_NUM_OF_SMALLER_IMAGES = self.APPLICATION_NUM_OF_SMALLER_IMAGES_ROWS * self.APPLICATION_NUM_OF_SMALLER_IMAGES_COLUMNS
        self.APPLICATION_NUM_OF_LARGER_IMAGES_ROWS = 2
        self.APPLICATION_NUM_OF_LARGER_IMAGES_COLUMNS = 3
        self.APPLICATION_MAX_NUM_OF_LARGER_IMAGES = self.APPLICATION_NUM_OF_LARGER_IMAGES_ROWS * self.APPLICATION_NUM_OF_LARGER_IMAGES_COLUMNS
        self.APPLICATION_NUM_OF_CLASSES_ROWS = 3
        self.APPLICATION_NUM_OF_CLASSES_COLUMNS = 4
        self.APPLICATION_ERROR_CLOSING_TIME = 5

        self.PROJECT_FOLDER_ABS_PATH = project_folder_abs_path

        self.WARMUP_VIDEO_FILE_ABS_PATH = os.path.join(self.PROJECT_FOLDER_ABS_PATH, self.WARMUP_VIDEO_FILE_NAME).replace(os.sep, "/")

        self.APPLICATION_ICON_ABS_PATH = os.path.join(project_folder_abs_path, self.APPLICATION_ICON_FILE_NAME).replace(os.sep, "/")

        self.CALIBRATION_FOLDER_REL_PATH = "calibration data"
        self.RECONSTRUCTION_FOLDER_REL_PATH = "reconstruction data"

        self.RECONSTRUCT_POINTCLOUDS_INSTEAD_OF_CALIBRATING_HAND_EYE = reconstruct_pointclouds_instead_of_calibrating_hand_eye

        if self.RECONSTRUCT_POINTCLOUDS_INSTEAD_OF_CALIBRATING_HAND_EYE == False:
            self.CALIBRATION_OR_RECONSTRUCTION_FOLDER_REL_PATH = self.CALIBRATION_FOLDER_REL_PATH
        else:
            self.CALIBRATION_OR_RECONSTRUCTION_FOLDER_REL_PATH = self.RECONSTRUCTION_FOLDER_REL_PATH
        
        self.CALIBRATION_OR_RECONSTRUCTION_FOLDER_ABS_PATH = os.path.join(self.PROJECT_FOLDER_ABS_PATH, self.CALIBRATION_OR_RECONSTRUCTION_FOLDER_REL_PATH).replace(\
            os.sep, "/")

        self.WARMUP_IMAGE_FILE_NAME = "warmup_image.jpg"
        self.WARMUP_IMAGE_FILE_ABS_PATH = os.path.join(self.PROJECT_FOLDER_ABS_PATH, self.WARMUP_IMAGE_FILE_NAME).replace(os.sep, "/")

        self.COLOR_IMAGES_FOLDER_REL_PATH = "color images"
        self.COLOR_IMAGES_FOLDER_ABS_PATH = os.path.join(self.CALIBRATION_OR_RECONSTRUCTION_FOLDER_ABS_PATH, self.COLOR_IMAGES_FOLDER_REL_PATH).replace(os.sep, "/")
        self.COMMON_PART_OF_COLOR_IMAGE_FILE_NAME = self.COLOR_IMAGES_FOLDER_REL_PATH.replace(" ", "_")[0:-1]

        self.VIDEOS_FOLDER_REL_PATH = "videos"
        self.VIDEOS_FOLDER_ABS_PATH = os.path.join(self.CALIBRATION_OR_RECONSTRUCTION_FOLDER_ABS_PATH, self.VIDEOS_FOLDER_REL_PATH).replace(os.sep, "/")
        self.COMMON_PART_OF_DETECTED_VIDEO_FILE_NAME = self.VIDEOS_FOLDER_REL_PATH.replace(" ", "_")[0:-1]

        self.DETECTED_COLOR_IMAGES_FOLDER_REL_PATH = "detected color images"
        self.DETECTED_COLOR_IMAGES_FOLDER_ABS_PATH = os.path.join(self.CALIBRATION_OR_RECONSTRUCTION_FOLDER_ABS_PATH, self.DETECTED_COLOR_IMAGES_FOLDER_REL_PATH).replace(\
            os.sep, "/")
        self.COMMON_PART_OF_DETECTED_COLOR_IMAGE_FILE_NAME = self.DETECTED_COLOR_IMAGES_FOLDER_REL_PATH.replace(" ", "_")[0:-1]

        self.DETECTED_VIDEOS_FOLDER_REL_PATH = "detected videos"
        self.DETECTED_VIDEOS_FOLDER_ABS_PATH = os.path.join(self.CALIBRATION_OR_RECONSTRUCTION_FOLDER_ABS_PATH, self.DETECTED_VIDEOS_FOLDER_REL_PATH).replace(os.sep, "/")
        self.COMMON_PART_OF_DETECTED_VIDEO_FILE_NAME = self.DETECTED_VIDEOS_FOLDER_REL_PATH.replace(" ", "_")[0:-1]

        self.DETECTED_LABELS_FOR_COLOR_IMAGES_FOLDER_REL_PATH = "detected labels for color images"
        self.DETECTED_LABELS_FOR_COLOR_IMAGES_FOLDER_ABS_PATH = os.path.join(self.CALIBRATION_OR_RECONSTRUCTION_FOLDER_ABS_PATH, \
            self.DETECTED_LABELS_FOR_COLOR_IMAGES_FOLDER_REL_PATH).replace(os.sep, "/")
        self.COMMON_PART_OF_DETECTED_LABELS_FOR_COLOR_IMAGES_FILE_NAME = self.DETECTED_LABELS_FOR_COLOR_IMAGES_FOLDER_REL_PATH.replace(" ", "_")[0:-1]

        self.DETECTED_LABELS_FOR_VIDEOS_FOLDER_REL_PATH = "detected labels for videos"
        self.DETECTED_LABELS_FOR_VIDEOS_FOLDER_ABS_PATH = os.path.join(self.CALIBRATION_OR_RECONSTRUCTION_FOLDER_ABS_PATH, \
            self.DETECTED_LABELS_FOR_VIDEOS_FOLDER_REL_PATH).replace(os.sep, "/")
        self.COMMON_PART_OF_DETECTED_LABELS_FOR_VIDEOS_FILE_NAME = self.DETECTED_LABELS_FOR_VIDEOS_FOLDER_REL_PATH.replace(" ", "_")[0:-1]

        self.DEPTH_ARRAYS_FOLDER_REL_PATH = "depth arrays"
        self.DEPTH_ARRAYS_FOLDER_ABS_PATH = os.path.join(self.CALIBRATION_OR_RECONSTRUCTION_FOLDER_ABS_PATH, self.DEPTH_ARRAYS_FOLDER_REL_PATH).replace(os.sep, "/")
        self.COMMON_PART_OF_DEPTH_ARRAY_FILE_NAME = self.DEPTH_ARRAYS_FOLDER_REL_PATH.replace(" ", "_")[0:-1]

        self.POINTCLOUDS_FOLDER_REL_PATH = "pointclouds"
        self.POINTCLOUDS_FOLDER_ABS_PATH = os.path.join(self.CALIBRATION_OR_RECONSTRUCTION_FOLDER_ABS_PATH, self.POINTCLOUDS_FOLDER_REL_PATH).replace(os.sep, "/")
        self.COMMON_PART_OF_POINTCLOUD_FILE_NAME = self.POINTCLOUDS_FOLDER_REL_PATH.replace(" ", "_")[0:-1]

        self.CAMERA_TRANSLATION_VECTORS_FOLDER_REL_PATH = "camera translation vectors"
        self.CAMERA_TRANSLATION_VECTORS_FOLDER_ABS_PATH = os.path.join(self.CALIBRATION_OR_RECONSTRUCTION_FOLDER_ABS_PATH, \
            self.CAMERA_TRANSLATION_VECTORS_FOLDER_REL_PATH).replace(os.sep, "/")
        self.COMMON_PART_OF_CAMERA_TRANS_VECTOR_FILE_NAME = self.CAMERA_TRANSLATION_VECTORS_FOLDER_REL_PATH.replace(" ", "_")[0:-1]

        self.CAMERA_ROTATION_VECTORS_FOLDER_REL_PATH = "camera rotation vectors"
        self.CAMERA_ROTATION_VECTORS_FOLDER_ABS_PATH = os.path.join(self.CALIBRATION_OR_RECONSTRUCTION_FOLDER_ABS_PATH, \
            self.CAMERA_ROTATION_VECTORS_FOLDER_REL_PATH).replace(os.sep, "/")
        self.COMMON_PART_OF_CAMERA_ROT_VECTOR_FILE_NAME = self.CAMERA_ROTATION_VECTORS_FOLDER_REL_PATH.replace(" ", "_")[0:-1]

        self.ROBOT_TRANSLATION_VECTORS_FOLDER_REL_PATH = "robot translation vectors"
        self.ROBOT_TRANSLATION_VECTORS_FOLDER_ABS_PATH = os.path.join(self.CALIBRATION_OR_RECONSTRUCTION_FOLDER_ABS_PATH, \
            self.ROBOT_TRANSLATION_VECTORS_FOLDER_REL_PATH).replace(os.sep, "/")
        self.COMMON_PART_OF_ROBOT_TRANS_VECTOR_FILE_NAME = self.ROBOT_TRANSLATION_VECTORS_FOLDER_REL_PATH.replace(" ", "_")[0:-1]

        self.ROBOT_ROTATION_VECTORS_FOLDER_REL_PATH = "robot rotation vectors"
        self.ROBOT_ROTATION_VECTORS_FOLDER_ABS_PATH = os.path.join(self.CALIBRATION_OR_RECONSTRUCTION_FOLDER_ABS_PATH, self.ROBOT_ROTATION_VECTORS_FOLDER_REL_PATH).replace(\
            os.sep, "/")
        self.COMMON_PART_OF_ROBOT_ROT_VECTOR_FILE_NAME = self.ROBOT_ROTATION_VECTORS_FOLDER_REL_PATH.replace(" ", "_")[0:-1]

        self.VISUALIZED_GEOMETRIES_IMAGES_TEMP_FOLDER_REL_PATH = "visualized geometries images"
        self.VISUALIZED_GEOMETRIES_IMAGES_TEMP_FOLDER_ABS_PATH = os.path.join(self.PROJECT_FOLDER_ABS_PATH, self.VISUALIZED_GEOMETRIES_IMAGES_TEMP_FOLDER_REL_PATH).replace(\
            os.sep, "/")

        self.HAND_EYE_TRANSLATION_VECTOR_FILE_NAME = ("hand eye translation vector").replace(" ", "_") + ".txt"
        self.HAND_EYE_TRANSLATION_VECTOR_FILE_ABS_PATH = os.path.join(self.PROJECT_FOLDER_ABS_PATH, self.HAND_EYE_TRANSLATION_VECTOR_FILE_NAME).replace(os.sep, "/")

        self.HAND_EYE_ROTATION_VECTOR_FILE_NAME = ("hand eye rotation vector").replace(" ", "_") + ".txt"
        self.HAND_EYE_ROTATION_VECTOR_FILE_ABS_PATH = os.path.join(self.PROJECT_FOLDER_ABS_PATH, self.HAND_EYE_ROTATION_VECTOR_FILE_NAME).replace(os.sep, "/")

        self.HAND_EYE_TRANSFORMATION_FILE_NAME = "hand_eye_transformation.jpg"
        self.HAND_EYE_TRANSFORMATION_FILE_ABS_PATH = os.path.join(self.PROJECT_FOLDER_ABS_PATH, self.HAND_EYE_TRANSFORMATION_FILE_NAME).replace(os.sep, "/")

        self.MODELS_TRAINED_ON_COLAB_FOLDER_REL_PATH = "models trained on Colab"
        self.MODELS_TRAINED_ON_COLAB_FOLDER_ABS_PATH = os.path.join(self.PROJECT_FOLDER_ABS_PATH, self.MODELS_TRAINED_ON_COLAB_FOLDER_REL_PATH).replace(os.sep, "/")

        self.SELECTED_WEIGHTS_FILE_NAME = "best.pt"
        self.SELECTED_WEIGHTS_FOLDER_REL_PATH = "weights"
        self.SELECTED_WEIGHTS_FILE_ABS_PATH = os.path.join(self.MODELS_TRAINED_ON_COLAB_FOLDER_ABS_PATH, self.SELECTED_TRAINED_YOLO_MODEL_FOLDER_REL_PATH, \
            self.SELECTED_WEIGHTS_FOLDER_REL_PATH, self.SELECTED_WEIGHTS_FILE_NAME).replace(os.sep, "/")

        self.CLASS_NAMES_LIST = ["Franck Apple Cinnamon Tea 60g", "Franck Black Currant Tea 60g", "Gavrilovic Pork Liver Pate 100g", "Giana Tuna Salad Mexico 185g", \
            "Kotanyi Curry Spices Mix 70g", "Kotanyi Garlic Granules 70g", "Nescafe Classic 225g", "Barcaffe Classic Original 225g", "Podravka Luncheon Meat 150g", \
            "Solana Pag Fine Sea Salt 250g", "Vindi Iso Sport Forest Strawberries 500mL", "Vindi Iso Sport Lemon-Grapefruit 500mL"]
        
        self.SELECTED_CLASS_INDICES_LIST = [selected_class_name_index for selected_class_name_index in range(0, len(self.CLASS_NAMES_LIST))]

        self.STREAMING_WINDOW_NAME = "Color and depth streaming"
        self.CHESSBOARD_WINDOW_NAME = "Chessboard pose"
        self.COMMON_PART_OF_PROCESSED_CHESSBOARD_WINDOW_NAME = "Processed chessboard pose: "

        self.KEY_FOR_PROPOSING_SAVE_OF_CALIBRATION_DATA = "s"
        self.KEY_FOR_ACCEPTING_SAVE_OF_CALIBRATION_DATA = "a"

        self.KEY_FOR_SAVING_RECONSTRUCTION_DATA = "s"

        self.KEY_FOR_EXITING_LOOP = "e"
        self.KEY_FOR_EXITING_POSE_VISUALIZATION = "w"