import os
import cv2
import numpy as np

from RS_and_3D_OD_common_programs import *
from RS_and_3D_OD_pointcloud_programs import *
from RS_and_3D_OD_constants import Constants

from PySide6.QtCore import Qt, QSize, QPoint, QTimer, QThread, Signal, Slot
from PySide6.QtGui import QPalette, QColor, QPainter, QImage, QPixmap, QIcon, QFont
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSpinBox, QGridLayout, QSizePolicy, QFrame

PROJECT_FOLDER_ABS_PATH = os.path.dirname(os.path.abspath("__file__")).replace(os.sep, "/")

const = Constants(project_folder_abs_path=PROJECT_FOLDER_ABS_PATH, reconstruct_pointclouds_instead_of_calibrating_hand_eye=True)



class CustomVideoThread(QThread):
    change_pixmap_signal = Signal(np.ndarray)

    def __init__(self):
        super(CustomVideoThread, self).__init__()
        self.capture_running_flag = True

    def run(self):
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, const.CAM_FRAME_SIZE[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, const.CAM_FRAME_SIZE[1])
        
        while self.capture_running_flag and capture.isOpened():
            (_, cv_image) = capture.read()
            self.change_pixmap_signal.emit(cv_image)
        
        capture.release()

    def stop(self):
        self.capture_running_flag = False
        self.wait()


class CustomVideoWidget(QLabel):
    def __init__(self):
        super(CustomVideoWidget, self).__init__()
        (self.display_width, self.display_height) = (const.CAM_FRAME_SIZE[0], const.CAM_FRAME_SIZE[1])
        self.resize(self.display_width, self.display_height)

        self.video_thread = CustomVideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

    @Slot(np.ndarray)
    def update_image(self, cv_image):
        qt_image = self.convert_cv_image_to_qt_image(cv_image)
        self.setPixmap(qt_image)
    
    def convert_cv_image_to_qt_image(self, cv_image):
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        (height, width, channels) = rgb_image.shape
        bytes_per_line = channels * width
        conversion_format = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        qt_image = QPixmap.fromImage(conversion_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio))

        return qt_image


class CustomColoredWidget(QWidget):
    def __init__(self, color_name_or_rgb_tuple="white"):
        super(CustomColoredWidget, self).__init__()
        self.setAutoFillBackground(True)
        palette = self.palette()

        if type(color_name_or_rgb_tuple) == str:
            palette.setColor(QPalette.Window, QColor(color_name_or_rgb_tuple))
        else:
            palette.setColor(QPalette.Window, QColor(*color_name_or_rgb_tuple))
        
        self.setPalette(palette)


class CustomEmptyLabel(QLabel):
    def __init__(self):
        super(CustomEmptyLabel, self).__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)


class CustomLabel(QLabel):
    def __init__(self, text="Text", font_style="Times", font_size=const.APPLICATION_LABEL_FONT_SIZE, font_bold_bool=False):
        super(CustomLabel, self).__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setText(text)
        if font_bold_bool == True:
            self.setFont(QFont(font_style, font_size, QFont.Bold))
        else:
            self.setFont(QFont(font_style, font_size))
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignCenter)


class CustomImage(QLabel):
    def __init__(self, image_path=None):
        super(CustomImage, self).__init__()
        if image_path != None:
            self.setFrameStyle(QFrame.StyledPanel)
            self.pixmap = QPixmap(image_path)

    def change_pixmap(self, image_path):
        self.pixmap = QPixmap(image_path)
        self.repaint()

    def paintEvent(self, event):
        image_size = self.size()
        painter = QPainter(self)
        point = QPoint(0, 0)
        scaledPix = self.pixmap.scaled(image_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        point.setX((image_size.width() - scaledPix.width())/2)
        point.setY((image_size.height() - scaledPix.height())/2)
        painter.drawPixmap(point, scaledPix)


class CustomButton(QPushButton):
    def __init__(self, text="Button", font_style="Times", font_size=const.APPLICATION_LABEL_FONT_SIZE, icon_path=None, icon_size_tuple=(0, 0), click_function_handle=None):
        super(CustomButton, self).__init__(QIcon(icon_path), text)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setIconSize(QSize(*icon_size_tuple))
        self.setText(text)
        self.setFont(QFont(font_style, font_size))
        self.clicked.connect(click_function_handle)


class CustomSpinBox(QSpinBox):
    def __init__(self, min_value=0, max_value=1, starting_value=0, font_style="Times", font_size=const.APPLICATION_LABEL_FONT_SIZE, value_change_function_handle=None):
        super(CustomSpinBox, self).__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setRange(min_value, max_value)
        self.setValue(starting_value)
        self.valueChanged.connect(value_change_function_handle)
        self.setFont(QFont(font_style, font_size))
        self.setAlignment(Qt.AlignCenter)




class CustomWindow(QMainWindow):

    def __init__(self, title="Title", font_style="Times", font_size=const.APPLICATION_LABEL_FONT_SIZE, icon_path=None, show_fullscreen=False):
        super(CustomWindow, self).__init__()
        
        self.const = Constants(project_folder_abs_path=PROJECT_FOLDER_ABS_PATH, reconstruct_pointclouds_instead_of_calibrating_hand_eye=True)

        self.starting_variables_already_defined = False
        self.visualize_pointclouds = False

        self.setWindowTitle(title)
        self.setWindowIcon(QIcon(icon_path))
        self.setMinimumSize(QSize(self.const.APPLICATION_ORIGINAL_WIDTH, self.const.APPLICATION_ORIGINAL_HEIGHT))
        self.setFont(QFont(font_style, font_size))

        if show_fullscreen == True:
            self.showFullScreen()
        else:
            self.showNormal()
        
        self.menu = self.menuBar().addMenu("Menu")
        self.fullscreen_action = self.menu.addAction("Show fullscreen", self.check_fullscreen)
        self.fullscreen_action.setCheckable(True)
        self.visualize_pointclouds_action = self.menu.addAction("Visualize pointclouds", self.check_visualizing_pointcloud)
        self.visualize_pointclouds_action.setCheckable(True)
        self.menu.addAction("Exit", self.close)
        self.menu.setFont(QFont(font_style, font_size))

        self.setCentralWidget(QWidget())

        self.main_display()


    def check_fullscreen(self):
        if self.fullscreen_action.isChecked():
            self.showFullScreen()
        else:
            self.showNormal()


    def check_visualizing_pointcloud(self):
        if self.visualize_pointclouds_action.isChecked():
            self.visualize_pointclouds = True
        else:
            self.visualize_pointclouds = False


    def edit_grid_layout(self, grid_layout, margins_size=0, spacing_size=0):
        if margins_size > 0:
            grid_layout.setContentsMargins(margins_size, margins_size, margins_size, margins_size)
        if spacing_size > 0:
            grid_layout.setSpacing(spacing_size)

        num_of_rows = grid_layout.rowCount()
        num_of_columns = grid_layout.columnCount()

        for index_row in range(0, num_of_rows):
            grid_layout.setRowMinimumHeight(index_row, self.minimumHeight())
        for index_column in range(0, num_of_columns):
            grid_layout.setColumnMinimumWidth(index_column, self.minimumWidth())


    def main_display(self):
        try:
            widget = CustomColoredWidget(self.const.APPLICATION_DARK_GRAY_COLOR_TUPLE)
            layout = QGridLayout()
    
            calibration_button = CustomButton(text="CALIBRATION", click_function_handle=self.calibration_display)
            reconstruction_button = CustomButton(text="RECONSTRUCTION \nAND HANDLING", click_function_handle=self.reconstruction_display)
    
            layout.addWidget(CustomEmptyLabel(), 0, 0, 1, 2)
            layout.addWidget(calibration_button, 1, 0, 1, 1)
            layout.addWidget(reconstruction_button, 1, 1, 1, 1)
            layout.addWidget(CustomEmptyLabel(), 2, 0, 1, 2)
            
            self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_GENERAL_SPACING_SIZE)
            widget.setLayout(layout)
            self.setCentralWidget(widget)
        
        except:
            self.error_display()
    
    
    def calibration_display(self):
        try:
            widget = CustomColoredWidget(self.const.APPLICATION_CALIBRATION_LIGHT_GRAY_COLOR_TUPLE)
            layout = QGridLayout()
    
            existing_data_button = CustomButton(text="CHOOSE EXISTING \nCALIBRATION DATA", click_function_handle=self.calibration_existing_data_display)
            new_data_button = CustomButton(text="CHOOSE NEW \nCALIBRATION DATA", click_function_handle=self.calibration_new_data_display)
            
            if self.starting_variables_already_defined == False:
                self.const = Constants(project_folder_abs_path=PROJECT_FOLDER_ABS_PATH, reconstruct_pointclouds_instead_of_calibrating_hand_eye=False)

                self.calibration_already_performed_bool = False
                self.new_images_data_value = self.const.MIN_NUM_OF_WANTED_COLOR_IMAGES

                self.starting_variables_already_defined = True
    
            layout.addWidget(CustomEmptyLabel(), 0, 0, 1, 2)
            layout.addWidget(existing_data_button, 1, 0, 1, 1)
            layout.addWidget(new_data_button, 1, 1, 1, 1)
            layout.addWidget(CustomEmptyLabel(), 2, 0, 1, 2)
            
            self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_GENERAL_SPACING_SIZE)
            widget.setLayout(layout)
            self.setCentralWidget(widget)
        
        except:
            self.error_display()
    
    
    def calibration_existing_data_display(self):
        try:
            widget = CustomColoredWidget(self.const.APPLICATION_CALIBRATION_LIGHT_GRAY_COLOR_TUPLE)
            layout = QGridLayout()
    
            calibration_button = CustomButton(text="Show calibration", click_function_handle=self.calibration_transformation_display)
            return_button = CustomButton(text="Return to menu", click_function_handle=self.calibration_display)
    
            create_folder_if_does_not_exists(self.const.COLOR_IMAGES_FOLDER_ABS_PATH)
    
            if len(sorted(os.listdir(self.const.COLOR_IMAGES_FOLDER_ABS_PATH))) < self.const.MIN_NUM_OF_WANTED_COLOR_IMAGES:
                self.calibration_new_data_display()
            else:
                if self.calibration_already_performed_bool == False:
                    self.hand_eye_transformation_matrix = calibrate_realsense_camera_and_universal_robot_with_hand_eye_method_and_get_matrix(self.const, \
                        calibrate_with_already_processed_data=True, num_of_wanted_chessboard_images=None)
    
                    self.calibration_already_performed_bool = True
    
                if len(sorted(os.listdir(self.const.DETECTED_COLOR_IMAGES_FOLDER_ABS_PATH))) > self.const.APPLICATION_MAX_NUM_OF_SMALLER_IMAGES:
                    image_file_names_list = sorted(os.listdir(self.const.DETECTED_COLOR_IMAGES_FOLDER_ABS_PATH))[0:self.const.APPLICATION_MAX_NUM_OF_SMALLER_IMAGES]
                else:
                    image_file_names_list = sorted(os.listdir(self.const.DETECTED_COLOR_IMAGES_FOLDER_ABS_PATH))
    
                images_list = []
                for (image_index, image) in enumerate(image_file_names_list):
                    image = CustomImage(image_path=(self.const.DETECTED_COLOR_IMAGES_FOLDER_ABS_PATH + "/" + image_file_names_list[image_index]))
                    images_list.append(image)
    
                while len(images_list) < self.const.APPLICATION_MAX_NUM_OF_SMALLER_IMAGES:
                    images_list.append(CustomEmptyLabel())
    
                images_list[-2] = CustomEmptyLabel()
                images_list[-1] = CustomEmptyLabel()
                images_array = np.reshape(images_list, (self.const.APPLICATION_NUM_OF_SMALLER_IMAGES_ROWS, self.const.APPLICATION_NUM_OF_SMALLER_IMAGES_COLUMNS))
    
                for (row_index, images_row) in enumerate(images_array):
                    for (column_index, image) in enumerate(images_row):
                        layout.addWidget(image, row_index, column_index, 1, 1)
    
                layout.addWidget(return_button, 3, 4, 1, 1)
                layout.addWidget(calibration_button, 3, 5, 1, 1)
    
                self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_IMAGES_SPACING_SIZE)
                widget.setLayout(layout)
                self.setCentralWidget(widget)
        
        except:
            self.error_display()
    
    
    def calibration_transformation_display(self):
        try:
            widget = CustomColoredWidget(self.const.APPLICATION_CALIBRATION_LIGHT_GRAY_COLOR_TUPLE)
            layout = QGridLayout()
    
            return_button = CustomButton(text="Return to menu", click_function_handle=self.calibration_display)
            
            transformation_matrix_object = TransformationMatrix(self.const)
            (translation_vector, rotation_vector) = transformation_matrix_object.convert_transformation_matrix_to_translation_and_rotation_vector(\
                self.hand_eye_transformation_matrix)
    
            transformation_image = CustomImage(image_path=self.const.HAND_EYE_TRANSFORMATION_FILE_ABS_PATH)
            translation_vector_label = CustomLabel("Translation vector from tool to camera: \n[{:.4f}, {:.4f}, {:.4f}]".format(*translation_vector))
            rotation_vector_label = CustomLabel("Rotation vector from tool to camera: \n[{:.4f}, {:.4f}, {:.4f}]".format(*rotation_vector))
    
            layout.addWidget(transformation_image, 0, 0, 2, 2)
            layout.addWidget(translation_vector_label, 0, 2, 1, 1)
            layout.addWidget(rotation_vector_label, 1, 2, 1, 1)
            layout.addWidget(return_button, 2, 0, 1, 1)
            layout.addWidget(CustomEmptyLabel(), 2, 1, 1, 2)
            
            self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_GENERAL_SPACING_SIZE)
            widget.setLayout(layout)
            self.setCentralWidget(widget)

        except:
            self.error_display()
    
    
    def calibration_new_data_display(self):
        try:
            widget = CustomColoredWidget(self.const.APPLICATION_CALIBRATION_LIGHT_GRAY_COLOR_TUPLE)
            layout = QGridLayout()
    
            new_images_data_label = CustomLabel(text="Define number of wanted chessboard images for \ncalibration (must be in range from {} to {}): ".format(\
                self.const.MIN_NUM_OF_WANTED_COLOR_IMAGES, self.const.MAX_NUM_OF_WANTED_COLOR_IMAGES))
            return_button = CustomButton(text="Return to menu", click_function_handle=self.calibration_display)
            self.new_images_data_spin_box = CustomSpinBox(min_value=self.const.MIN_NUM_OF_WANTED_COLOR_IMAGES, max_value=self.const.MAX_NUM_OF_WANTED_COLOR_IMAGES, \
                starting_value=self.const.MIN_NUM_OF_WANTED_COLOR_IMAGES, value_change_function_handle=self.calibration_spin_box_value_change)
            self.new_images_data_value = self.new_images_data_spin_box.value()
            streaming_button = CustomButton(text="Start calibration", click_function_handle=self.calibration_streaming_display)
            
            layout.addWidget(new_images_data_label, 0, 0, 1, 1)
            layout.addWidget(self.new_images_data_spin_box, 0, 1, 1, 1)
            layout.addWidget(CustomEmptyLabel(), 1, 0, 1, 2)
            layout.addWidget(return_button, 2, 0, 1, 1)
            layout.addWidget(streaming_button, 2, 1, 1, 1)
    
            self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_GENERAL_SPACING_SIZE)
            widget.setLayout(layout)
            self.setCentralWidget(widget)
        
        except:
            self.error_display()
    
    
    def calibration_spin_box_value_change(self):
        self.new_images_data_value = self.new_images_data_spin_box.value()
    
    
    def calibration_streaming_display(self):
        try:
            widget = CustomColoredWidget(self.const.APPLICATION_CALIBRATION_LIGHT_GRAY_COLOR_TUPLE)
            layout = QGridLayout()
    
            proposing_label = CustomLabel(text="Propose valid calibration image for saving with key:")
            proposing_key_label = CustomLabel(text="{}".format(self.const.KEY_FOR_PROPOSING_SAVE_OF_CALIBRATION_DATA.upper()), font_bold_bool=True)
            proceeding_label = CustomLabel(text="Proceed with current number of images (three are minimum) with key:")
            proceeding_key_label = CustomLabel(text="{}".format(self.const.KEY_FOR_EXITING_LOOP.upper()), font_bold_bool=True)
            accepting_label = CustomLabel(text="Accept proposed calibration image with key:")
            accepting_key_label = CustomLabel(text="{}".format(self.const.KEY_FOR_ACCEPTING_SAVE_OF_CALIBRATION_DATA.upper()), font_bold_bool=True)
            declining_label = CustomLabel(text="Decline proposed calibration image with key:")
            declining_key_label = CustomLabel(text="{}".format(self.const.KEY_FOR_EXITING_POSE_VISUALIZATION.upper()), font_bold_bool=True)
            
            layout.addWidget(proposing_label, 0, 0, 1, 1)
            layout.addWidget(CustomEmptyLabel(), 0, 1, 1, 1)
            layout.addWidget(accepting_label, 0, 2, 1, 1)
            layout.addWidget(proposing_key_label, 1, 0, 1, 1)
            layout.addWidget(CustomEmptyLabel(), 1, 1, 1, 1)
            layout.addWidget(accepting_key_label, 1, 2, 1, 1)
            layout.addWidget(proceeding_label, 2, 0, 1, 1)
            layout.addWidget(CustomEmptyLabel(), 2, 1, 1, 1)
            layout.addWidget(declining_label, 2, 2, 1, 1)
            layout.addWidget(proceeding_key_label, 3, 0, 1, 1)
            layout.addWidget(CustomEmptyLabel(), 3, 1, 1, 1)
            layout.addWidget(declining_key_label, 3, 2, 1, 1)
            
            self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_GENERAL_SPACING_SIZE)
            widget.setLayout(layout)
            self.setCentralWidget(widget)
    
            self.hand_eye_transformation_matrix = calibrate_realsense_camera_and_universal_robot_with_hand_eye_method_and_get_matrix(self.const, \
                calibrate_with_already_processed_data=False, num_of_wanted_chessboard_images=self.new_images_data_value)
            
            self.calibration_already_performed_bool = True
            
            self.calibration_existing_data_display()
        
        except:
            self.error_display()
    
    
    def reconstruction_display(self):
        try:
            widget = CustomColoredWidget(self.const.APPLICATION_RECONSTRUCTION_LIGHT_GRAY_COLOR_TUPLE)
            layout = QGridLayout()
    
            existing_data_button = CustomButton(text="CHOOSE EXISTING \nRECONSTRUCTION DATA", click_function_handle=self.reconstruction_existing_data_display)
            new_data_button = CustomButton(text="CHOOSE NEW \nRECONSTRUCTION DATA", click_function_handle=self.reconstruction_new_data_display)
            
            if self.starting_variables_already_defined == False:
                self.const = Constants(project_folder_abs_path=PROJECT_FOLDER_ABS_PATH, reconstruct_pointclouds_instead_of_calibrating_hand_eye=True)
                self.ur_communication_object = UniversalRobotCommunication(const, connection_staying_opened_bool=True)

                self.detection_already_performed_bool = False
                self.port_from_new_value = self.const.COMMUNICATION_PC_PORT

                self.starting_variables_already_defined = True
    
            layout.addWidget(CustomEmptyLabel(), 0, 0, 1, 2)
            layout.addWidget(existing_data_button, 1, 0, 1, 1)
            layout.addWidget(new_data_button, 1, 1, 1, 1)
            layout.addWidget(CustomEmptyLabel(), 2, 0, 1, 2)
    
            self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_GENERAL_SPACING_SIZE)
            widget.setLayout(layout)
            self.setCentralWidget(widget)
        
        except:
            self.error_display()
    
    
    def reconstruction_existing_data_display(self):
        try:
            widget = CustomColoredWidget(self.const.APPLICATION_RECONSTRUCTION_LIGHT_GRAY_COLOR_TUPLE)
            layout = QGridLayout()
    
            reconstruction_button = CustomButton(text="Reconstruct", click_function_handle=self.reconstruction_combining_display)
            return_button = CustomButton(text="Return to menu", click_function_handle=self.reconstruction_display)
    
            create_folder_if_does_not_exists(self.const.COLOR_IMAGES_FOLDER_ABS_PATH)
    
            if len(sorted(os.listdir(self.const.COLOR_IMAGES_FOLDER_ABS_PATH))) < 1:
                self.reconstruction_new_data_display()
            else:
                if self.detection_already_performed_bool == False:
                    detect_objects_on_data_from_realsense_camera_and_universal_robot(self.const, self.ur_communication_object, detect_with_already_processed_data=True, \
                        server_port=None)
    
                    self.detection_already_performed_bool = True
                
                if len(sorted(os.listdir(self.const.DETECTED_COLOR_IMAGES_FOLDER_ABS_PATH))) > self.const.APPLICATION_MAX_NUM_OF_SMALLER_IMAGES:
                    image_file_names_list = sorted(os.listdir(self.const.DETECTED_COLOR_IMAGES_FOLDER_ABS_PATH))[0:self.const.APPLICATION_MAX_NUM_OF_SMALLER_IMAGES]
                else:
                    image_file_names_list = sorted(os.listdir(self.const.DETECTED_COLOR_IMAGES_FOLDER_ABS_PATH))
        
                images_list = []
                for (image_index, image) in enumerate(image_file_names_list):
                    image = CustomImage(image_path=(self.const.DETECTED_COLOR_IMAGES_FOLDER_ABS_PATH + "/" + image_file_names_list[image_index]))
                    images_list.append(image)
                
                while len(images_list) < self.const.APPLICATION_MAX_NUM_OF_SMALLER_IMAGES:
                    images_list.append(CustomEmptyLabel())
                
                images_list[-2] = CustomEmptyLabel()
                images_list[-1] = CustomEmptyLabel()
                images_array = np.reshape(images_list, (self.const.APPLICATION_NUM_OF_SMALLER_IMAGES_ROWS, self.const.APPLICATION_NUM_OF_SMALLER_IMAGES_COLUMNS))
        
                for (row_index, images_row) in enumerate(images_array):
                    for (column_index, image) in enumerate(images_row):
                        layout.addWidget(image, row_index, column_index, 1, 1)
        
                layout.addWidget(return_button, 3, 4, 1, 1)
                layout.addWidget(reconstruction_button, 3, 5, 1, 1)
        
                self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_IMAGES_SPACING_SIZE)
                widget.setLayout(layout)
                self.setCentralWidget(widget)
        
        except:
            self.error_display()
    
    
    def reconstruction_combining_display(self):
        try:
            widget = CustomColoredWidget(self.const.APPLICATION_RECONSTRUCTION_LIGHT_GRAY_COLOR_TUPLE)
            layout = QGridLayout()
    
            self.combined_pointcloud = reconstruct_and_register_pointclouds_and_get_combined_pointcloud(self.const, use_registration_for_pointclouds_reconstruction=False)
    
            create_folder_if_does_not_exists_or_empty_it_if_exists(self.const.VISUALIZED_GEOMETRIES_IMAGES_TEMP_FOLDER_ABS_PATH)
    
            for front_view_vector in self.const.VISUALIZED_GEOMETRIES_IMAGES_FRONT_VIEW_VECTORS_LIST:
                time.sleep(self.const.VISUALIZED_GEOMETRIES_IMAGES_SLEEPING_TIME)
                save_geometries_image([self.combined_pointcloud], self.const.VISUALIZED_GEOMETRIES_IMAGES_TEMP_FOLDER_ABS_PATH + "/" + \
                    get_timestamp_with_unique_id_string(), self.const.VISUALIZED_GEOMETRIES_IMAGES_ZOOM_FACTOR, self.const.VISUALIZED_GEOMETRIES_IMAGES_UP_VIEW_VECTOR, \
                    front_view_vector, self.const.VISUALIZED_GEOMETRIES_IMAGES_TRANSLATION_VECTOR)
    
            if len(sorted(os.listdir(self.const.VISUALIZED_GEOMETRIES_IMAGES_TEMP_FOLDER_ABS_PATH))) > self.const.APPLICATION_MAX_NUM_OF_LARGER_IMAGES:
                image_file_names_list = sorted(os.listdir(self.const.VISUALIZED_GEOMETRIES_IMAGES_TEMP_FOLDER_ABS_PATH))[0:self.const.APPLICATION_MAX_NUM_OF_LARGER_IMAGES]
            else:
                image_file_names_list = sorted(os.listdir(self.const.VISUALIZED_GEOMETRIES_IMAGES_TEMP_FOLDER_ABS_PATH))
    
            images_list = []
            for (image_index, image) in enumerate(image_file_names_list):
                image = CustomImage(image_path=(self.const.VISUALIZED_GEOMETRIES_IMAGES_TEMP_FOLDER_ABS_PATH + "/" + image_file_names_list[image_index]))
                images_list.append(image)
            
            while len(images_list) < self.const.APPLICATION_MAX_NUM_OF_LARGER_IMAGES:
                images_list.append(CustomEmptyLabel())
            
            delete_folder_if_exists(self.const.VISUALIZED_GEOMETRIES_IMAGES_TEMP_FOLDER_ABS_PATH)
    
            class_selection_button = CustomButton(text="Class selection", click_function_handle=self.reconstruction_class_selection_display)
            return_button = CustomButton(text="Return to menu", click_function_handle=self.reconstruction_display)
    
            layout.addWidget(images_list[0], 0, 0, 2, 2)
            layout.addWidget(images_list[1], 0, 2, 2, 2)
            layout.addWidget(images_list[2], 0, 4, 2, 2)
            layout.addWidget(images_list[3], 2, 0, 2, 2)
            layout.addWidget(images_list[4], 2, 2, 2, 2)
            layout.addWidget(images_list[5], 2, 4, 2, 2)
            layout.addWidget(CustomEmptyLabel(), 4, 0, 1, 4)
            layout.addWidget(return_button, 4, 4, 1, 1)
            layout.addWidget(class_selection_button, 4, 5, 1, 1)
    
            self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_IMAGES_SPACING_SIZE)
            widget.setLayout(layout)
            self.setCentralWidget(widget)
    
            if self.visualize_pointclouds == True:
                visualize_geometries([self.combined_pointcloud], visualize_coordinate_system=True)
        
        except:
            self.error_display()
    
    
    def reconstruction_class_selection_display(self):
        try:
            widget = CustomColoredWidget(self.const.APPLICATION_RECONSTRUCTION_LIGHT_GRAY_COLOR_TUPLE)
            layout = QGridLayout()
    
            class_buttons_list = []
            for class_name in self.const.CLASS_NAMES_LIST:
                class_button = CustomButton(text=class_name, click_function_handle=self.reconstruction_analysis_display)
                class_buttons_list.append(class_button)
            
            class_buttons_array = np.reshape(class_buttons_list, (self.const.APPLICATION_NUM_OF_CLASSES_ROWS, self.const.APPLICATION_NUM_OF_CLASSES_COLUMNS))
            return_button = CustomButton(text="Return to menu", click_function_handle=self.reconstruction_display)
    
            for (row_index, class_buttons_row) in enumerate(class_buttons_array):
                for (column_index, class_button) in enumerate(class_buttons_row):
                    layout.addWidget(class_button, row_index, column_index, 1, 1)
            
            layout.addWidget(return_button, 3, 0, 1, 1)
            layout.addWidget(CustomEmptyLabel(), 3, 1, 1, 3)
    
            self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_GENERAL_SPACING_SIZE)
            widget.setLayout(layout)
            self.setCentralWidget(widget)
        
        except:
            self.error_display()
    
    
    def reconstruction_analysis_display(self):
        try:
            widget = CustomColoredWidget(self.const.APPLICATION_RECONSTRUCTION_LIGHT_GRAY_COLOR_TUPLE)
            layout = QGridLayout()
    
            selected_class_button = self.sender()
            selected_class_name = selected_class_button.text()
            selected_class_index = self.const.CLASS_NAMES_LIST.index(selected_class_name)
    
            if len(np.asarray(self.combined_pointcloud.points)) > 0:
                (self.detected_pointcloud, detected_oriented_bounding_boxes_list) = \
                    run_detection_on_combined_pointcloud_for_selected_class_index_and_get_detected_pointcloud_and_bounding_boxes_list(self.const, self.combined_pointcloud, \
                        selected_class_index)
        
                if len(np.asarray(self.detected_pointcloud.points)) > 0:
                    (self.hull_pointcloud, hull_oriented_bounding_box, hull_sides_points_lineset, hull_center_point, hull_sides_center_points_list, hull_PCA_vectors_list) = \
                        run_PCA_on_detected_pointcloud_and_get_sides_center_points_with_normals(self.const, self.detected_pointcloud)
        
                    try:
                        (self.tool_target_approach_point, self.tool_target_point, self.tool_needed_opened_gripper_width, tool_partially_opened_gripper_height, \
                            tool_rotation_matrix, self.tool_rotation_rotvec, self.tool_target_coordinate_system_arrows, \
                            self.tool_target_approach_coordinate_system_arrows) = process_sides_center_points_with_normals_and_get_handling_data(self.const, \
                            hull_center_point, hull_sides_center_points_list, hull_PCA_vectors_list)

                        (tool_mesh, ground_level_mesh) = create_tool_and_ground_level_visualization_and_get_their_geometry(self.const, tool_rotation_matrix, \
                            self.tool_target_point, self.tool_needed_opened_gripper_width, tool_partially_opened_gripper_height)

                        create_folder_if_does_not_exists_or_empty_it_if_exists(self.const.VISUALIZED_GEOMETRIES_IMAGES_TEMP_FOLDER_ABS_PATH)

                        for front_view_vector in self.const.VISUALIZED_GEOMETRIES_IMAGES_FRONT_VIEW_VECTORS_LIST:
                            time.sleep(self.const.VISUALIZED_GEOMETRIES_IMAGES_SLEEPING_TIME)
                            save_geometries_image([self.combined_pointcloud, tool_mesh, ground_level_mesh, self.hull_pointcloud], \
                                self.const.VISUALIZED_GEOMETRIES_IMAGES_TEMP_FOLDER_ABS_PATH + "/" + get_timestamp_with_unique_id_string(), \
                                self.const.VISUALIZED_GEOMETRIES_IMAGES_ZOOM_FACTOR, self.const.VISUALIZED_GEOMETRIES_IMAGES_UP_VIEW_VECTOR, front_view_vector, \
                                self.const.VISUALIZED_GEOMETRIES_IMAGES_TRANSLATION_VECTOR)

                        if len(sorted(os.listdir(self.const.VISUALIZED_GEOMETRIES_IMAGES_TEMP_FOLDER_ABS_PATH))) > self.const.APPLICATION_MAX_NUM_OF_LARGER_IMAGES:
                            image_file_names_list = sorted(os.listdir(self.const.VISUALIZED_GEOMETRIES_IMAGES_TEMP_FOLDER_ABS_PATH))\
                                [0:self.const.APPLICATION_MAX_NUM_OF_LARGER_IMAGES]
                        else:
                            image_file_names_list = sorted(os.listdir(self.const.VISUALIZED_GEOMETRIES_IMAGES_TEMP_FOLDER_ABS_PATH))

                        images_list = []
                        for (image_index, image) in enumerate(image_file_names_list):
                            image = CustomImage(image_path=(self.const.VISUALIZED_GEOMETRIES_IMAGES_TEMP_FOLDER_ABS_PATH + "/" + image_file_names_list[image_index]))
                            images_list.append(image)

                        while len(images_list) < self.const.APPLICATION_MAX_NUM_OF_LARGER_IMAGES:
                            images_list.append(CustomEmptyLabel())

                        delete_folder_if_exists(self.const.VISUALIZED_GEOMETRIES_IMAGES_TEMP_FOLDER_ABS_PATH)

                        preview_button = CustomButton(text="Preview handling", click_function_handle=self.reconstruction_preview_display)
                        return_to_class_button = CustomButton(text="Return to \nclass selection", click_function_handle=self.reconstruction_class_selection_display)
                        return_button = CustomButton(text="Return to menu", click_function_handle=self.reconstruction_display)

                        layout.addWidget(images_list[0], 0, 0, 2, 2)
                        layout.addWidget(images_list[1], 0, 2, 2, 2)
                        layout.addWidget(images_list[2], 0, 4, 2, 2)
                        layout.addWidget(images_list[3], 2, 0, 2, 2)
                        layout.addWidget(images_list[4], 2, 2, 2, 2)
                        layout.addWidget(images_list[5], 2, 4, 2, 2)
                        layout.addWidget(CustomEmptyLabel(), 4, 0, 1, 3)
                        layout.addWidget(return_button, 4, 3, 1, 1)
                        layout.addWidget(return_to_class_button, 4, 4, 1, 1)
                        layout.addWidget(preview_button, 4, 5, 1, 1)
    
                        self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_IMAGES_SPACING_SIZE)
                        widget.setLayout(layout)
                        self.setCentralWidget(widget)

                        if self.visualize_pointclouds == True:
                            visualize_geometries([self.combined_pointcloud, tool_mesh, ground_level_mesh, self.hull_pointcloud], visualize_coordinate_system=True)
                    
                    except:
                        not_handling_label = CustomLabel(text="There is no any feasible handling orientation.".format(selected_class_name))
                        return_to_class_button = CustomButton(text="Return to class selection", click_function_handle=self.reconstruction_class_selection_display)
                        return_button = CustomButton(text="Return to menu", click_function_handle=self.reconstruction_display)

                        layout.addWidget(not_handling_label, 0, 0, 1, 2)
                        layout.addWidget(CustomEmptyLabel(), 1, 0, 1, 2)
                        layout.addWidget(return_button, 2, 0, 1, 1)
                        layout.addWidget(return_to_class_button, 2, 1, 1, 1)
    
                        self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_GENERAL_SPACING_SIZE)
                        widget.setLayout(layout)
                        self.setCentralWidget(widget)
        
                else:
                    not_detected_label = CustomLabel(text="There is no any detected object of class '{}'.".format(selected_class_name))
                    return_to_class_button = CustomButton(text="Return to class selection", click_function_handle=self.reconstruction_class_selection_display)
                    return_button = CustomButton(text="Return to menu", click_function_handle=self.reconstruction_display)
        
                    layout.addWidget(not_detected_label, 0, 0, 1, 2)
                    layout.addWidget(CustomEmptyLabel(), 1, 0, 1, 2)
                    layout.addWidget(return_button, 2, 0, 1, 1)
                    layout.addWidget(return_to_class_button, 2, 1, 1, 1)
    
                    self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_GENERAL_SPACING_SIZE)
                    widget.setLayout(layout)
                    self.setCentralWidget(widget)

            else:
                not_any_detected_label = CustomLabel(text="There is no any detected object at all.".format(selected_class_name))
                return_button = CustomButton(text="Return to menu", click_function_handle=self.reconstruction_display)
        
                layout.addWidget(not_any_detected_label, 0, 0, 1, 2)
                layout.addWidget(CustomEmptyLabel(), 1, 0, 1, 2)
                layout.addWidget(return_button, 2, 0, 1, 1)
                layout.addWidget(CustomEmptyLabel(), 2, 1, 1, 1)
    
                self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_GENERAL_SPACING_SIZE)
                widget.setLayout(layout)
                self.setCentralWidget(widget)
    
        
        except:
            self.error_display()
    
    
    def reconstruction_preview_display(self):
        try:
            widget = CustomColoredWidget(self.const.APPLICATION_RECONSTRUCTION_LIGHT_GRAY_COLOR_TUPLE)
            layout = QGridLayout()
    
            return_to_class_button = CustomButton(text="Return to class selection", click_function_handle=self.reconstruction_class_selection_display)
            tool_target_approach_label = CustomLabel("Tool approach point: \n[{:.4f}, {:.4f}, {:.4f}]".format(*self.tool_target_approach_point))
            tool_target_label = CustomLabel("Tool target point: \n[{:.4f}, {:.4f}, {:.4f}]".format(*self.tool_target_point))
            tool_rotation_label = CustomLabel("Tool rotation vector: \n[{:.4f}, {:.4f}, {:.4f}]".format(*self.tool_rotation_rotvec))
            tool_needed_width_label = CustomLabel("Needed gripper width: \n{:.2f}".format(self.tool_needed_opened_gripper_width))
    
            layout.addWidget(CustomEmptyLabel(), 0, 0, 1, 1)
            layout.addWidget(tool_target_approach_label, 0, 1, 1, 1)
            layout.addWidget(tool_rotation_label, 0, 2, 1, 1)
            layout.addWidget(CustomEmptyLabel(), 1, 0, 1, 1)
            layout.addWidget(tool_target_label, 1, 1, 1, 1)
            layout.addWidget(tool_needed_width_label, 1, 2, 1, 1)
            layout.addWidget(return_to_class_button, 2, 0, 1, 1)
            layout.addWidget(CustomEmptyLabel(), 2, 1, 1, 1)
    
            if self.ur_communication_object.get_connection_currently_opened_bool() == True:
                handling_button = CustomButton(text="Handle object", click_function_handle=self.reconstruction_handling_display)
                layout.addWidget(handling_button, 2, 2, 1, 1)
            else:
                connecting_button = CustomButton(text="Connect Universal Robot", click_function_handle=self.reconstruction_connection_from_existing_display)
                layout.addWidget(connecting_button, 2, 2, 1, 1)
    
            self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_GENERAL_SPACING_SIZE)
            widget.setLayout(layout)
            self.setCentralWidget(widget)
        
        except:
            self.error_display()
    
    
    def reconstruction_handling_display(self):
        try:
            widget = CustomColoredWidget(self.const.APPLICATION_RECONSTRUCTION_LIGHT_GRAY_COLOR_TUPLE)
            layout = QGridLayout()
    
            waiting_label = CustomLabel(text="Universal Robot is handling object. Wait for it to end.")
    
            layout.addWidget(CustomEmptyLabel(), 0, 0, 1, 2)
            layout.addWidget(waiting_label, 1, 0, 1, 2)
            layout.addWidget(CustomEmptyLabel(), 2, 0, 1, 2)
    
            self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_GENERAL_SPACING_SIZE)
            widget.setLayout(layout)
            self.setCentralWidget(widget)
    
            send_handling_data_to_robot(self.ur_communication_object, self.tool_target_approach_point, self.tool_target_point, self.tool_rotation_rotvec, \
                self.tool_needed_opened_gripper_width, server_port=self.port_from_new_value)
    
            self.reconstruction_finished_display()
        
        except:
            self.error_display()
            
    
    def reconstruction_finished_display(self):
        try:
            widget = CustomColoredWidget(self.const.APPLICATION_RECONSTRUCTION_LIGHT_GRAY_COLOR_TUPLE)
            layout = QGridLayout()
    
            finished_label = CustomLabel(text="Universal Robot has finished handling object.")
            return_to_class_button = CustomButton(text="Return to class selection", click_function_handle=self.reconstruction_class_selection_display)
            return_button = CustomButton(text="Return to menu", click_function_handle=self.reconstruction_display)
    
            layout.addWidget(finished_label, 0, 0, 1, 2)
            layout.addWidget(CustomEmptyLabel(), 1, 0, 1, 2)
            layout.addWidget(return_button, 2, 0, 1, 1)
            layout.addWidget(return_to_class_button, 2, 1, 1, 1)
    
            self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_GENERAL_SPACING_SIZE)
            widget.setLayout(layout)
            self.setCentralWidget(widget)
        
        except:
            self.error_display()
    
    
    def reconstruction_new_data_display(self):
        try:
            if self.ur_communication_object.get_connection_currently_opened_bool() == True:
                self.reconstruction_streaming_display()
            
            else:
                widget = CustomColoredWidget(self.const.APPLICATION_RECONSTRUCTION_LIGHT_GRAY_COLOR_TUPLE)
                layout = QGridLayout()
    
                port_from_new_label = CustomLabel(text="Define PC's port you want Universal Robot to connect to: ")
                return_button = CustomButton(text="Return to menu", click_function_handle=self.reconstruction_display)
                self.port_from_new_spin_box = CustomSpinBox(min_value=self.const.COMMUNICATION_MIN_PORT_NUMBER, max_value=self.const.COMMUNICATION_MAX_PORT_NUMBER, \
                    starting_value=self.const.COMMUNICATION_PC_PORT, value_change_function_handle=self.reconstruction_spin_box_from_new_value_change)
                self.port_from_new_value = self.port_from_new_spin_box.value()
    
                streaming_button = CustomButton(text="Connect Universal Robot in {} seconds and start \nstreaming (make sure that the firewall is disabled)".format(\
                    self.const.COMMUNICATION_CONNECTION_TIMEOUT_TIME), click_function_handle=self.reconstruction_streaming_display)
    
                layout.addWidget(port_from_new_label, 0, 0, 1, 1)
                layout.addWidget(self.port_from_new_spin_box, 0, 1, 1, 1)
                layout.addWidget(CustomEmptyLabel(), 1, 0, 1, 2)
                layout.addWidget(return_button, 2, 0, 1, 1)
                layout.addWidget(streaming_button, 2, 1, 1, 1)
    
                self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_GENERAL_SPACING_SIZE)
                widget.setLayout(layout)
                self.setCentralWidget(widget)
        
        except:
            self.error_display()
    
    
    def reconstruction_spin_box_from_new_value_change(self):
        self.port_from_new_value = self.port_from_new_spin_box.value()
    
    
    def reconstruction_connection_from_existing_display(self):
        try:
            if self.ur_communication_object.get_connection_currently_opened_bool() == True:
                self.reconstruction_handling_display()
            
            else:
                widget = CustomColoredWidget(self.const.APPLICATION_RECONSTRUCTION_LIGHT_GRAY_COLOR_TUPLE)
                layout = QGridLayout()
    
                port_from_existing_label = CustomLabel(text="Define PC's port you want Universal Robot to connect to: ")
                return_button = CustomButton(text="Return to menu", click_function_handle=self.reconstruction_display)
                self.port_from_existing_spin_box = CustomSpinBox(min_value=self.const.COMMUNICATION_MIN_PORT_NUMBER, max_value=self.const.COMMUNICATION_MAX_PORT_NUMBER, \
                    starting_value=self.const.COMMUNICATION_PC_PORT, value_change_function_handle=self.reconstruction_spin_box_from_existing_value_change)
                self.port_from_existing_value = self.port_from_existing_spin_box.value()
    
                handling_button = CustomButton(text="Connect Universal Robot in {} seconds and start \nhandling (make sure that the firewall is disabled)".format(\
                    self.const.COMMUNICATION_CONNECTION_TIMEOUT_TIME), click_function_handle=self.reconstruction_handling_display)
    
                layout.addWidget(port_from_existing_label, 0, 0, 1, 1)
                layout.addWidget(self.port_from_existing_spin_box, 0, 1, 1, 1)
                layout.addWidget(CustomEmptyLabel(), 1, 0, 1, 2)
                layout.addWidget(return_button, 2, 0, 1, 1)
                layout.addWidget(handling_button, 2, 1, 1, 1)
    
                self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_GENERAL_SPACING_SIZE)
                widget.setLayout(layout)
                self.setCentralWidget(widget)
            
        except:
            self.error_display()
    
    
    def reconstruction_spin_box_from_existing_value_change(self):
        self.port_from_existing_value = self.port_from_existing_spin_box.value()
    
    
    def reconstruction_streaming_display(self):
        try:
            widget = CustomColoredWidget(self.const.APPLICATION_RECONSTRUCTION_LIGHT_GRAY_COLOR_TUPLE)
            layout = QGridLayout()

            waiting_label = CustomLabel(text="Universal Robot is taking pictures and reconstructing pointclouds. Wait for it to end.")

            layout.addWidget(CustomEmptyLabel(), 0, 0, 1, 2)
            layout.addWidget(waiting_label, 1, 0, 1, 2)
            layout.addWidget(CustomEmptyLabel(), 2, 0, 1, 2)

            self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_GENERAL_SPACING_SIZE)
            widget.setLayout(layout)
            self.setCentralWidget(widget)

            detect_objects_on_data_from_realsense_camera_and_universal_robot(self.const, self.ur_communication_object, detect_with_already_processed_data=False, server_port=\
                self.port_from_new_value)

            self.detection_already_performed_bool = True

            self.reconstruction_existing_data_display()
        
        except:
            self.error_display()


    def error_display(self):
        widget = CustomColoredWidget(self.const.APPLICATION_ERROR_LIGHT_GRAY_COLOR_TUPLE)
        layout = QGridLayout()

        error_label = CustomLabel(text="Unexpected error occured. Application will close in {} seconds.".format(self.const.APPLICATION_ERROR_CLOSING_TIME))
        
        layout.addWidget(CustomEmptyLabel(), 0, 0, 1, 2)
        layout.addWidget(error_label, 1, 0, 1, 2)
        layout.addWidget(CustomEmptyLabel(), 2, 0, 1, 2)
        
        self.edit_grid_layout(layout, margins_size=self.const.APPLICATION_MARGINS_SIZE, spacing_size=self.const.APPLICATION_GENERAL_SPACING_SIZE)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.timeoutTimer = QTimer()
        self.timeoutTimer.timeout.connect(self.close)
        self.timeoutTimer.setSingleShot(True)
        self.timeoutTimer.start(self.const.APPLICATION_ERROR_CLOSING_TIME * 1000)

        print("Application was closed due to unexpected error!")




if __name__ == "__main__":
    warmup_object_detection(const)

    if not QApplication.instance():
        application = QApplication([])
    else:
        application = QApplication.instance()

    widget = CustomWindow(title="Robotic object handling", icon_path=const.APPLICATION_ICON_ABS_PATH)
    widget.show()
    application.exec()