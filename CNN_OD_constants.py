import os


class Constants:
    def __init__(self, project_folder_abs_path):

        # CHANGEABLE VARIABLES:

        self.SELECTED_DATASET_NAMES_LIST = ["household items", "test household items", "reduced household items"]

        self.SELECTED_GENUINE_CLASS_NAMES_LIST = ["Franck Apple Cinnamon Tea 60g", "Franck Black Currant Tea 60g", "Gavrilovic Pork Liver Pate 100g", \
            "Giana Tuna Salad Mexico 185g", "Kotanyi Curry Spices Mix 70g", "Kotanyi Garlic Granules 70g", "Nescafe Classic 225g", "Barcaffe Classic Original 225g", \
            "Podravka Luncheon Meat 150g", "Solana Pag Fine Sea Salt 250g", "Vindi Iso Sport Forest Strawberries 500mL", "Vindi Iso Sport Lemon-Grapefruit 500mL"]


        # FIXED VARIABLES:

        self.PREPARED_IMAGE_SIZE = (640, 480)  # do not change the prepared image size because the labels will no longer be correct

        self.TESTING_PERCENTAGE_OF_DATASET = 0.15
        self.VALIDATION_PERCENTAGE_OF_DATASET = 0.15

        selected_dataset_name = input("Select dataset from '" + "', '".join([input_dataset_name for input_dataset_name in self.SELECTED_DATASET_NAMES_LIST]) + "': ")
        
        while selected_dataset_name not in self.SELECTED_DATASET_NAMES_LIST:
            selected_dataset_name = input("Select dataset from '" + "', '".join([input_dataset_name for input_dataset_name in self.SELECTED_DATASET_NAMES_LIST]) + "': ")
        
        self.SELECTED_DATASET_NAME = selected_dataset_name

        self.OD_CONSTANTS_FILE_NAME = "OD_constants.py"
        self.OD_CONSTANTS_FILE_ABS_PATH = os.path.join(project_folder_abs_path, self.OD_CONSTANTS_FILE_NAME).replace(os.sep, "/")

        self.YOLO_MODELS_URL = "https://github.com/ultralytics/yolov5"
        self.YOLO_MODELS_FOLDER_REL_PATH = "YOLO v5"
        self.YOLO_MODELS_FOLDER_ABS_PATH = os.path.join(project_folder_abs_path, self.YOLO_MODELS_FOLDER_REL_PATH).replace(os.sep, "/")

        self.GOOGLE_DRIVE_FILES_FOLDER_REL_PATH = "FILES FOR GOOGLE DRIVE"
        self.GOOGLE_DRIVE_FILES_FOLDER_ABS_PATH = os.path.join(project_folder_abs_path, self.GOOGLE_DRIVE_FILES_FOLDER_REL_PATH).replace(os.sep, "/")

        self.LOCAL_PC_FILES_FOLDER_REL_PATH = "FILES FOR LOCAL PC"
        self.LOCAL_PC_FILES_FOLDER_ABS_PATH = os.path.join(project_folder_abs_path, self.LOCAL_PC_FILES_FOLDER_REL_PATH).replace(os.sep, "/")

        self.SHOT_IMAGES_FOLDER_REL_PATH = "shot images for training"

        self.SELECTED_SHOT_IMAGES_FOLDER_REL_PATH = self.SHOT_IMAGES_FOLDER_REL_PATH + " - " + self.SELECTED_DATASET_NAME + " dataset"
        self.SELECTED_SHOT_IMAGES_FOLDER_ABS_PATH = os.path.join(project_folder_abs_path, self.SHOT_IMAGES_FOLDER_REL_PATH, \
            self.SELECTED_SHOT_IMAGES_FOLDER_REL_PATH).replace(os.sep, "/")

        self.PREPARED_IMAGES_AND_LABELS_FOLDER_REL_PATH = "prepared images and labels for training"

        self.SELECTED_PREPARED_IMAGES_AND_LABELS_FOLDER_REL_PATH = self.PREPARED_IMAGES_AND_LABELS_FOLDER_REL_PATH + " - " + self.SELECTED_DATASET_NAME + " dataset"
        self.SELECTED_PREPARED_IMAGES_AND_LABELS_FOLDER_ABS_PATH = os.path.join(project_folder_abs_path, self.PREPARED_IMAGES_AND_LABELS_FOLDER_REL_PATH, \
            self.SELECTED_PREPARED_IMAGES_AND_LABELS_FOLDER_REL_PATH).replace(os.sep, "/")

        self.SELECTED_PREPARED_IMAGES_FOLDER_ABS_PATH = os.path.join(self.SELECTED_PREPARED_IMAGES_AND_LABELS_FOLDER_ABS_PATH, \
            "prepared images for training").replace(os.sep, "/")
        self.SELECTED_XML_LABELS_FOR_PREPARED_IMAGES_FOLDER_ABS_PATH = os.path.join(self.SELECTED_PREPARED_IMAGES_AND_LABELS_FOLDER_ABS_PATH, \
            "xml labels for prepared images for training").replace(os.sep, "/")
        self.SELECTED_TXT_LABELS_FOR_PREPARED_IMAGES_FOLDER_ABS_PATH = os.path.join(self.SELECTED_PREPARED_IMAGES_AND_LABELS_FOLDER_ABS_PATH, \
            "txt labels for prepared images for training").replace(os.sep, "/")

        self.LABELIMG_URL = "https://github.com/tzutalin/labelImg"
        self.LABELIMG_FOLDER_ABS_PATH = os.path.join(project_folder_abs_path, "labelimg").replace(os.sep, "/")

        self.LABELED_SECTIONS_FOLDER_REL_PATH = "labeled sections for training"
        self.LABELED_SECTIONS_FOLDER_ABS_PATH = os.path.join(project_folder_abs_path, self.LABELED_SECTIONS_FOLDER_REL_PATH).replace(os.sep, "/")

        self.SELECTED_LABELED_SECTIONS_FOLDER_REL_PATH = self.LABELED_SECTIONS_FOLDER_REL_PATH + " - " + self.SELECTED_DATASET_NAME + " dataset"
        self.SELECTED_LABELED_SECTIONS_FOLDER_REL_PATH = os.path.join(self.LABELED_SECTIONS_FOLDER_REL_PATH, \
            self.SELECTED_LABELED_SECTIONS_FOLDER_REL_PATH).replace(os.sep, "/")
        self.SELECTED_LABELED_SECTIONS_FOLDER_ABS_PATH = os.path.join(project_folder_abs_path, self.SELECTED_LABELED_SECTIONS_FOLDER_REL_PATH).replace(os.sep, "/")

        self.SELECTED_TRAINING_SECTION_FOLDER_REL_PATH = os.path.join(self.SELECTED_LABELED_SECTIONS_FOLDER_REL_PATH, "training section").replace(os.sep, "/")
        self.SELECTED_TRAINING_SECTION_FOLDER_ABS_PATH = os.path.join(project_folder_abs_path, self.SELECTED_TRAINING_SECTION_FOLDER_REL_PATH).replace(os.sep, "/")

        self.SELECTED_TESTING_SECTION_FOLDER_REL_PATH = os.path.join(self.SELECTED_LABELED_SECTIONS_FOLDER_REL_PATH, "testing section").replace(os.sep, "/")
        self.SELECTED_TESTING_SECTION_FOLDER_ABS_PATH = os.path.join(project_folder_abs_path, self.SELECTED_TESTING_SECTION_FOLDER_REL_PATH).replace(os.sep, "/")

        self.SELECTED_VALIDATION_SECTION_FOLDER_REL_PATH = os.path.join(self.SELECTED_LABELED_SECTIONS_FOLDER_REL_PATH, "validation section").replace(os.sep, "/")
        self.SELECTED_VALIDATION_SECTION_FOLDER_ABS_PATH = os.path.join(project_folder_abs_path, self.SELECTED_VALIDATION_SECTION_FOLDER_REL_PATH).replace(os.sep, "/")

        self.SELECTED_TRAINING_SECTION_IMAGES_FOLDER_ABS_PATH = os.path.join(self.SELECTED_TRAINING_SECTION_FOLDER_ABS_PATH, "images").replace(os.sep, "/")
        self.SELECTED_TRAINING_SECTION_LABELS_FOLDER_ABS_PATH = os.path.join(self.SELECTED_TRAINING_SECTION_FOLDER_ABS_PATH, "labels").replace(os.sep, "/")

        self.SELECTED_TESTING_SECTION_IMAGES_FOLDER_ABS_PATH = os.path.join(self.SELECTED_TESTING_SECTION_FOLDER_ABS_PATH, "images").replace(os.sep, "/")
        self.SELECTED_TESTING_SECTION_LABELS_FOLDER_ABS_PATH = os.path.join(self.SELECTED_TESTING_SECTION_FOLDER_ABS_PATH, "labels").replace(os.sep, "/")

        self.SELECTED_VALIDATION_SECTION_IMAGES_FOLDER_ABS_PATH = os.path.join(self.SELECTED_VALIDATION_SECTION_FOLDER_ABS_PATH, "images").replace(os.sep, "/")
        self.SELECTED_VALIDATION_SECTION_LABELS_FOLDER_ABS_PATH = os.path.join(self.SELECTED_VALIDATION_SECTION_FOLDER_ABS_PATH, "labels").replace(os.sep, "/")

        self.SELECTED_DATASET_CONFIG_FILE_NAME = self.SELECTED_DATASET_NAME.replace(" ", "_") + "_config" + ".yaml"
        self.SELECTED_DATASET_CONFIG_FOLDER_ABS_PATH = os.path.join(self.YOLO_MODELS_FOLDER_ABS_PATH, "data").replace(os.sep, "/")
        self.SELECTED_DATASET_CONFIG_FILE_ABS_PATH = os.path.join(self.SELECTED_DATASET_CONFIG_FOLDER_ABS_PATH, \
            self.SELECTED_DATASET_CONFIG_FILE_NAME).replace(os.sep, "/")

        self.SELECTED_DATASET_HYPERPARAMETERS_FILE_NAME = self.SELECTED_DATASET_NAME.replace(" ", "_") + "_hyps" + ".yaml"
        self.SELECTED_DATASET_HYPERPARAMETERS_FOLDER_ABS_PATH = os.path.join(self.YOLO_MODELS_FOLDER_ABS_PATH, "data", "hyps").replace(os.sep, "/")
        self.SELECTED_DATASET_HYPERPARAMETERS_FILE_ABS_PATH = os.path.join(self.SELECTED_DATASET_HYPERPARAMETERS_FOLDER_ABS_PATH, \
            self.SELECTED_DATASET_HYPERPARAMETERS_FILE_NAME).replace(os.sep, "/")

        self.SELECTED_DATASET_MODEL_NAME = self.SELECTED_DATASET_NAME.replace(" ", "_") + "_model"

        self.MODELS_TRAINED_ON_COLAB_FOLDER_REL_PATH = "models trained on Colab"
        self.MODELS_TRAINED_ON_COLAB_FOLDER_ABS_PATH = os.path.join(project_folder_abs_path, self.MODELS_TRAINED_ON_COLAB_FOLDER_REL_PATH).replace(os.sep, "/")