import os
import sys
import torch
import cv2
from pathlib import Path
import csv
import time

PROJECT_FOLDER_ABS_PATH = os.path.dirname(os.path.abspath("__file__")).replace(os.sep, "/")
YOLO_MODELS_FOLDER_REL_PATH = "YOLO v5"
YOLO_MODELS_FOLDER_ABS_PATH = PROJECT_FOLDER_ABS_PATH + "/" + YOLO_MODELS_FOLDER_REL_PATH

sys.path.append(YOLO_MODELS_FOLDER_ABS_PATH)

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, colorstr, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_sync



@torch.no_grad()
def detect_objects(source_abs_path_or_zero="0", detected_images_or_videos_folder_abs_path=None, detected_labels_folder_abs_path=None, weights_file_abs_path="yolov5s.pt", \
    selected_class_indices_list=None, image_size=[480, 480], confidence_threshold=0.5, nms_iou_threshold=0.4, max_detections=1000):
    
    if isinstance(image_size, int):
        image_size = [image_size]
    
    if len(list(image_size)) != 2:
        # best image_size is one on which the model was trained, for used case is that [480, 480]
        image_size = [image_size[0], image_size[0]]

    webcam_bool = source_abs_path_or_zero.isnumeric() or source_abs_path_or_zero.lower().startswith(("http://", "https://"))

    set_logging()
    device_name = select_device("")


    # load model
    model = attempt_load(weights_file_abs_path, map_location=device_name)  # load model
    stride_size = int(model.stride.max())  # get model stride
    class_names = model.module.names if hasattr(model, "module") else model.names  # get class names
    image_size = check_img_size(image_size, s=stride_size)  # check image size

    # load dataset
    if webcam_bool == True:
        torch.backends.cudnn.benchmark = True  # speed up constant image size inference
        dataset = LoadStreams(source_abs_path_or_zero, img_size=image_size, stride=stride_size, auto=True)  # load dataset
        batch_size = len(dataset)
    else:
        dataset = LoadImages(source_abs_path_or_zero, img_size=image_size, stride=stride_size, auto=True)  # load dataset
        batch_size = 1
    
    (detected_video_abs_path, detected_video_writer) = ([None] * batch_size, [None] * batch_size)

    # run inference
    if device_name.type != "cpu":
        model(torch.zeros(1, 3, *image_size).to(device_name).type_as(next(model.parameters())))  # run inference once
    
    starting_time = time.time()


    for (image_path, resized_image, image, video_capture_bool) in dataset:
        resized_image = torch.from_numpy(resized_image).to(device_name)
        resized_image = resized_image / 255.0  # normalize image
        
        if len(resized_image.shape) == 3:
            resized_image = resized_image[None]  # expand image for batch dimension

        # process inference
        before_inference_time = time_sync()
        predictions = model(resized_image, augment=False, visualize=False)[0]
        predictions = non_max_suppression(predictions, confidence_threshold, nms_iou_threshold, selected_class_indices_list, False, max_det=max_detections)
        after_inference_time = time_sync()

        # process predictions
        for (index, detections) in enumerate(predictions):
            if webcam_bool == True:
                (detected_image_file_name, detected_image, index_of_detected_image) = (image_path[index], image[index].copy(), dataset.count)
            else:
                (detected_image_file_name, detected_image, index_of_detected_image) = (image_path, image.copy(), getattr(dataset, "frame", 0))

            if (detected_images_or_videos_folder_abs_path != None) and (detected_labels_folder_abs_path != None):
                detected_image_file_name = Path(detected_image_file_name)
                detected_image_file_abs_path = detected_images_or_videos_folder_abs_path +  "/" + detected_image_file_name.name
                detected_labels_file_abs_path = detected_labels_folder_abs_path + "/" + detected_image_file_name.stem + ("" if dataset.mode == "image" else \
                    f"_{index_of_detected_image}")


            if len(detections) > 0:
                normalization_gain = torch.tensor(detected_image.shape)[[1, 0, 1, 0]]

                # rescale detections from resized_image size to detected_image size
                detections[:, :4] = scale_coords(resized_image.shape[2:], detections[:, :4], detected_image.shape).round()

                # save detected labels
                if (detected_images_or_videos_folder_abs_path != None) and (detected_labels_folder_abs_path != None):
                    with open(detected_labels_file_abs_path + ".txt", "w", newline="") as detected_labels_file:
                        for (*detected_starting_xy_and_ending_xy, detected_confidence_tensor, detected_class_float) in reversed(detections):
                            detected_center_xy_and_wh_scaled_float = (xyxy2xywh(torch.tensor(detected_starting_xy_and_ending_xy).view(1, 4)) / normalization_gain).view(-1).\
                                tolist()

                            detected_class_int = int(detected_class_float)
                            detected_confidence_float = detected_confidence_tensor.cpu().numpy()
                            detected_center_xy_and_wh_int = [int(detected_center_xy_and_wh_scaled_float[0] * detected_image.shape[1]), \
                                int(detected_center_xy_and_wh_scaled_float[1] * detected_image.shape[0]), int(detected_center_xy_and_wh_scaled_float[2] * \
                                detected_image.shape[1]), int(detected_center_xy_and_wh_scaled_float[3] * detected_image.shape[0])]

                            detected_class_name = f"{class_names[detected_class_int]} {detected_confidence_float:.2f}"

                            detected_image = plot_one_box(detected_starting_xy_and_ending_xy, detected_image, label=detected_class_name, color=colors(detected_class_int, \
                                True), line_width=2)

                            detected_labels = [detected_class_int, *detected_center_xy_and_wh_int, detected_confidence_float]

                            # labels are saved as class number, center x, center y, width, height and confidence
                            detected_labels_writer = csv.writer(detected_labels_file, delimiter=",")
                            detected_labels_writer.writerow(detected_labels)
                
                else:
                    for (*detected_starting_xy_and_ending_xy, detected_confidence_tensor, detected_class_float) in reversed(detections):
                        detected_center_xy_and_wh_scaled_float = (xyxy2xywh(torch.tensor(detected_starting_xy_and_ending_xy).view(1, 4)) / normalization_gain).view(-1).\
                            tolist()

                        detected_class_int = int(detected_class_float)
                        detected_confidence_float = detected_confidence_tensor.cpu().numpy()
                        detected_center_xy_and_wh_int = [int(detected_center_xy_and_wh_scaled_float[0] * detected_image.shape[1]), \
                            int(detected_center_xy_and_wh_scaled_float[1] * detected_image.shape[0]), int(detected_center_xy_and_wh_scaled_float[2] * \
                            detected_image.shape[1]), int(detected_center_xy_and_wh_scaled_float[3] * detected_image.shape[0])]

                        detected_class_name = f"{class_names[detected_class_int]} {detected_confidence_float:.2f}"

                        detected_image = plot_one_box(detected_starting_xy_and_ending_xy, detected_image, label=detected_class_name, color=colors(detected_class_int, \
                            True), line_width=2)


            # print inference time
            print(f"Inference lasted {after_inference_time - before_inference_time:.3f} s.")

            # stream webcam detected video
            if webcam_bool == True:
                cv2.imshow("video", detected_image)
                cv2.waitKey(1)

            if (detected_images_or_videos_folder_abs_path != None) and (detected_labels_folder_abs_path != None):
                # save detected images or video
                if dataset.mode == "image":
                    cv2.imwrite(detected_image_file_abs_path, detected_image)
                else:
                    if detected_video_abs_path[index] != detected_image_file_abs_path:
                        detected_video_abs_path[index] = detected_image_file_abs_path
                        if isinstance(detected_video_writer[index], cv2.VideoWriter):
                            detected_video_writer[index].release()

                        if video_capture_bool == True:
                            video_fps = video_capture_bool.get(cv2.CAP_PROP_FPS)
                            video_width = int(video_capture_bool.get(cv2.CAP_PROP_FRAME_WIDTH))
                            video_height = int(video_capture_bool.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            (video_fps, video_width, video_height) = (30, detected_image.shape[1], detected_image.shape[0])

                        detected_video_writer[index] = cv2.VideoWriter(detected_image_file_abs_path + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (video_width, \
                            video_height))

                    detected_video_writer[index].write(detected_image)


    if (detected_images_or_videos_folder_abs_path != None) and (detected_labels_folder_abs_path != None):
        print(f"\nResults are saved to {colorstr('bold', detected_images_or_videos_folder_abs_path)}.")
    else:
        print(f"\nResults are not saved.")

    print(f"Total inference lasted {time.time() - starting_time:.3f} s.\n")