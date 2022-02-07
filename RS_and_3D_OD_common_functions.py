import os
import shutil
import time
import random
import string



def get_timestamp_with_unique_id_string():
    timestamp_with_unique_id_string = str(time.strftime("%d-%m-%Y_%H-%M-%S")) + "-" + random.choice(string.ascii_uppercase) + random.choice(string.ascii_uppercase)
    return timestamp_with_unique_id_string



def start_timer():
    return time.time()



def split_timer(updated_time):
    print("Time elapsed: {} ms\n".format(1000 * (time.time() - updated_time)))
    return time.time()



def create_folder_if_does_not_exists(folder_abs_path):
    if not os.path.exists(folder_abs_path):
        os.makedirs(folder_abs_path)



def delete_folder_if_exists(folder_abs_path):
    if os.path.exists(folder_abs_path):
        shutil.rmtree(folder_abs_path)



def create_folder_if_does_not_exists_or_empty_it_if_exists(original_folder_abs_path):
    if os.path.exists(original_folder_abs_path):
        shutil.rmtree(original_folder_abs_path)

    os.makedirs(original_folder_abs_path)



def backup_and_empty_folder_if_exists(original_folder_abs_path, timestamp_string):
    backup_folder_abs_path = original_folder_abs_path + " - backup" + "/" + "backed up data - " + timestamp_string

    if os.path.exists(original_folder_abs_path):
        shutil.copytree(original_folder_abs_path, backup_folder_abs_path, dirs_exist_ok=True)
        shutil.rmtree(original_folder_abs_path)
        os.makedirs(original_folder_abs_path)