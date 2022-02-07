import os
import subprocess
import time


def main():
    project_folder_abs_path = os.path.dirname(os.path.abspath("__file__")).replace(os.sep, "/")

    build_folder_abs_path = project_folder_abs_path + "/build"
    if not os.path.exists(build_folder_abs_path):
        os.makedirs(build_folder_abs_path)
    cmake_command = subprocess.Popen("cmake " + chr(34) + project_folder_abs_path + chr(34) + " -B " + chr(34) + \
        build_folder_abs_path + chr(34), shell=True)

    time.sleep(1)

    for solution_file_name in os.listdir(build_folder_abs_path):
        if solution_file_name.endswith(".sln"):
            solution_file_abs_path = build_folder_abs_path + "/" + solution_file_name
            devenc_command = subprocess.Popen("devenv " + chr(34) + solution_file_abs_path + chr(34) + " /Build " + chr(34) + \
                "Release|x64" + chr(34), shell=True)
    
    input()


if __name__ == "__main__":
    main()