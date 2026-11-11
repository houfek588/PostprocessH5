import file_processing.read_Nastran_files as ns
import json
from distutils.util import strtobool
# source_file = "C:\\Projects\\MKP\\"
#
#
# b1 = ns.ParseBdfFile("C:\\Projects\MKP\\Vigil_v10\\mkp_02\\box2_freq_Z.bdf")
# # b1 = ns.ParseBdfFile("C:\\Projects\MKP\\Vigil_v10\\mkp_02\\box2_weight.bdf")
# b1.print_statistics()
# b1.print_materials()
# keys = b1.print_loads()
# print(keys)
# # b1.update_load_orientation(1,[1,0,3], "ABCD")
# b1.update_load_orientation(keys[0],[1, 0, 0], "_ABCD")

# b1.print_loads()
# b1.update_material(1, 0, 3111, 0)
# b1.save_bdf_file("edit.bdf")
#
# b1.print_loads()

def gravity_files_change(file_path, file_name):
    print("static file processing...")
    # file_path = "C:\\Projects\MKP\\Vigil_v10\\mkp_02\\"
    # file_name = "box2_weight.bdf"
    print(f"source file: {file_path + file_name}")
    bdf = ns.ParseBdfFile(file_path + file_name)

    load_dict = {"1g_x_ch": [1, 0, 0],
                 "1g_y_ch": [0, 1, 0],
                 "1g_z_ch": [0, 0, -1],
                 "20g_x": [20, 0, 0],
                 "20g_y": [0, 20, 0],
                 "20g_z": [0, 0, -20]}

    just_name = file_name.split(".")[0]
    type = file_name.split(".")[1]

    for n in load_dict.keys():
        bdf.update_load_orientation(1, load_dict[n])
        new_name = just_name + n + "." + type
        bdf.save_bdf_file(file_path + new_name)
        print(f"created: {file_path + new_name}")


def acc_files_change(file_path, file_name):
    print("dynamic file processing...")
    # file_path = "C:\\Projects\MKP\\Vigil_v10\\mkp_02\\"
    # file_name = "box2_freq.bdf"
    print(f"source file: {file_path + file_name}")
    bdf = ns.ParseBdfFile(file_path + file_name)

    load_dict = {"_x": [1, 0, 0],
                 "_y": [0, 1, 0],
                 "_z": [0, 0, 1]}

    just_name = file_name.split(".")[0]
    type = file_name.split(".")[1]

    for n in load_dict.keys():
        bdf.update_load_orientation(6, load_dict[n])
        new_name = just_name + n + "." + type
        bdf.save_bdf_file(file_path + new_name)
        print(f"created: {file_path + new_name}")


# main code
print("---- SCRIPT DONE ----")
config_file = "par_bdf.json"
with open(config_file, "r") as file:
    parameters = json.load(file)
    print(f"configuration loaded from file {config_file}")


# gravity_files_change()
# acc_files_change()
if bool(strtobool(parameters["static_file"])):
    gravity_files_change(parameters["grav_file_path"], parameters["grav_file_name"])

if bool(strtobool(parameters["dynamic_file"])):
    acc_files_change(parameters["dyn_file_path"], parameters["dyn_file_name"])

print("---- SCRIPT DONE ----")
input("Press Enter to continue...")