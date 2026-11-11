from file_processing import read_Nastran_files, load_measured_data
from visualization import graph_processing as gr
import runNastran
import json
from distutils.util import strtobool

# /opt/msc/nastran/2024.1/bin/msc20241 nastran vigil_geometry_3D_freq_rel.bdf solve=auto    skript na emě

with open("parameters2.json", "r") as file:
    parameters = json.load(file)

nastran_input_file = parameters["bdf_file_path"]
nastr_path = parameters["nastran_path"]

# run nastran solver to get new results
if bool(strtobool(parameters["run_nastran"])):
    runNastran.run_nastran(nastran_input_file, nastr_path)
    runNastran.wait_for_file("random_ram_edit.f06")


# read results from OP2 file
dataOP2 = read_Nastran_files.ParseOP2File(parameters["op2_file_path"])
save_plot_to_file = bool(strtobool(parameters["save_plot_to_file"]))


# plot graphs with results from set nodes and compare it with measured data
gr.plot_immediately = False
for item in parameters["nodes"].items():

    Data2 = load_measured_data.LoadXLS(item[1]["measure_file_path"])

    axis_x = gr.DataAcc(dataOP2.get_frequencies(), dataOP2.get_magnitude_acceleration(item[1]["node"]), "FEM, node " + str(item[1]["node"]), 1/9810)
    axis_y = gr.DataAcc(Data2.get_data_row(0), Data2.get_data_row(4), "measured")

    plot_title = "Acceleration " + str(item[0])
    plot_file_name = "plot_" + str(item[0]) + ".png"
    gr.plot_different_data_len(axis_x, axis_y, plot_title, save_plot_to_file, plot_file_name)


# gr.plot_data(dataOP2.get_frequencies(), [dataOP2.get_acc_node_data(node1,"X"),dataOP2.get_acc_node_data(node1,"Y"),dataOP2.get_acc_node_data(node1,"Z")], "Acceleration real",False)
# gr.plot_data(dataOP2.get_frequencies(), dataOP2.get_acc_node_data(node1,"X"), "Acceleration Y",False)
# gr.plot_data(dataOP2.get_frequencies(), dataOP2.get_acc_node_data(node1,"Y"), "Acceleration Y",False)
# gr.plot_data(dataOP2.get_frequencies(), dataOP2.get_acc_node_data(node1,"Z"), "Acceleration Z",False)



gr.plot_all_graphs()