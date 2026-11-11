# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
from file_processing import load_MSC_data, load_measured_data

# file_data = {
#     "cover_middle": ["../MKP_02/acc_sweep5g01_mid_cover.xyd", "../Shaker_data/t13.xls"],
#     "PCB_middle":   ["../MKP_02/acc_sweep5g01_mid_PCB.xyd", "../Shaker_data/t11.xls"],
#     "PCB_right":    ["../MKP_02/acc_sweep5g01_right_PCB.xyd", "../Shaker_data/t12.xls"]
# }

file_data = {
    "cover_middle": ["../MKP_03/acc_sweep5g_s1_mid_cover.xyd", "../Shaker_data/t13.xls"],
    "PCB_middle":   ["../MKP_03/acc_sweep5g_s1_mid_PSB.xyd", "../Shaker_data/t11.xls"],
    "PCB_right":    ["../MKP_03/acc_sweep5g_s1_right_PSB.xyd", "../Shaker_data/t12.xls"],
    "inner_middle":   ["../MKP_03/acc_sweep5g_s1_mid_inner.xyd", "../Shaker_data/t15.xls"],
}

chosen_data = "cover_middle"



# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
def plot_compare_graph(file_data, chosen_data):

    file_path_FEM = file_data[chosen_data][0]
    file_path_measured = file_data[chosen_data][1]

    graph_title = "Acceleration: " + chosen_data

    Data1 = load_MSC_data.LoadMSC(file_path_FEM)
    Data2 = load_measured_data.LoadXLS(file_path_measured)

    print(Data2.get_titles())



    print(type(Data1.get_Xdata()[0]))
    print(Data1.get_Xdata())
    # plt.style.use('dark_background')
    fig = plt.figure(figsize=(4, 3))
    plt.plot(Data1.get_Xdata(), Data1.get_Ydata_scaled(1/9810.0), linestyle='-', color='b', label='FEM')
    plt.plot(Data2.get_data_row(0), Data2.get_data_row(4), linestyle='--', color='r', label='measured')

    # Add labels and title
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Acceleration [g]")
    plt.title(graph_title)
    plt.legend()  # Add legend for clarity


    # Show the plot
    plt.grid(True)  # Optional: Add gridlines
    return fig

print(file_data.keys())
for gr in file_data.keys():
    plot_compare_graph(file_data, gr)
plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
