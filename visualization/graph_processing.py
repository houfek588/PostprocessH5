import math
import matplotlib.pyplot as plt

plot_immediately = False


class DataAcc:
    def __init__(self, data_x, data_y, title, scale=0):
        self.dataX = data_x
        self.dataY = data_y
        self.title = title
        self.scale = 0

        self.set_y_data_scale(scale)

    def get_x(self):
        return self.dataX

    def get_y(self):
        return self.dataY

    def get_y_data_scaled(self):
        if self.scale == 0:
            print(f"{self.title}: scale is 0, returned Y data without scaling")
            return self.dataY
        else:
            return [(value * self.scale) for value in self.dataY]

    def get_title(self):
        return self.title

    def get_scale(self):
        return self.scale

    def set_y_data_scale(self, scale):
        self.scale = scale


def plot_data(dataX, dataY, title="title", save_to_file: bool = False, file_name="plot.png"):
    """
    Plots nodal accelerations from acceleration data.

    Args:
        acceleration_data (dict): Acceleration results from OP2 file.
    """
    fig = plt.figure(figsize=(4, 3))

    if len(dataY) < 5:
        for i in range(0,len(dataY)):
            plt.plot(dataX, dataY[i])
    else:
        plt.plot(dataX, dataY)


    plt.xlabel("Frequency")
    plt.ylabel("Acceleration")
    plt.title(title)
    plt.grid(True)

    if save_to_file:
        # Save the plot to a file
        plt.savefig(file_name, dpi=300) # Save as PNG file

    if plot_immediately:
        plt.show()


def plot_different_data_len(data1: DataAcc, data2: DataAcc, name="Nodal Accelerations", save_to_file: bool = False, file_name="plot.png"):
    """
    Plots nodal accelerations from acceleration data.

    Args:
        acceleration_data (dict): Acceleration results from OP2 file.
    """
    fig = plt.figure(figsize=(4, 3))
    plt.plot(data1.get_x(), data1.get_y_data_scaled(), label=data1.get_title(), linestyle='-', color='b')
    plt.plot(data2.get_x(), data2.get_y_data_scaled(), label=data2.get_title(), linestyle='--', color='r')

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Acceleration [g]")
    plt.title(name)
    plt.legend()
    plt.grid(True)

    if save_to_file:
        # Save the plot to a file
        plt.savefig(file_name, dpi=300) # Save as PNG file

    if plot_immediately:
        plt.show()


def plot_all_graphs():
    plt.show()


# # Example usage
# op2_file = "random_ram_edit.op2"
# results = read_op2_file(op2_file)
#
# for subcase, accelerations in results.items():
#     df = accelerations.freqs
#
#     real_part = accelerations.data.real
#     accX = get_node_data(1, "X", real_part)
#     accY = get_node_data(1, "Y", real_part)
#     accZ = get_node_data(1, "Z", real_part)
#
#     # acc = [math.sqrt(pow(accX[i],2)+pow(accY[i],2)+pow(accZ[i],2))] for i in range(0, len(accX))
#     acc = []
#     for i in range(0, len(accX)):
#         acc.append(math.sqrt(pow(accX[i], 2) + pow(accY[i], 2) + pow(accZ[i], 2)))
#
#     Acc_X = DataAcc(df, accX, "acc X")
#     Acc_Y = DataAcc(df, accY, "acc Y")
#
#
# a1 = [accX, accY, accZ]
#
#
# plot_different_data_len(Acc_X, Acc_Y)
# Plot accelerations if available
# if results:
#     plot_accelerations(df, a1, True)