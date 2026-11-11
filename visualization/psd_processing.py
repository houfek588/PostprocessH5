import graph_processing as gr
from file_processing import read_Nastran_files, load_measured_data

def lin_interpolate_curve_points(x, curve_points):
    """
    Perform linear interpolation for a given x based on provided curve points.

    Args:
        x (float): The x-value to interpolate.
        curve_points (list of tuples): A list of (x, y) coordinate pairs.

    Returns:
        float: Interpolated y-value.
    """
    # Ensure we have at least two points for interpolation
    if len(curve_points) < 2:
        raise ValueError("At least two points are required for interpolation.")

    # Sort points by x-value (in case they are unordered)
    curve_points = sorted(curve_points, key=lambda p: p[0])

    # Handle cases where x is outside the provided range
    if x <= curve_points[0][0]:
        return curve_points[0][1]  # Return first y-value (constant extrapolation)
    if x >= curve_points[-1][0]:
        return curve_points[-1][1]  # Return last y-value (constant extrapolation)

    # Iterate through curve points to find the correct segment
    for i in range(len(curve_points) - 1):
        x1, y1 = curve_points[i]
        x2, y2 = curve_points[i + 1]

        if x1 <= x < x2:  # Find the correct interval
            a = (y2 - y1) / (x2 - x1)  # Slope
            b = y1 - a * x1  # Intercept
            return a * x + b  # Linear interpolation

    # Default return (should never be reached)
    return None


def random_psd(freq, input, psd_curve):
    pass
    

# psd_input = [(20, 0.4811805),
#              (90, 9.62361),
#              (400, 9.62361),
#              (2000, 0.3849444)]

# freq = range(1, 2200, 10)
# n = len(freq)
#
# p = []
# r = []
# q = []
# a1 = (psd_input[1][1] - psd_input[0][1])/(psd_input[1][0] - psd_input[0][0])
# b1 = psd_input[0][1] - a1 * psd_input[0][0]
#
# a2 = (psd_input[2][1] - psd_input[1][1])/(psd_input[2][0] - psd_input[1][0])
# b2 = psd_input[1][1] - a2 * psd_input[1][0]
#
# a3 = (psd_input[3][1] - psd_input[2][1])/(psd_input[3][0] - psd_input[2][0])
# b3 = psd_input[2][1] - a3 * psd_input[2][0]
#
# print(f"a1 = {a1}, b1 = {b1}")
# print(f"a2 = {a2}, b2 = {b2}")
# print(f"a3 = {a3}, b3 = {b3}")
#
# for f in freq:
#     if f < psd_input[0][0]:
#         p.append(0)
#     elif f < psd_input[1][0]:
#         # psd_input[0][1]
#
#         p.append(a1 * f + b1)
#     elif f < psd_input[2][0]:
#         # p.append(psd_input[1][1])
#         p.append(a2 * f + b2)
#     elif f < psd_input[3][0]:
#         # p.append(psd_input[2][1])
#         p.append(a3 * f + b3)
#     else:
#         p.append(0)
#
#     r.append(lin_interpolate_curve_points1(f, psd_input))
#     q.append(lin_interpolate_curve_points(f, psd_input))




# axis_x = gr.DataAcc(dataOP2.get_frequencies(), dataOP2.get_magnitude_acceleration(item[1]["node"]), "FEM, node " + str(item[1]["node"]), 1/9810)
# axis_y = gr.DataAcc(Data2.get_data_row(0), Data2.get_data_row(4), "measured")
#
# plot_title = "Acceleration " + str(item[0])
# plot_file_name = "plot_" + str(item[0]) + ".png"
# gr.plot_different_data_len(axis_x, axis_y, plot_title, save_plot_to_file, plot_file_name)
path = "../../../Vigil_v10/mkp_01/freq_results/box_frequency_pcb.op2"
dataOP2 = read_Nastran_files.ParseOP2File(path)

node = 6126275
acc = dataOP2.get_magnitude_acceleration(node)
fr = dataOP2.get_frequencies()

gr.plot_data(fr, acc, "acc node " + str(node))
# gr.plot_data(freq,r,"psd1")
# gr.plot_data(freq,q,"psd2")

gr.plot_all_graphs()