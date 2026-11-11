from pathlib import Path
import matplotlib.pyplot as plt
import csv


class LoadMSC:
    def __init__(self, file_path):
        self.file_path = Path(file_path)

        # Initialize attributes
        self.x = []
        self.y = []
        self.x_title = None
        self.y_title = None

        try:
            # Open the file and parse the content
            with open(self.file_path, 'r') as file:
                content = csv.reader(file, delimiter=' ', quotechar='|')

                for i, row in enumerate(content):
                    if i == 0:  # First row is the title
                        self.x_title = row[0]
                        self.y_title = row[1]
                    else:  # Data rows
                        try:
                            self.x.append(float(row[0]))
                            self.y.append(float(row[1]))
                        except ValueError as e:
                            print(f"Warning: Skipped row {i} due to error: {e}")

            print(f"File {self.file_path} was successfully read.")

        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def get_data(self):
        return [self.x, self.y]

    def get_Xdata(self):
        return self.x

    def get_Ydata(self):
        return self.y

    def get_Ydata_scaled(self, scale):
        y = []
        for value in self.y:
            y.append(value * scale)
        return y

    def print_data(self, scale):
        # Create the plot
        y = []
        for value in self.y:
            y.append(value * scale)

        plt.plot(self.x, y, linestyle='-', color='b', label='y = 2x')

        # Add labels and title
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("XY Data Plot")
        plt.legend()  # Add legend for clarity

        # Show the plot
        plt.grid(True)  # Optional: Add gridlines
        plt.show()


