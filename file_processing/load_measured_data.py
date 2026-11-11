from pathlib import Path
import matplotlib.pyplot as plt
import csv


class LoadXLS:
    def __init__(self, file_path):
        self.file_path = Path(file_path)

        # Initialize attributes
        self.titles = None
        self.data = []

        try:
            with open(self.file_path, 'r') as file:
                content = csv.reader(file, delimiter='\t', quotechar='|')

                # Initialize title and lines
                title = []
                lines = []

                # Process rows in the CSV
                for row in content:
                    if self.is_float(row[0]):
                        try:
                            # Convert all valid values in the row to floats
                            frow = [float(i) for i in row]
                            lines.append(frow)
                        except ValueError as e:
                            # Handle invalid values and provide feedback
                            print(f"Row was unread due to invalid value: {row} | Error: {e}")
                    else:
                        title = row  # Assuming the first non-numeric row is the title

                # Transpose the data
                transposed = list(zip(*lines))

                # Set titles and data
                self.titles = title
                self.data = transposed

            print(f"File {self.file_path} was successfully read.")

        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def is_float(self, value):
        """Check if a value can be converted to a float."""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def get_data(self):
        return self.data

    def get_data_row(self, row):
        return self.data[row]

    def get_titles(self):
        return self.titles

    def print_data(self, data_row, scale):

        # Create the plot
        y = []
        for value in self.data[data_row]:
            y.append(value * scale)

        print(type(y[0]))

        plt.plot(self.data[0], y, linestyle='-', color='b', label='y = 2x')

        # Add labels and title
        plt.xlabel(self.titles[0])
        plt.ylabel(self.titles[data_row])
        plt.title("XY Data Plot")
        plt.legend()  # Add legend for clarity

        # Show the plot
        plt.grid(True)  # Optional: Add gridlines
        plt.show()


