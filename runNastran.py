import subprocess
import os
import time


def run_nastran(input_file, nastran_path):
    """
    Run a Nastran job using subprocess.

    Args:
        input_file (str): Path to the Nastran input file (.bdf or .dat).
        nastran_path (str): Path to the Nastran executable. Default assumes it's in PATH.

    Returns:
        None
    """
    try:
        # Command to execute Nastran
        command = [nastran_path, input_file]

        # Run the command
        result = subprocess.run(command, check=True, capture_output=True, text=True)


        # Print Nastran's output
        print("Nastran Output:")
        print(result.args)
        # print(result)


    except subprocess.CalledProcessError as e:
        print(f"Error running Nastran: {e}")
        print(f"Standard Output:\n{e.stdout}")
        print(f"Standard Error:\n{e.stderr}")
    except FileNotFoundError:
        print(f"Error: Nastran executable not found at {nastran_path}. Ensure it's installed and in PATH.")


def wait_for_file(file_path, timeout=600):
    """
    Wait for a file to be created and stop changing size (indicating completion).
    """
    start_time = time.time()
    last_size = -1

    while time.time() - start_time < timeout:
        if os.path.exists(file_path):
            current_size = os.path.getsize(file_path)
            if current_size == last_size:
                print(f"File {file_path} is ready.")
                return True
            else:
                last_size = current_size
        else:
            print(f"Waiting for file {file_path} to appear...")

        time.sleep(5)  # Check every 5 seconds

    print(f"Timeout: File {file_path} did not become ready within {timeout} seconds.")
    return False

# Example usage
# input_file = "Random_ram_edit.bdf"  # Replace with your Nastran input file
# path = "C:\\NastranFiles\\MSC_Nastran\\2024.1\\bin\\nast20241w"
#
# print(input_file)
# print(path)
# run_nastran(input_file, path)
# wait_for_file("random_ram_edit.f06")
# print("Nastran completed")