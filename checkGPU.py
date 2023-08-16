import subprocess
import time

# Define the bash command to be checked
bash_command = "test `nvidia-smi | grep % | sed 's/.*\\(..\\)%.*/\\1/g'` -eq 0"

# Define the duration for which the condition should be checked (in seconds)
check_duration = 300  # 5 minutes

while True:
    # Get the current timestamp
    start_time = time.time()

    # Flag to track if the condition is true for the entire duration
    condition_met = True

    while time.time() - start_time < check_duration:
        # Execute the bash command and capture its output
        result = subprocess.run(bash_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check the exit code of the command and the extracted percentage
        if result.returncode == 0 :
            print("Condition is true.")
        else:
            print("Condition is false.")
            condition_met = False
            break  # Exit the inner loop if condition is not met

        # Wait for a second before checking again
        time.sleep(1)

    if condition_met:
        print("Condition held true for the last 5 minutes.")
        subprocess.run(["sudo", "shutdown", "-h", "now"])

    # Wait for a short interval before starting the next check
    time.sleep(1)  # Sleep for the remaining time (1 minute) before starting next check

