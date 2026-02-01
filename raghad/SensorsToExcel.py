import pandas as pd
import time
import FlexSensors  


def read_flex_sensor():
    flex = FlexSensors.FlexSensors()
    voltages = flex.read_voltage()
    # we should add the flex sensor code here 
    return voltages
    #return [0.2, 0.3, 0.1, 0.5, 0.6]

#Input file with words (try with test file first)
input_file = "C:/Users/LENOVO/OneDrive - University Of Jordan/Desktop/Grad Proj/Book1.xlsx" 
df_input = pd.read_excel(input_file, header=None)  # read the file 

output_file = "output.xlsx"   # the file that contains words + sensors values 
results = []

num_trials = 1       # Number of trials per word
trial_duration = 2    # Seconds per trial
sample_interval = 1   

# read the words from input file line by line 
for index, row in df_input.iterrows():
    word = str(row[0])  
    print(f"\nCurrent word: {word}")

    for trial_num in range(1, num_trials + 1):
        input(f"\nPress Enter to start recording trial {trial_num}/{num_trials} (will take {trial_duration} seconds)...")
       
        start_time = time.time()
        last_reading = None

        while time.time() - start_time < trial_duration:
            last_reading = read_flex_sensor()  # read sensor
            print(f"Reading: {last_reading}")
            time.sleep(sample_interval)
            
        results.append([word] + last_reading)
        print(f"Trial {trial_num}/{num_trials} for word '{word}' is done! Press Enter for next trial.")

# save all results to Excel file 
columns = ["Word", "Thumb", "Index", "Middle", "Ring", "Pinky"]
df_output = pd.DataFrame(results, columns=columns)
df_output.to_excel(output_file, index=False)
print(f"\n Result is saved in : ", output_file)
import os
print(os.path.abspath(output_file))