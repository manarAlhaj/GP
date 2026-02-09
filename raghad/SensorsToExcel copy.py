import pandas as pd
import time
from Sensors import CollectData


input_file = "C:/Users/LENOVO/OneDrive - University Of Jordan/Desktop/Grad Proj/Book1.xlsx" 
df_input = pd.read_excel(input_file, header=None)  # read the file 

output_file = "output.xlsx"     
results = []

num_trials = 1      
trial_duration = 2   
sample_interval = 1   

for index, row in df_input.iterrows():
    word = str(row[0])  
    print(f"\nCurrent word: {word}")

    for trial_num in range(1, num_trials + 1):
        input(f"\nPress Enter to start recording trial {trial_num}/{num_trials} (will take {trial_duration} seconds)...")
        print("Collecting Flex + IMU data...")

        samples = CollectData() # the function that has the data
        
        if samples is None or len(samples) == 0:            # no data 
            print("No data collected, skipping trial.")
            continue
        
        for sample in samples:
            results.append([word, trial_num] + sample)
        print(f"Trial {trial_num}/{num_trials} for word '{word}' is done! Press Enter for next trial.")

# save all results to Excel file 
columns = ["Word", "Thumb", "Index", "Middle", "Ring", "Pinky" , "sinY" , "cosY", "sinP" ,"cosP" ,"sinR", "cosR", "deltaY","deltaP" ,"deltaR"]
df_output = pd.DataFrame(results, columns=columns)
df_output.to_excel(output_file, index=False)
print(f"\n Result is saved in : ", output_file)
