import pandas as pd
import time

# --- Simulated Flex Sensor function ---
def read_flex_sensor():
    # return the readings you provided
    return [0.2, 0.3, 0.1, 0.5, 0.6]

# --- Setup test Excel file ---
# We'll create a dummy Excel file with 2 words for testing
test_words = ["سلام", "مرحبا"]
df_input = pd.DataFrame(test_words)
input_file = "test_input.xlsx"
df_input.to_excel(input_file, index=False, header=False)

output_file = "test_output.xlsx"
results = []

num_trials = 3        # For testing, do 3 trials per word
trial_duration = 5    # For testing, each trial lasts 5 seconds
sample_interval = 1   # Read every second

# --- Loop through words ---
for index, row in df_input.iterrows():
    word = str(row[0])
    print(f"\nCurrent word: {word}")

    for trial_num in range(1, num_trials + 1):
        input(f"\nPress Enter to start trial {trial_num}/{num_trials} (will last {trial_duration} seconds)...")

        start_time = time.time()
        while time.time() - start_time < trial_duration:
            readings = read_flex_sensor()
            results.append([word] + readings)
            print(f"Readings: {readings}")
            time.sleep(sample_interval)

        print(f"Trial {trial_num}/{num_trials} for '{word}' finished! Press Enter to start the next trial.")

# --- Save results to Excel ---
columns = ["Word", "Thumb", "Index", "Middle", "Ring", "Pinky"]
df_output = pd.DataFrame(results, columns=columns)
df_output.to_excel(output_file, index=False)

print("\nAll readings saved to:", output_file)
import os
print("Saved at:", os.path.abspath("test_output.xlsx"))

