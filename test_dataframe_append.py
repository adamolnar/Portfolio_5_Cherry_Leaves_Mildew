import pandas as pd

# Initialize a DataFrame
df = pd.DataFrame(columns=['Name', 'Result'])

# Data to append
new_row = {'Name': 'Sample Name', 'Result': 'Positive'}

# Append the new row to the DataFrame
df = df.append(new_row, ignore_index=True)

# Print the DataFrame to verify the append operation
print(df)
