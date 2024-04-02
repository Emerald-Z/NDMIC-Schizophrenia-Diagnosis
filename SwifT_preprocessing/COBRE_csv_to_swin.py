import pandas as pd

# Read the original CSV file without headers
df = pd.read_csv('COBRE_phenotypic_data.csv', header=None, skiprows=1)

# Create an empty list to store rows for the new DataFrame
new_data = []

for index, row in df.iterrows():
    subjectkey = row[0]  # Assuming the subject column is the first column
    schizo = 1 if row[4] == 'Patient' else 0  # Assuming 'Subject Type' is the fifth column
    new_data.append({'subjectkey': subjectkey, 'schizo': schizo})

# Convert the list of dictionaries to a DataFrame
new_df = pd.DataFrame(new_data)

# Save the new DataFrame to a new CSV file
new_df.to_csv('/Users/ejzhang/Downloads/NDMIC/COBRE_MNI_to_TRs/metadata/COBRE_test.csv', index=False)
