import os

def extract_and_store_file_names(folder_path, output_file):
    try:
        # Get the list of files in the specified folder
        files = os.listdir(folder_path)

        # Open the output file in write mode
        with open(output_file, 'w') as file:
            # Write each file name to the text file
            for file_name in files:
                file.write(file_name.split('.')[0] + '\n')

        print(f"File names extracted and stored in {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Specify the folder path and the output file name
folder_path = r'C:\Users\DELL\Desktop\Stock Market\archive\stocks'
output_file = r'C:\Users\DELL\Desktop\Stock Market\archive\stocks\companies.txt'

# Call the function with the specified folder path and output file name
extract_and_store_file_names(folder_path, output_file)