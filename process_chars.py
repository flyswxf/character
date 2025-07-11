import pandas as pd

def process_characters():
    txt_file_path = 'chinese/3500常用字.txt'
    csv_file_path = 'four_corner_data.csv'

    # Read characters from the text file
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        all_chars = f.read()
        # Use a set to get unique characters
        unique_chars = set(list(all_chars.replace('\n', '')))

    # Create a new DataFrame with the unique characters
    df = pd.DataFrame(list(unique_chars), columns=['character'])
    df['four_corner'] = '' # Add an empty column for four_corner codes

    # Write to the CSV file, overwriting existing content but keeping the header
    df.to_csv(csv_file_path, index=False, encoding='utf-8')

    print(f"Successfully processed {len(unique_chars)} unique characters into {csv_file_path}")

if __name__ == "__main__":
    process_characters()