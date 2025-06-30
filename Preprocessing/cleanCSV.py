#Step 1: Clean Special Chars, Brackets etc and keep the words and letters intact
#Sample Python Command: python cleanCSV.py /Users/jainamdoshi/Desktop/Projects/DefactoAI/Dataset/fake.csv
#This Script works for title column only

import csv
import string
import sys
import re

def clean_title(title):
    # Remove text inside any type of brackets
    title = re.sub(r'\(.*?\)|\[.*?\]|\{.*?\}|<.*?>', '', title)

    # Define allowed characters: A-Z, a-z, 0-9, punctuation (excluding brackets)
    allowed_chars = set(string.ascii_letters + string.digits + string.punctuation)
    brackets_to_exclude = set("(){}[]<>")
    allowed_chars -= brackets_to_exclude

    # Remove any character not in allowed set and not a whitespace
    cleaned = ''.join(c for c in title if c in allowed_chars or c.isspace())
    
    return cleaned.strip()

def clean_csv_titles(csv_file_path):
    cleaned_rows = []

    try:
        # Read and clean
        with open(csv_file_path, 'r', encoding='utf-8', newline='') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            if 'title' not in fieldnames:
                print("The CSV file does not contain a 'title' column.")
                return

            for row in reader:
                row['title'] = clean_title(row['title'])
                cleaned_rows.append(row)

        # Write cleaned rows back to the same file
        with open(csv_file_path, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cleaned_rows)

        print(f"Cleaned titles written back to '{csv_file_path}'.")

    except FileNotFoundError:
        print("File not found. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_titles.py path/to/file.csv")
    else:
        clean_csv_titles(sys.argv[1])
