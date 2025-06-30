#Step 2: Fill in the Incomplete Words in Title Column
#Sample Python Command: python cleanCSV.py /Users/jainamdoshi/Desktop/Projects/DefactoAI/Dataset/fake.csv
#This Script works for title column only

import csv
import re
import sys

# Define abusive words and generate flexible regex patterns
abusive_words = {
    "fuck": ["f\\*+k", "f\\*+ck", "f\\*+k+", "f\\*+u\\*+k", "f\\*+u\\*+c\\*+k"],
    "fucking": ["f\\*+king", "f\\*+cking", "f\\*+u\\*+c\\*+k\\*+ing"],
    "fucker": ["f\\*+cker", "f\\*+ker"],
    "shit": ["s\\*+t", "s\\*+h\\*+i\\*+t", "sh\\*+t"],
    "ass": ["a\\*+", "a\\*+s"],
    "asshole": ["a\\*+hole", "a\\*+shole"],
    "bitch": ["b\\*+ch", "b\\*+tch", "b\\*+\\*+h"],
    "damn": ["d\\*+n"],
    "cunt": ["c\\*+t"],
    "hell": ["h\\*+l"],
    "bastard": ["b\\*+tard"],
    "mother": ["m\\*+r"],
    "motherfucker": ["m\\*+f\\*+r", "m\\*+\\*+f\\*+\\*+r"],
    "dick": ["d\\*+k"],
    "pussy": ["p\\*+sy", "p\\*+\\*+y"],
    "cock": ["c\\*+k"],
    "nigga": ["n\\*+a"],
    "nigger": ["n\\*+r"]
}

# Compile regex patterns for all
pattern_map = {
    re.compile(rf"\b{pattern}\b", re.IGNORECASE): word
    for word, patterns in abusive_words.items()
    for pattern in patterns
}

def replace_abusive_words(text):
    for pattern, replacement in pattern_map.items():
        text = pattern.sub(replacement, text)
    return text

def clean_abusive_titles(csv_file_path):
    cleaned_rows = []

    try:
        with open(csv_file_path, 'r', encoding='utf-8', newline='') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            if 'title' not in fieldnames:
                print("The CSV file does not contain a 'title' column.")
                return

            for row in reader:
                row['title'] = replace_abusive_words(row['title'])
                cleaned_rows.append(row)

        with open(csv_file_path, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cleaned_rows)

        print(f"Abusive words in titles cleaned in: {csv_file_path}")

    except FileNotFoundError:
        print("File not found. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Command-line usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_abuse.py path/to/file.csv")
    else:
        clean_abusive_titles(sys.argv[1])
