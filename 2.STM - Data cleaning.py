import pandas as pd
import jieba
import re
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter


def check_file_exists(filepath):
    """Check if file exists"""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' does not exist!")
        print(f"Current working directory: {os.getcwd()}")
        print("Files in current directory:")
        print('\n'.join(os.listdir('.')))
        return False
    return True


def try_read_first_line(filepath):
    """Try different encodings to read the first line of the file"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'ascii']

    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                first_line = f.readline().strip()
                print(f"\nSuccessfully read first line using {encoding} encoding:")
                print(first_line)
                return encoding
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Encountered other error while reading: {str(e)}")

    print("Unable to read file with any encoding!")
    return None


def read_csv_file(filepath, encoding):
    """Try to read CSV file with different delimiters"""
    delimiters = [',', '\t', ';', '|']

    for delimiter in delimiters:
        try:
            df = pd.read_csv(filepath, delimiter=delimiter, encoding=encoding)
            print(f"\nSuccessfully read file using delimiter '{delimiter}'")
            return df
        except pd.errors.EmptyDataError:
            print(f"File is empty!")
            return None
        except Exception as e:
            continue

    print("Unable to read file with any common delimiter!")
    return None


def load_stopwords(stopwords_file):
    """Load stopwords"""
    try:
        if not os.path.exists(stopwords_file):
            print(f"Warning: Stopwords file '{stopwords_file}' does not exist, will not use stopwords")
            return set()

        with open(stopwords_file, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f])
            print(f"Successfully loaded {len(stopwords)} stopwords")
            return stopwords
    except Exception as e:
        print(f"Error loading stopwords file: {str(e)}")
        return set()


def load_keywords(keywords_file):
    """Load keywords"""
    try:
        if not os.path.exists(keywords_file):
            print(f"Warning: Keywords file '{keywords_file}' does not exist!")
            return set()

        with open(keywords_file, 'r', encoding='utf-8') as f:
            keywords = set([line.strip() for line in f])
            print(f"Successfully loaded {len(keywords)} keywords")
            # Add keywords to jieba dictionary
            for keyword in keywords:
                jieba.add_word(keyword)
            return keywords
    except Exception as e:
        print(f"Error loading keywords file: {str(e)}")
        return set()


def preprocess_text(text, stopwords):
    """Text preprocessing"""
    try:
        # Convert to string
        text = str(text)

        # Keep only Chinese characters
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)

        # Word segmentation
        words = jieba.cut(text)

        # Remove stopwords
        words = [word for word in words if word not in stopwords and len(word.strip()) > 0]

        # Join words with spaces
        return ' '.join(words)
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return ""


def process_date(date_str):
    """Process date string, standardize format to YYYY-MM-DD"""
    try:
        if pd.isna(date_str):
            return None

        date_str = str(date_str)
        if '年' in date_str:
            # Full date format: "2023年03月22日 13:29"
            date_part = re.match(r'(\d{4})年(\d{2})月(\d{2})日', date_str)
        else:
            # Short date format: "03月22日 13:29"
            date_part = re.match(r'(\d{2})月(\d{2})日', date_str)
            if date_part:
                return f"2024-{date_part.group(1)}-{date_part.group(2)}"

        if date_part:
            if len(date_part.groups()) == 3:
                return f"{date_part.group(1)}-{date_part.group(2)}-{date_part.group(3)}"
    except Exception as e:
        print(f"Error processing date: {date_str}, {str(e)}")
    return None


def get_monthly_stats(dates):
    """Count articles by month"""
    monthly_counts = Counter([date[:7] for date in dates if pd.notna(date)])
    return dict(sorted(monthly_counts.items()))


def main():
    # Set file paths
    input_file = 'everything.csv'
    stopwords_file = 'stopwords.txt'
    keywords_file = 'keywords.txt'
    output_file = 'tokenized.csv'

    # Check if input file exists
    if not check_file_exists(input_file):
        sys.exit(1)

    # Try to read file and determine encoding
    encoding = try_read_first_line(input_file)
    if not encoding:
        sys.exit(1)

    # Read CSV file
    df = read_csv_file(input_file, encoding)
    if df is None:
        sys.exit(1)

    # Print DataFrame information
    print("\nDataFrame info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())

    # Rename columns
    column_mapping = {
        '正文': 'text',
        '性质': 'event',
        'total_include': 'fan'
    }

    # Check and rename existing columns
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
            print(f"\nRenamed column '{old_col}' to '{new_col}'")

    # Check if necessary columns exist
    if 'text' not in df.columns:
        print(f"Error: No 'text' column in data!")
        print("Available columns:", df.columns.tolist())
        sys.exit(1)

    # Load stopwords
    stopwords = load_stopwords(stopwords_file)

    # Load keywords and add to jieba dictionary
    keywords = load_keywords(keywords_file)

    # Process text
    print("\nStarting text processing...")
    df['text'] = df['text'].apply(lambda x: preprocess_text(x, stopwords))

    # Remove empty text
    original_len = len(df)
    df = df[df['text'].str.len() > 0]
    removed_count = original_len - len(df)
    if removed_count > 0:
        print(f"\nRemoved {removed_count} records with empty text")

    # Remove rows where Nationalism is 0
    if 'Nationalism' in df.columns:
        original_len = len(df)
        df = df[df['Nationalism'] != 0]
        nationalism_removed = original_len - len(df)
        print(f"\nRemoved {nationalism_removed} records with Nationalism = 0")
        print(f"Remaining records: {len(df)}")

    # Process date column
    if '发布日期' in df.columns:
        print("\nProcessing dates...")
        df['Date'] = df['发布日期'].apply(process_date)

        # Count by month
        monthly_stats = get_monthly_stats(df['Date'])
        print("\nMonthly article count statistics:")
        for month, count in monthly_stats.items():
            print(f"{month}: {count} articles")

        # Adjust column order
        cols = df.columns.tolist()
        pub_date_idx = cols.index('发布日期')
        cols.insert(pub_date_idx + 1, 'Date')
        cols.remove('Date')  # Remove Date from original position
        df = df[cols]
    else:
        print("Could not find '发布日期' column")

    # Save processed data
    try:
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nSuccessfully saved processed data to: {output_file}")
        print(f"Number of documents after processing: {len(df)}")
        print("\nExamples of processed text:")
        print(df['text'].head())
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Program execution error: {str(e)}")
        sys.exit(1)
