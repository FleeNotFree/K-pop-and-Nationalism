import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import logging
import glob
import warnings
import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from langchain_community.document_loaders import TextLoader, CSVLoader, UnstructuredExcelLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

warnings.filterwarnings('ignore')

os.environ['NUMEXPR_MAX_THREADS'] = '16'

# Set up logs folder and file
os.makedirs('logs', exist_ok=True)
log_filename = f"logs/processing_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
progress_filename = 'logs/last_processed_row.txt'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            log_filename,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)

logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

# Set API keys
OPENAI_API_KEY = 'OPENAI_API_KEY'
HF_API_KEY = 'HUGGING_FACE_API_KEY'
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Set up embedding model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    embedding_model_name = "maidalun1020/bce-embedding-base_v1"
    embedding_model_kwargs = {'device': 'cpu'}
    embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs
    )


def get_loader_for_file(file_path):
    """Return appropriate loader based on file extension"""
    extension = file_path.lower().split('.')[-1]
    loaders = {
        'csv': CSVLoader,
        'xlsx': UnstructuredExcelLoader,
        'xls': UnstructuredExcelLoader,
        'pdf': PyPDFLoader,
        'txt': TextLoader
    }
    return loaders.get(extension)


def detect_delimiter(file_path):
    """Detect the delimiter used in a CSV file"""
    import csv
    from collections import Counter

    # Common delimiters to check
    possible_delimiters = [',', ';', '\t', '|']

    # Read first few lines to detect delimiter
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read up to 5 lines
        sample = ''.join([next(f) for _ in range(5)])

    # Count occurrences of each delimiter in the sample
    counts = Counter(c for c in sample if c in possible_delimiters)

    # If no common delimiter found, default to comma
    if not counts:
        logging.warning(f"No common delimiter found in {file_path}, defaulting to comma")
        return ','

    # Get the most common delimiter
    delimiter = counts.most_common(1)[0][0]
    logging.info(f"Detected delimiter '{delimiter}' for {file_path}")
    return delimiter


def load_documents(file_path):
    """Load documents with appropriate loader and handle CSV delimiter detection"""
    try:
        extension = file_path.lower().split('.')[-1]

        # Special handling for CSV files
        if extension == 'csv':
            delimiter = detect_delimiter(file_path)
            logging.info(f"Loading CSV file {file_path} with delimiter: {delimiter}")
            return CSVLoader(file_path, csv_args={'delimiter': delimiter}).load()

        # Handle other file types
        loader_class = get_loader_for_file(file_path)
        if loader_class:
            loader = loader_class(file_path)
            return loader.load()
        else:
            logging.warning(f"No loader found for file type: {file_path}")
            return []

    except Exception as e:
        logging.error(f"Error loading file {file_path}: {str(e)}")
        return []


def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)


def get_all_reference_files(reference_dir):
    """Get all supported files from reference directory"""
    supported_extensions = ['.pdf', '.csv', '.xlsx', '.xls', '.txt']
    reference_files = []

    for ext in supported_extensions:
        pattern = os.path.join(reference_dir, f'**/*{ext}')
        reference_files.extend(glob.glob(pattern, recursive=True))

    logging.info(f"Found reference files: {reference_files}")
    return reference_files


def load_reference_materials(reference_dir, persist_directory):
    """Load all reference materials from directory into vector database"""
    # Create directories if they don't exist
    os.makedirs(reference_dir, exist_ok=True)
    os.makedirs(persist_directory, exist_ok=True)

    # Check if vector store already exists
    if os.path.exists(os.path.join(persist_directory, 'chroma.sqlite3')):
        logging.info("Loading existing vector store...")
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    reference_files = get_all_reference_files(reference_dir)
    if not reference_files:
        raise ValueError(f"No supported files found in {reference_dir}")

    all_docs = []
    for file in reference_files:
        logging.info(f"Processing reference file: {file}")
        docs = load_documents(file)
        if docs:
            split_docs = split_documents(docs)
            all_docs.extend(split_docs)
            logging.info(f"Successfully processed {file}")
        else:
            logging.warning(f"No documents loaded from {file}")

    if not all_docs:
        raise ValueError("No documents were successfully loaded")

    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    logging.info("Reference materials loaded into vector database.")
    return vectordb


def setup_retrieval_chain(vectordb):
    retriever = vectordb.as_retriever(search_type='similarity', search_kwargs={'k': 7})

    template = r'''====Prompt=======

    Post content：{question}

    Context：{context}
    '''

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.3)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": QA_CHAIN_PROMPT,
            "document_prompt": PromptTemplate(input_variables=["page_content"], template="{page_content}"),
            "document_variable_name": "context",
            "document_separator": "\n\n"
        }
    )

    return qa_chain


def process_csv(input_file, output_file, qa_chain):
    import time

    input_delimiter = detect_delimiter(input_file)
    logging.info(f"Detected delimiter '{input_delimiter}' for input file: {input_file}")

    last_processed_row = 0
    try:
        with open(progress_filename, 'r') as f:
            last_processed_row = int(f.read().strip())
        logging.info(f"Resuming from row {last_processed_row}")
    except FileNotFoundError:
        logging.info("Starting new processing")

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'a' if last_processed_row > 0 else 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile, delimiter=input_delimiter)
        writer = csv.writer(outfile, delimiter=input_delimiter)

        header = next(reader)
        if last_processed_row == 0:
            new_variables = ['In1', 'In2', 'In3', 'In4', 'Out1', 'Out2', 'Out3', 'Out4', 'Total', 'In_Total',
                             'Out_Total', 'Nationalism']
            header.extend(new_variables)
            writer.writerow(header)

        # Skip processed rows
        for _ in range(last_processed_row):
            next(reader)

        for row_index, row in enumerate(reader, start=last_processed_row + 1):
            max_retries = 6
            retry_count = 0

            while retry_count < max_retries:
                try:
                    # 1. Get response
                    post_content = row[2]
                    logging.info(
                        f"Processing row {row_index}" + (f" (Retry {retry_count + 1})" if retry_count > 0 else ""))

                    result = qa_chain.invoke({
                        "query": f"Please code the following Weibo post:\n\n{post_content}"
                    })
                    encoded_result = result['result']

                    logging.debug(f"GPT returned for row {row_index}: {encoded_result}")

                    # 2. Parse
                    parsed_result = {}
                    for line in encoded_result.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            parsed_result[key.strip()] = value.strip()

                    logging.debug(f"Parsed results for row {row_index}: {parsed_result}")

                    # 3. Validate
                    new_variables = ['In1', 'In2', 'In3', 'In4', 'Out1', 'Out2', 'Out3', 'Out4']
                    for var in new_variables:
                        if var not in parsed_result or parsed_result[var] not in ['0', '1']:
                            raise ValueError(f"Missing or invalid value for {var}")

                    # 4. Calculate total score
                    in_total = sum(int(parsed_result[f'In{i}']) for i in range(1, 5))
                    out_total = sum(int(parsed_result[f'Out{i}']) for i in range(1, 5))
                    total = in_total + out_total

                    # 5. Calculate Nationalism type
                    if total == 0:
                        nationalism = 0
                    elif in_total > out_total:
                        nationalism = 1
                    elif in_total == out_total and total > 0:
                        nationalism = 2
                    else:
                        nationalism = 3

                    # 6. Prepare values to write into csv
                    new_values = [parsed_result[var] for var in new_variables]
                    new_values.extend([str(total), str(in_total), str(out_total), str(nationalism)])

                    complete_row = row.copy()
                    complete_row.extend(new_values)

                    # 7. Write and update progress
                    import tempfile
                    import shutil
                    import os

                    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_progress:
                        temp_progress.write(str(row_index))

                    writer.writerow(complete_row)
                    outfile.flush()
                    shutil.move(temp_progress.name, progress_filename)

                    logging.info(f"Successfully processed row {row_index}")
                    break

                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        logging.error(f"Error processing row {row_index} (Attempt {retry_count}): {str(e)}")
                        logging.info(f"Retrying row {row_index}...")
                        time.sleep(1)
                    else:
                        logging.error(f"Failed to process row {row_index} after {max_retries} attempts: {str(e)}")
                        return


def main():
    # Set file paths
    reference_dir = "reference"  # Directory containing reference materials
    persist_directory = "vector_db"
    input_csv = "post.csv"
    output_csv = "post_coded.csv"

    try:
        # Load reference materials into vector database
        logging.info("Loading reference materials...")
        vectordb = load_reference_materials(reference_dir, persist_directory)

        # Set up retrieval chain
        qa_chain = setup_retrieval_chain(vectordb)

        # Process CSV file
        logging.info("Starting CSV processing...")
        process_csv(input_csv, output_csv, qa_chain)

        total_rows = sum(1 for line in open(input_csv)) - 1
        last_row = int(open(progress_filename).read().strip())

        if last_row >= total_rows:
            if os.path.exists(progress_filename):
                os.remove(progress_filename)
            logging.info("All rows processed. Progress file removed.")
        else:
            logging.info(f"Processed up to row {last_row} of {total_rows}. Progress file kept.")

        logging.info("Processing complete!")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.info("Progress file kept for next run")
        raise


if __name__ == "__main__":
    main()