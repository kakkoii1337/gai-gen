import os, zipfile
import tempfile,shutil,re
from . import constants,utils
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Remove most non-alphanumeric characters from a filename
def clean_paths(file_path_or_paths):
    if (isinstance(file_path_or_paths,list)):
        paths = []
        for file_path in file_path_or_paths:
            paths.append(clean_paths(file_path))
        return paths
    return file_path_or_paths.replace("/","_").replace("\\","_").replace(" ","_").replace(":","").replace(",","").replace("'","").replace('"','').lower()

def flatten_abs_paths(dir_or_file):
    abs_file_paths = []
    if os.path.isfile(dir_or_file):
        abs_file_paths.append(os.path.abspath(dir_or_file))
        return abs_file_paths
    
    for dirpath, _, filenames in os.walk(dir_or_file):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            abs_file_paths.append(os.path.abspath(file_path))
    return abs_file_paths

def unzip_and_remove(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)  # remove the .zip file after extraction

def _unzip_temp(temp_dir):
    for root, dirs, files in os.walk(temp_dir):
        for filename in files:
            if filename.endswith(".zip"):
                zip_file_path = os.path.join(root, filename)
                unzip_and_remove(zip_path=zip_file_path, extract_to=root)
                _unzip_temp(root) # recursive call to handle nested zip files

def unzip_all(file_or_dir, dest_dir=None):
    # Copy all into a temp dir
    temp_dir = tempfile.mkdtemp()
    shutil.copytree(file_or_dir, temp_dir, dirs_exist_ok=True)

    # Recursively unzip zipped files
    _unzip_temp(temp_dir)

    # Move all files to dest_dir (if exists)
    if dest_dir:
        shutil.copytree(temp_dir, dest_dir, dirs_exist_ok=True)
        shutil.rmtree(temp_dir)
        return dest_dir
    
    return temp_dir

# Purpose: Create a directory using the url or path in ~/.locallm/chunks directory
def get_chunk_dir(chunk_dir, path_or_url):
    if (utils.is_url(path_or_url)):
        path_or_url = path_or_url.replace("://","_").replace("/","_").replace(".","_")
    else:
        path_or_url = os.path.basename(path_or_url).split('.')[0]
    chunk_name=re.sub(r'^_+', '', path_or_url)
    return os.path.join(chunk_dir, chunk_name)

# Purpose: Create a chunk id using sha256 of its context
def create_chunk_id(text):
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# Purpose: Split text using LangChain's recursive text splitter into chunks in the chunks_dir
def split_chunks(text, chunks_dir=None,chunk_size=2000,chunk_overlap=200):
    
    # Remove existing chunks
    if (chunks_dir != None):
        if os.path.exists(chunks_dir):
            shutil.rmtree(chunks_dir)    
        os.makedirs(chunks_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,        # approx 512 tokens
        chunk_overlap=chunk_overlap,     # 10% overlap
        length_function=len,
        is_separator_regex=False
    )
    chunks = splitter.create_documents([text])
    for chunk in chunks:
        chunk_id = create_chunk_id(chunk.page_content)
        chunk.metadata = {
            "chunk_id": chunk_id,
            "chunk_size": len(chunk.page_content)
        }
        if (chunks_dir != None):
                chunk_fname = os.path.join(chunks_dir,chunk_id)
                with open(chunk_fname,'w') as f:
                    f.write(chunk.page_content)
    return chunks

