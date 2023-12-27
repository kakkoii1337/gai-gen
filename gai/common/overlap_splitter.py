import os
from gai.common import logging
logger = logging.getLogger(__name__)

class OverlapChunkSplitter:

    def __init__(self, overlap=0.1, max_chunk_size=2048):
        self.overlap = overlap
        self.max_chunk_size = max_chunk_size # context size is 512 tokens approx. 4 x 512 token = 2048 chars

    # Max size of each chunk if 512
    # The text will be split into n chunks such that each chunk is less than 512k but the sum of any 2 chunks is greater than 512
    # Each chunk should also be as similar in size as possible.

    def split(self, text):

        # Find the least number of chunks that will fit the text
        # Each chunk should be less than 512k
        file_size = len(text)
        self.chunk_count = (file_size // self.max_chunk_size) + 1

        chunks = []
        block_size = file_size // self.chunk_count

        # calculate 10% of block size for context overlap
        overlap_size = int(block_size * self.overlap) 
        logger.debug('overlap:'+str(overlap_size) )
        if block_size <= overlap_size:
            raise ValueError("Block size must be larger than overlap")

        start_position = 0
        for i in range(self.chunk_count):

            #last chunk is the longest
            end_position = start_position + block_size
            if i == self.chunk_count - 1:
                chunk = text[start_position:]
            else:
                logger.debug('block_size:'+str(block_size))
                logger.debug('start_position:'+str(start_position))
                chunk = text[start_position:end_position]
            chunks.append(chunk)
            
            # If next run is the last chunk, then don't subtract overlap
            new_position = end_position
            if (new_position + block_size < len(text)):
                new_position = end_position - overlap_size
            start_position = new_position

        return chunks

    def split_by_count(self,input_file_path):
        logger.debug('split: input_file_path='+input_file_path)
        input_file_name = os.path.splitext(os.path.basename(input_file_path))[0]
        output_dir = utils.mkdir_cache('chunks')
        output_file_prefix = os.path.join(output_dir, input_file_name)

        with open(input_file_path, 'r') as f:
            text = f.read()

        chunks = self.split(text)
        count = len(chunks)
        for i in range(count):
            output_file_path = f"{output_file_prefix}.{str(i).zfill(3)}"
            logger.debug(f'[{i}/{count}] Creating {output_file_path}...{len(chunks[i])} chars')
            with open(output_file_path, 'w') as output_file:
                output_file.write(chunks[i])
        
        logger.debug('Done.')
