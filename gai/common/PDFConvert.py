from unstructured.partition.pdf import partition_pdf
import re

class PDFConvert:

    @staticmethod
    def pdf_to_text(pdf_file_path, clean=True):
        elements = partition_pdf(pdf_file_path)
        text="\n\n".join([str(el) for el in elements])
        text = re.sub(r'\(cid:[^\)]*\)', '', text)
        if clean:
            text = re.sub(r'\s+', ' ', text)
        return text


