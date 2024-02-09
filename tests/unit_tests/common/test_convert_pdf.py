from gai.common.PDFConvert import PDFConvert
from gai.common.utils import this_dir
import os

src = os.path.join(this_dir(__file__),"attention-is-all-you-need.pdf")
target = os.path.join(this_dir(__file__),"attention-is-all-you-need.txt")
text=PDFConvert.pdf_to_text(src,False)
with open(target,"w") as f:
    f.write(text)