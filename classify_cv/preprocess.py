# classify_cv/preprocess.py
import re

def preprocess_cv(cv_text: str) -> str:
    sections = re.findall(
        r"(Summary|Professional skills|Projects|Work Experience)(.*?)(Education|Certificates|Soft Skills|$)",
        cv_text,
        re.DOTALL | re.IGNORECASE
    )

    if sections:
        extracted = " ".join([s[1] for s in sections])
    else:
        extracted = cv_text  

    extracted = re.sub(r"\n+", " ", extracted)        
    extracted = re.sub(r"[^a-zA-Z0-9\s.,/()+-]", " ", extracted)  
    extracted = re.sub(r"\s+", " ", extracted).strip() 
    extracted = extracted.lower()

    return extracted
