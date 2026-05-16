import os
from pathlib import Path


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv()

API_KEY = os.getenv("API_KEY", "EMPTY")
API_BASE_URL = os.getenv("API_BASE_URL")
CHAT_BASE_URL = os.getenv("CHAT_BASE_URL", API_BASE_URL)  
CHAT_API_KEY = os.getenv("CHAT_API_KEY", API_KEY) 

BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-3.1-70B")
CHAT_MODEL = os.getenv("CHAT_MODEL", "meta-llama/Llama-3.1-70B-Instruct")

ALPHA = 50.0        
T0 = 3               
T_MIN = 0.001           
BETA = 0.98             
K_INIT = 8              
NUM_ICM_ITERATIONS = 600  
CONTEXT_SIZE = 256      
MAX_FEW_SHOT = 48      
LOGPROBS_TOP = 5    

COUNTRIES = [
    "United States",
    "Britain",
    "Germany",
    "Japan",
]

TRAIN_RATIO = 0.7

MAX_TRAIN = 256      
MAX_TEST = 100       

FEW_SHOT_COUNTS = [4, 16, 32, 48]
SEED = 42
