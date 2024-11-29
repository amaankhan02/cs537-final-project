from src.dataset import DatasetName
from src.llm import GPTMini, Gemini, LlamaMini

dataset_file_paths = {
    DatasetName.DIFFICULT_DONOTANSWER: "data/difficult_donotanswer.json",
    DatasetName.SMARTHOME_CHILD_SAFETY: "data/dataset1.json"  # TODO: change the filepath for this later
}

llm_models = {
    "gpt": GPTMini,
    "gemini": Gemini,
    "llama": LlamaMini
}