import os

from huggingface_hub import login

from datasets import load_dataset, DatasetDict, Audio

from transformers import WhisperTokenizer
from transformers import WhisperFeatureExtractor
from transformers import WhisperProcessor


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


def get_dir_size(path='/common_voice/train'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


login("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", add_to_git_credential=True)

common_voice = DatasetDict()
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "sv-SE", split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "sv-SE", split="test", use_auth_token=True)


common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Swedish", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Swedish", task="transcribe")

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)

common_voice.save_to_disk("common_voice")
cc = DatasetDict.load_from_disk("common_voice")

# %%
os.system("gsutil cp -r ./common_voice  gs://bucket-common-voice")