import os
import logging
import numpy as np
import librosa
import onnxruntime as ort

logger = logging.getLogger(__name__)
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "asrassets"))

class IndicASRPipeline:
    """Multilingual Speech-to-Text Pipeline using NeMo IndicConformer models running on ONNX."""

    def __init__(self):
        self.sessions = {}
        self.tokens = {}
        self.supported_languages = ["hi", "kn", "en", "mr", "ta", "te", "ml"]
        self._load_models()

    def _load_models(self):
        for lang in ["hi", "kn"]:
            lang_dir = os.path.join(MODELS_DIR, lang)
            model_path = os.path.join(lang_dir, "model.int8.onnx")
            tokens_path = os.path.join(lang_dir, "tokens.txt")
            
            if os.path.exists(model_path) and os.path.exists(tokens_path):
                logger.info(f"Loading {lang} ASR model from {model_path}...")
                session_opts = ort.SessionOptions()
                session_opts.intra_op_num_threads = 2
                self.sessions[lang] = ort.InferenceSession(
                    model_path,
                    sess_options=session_opts,
                    providers=["CPUExecutionProvider"]
                )
                with open(tokens_path, "r", encoding="utf-8") as f:
                    self.tokens[lang] = [line.strip().split()[0] for line in f.readlines()]
            else:
                logger.info(f"Skipping {lang}: Model or tokens not found in {lang_dir}")

    def transcribe(self, lang: str, audio_array: np.ndarray) -> str:
        """Transcribes 16kHz audio array into text using IndicConformer CTC decoding."""
        if lang not in self.sessions:
            return ""
            
        if audio_array is None or len(audio_array) == 0:
            return ""
            
        # Audio Preprocessing: Pre-emphasis + Mel Spectrogram (NeMo parameters)
        audio_array = np.append(audio_array[0], audio_array[1:] - 0.97 * audio_array[:-1])
        melspec = librosa.feature.melspectrogram(
            y=audio_array, 
            sr=16000, 
            n_fft=512, 
            hop_length=160, 
            win_length=320,
            window='hann',
            center=True,
            n_mels=80,
            fmin=0.0,
            fmax=8000.0
        )
        log_melspec = np.log(melspec + 1e-5).astype(np.float32)
        mean = np.mean(log_melspec, axis=1, keepdims=True)
        std = np.std(log_melspec, axis=1, keepdims=True)
        log_melspec = (log_melspec - mean) / (std + 1e-5)
        
        audio_signal = np.expand_dims(log_melspec, axis=0)
        length_val = np.array([audio_signal.shape[2]], dtype=np.int64)
        
        inputs = {"audio_signal": audio_signal, "length": length_val}
        outputs = self.sessions[lang].run(None, inputs)
        logprobs = outputs[0]
        
        return self._decode_ctc(logprobs, self.tokens[lang])

    def _decode_ctc(self, logprobs: np.ndarray, vocab: list) -> str:
        predictions = np.argmax(logprobs[0], axis=-1)
        decoded_tokens = []
        prev_token = None
        blank_id = len(vocab) - 1
        if "<blk>" in vocab:
            blank_id = vocab.index("<blk>")
            
        for token_id in predictions:
            if token_id != prev_token and token_id != blank_id:
                if token_id < len(vocab):
                    decoded_tokens.append(vocab[token_id])
            prev_token = token_id
            
        text = "".join(decoded_tokens).replace("\u2581", " ").replace("<unk>", "").strip()
        return text

asr_pipeline = IndicASRPipeline()
