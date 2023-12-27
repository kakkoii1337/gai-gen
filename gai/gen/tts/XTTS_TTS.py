import os, gc
#from TTS.api import TTS
from gai.common import generators_utils, logging
logger = logging.getLogger(__name__)
from gai.common.utils import get_config_path
import torch

class XTTS_TTS:

    def __init__(self, model_config):
        self.model = None
        self.tokenizer = None
        pass

    def load(self):
        logger.info("Loading XTTS...")
        print("Loading XTTS...")
        
        os.environ["COQUI_TOS_AGREED"] = "1"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_dir = f"{get_config_path()}/models"
        model_path=os.path.join(base_dir, "tts/tts_models--multilingual--multi-dataset--xtts_v2")
        config_path = os.path.join(model_path,"config.json")

        # Load Config
        from TTS.tts.configs.xtts_config import XttsConfig
        self.config = XttsConfig()
        self.config.load_json(config_path)

        # Load Model
        from TTS.tts.models.xtts import Xtts
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=model_path, eval=True, use_deepspeed=True if device == "cuda" else False)
        self.model.to(device)
        logger.info("XTTS Loaded.")
        return self

    def unload(self):
        try:
            del self.model
            del self.tokenizer
        except :
            pass
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        return self

    def _postprocess(self,wav):
        import numpy as np

        """Post process the output waveform"""
        if isinstance(wav, list):
            wav = torch.cat(wav, dim=0)
        wav = wav.clone().detach().cpu().numpy()
        wav = wav[None, : int(wav.shape[0])]
        wav = np.clip(wav, -1, 1)
        wav = (wav * 32767).astype(np.int16)
        return wav

    def _encode_audio_common(self,
        frame_input, encode_base64=True, sample_rate=24000, sample_width=2, channels=1
    ):
        import io
        import wave
        import base64

        """Return base64 encoded audio"""
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as vfout:
            vfout.setnchannels(channels)
            vfout.setsampwidth(sample_width)
            vfout.setframerate(sample_rate)
            vfout.writeframes(frame_input)

        wav_buf.seek(0)
        if encode_base64:
            b64_encoded = base64.b64encode(wav_buf.getbuffer()).decode("utf-8")
            return b64_encoded
        else:
            return wav_buf.read()

    def _generating(self,chunks,add_wav_header: bool = True):
        content = []
        for i, chunk in enumerate(chunks):
            try:
                chunk = self._postprocess(chunk)
                if i == 0 and add_wav_header:
                    header = self._encode_audio_common(b"", encode_base64=False)
                    content.append(header)
                    content.append(chunk.tobytes())
                else:
                    content.append(chunk.tobytes())
            except Exception as e:
                logger.error(e)
        content = b''.join(content)
        return content

    def _streaming(self,chunks,add_wav_header: bool = True):
        for i, chunk in enumerate(chunks):
            try:
                chunk = self._postprocess(chunk)
                if i == 0 and add_wav_header:
                    header = self._encode_audio_common(b"", encode_base64=False)
                    yield header
                    yield chunk.tobytes()
                else:
                    yield chunk.tobytes()
            except Exception as e:
                logger.error(e)

    def create(self,**model_params):
        logger.info("XTTS generating...")

        if not self.model:
            self.load()
       
        input = model_params.pop("input",None)
        if input is None:
            raise Exception("Missing input parameter")

        voice = model_params.pop("voice",None)
        if voice is None:
            voice = "Vjollca Johnnie"

        language = model_params.pop("language",None)
        if language is None:
            language = "en"

        stream = model_params.pop("stream",False)

        from TTS.tts.utils.speakers import SpeakerManager
        speaker_manager = SpeakerManager(speaker_id_file_path=f"{get_config_path()}/models/tts/tts_models--multilingual--multi-dataset--xtts_v2/speakers_xtts.pth")
        speaker = speaker_manager.get_speakers()[voice]
        gpt_cond_latent, speaker_embedding=speaker.values()
        chunks = self.model.inference_stream(input, language, gpt_cond_latent, speaker_embedding, **model_params)

        if not stream:
            response = self._generating(chunks)
        else:
            response = self._streaming(chunks)

        logger.info("XTTS completed.")
        return response


