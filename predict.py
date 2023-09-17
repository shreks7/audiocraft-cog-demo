# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os

# We need to set `TRANSFORMERS_CACHE` before any imports, which is why this is up here.
MODEL_PATH = "/src/models/"
os.environ["TRANSFORMERS_CACHE"] = MODEL_PATH
os.environ["TORCH_HOME"] = MODEL_PATH

import shutil
import random

from tempfile import TemporaryDirectory
from distutils.dir_util import copy_tree
from typing import Optional, Iterator, List
from cog import BasePredictor, Input, Path, BaseModel
import torch
import datetime

# Model specific imports
import torchaudio
import subprocess
import typing as tp

from audiocraft.models import MusicGen
from audiocraft.models.loaders import (
    load_compression_model,
    load_lm_model,
)
from audiocraft.data.audio import audio_write
from audiocraft.data.audio import audio_write

from BeatNet.BeatNet import BeatNet
import madmom.audio.filters

# Hack madmom to work with recent python
madmom.audio.filters.np.float = float

import soundfile as sf
import librosa
import numpy as np
import pyrubberband as pyrb


class Outputs(BaseModel):
    variation_01: Optional[Path] = None
    variation_02: Optional[Path] = None
    variation_03: Optional[Path] = None
    variation_04: Optional[Path] = None
    variation_05: Optional[Path] = None
    variation_06: Optional[Path] = None
    variation_07: Optional[Path] = None
    variation_08: Optional[Path] = None
    variation_09: Optional[Path] = None
    variation_10: Optional[Path] = None
    variation_11: Optional[Path] = None
    variation_12: Optional[Path] = None
    variation_13: Optional[Path] = None
    variation_14: Optional[Path] = None
    variation_15: Optional[Path] = None
    variation_16: Optional[Path] = None
    variation_17: Optional[Path] = None
    variation_18: Optional[Path] = None
    variation_19: Optional[Path] = None
    variation_20: Optional[Path] = None


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.medium_model = self._load_model(
            model_path=MODEL_PATH,
            cls=MusicGen,
            model_id="facebook/musicgen-medium",
        )

        self.large_model = self._load_model(
            model_path=MODEL_PATH,
            cls=MusicGen,
            model_id="facebook/musicgen-large",
        )

        self.beatnet = BeatNet(
            1,
            mode="offline",
            inference_model="DBN",
            plot=[],
            thread=False,
            device="cuda:0",
        )

    def _load_model(
        self,
        model_path: str,
        cls: Optional[any] = None,
        load_args: Optional[dict] = {},
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> MusicGen:

        if device is None:
            device = self.device


        name = model_id
        print("Loading model:"+name)
        compression_model = load_compression_model(
            name, device=device, cache_dir=model_path
        )
        lm = load_lm_model(name, device=device, cache_dir=model_path)

        return MusicGen(name, compression_model, lm)

    def predict(
        self,
        prompt: str = Input(
            description="A description of the music you want to generate."
        ),
        bpm: float = Input(
            description="Tempo in beats per minute",
            default=140.0,
            ge=40,
            le=300,
        ),
        variations: int = Input(
            description="Number of variations to generate",
            default=4,
            ge=1,
            le=20,
        ),
        max_duration: int = Input(
            description="Maximum duration of the generated loop in seconds.",
            default=8,
            le=20,
            ge=2,
        ),
        model_version: str = Input(
            description="Model to use for generation. .",
            default="medium",
            choices=["medium", "large"],
        ),
        top_k: int = Input(
            description="Reduces sampling to the k most likely tokens.", default=250
        ),
        top_p: float = Input(
            description="Reduces sampling to tokens with cumulative probability of p. When set to  `0` (default), top_k sampling is used.",
            default=0.0,
        ),
        temperature: float = Input(
            description="Controls the 'conservativeness' of the sampling process. Higher temperature means more diversity.",
            default=1.0,
        ),
        classifier_free_guidance: int = Input(
            description="Increases the influence of inputs on the output. Higher values produce lower-varience outputs that adhere more closely to inputs.",
            default=3,
        ),
        output_format: str = Input(
            description="Output format for generated audio.",
            default="wav",
            choices=["wav", "mp3"],
        ),
        seed: int = Input(
            description="Seed for random number generator. If None or -1, a random seed will be used.",
            default=-1,
        ),
    ) -> Outputs:
        prompt = prompt + f", {bpm} bpm"

        model = self.medium_model if model_version == "medium" else self.large_model

        model.set_generation_params(
            duration=max_duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=classifier_free_guidance,
        )

        if not seed or seed == -1:
            seed = torch.seed() % 2**32 - 1
            set_all_seeds(seed)
        set_all_seeds(seed)
        print(f"Using seed {seed}")

        print("Generating variation 1")

        wav = model.generate([prompt], progress=True).cpu().numpy()[0, 0]
        # normalize
        wav = wav / np.abs(wav).max()

        beats = self.estimate_beats(wav, model.sample_rate)
        start_time, end_time = self.get_loop_points(beats)
        loop_seconds = end_time - start_time

        print("Beats:\n", beats)
        print(f"{start_time}, {end_time}")

        num_beats = len(beats[(beats[:, 0] >= start_time) & (beats[:, 0] < end_time)])
        duration = end_time - start_time
        actual_bpm = num_beats / duration * 60
        if (
            abs(actual_bpm - bpm) > 10
            and abs(actual_bpm / 2 - bpm) > 10
            and abs(actual_bpm * 2 - bpm) > 10
        ):
            raise ValueError(
                f"Failed to generate a loop in the requested {bpm} bpm. Please try again."
            )

        # Allow octave errors
        if abs(actual_bpm / 2 - bpm) <= 10:
            actual_bpm = actual_bpm / 2
        elif abs(actual_bpm * 2 - bpm) <= 10:
            actual_bpm = actual_bpm * 2

        start_sample = int(start_time * model.sample_rate)
        end_sample = int(end_time * model.sample_rate)
        loop = wav[start_sample:end_sample]

        # do a quick blend with the lead-in do avoid clicks
        num_lead = 100
        lead_start = start_sample - num_lead
        lead = wav[lead_start:start_sample]
        num_lead = len(lead)
        loop[-num_lead:] *= np.linspace(1, 0, num_lead)
        loop[-num_lead:] += np.linspace(0, 1, num_lead) * lead

        stretched = pyrb.time_stretch(loop, model.sample_rate, bpm / actual_bpm)

        outputs = Outputs()
        add_output(
            outputs, self.write(stretched, model.sample_rate, output_format, "out-0")
        )

        if variations > 1:
            # Use last 4 beats as audio prompt
            last_4beats = beats[beats[:, 0] <= end_time][-5:]
            audio_prompt_start_time = last_4beats[0][0]
            audio_prompt_end_time = last_4beats[-1][0]
            audio_prompt_start_sample = int(audio_prompt_start_time * model.sample_rate)
            audio_prompt_end_sample = int(audio_prompt_end_time * model.sample_rate)
            audio_prompt_seconds = audio_prompt_end_time - audio_prompt_start_time
            audio_prompt = torch.tensor(
                wav[audio_prompt_start_sample:audio_prompt_end_sample]
            )[None]
            audio_prompt_duration = audio_prompt_end_sample - audio_prompt_start_sample

            model.set_generation_params(
                duration=loop_seconds + audio_prompt_seconds + 0.1,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                cfg_coef=classifier_free_guidance,
            )

            for i in range(1, variations):
                print(f"\nGenerating variation {i + 1}")

                continuation = model.generate_continuation(
                    prompt=audio_prompt,
                    prompt_sample_rate=model.sample_rate,
                    descriptions=[prompt],
                    progress=True,
                )
                variation_loop = continuation.cpu().numpy()[
                    0, 0, audio_prompt_duration : audio_prompt_duration + len(loop)
                ]
                variation_loop[-num_lead:] *= np.linspace(1, 0, num_lead)
                variation_loop[-num_lead:] += np.linspace(0, 1, num_lead) * lead

                variation_stretched = pyrb.time_stretch(
                    variation_loop, model.sample_rate, bpm / actual_bpm
                )
                add_output(
                    outputs,
                    self.write(
                        variation_stretched,
                        model.sample_rate,
                        output_format,
                        f"out-{i}",
                    ),
                )

        return outputs

    def estimate_beats(self, wav, sample_rate):
        # resample to BeatNet's sample rate
        beatnet_input = librosa.resample(
            wav,
            orig_sr=sample_rate,
            target_sr=self.beatnet.sample_rate,
        )
        return self.beatnet.process(beatnet_input)

    def get_loop_points(self, beats):
        # extract an even number of bars
        downbeat_times = beats[:, 0][beats[:, 1] == 1]
        num_bars = len(downbeat_times) - 1

        if num_bars < 1:
            raise ValueError(
                "Less than one bar detected. Try increasing max_duration, or use a different seed."
            )

        even_num_bars = int(2 ** np.floor(np.log2(num_bars)))
        start_time = downbeat_times[0]
        end_time = downbeat_times[even_num_bars]

        return start_time, end_time

    def write(self, audio, sample_rate, output_format, name):
        wav_path = name + ".wav"
        sf.write(wav_path, audio, sample_rate)

        if output_format == "mp3":
            mp3_path = name + ".mp3"
            subprocess.call(
                ["ffmpeg", "-loglevel", "error", "-y", "-i", wav_path, mp3_path]
            )
            os.remove(wav_path)
            path = mp3_path
        else:
            path = wav_path

        return Path(path)


def add_output(outputs, path):
    for i in range(1, 21):
        field = f"variation_{i:02d}"
        if getattr(outputs, field) is None:
            setattr(outputs, field, path)
            return
    raise ValueError("Failed to add output")


# From https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
