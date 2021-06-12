import logging
import audioop
import math
import typing
from collections import deque
from queue import Queue

from dataclasses import dataclass, field
from enum import Enum

import webrtcvad
import pyaudio
import io
import wave


_LOGGER = logging.getLogger("rhasspymicrophone_pyaudio_hermes")


def to_wav_bytes(
        audio_data: bytes,
        sample_rate: typing.Optional[int] = 16000,
        sample_width: typing.Optional[int] = 2,
        channels: typing.Optional[int] = 1,
) -> bytes:
    """Wrap raw audio data in WAV."""
        
    with io.BytesIO() as wav_buffer:
        wav_file: wave.Wave_write = wave.open(wav_buffer, mode="wb")
        with wav_file:
            wav_file.setframerate(sample_rate)
            wav_file.setsampwidth(sample_width)
            wav_file.setnchannels(channels)
            wav_file.writeframes(audio_data)

        return wav_buffer.getvalue()



class VoiceCommandResult(str, Enum):
    """Success/failure of voice command recognition."""

    SUCCESS = "success"
    FAILURE = "failure"


class VoiceCommandEventType(str, Enum):
    """Possible event types during voice command recognition."""

    STARTED = "started"
    SPEECH = "speech"
    SILENCE = "silence"
    STOPPED = "stopped"
    TIMEOUT = "timeout"

@dataclass
class VoiceCommandEvent:
    """Speech/silence events."""

    type: VoiceCommandEventType
    time: float

@dataclass
class VoiceCommand:
    """Result of voice command recognition."""

    result: VoiceCommandResult
    audio_data: typing.Optional[bytes] = None
    events: typing.List[VoiceCommandEvent] = field(default_factory=list)
           

class SilenceMethod(str, Enum):
    """Method used to determine if an audio frame contains silence.
    Values
    ------
    VAD_ONLY
      Only use webrtcvad
    RATIO_ONLY
      Only use max/current energy ratio threshold
    CURRENT_ONLY
      Only use current energy threshold
    VAD_AND_RATIO
      Use webrtcvad and max/current energy ratio threshold
    VAD_AND_CURRENT
      Use webrtcvad and current energy threshold
    ALL
      Use webrtcvad, max/current energy ratio, and current energy threshold
    """

    VAD_ONLY = "vad_only"
    RATIO_ONLY = "ratio_only"
    CURRENT_ONLY = "current_only"
    VAD_AND_RATIO = "vad_and_ratio"
    VAD_AND_CURRENT = "vad_and_current"
    ALL = "all"

class RpizCommandRecorder:
    """ Simple recorder for Raspberry Pi Zero + Respeaker Pi HAT 2-mic 
    
        Many codes are borrowed from Rhasspy project.
    """
    def __init__(self,
                 mic,
                silence_method: SilenceMethod = SilenceMethod.VAD_ONLY,
    ):
        self.sample_rate = 16000
        self.sample_width = 2
        self.channels = 1
        self.chunk_size = 960
        self.vad_mode = 3
        
        # at most 20 seconds
        self.max_timeout = 20
        
        self.mic = mic
        
        self.silence_method = silence_method

        # using the seeed voice card
        self.device_index = 0

        self.frames_per_buffer = 1024
        #self.chunk_queue: Queue = Queue()
            
            
        self.skip_seconds = 0
        self.min_seconds = 1
        self.max_seconds = 30
        self.speech_seconds = 0.3
        self.silence_seconds = 0.5
        self.before_seconds = 0.5

        self.max_energy = None
        self.dynamic_max_energy = self.max_energy is None
        self.max_current_ratio_threshold = None
        self.current_energy_threshold = None 
            
        if self.silence_method in [
            SilenceMethod.VAD_ONLY,
            SilenceMethod.VAD_AND_RATIO,
            SilenceMethod.VAD_AND_CURRENT,
            SilenceMethod.ALL,
        ]:
            self.use_vad = True
        else:
            self.use_vad = False    
            
        
        if self.silence_method in [
            SilenceMethod.VAD_AND_RATIO,
            SilenceMethod.RATIO_ONLY,
            SilenceMethod.ALL,
        ]:
            self.use_ratio = True
            assert (
                self.max_current_ratio_threshold is not None
            ), "Max/current ratio threshold is required"
        else:
            self.use_ratio = False

        if self.silence_method in [
            SilenceMethod.VAD_AND_CURRENT,
            SilenceMethod.CURRENT_ONLY,
            SilenceMethod.ALL,
        ]:
            self.use_current = True
            assert (
                self.current_energy_threshold is not None
            ), "Current energy threshold is required"
        else:
            self.use_current = False
        
        
        # Voice detector
        self.vad: typing.Optional[webrtcvad.Vad] = None
        if self.use_vad:
            assert self.vad_mode in range(
                1, 4
            ), f"VAD mode must be 1-3 (got {vad_mode})"

            chunk_ms = 1000 * ((self.chunk_size / 2) / self.sample_rate)
            assert chunk_ms in [10, 20, 30], (
                "Sample rate and chunk size must make for 10, 20, or 30 ms buffer sizes,"
                + f" assuming 16-bit mono audio (got {chunk_ms} ms)"
            )

            self.vad = webrtcvad.Vad()
            self.vad.set_mode(self.vad_mode)
            
       
    
    def reset(self):
        self.chunk_queue: Queue = Queue()

        # some buffers --- do we still need them?
        self.seconds_per_buffer = self.chunk_size / self.sample_rate

        # Store some number of seconds of audio data immediately before voice command starts
        self.before_buffers = int(
            math.ceil(self.before_seconds / self.seconds_per_buffer)
        )

        # Pre-compute values
        self.speech_buffers = int(
            math.ceil(self.speech_seconds / self.seconds_per_buffer)
        )

        self.skip_buffers = int(math.ceil(self.skip_seconds / self.seconds_per_buffer))   
        
        # State
        self.events: typing.List[VoiceCommandEvent] = []
        self.before_phrase_chunks: typing.Deque[bytes] = deque(
            maxlen=self.before_buffers
        )
        self.phrase_buffer: bytes = bytes()

        self.max_buffers: typing.Optional[int] = None
        self.min_phrase_buffers: int = 0
        self.skip_buffers_left: int = 0
        self.speech_buffers_left: int = 1
        self.last_speech: bool = False
        self.in_phrase: bool = False
        self.after_phrase: bool = False
        self.silence_buffers: int = 1
        self.current_seconds: float = 0
        self.current_chunk: bytes = bytes()
            
        self.events = []
        self.current_chunk = bytes()
        self.before_phrase_chunks = deque(
            maxlen=self.before_buffers
        )    
       
        self.first_chunk = True 
    
    def process_chunk(self, audio_chunk: bytes) -> typing.Optional[VoiceCommand]:
        """Process a single chunk of audio data."""

        # Add to overall buffer
        self.current_chunk += audio_chunk

        # Process audio in exact chunk(s)
        while len(self.current_chunk) > self.chunk_size:
            # Extract chunk
            chunk = self.current_chunk[: self.chunk_size]
            self.current_chunk = self.current_chunk[self.chunk_size :]

            if self.skip_buffers_left > 0:
                # Skip audio at beginning
                self.skip_buffers_left -= 1
                continue

            if self.in_phrase:
                self.phrase_buffer += chunk
            else:
                self.before_phrase_chunks.append(chunk)

            self.current_seconds += self.seconds_per_buffer

            # Check maximum number of seconds to record
            if self.max_buffers:
                self.max_buffers -= 1
                if self.max_buffers <= 0:
                    # Timeout
                    self.events.append(
                        VoiceCommandEvent(
                            type=VoiceCommandEventType.TIMEOUT,
                            time=self.current_seconds,
                        )
                    )
                    return VoiceCommand(
                        result=VoiceCommandResult.FAILURE, events=self.events
                    )

            # Detect speech in chunk
            
            if self.first_chunk:
                # skip the first chunk
                self.first_chunk = False
                continue
            
            is_speech = not self.is_silence(chunk)
            
            
            if not is_speech and not self.in_phrase:
                # any silence before in_phrase is tolerated
                self.events.append(
                    VoiceCommandEvent(
                        type=VoiceCommandEventType.SILENCE, time=self.current_seconds
                    )
                )
                continue
            
            if is_speech and not self.last_speech:
                # Silence -> speech
                self.events.append(
                    VoiceCommandEvent(
                        type=VoiceCommandEventType.SPEECH, time=self.current_seconds
                    )
                )
            elif not is_speech and self.last_speech:
                # Speech -> silence
                self.events.append(
                    VoiceCommandEvent(
                        type=VoiceCommandEventType.SILENCE, time=self.current_seconds
                    )
                )

            self.last_speech = is_speech

            # Handle state changes
            if is_speech and self.speech_buffers_left > 0:
                self.speech_buffers_left -= 1
            elif is_speech and not self.in_phrase:
                # Start of phrase
                self.events.append(
                    VoiceCommandEvent(
                        type=VoiceCommandEventType.STARTED, time=self.current_seconds
                    )
                )
                self.in_phrase = True
                self.after_phrase = False
                self.min_phrase_buffers = int(
                    math.ceil(self.min_seconds / self.seconds_per_buffer)
                )
            elif self.in_phrase and (self.min_phrase_buffers > 0):
                # In phrase, before minimum seconds
                self.min_phrase_buffers -= 1
            elif not is_speech:
                # Outside of speech
                if not self.in_phrase:
                    # Reset
                    self.speech_buffers_left = self.speech_buffers
                elif self.after_phrase and (self.silence_buffers > 0):
                    # After phrase, before stop
                    self.silence_buffers -= 1
                elif self.after_phrase and (self.silence_buffers <= 0):
                    # Phrase complete
                    self.events.append(
                        VoiceCommandEvent(
                            type=VoiceCommandEventType.STOPPED,
                            time=self.current_seconds,
                        )
                    )

                    # Merge before/during command audio data
                    before_buffer = bytes()
                    for before_chunk in self.before_phrase_chunks:
                        before_buffer += before_chunk

                    return VoiceCommand(
                        result=VoiceCommandResult.SUCCESS,
                        audio_data=before_buffer + self.phrase_buffer,
                        # audio_data = self.current_chunk
                        events=self.events,
                    )
                elif self.in_phrase and (self.min_phrase_buffers <= 0):
                    # Transition to after phrase
                    self.after_phrase = True
                    self.silence_buffers = int(
                        math.ceil(self.silence_seconds / self.seconds_per_buffer)
                    )

        return None
    
    
    def record(self) -> typing.Optional[VoiceCommand]:
        """Record audio from PyAudio device."""
        try:
            
            self.reset()
            
            

            """
            # init pyaudio, slow
            self.audio = pyaudio.PyAudio()
            
            # Open device
            mic = self.audio.open(
                input_device_index=self.device_index,
                channels=self.channels,
                format=self.audio.get_format_from_width(self.sample_width),
                rate=self.sample_rate,
                frames_per_buffer=self.frames_per_buffer,
                input=True,
                #stream_callback=callback,
            )

            assert mic is not None
            """
            
            print("* recording")
            
            for i in range(0, int(self.sample_rate / self.chunk_size * self.max_timeout)):
                data = self.mic.read(self.chunk_size, exception_on_overflow = False)
                rslt = self.process_chunk(data)
                if rslt is not None:
                    #mic.stop_stream()
                    #self.audio.terminate()
                    print("* done recording")
                    return rslt
            
        except Exception as e:
            _LOGGER.exception("record error")
        
        return None
    
    
    def is_silence(self, chunk: bytes) -> bool:
        """True if audio chunk contains silence."""
        all_silence = True

        if self.use_vad:
            # Use VAD to detect speech
            assert self.vad is not None
            all_silence = all_silence and (
                not self.vad.is_speech(chunk, self.sample_rate)
            )

        if self.use_ratio or self.use_current:
            # Compute debiased energy of audio chunk
            energy = WebRtcVadRecorder.get_debiased_energy(chunk)
            if self.use_ratio:
                # Ratio of max/current energy compared to threshold
                if self.dynamic_max_energy:
                    # Overwrite max energy
                    if self.max_energy is None:
                        self.max_energy = energy
                    else:
                        self.max_energy = max(energy, self.max_energy)

                assert self.max_energy is not None
                if energy > 0:
                    ratio = self.max_energy / energy
                else:
                    # Not sure what to do here
                    ratio = 0

                assert self.max_current_ratio_threshold is not None
                all_silence = all_silence and (ratio > self.max_current_ratio_threshold)
            elif self.use_current:
                # Current energy compared to threshold
                assert self.current_energy_threshold is not None
                all_silence = all_silence and (energy < self.current_energy_threshold)

        return all_silence

    # -------------------------------------------------------------------------

    @staticmethod
    def get_debiased_energy(audio_data: bytes) -> float:
        """Compute RMS of debiased audio."""
        # Thanks to the speech_recognition library!
        # https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py
        energy = -audioop.rms(audio_data, 2)
        energy_bytes = bytes([energy & 0xFF, (energy >> 8) & 0xFF])
        debiased_energy = audioop.rms(
            audioop.add(audio_data, energy_bytes * (len(audio_data) // 2), 2), 2
        )

        # Probably actually audio if > 30
        return debiased_energy