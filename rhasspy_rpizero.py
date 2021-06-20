# A much simplified Rhasspy service for Raspberry Pi Zero
# Codes are derived from Rhasspy https://github.com/rhasspy
# License: MIT

# Do not support SSL

# Python 3
# Require: pyaudio, pvporcupine, webrtcvad
# e.g., install them by
# pip3 install pvporcupine pyaudio webrtcvad

# -------------------
# Configuration
# -------------------

# change the following to your setting
site_id = 'rpizero_study'
asr_url = "http://192.168.1.103:12101/api/speech-to-text"
nlu_url = "http://192.168.1.103:12101/api/text-to-intent"
wake_response_wav_file = "/home/pi/beep_ask.wav" 
intent_response_wave_file = "/home/pi/beep_confirm.wav" 

# wake word: one or more of the following:
# 'alexa',
# 'americano',
# 'blueberry',
# 'bumblebee',
# 'computer',
# 'grapefruit',
# 'grasshopper',
# 'hey google',
# 'hey siri',
# 'jarvis',
# 'ok google',
# 'pico clock',
# 'picovoice',
# 'porcupine',
# 'terminator'
keywords = ['bumblebee']

# for Raspberry Pi Zero + Raspberry Pi OS + Respeaker Pi HAT 2-mic
# the device index is usual zero
# you can check it by arecord -l
input_device_index = 0


# -----------------
# initialization
# -----------------

import rpizrecorder
import typing
import os
import struct
from datetime import datetime
import pvporcupine
import pyaudio
import requests


# set up wake words
if keywords is None:
    keywords = sorted(pvporcupine.KEYWORDS)
    
porcupine = pvporcupine.create(keywords=keywords)

# open audio stream
audio = pyaudio.PyAudio()
mic = audio.open(
                input_device_index=input_device_index,
                channels=1, # mono
                format=audio.get_format_from_width(2),
                rate=16000,
                frames_per_buffer=1024,
                input=True,
)

# a simplified command recorder
recorder = rpizrecorder.RpizCommandRecorder(mic)

# init a http session
http_session = requests.Session()



# -------------------
# start the main loop
# -------------------

try:
    
    params = {"siteId": site_id}

    while True:
        detected = None
        while True:
            pcm = mic.read(porcupine.frame_length,exception_on_overflow = False)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

            result = porcupine.process(pcm)
            if result >= 0:
                print('[%s] Detected %s' % (str(datetime.now()), keywords[result]))
                detected = keywords[result]
                break

        #if detected == 'terminator':
        #    break;

        # play a sound to signal wakeup 
        os.system(f'aplay -Dhw:0 {wake_response_wav_file}')

        
        # record a command
        rslt = recorder.record()

        # speech to text
        if rslt is not None:
            wav_bytes = rpizrecorder.to_wav_bytes(
                 rslt.audio_data, recorder.sample_rate, recorder.sample_width, recorder.channels
            )

            with http_session.post(
                                asr_url,
                                data=wav_bytes,
                                headers={"Content-Type": "audio/wav", "Accept": "application/json"},
                                params=params,
                                #ssl=ssl_context,
                                timeout=10,
            ) as response:
                                response.raise_for_status()
                                transcription_dict = response.json()

            input_text = transcription_dict['text']

            # intent recognition
            with http_session.post(
               nlu_url, data=input_text, params=params, timeout=10
            ) as response:
                response.raise_for_status()
                intent_dict = response.json()   

            # play a sound to signal intent recognized
            os.system(f'aplay -Dhw:0 {intent_response_wave_file}')
            
except KeyboardInterrupt:
    print('Exit!')

    
# clean up
http_session.close()
porcupine.delete()
mic.stop_stream()
audio.terminate()
