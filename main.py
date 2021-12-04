import sys

from jetson_voice.utils import load_resource


def ASR(resource, *args, **kwargs):
    """
    Loads a streaming ASR service or model.
    See the ASRService class for the signature that implementations use.
    """
    factory_map = {
        'riva' : 'jetson_voice.backends.riva.RivaASRService',
        'tensorrt' : 'jetson_voice.models.asr.ASREngine',
        'onnxruntime' : 'jetson_voice.models.asr.ASREngine'
    }
    
    return load_resource(resource, factory_map, *args, **kwargs)

    
class ASRService():
    """
    Streaming ASR service base class.
    """
    def __init__(self, config, *args, **kwargs):
        self.config = config
        
    def __call__(self, samples):
        """
        Transcribe streaming audio samples to text, returning the running phrase.
        Phrases are broken up when a break in the audio is detected (i.e. end of sentence)
        
        Parameters:
          samples (array) -- Numpy array of audio samples.
        Returns a list[dict] of the running transcripts with the following keys:
        
          text (string) -- the transcript of the current sentence
          words (list[dict]) -- a list of word dicts that make up the sentence
          end (bool) -- if true, end-of-sentence due to silence
          
        Each transcript represents one phrase/sentence.  When a sentence has been determined
        to be ended, it will be marked with end=True.  Multiple sentence transcripts can be 
        returned if one just ended and another is beginning. 
        """
        pass
    
    @property
    def classification(self):
        """
        Returns true if this is an ASR classification model (e.g. for VAD or keyword spotting)
        Otherwise, this is an ASR transcription model that converts audio to text.
        """
        return False
        
    @property
    def sample_rate(self):
        """
        The sample rate that the model runs at (in Hz)
        Input audio should be resampled to this rate.
        """
        pass
    
    @property
    def frame_length(self):
        """
        Duration in seconds per frame / chunk.
        """
        pass
        
    @property
    def chunk_size(self):
        """
        Number of samples per frame/chunk (equal to frame_length * sample_rate)
        """
        pass

from jetson_voice import list_audio_devices, AudioInput, ConfigArgParser
list_audio_devices()
asr = ASR("vad_marblenet")

stream = AudioInput(wav=None, mic=1, 
                     sample_rate=16000, 
                     chunk_size=1600)
