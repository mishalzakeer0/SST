import torch
from translate import Translator
from glob import glob
from gtts import gTTS
import pyaudio
import wave
import ssl

sec = int(input("Enter seconds to record"))

def recording(s):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Start Recording")
    frames = []
    seconds = s
    for i in range(0, int(RATE/CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Stop Recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open("record.wav", "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

recording(sec)


def sts(FROM_LANG, TO_LANG):

    try:    
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context



    device = 'cpu'  # gpu also works, but our models are fast enough for CPU
    model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                           model='silero_stt',
                                           language=FROM_LANG,
                                           device=device)
    (read_batch, split_into_batches,
    read_audio, prepare_model_input) = utils  # see function signature for details

    # download a single file in any format compatible with TorchAudio
    # torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav',
    #                                dst ='speech_orig.wav', progress=True)
    test_files = glob('record.wav')
    batches = split_into_batches(test_files, batch_size=10)
    input = prepare_model_input(read_batch(batches[0]),
                                device=device)

    output = model(input)
    for example in output:
        print(decoder(example.cpu()))
        translator = Translator(to_lang=TO_LANG)
        malayalam_text = translator.translate(decoder(example.cpu()))
        # print(malayalam_text)
        tts = gTTS(text=malayalam_text, lang=TO_LANG, slow=False)
        tts.save('output.mp3')

sts("en", "ml")