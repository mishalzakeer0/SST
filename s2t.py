import torch
from translate import Translator
from glob import glob
from gtts import gTTS

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass

else:
    ssl._create_default_https_context = _create_unverified_https_context
    
FROM_LANG = 'en'

TO_LANG = 'ml'

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
test_files = glob('speech_orig.wav')
batches = split_into_batches(test_files, batch_size=10)
input = prepare_model_input(read_batch(batches[0]),
                            device=device)

output = model(input)
for example in output:
    # print(decoder(example.cpu()))
    translator = Translator(to_lang=TO_LANG)
    malayalam_text = translator.translate(decoder(example.cpu()))
    print(malayalam_text)
    tts = gTTS(text=malayalam_text, lang=TO_LANG, slow=False)
    tts.save('output.mp3')