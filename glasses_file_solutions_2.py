from openwakeword.model import Model
import numpy as np

model = Model()

# Fake audio chunk (silence)
audio = np.zeros(16000, dtype=np.int16)

prediction = model.predict(audio)

print(prediction)