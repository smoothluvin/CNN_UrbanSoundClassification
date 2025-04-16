import torch
import torchaudio
from cnn import CNNNetwork
from urbansounddataset import UrbanSoundDataset
from train import ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

def predict(model, input, target, class_mapping):
    model.eval()

    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

if __name__ == "__main__":
    # Loading CNNNetwork
    cnn = CNNNetwork()
    state_dict = torch.load("./models/cnn.pth")
    cnn.load_state_dict(state_dict)

    # Instantiating our dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 1024,
        hop_length = 512,
        n_mels = 64
    )
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram, 
                            SAMPLE_RATE, 
                            NUM_SAMPLES,
                            "cpu")

    # Grabbing a sample from UrbanSoundDataset for reference
    input, target = usd[0][0], usd[0][1] # Tensor [BATCH_SIZE, num_channels, freq_axis, time_axis]
    input.unsqueeze_(0) # This is where we want to introduce the new dimension

    # Making an inference
    predicted, expected = predict(cnn, input, target, class_mapping)
    print(f"Predicted: {predicted}, expected: {expected}")
