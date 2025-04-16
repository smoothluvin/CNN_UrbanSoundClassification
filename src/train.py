import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from urbansounddataset import UrbanSoundDataset
from cnn import CNNNetwork


# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EPOCHS = 10
ANNOTATIONS_FILE = "./data/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "./data/UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

# Creating a data loader for custom dataset
def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size = BATCH_SIZE)
    return train_dataloader

def train_one_epoch(model, data_loader, loss_function, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Calculating loss
        predictions = model(inputs)
        loss = loss_function(predictions, targets)

        # Back propogate loss and update weights
        optimizer.zero_grad()
        loss.backward()

        # Updating the weights
        optimizer.step()

    print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_function, optimizer, device, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_one_epoch(model, data_loader, loss_function, optimizer, device)
        print("-----------------------------------------")
    print("Training is complete")

if __name__ == "__main__":
    
    # Utilizing GPU
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device = {device}")

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
                            device)
    train_dataloader = create_data_loader(usd, BATCH_SIZE)

    # Constructing the model and assigning it to my GPU
    cnn = CNNNetwork().to(device)
    print(cnn)

    # Instantiating the optimizer and loss function
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(),
                                 lr = LEARNING_RATE)
    
    # Training model
    train(cnn, train_dataloader, loss_function, optimizer, device, EPOCHS)

    # Saving the model
    torch.save(cnn.state_dict(), "./models/cnn.pth")
    print("Model trained and saved!")
