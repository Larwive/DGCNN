import torch
import torch.nn as nn
import torch.optim as optim
from model import DGCNN
from torch.utils.data import DataLoader, TensorDataset
from numpy import concatenate, array, float32
from tqdm import tqdm
from time import process_time_ns

from DREAMER_extract import read_raw, get_features, read_valence_arousal_dominance


def format_time(seconds):
    seconds = int(seconds)
    days = seconds // (24 * 3600)
    hours = (seconds % (24 * 3600)) // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days > 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
    if seconds > 0:
        parts.append(f"{seconds} second{'s' if seconds > 1 else ''}")

    if not parts:
        return "0 seconds"

    return ", ".join(parts)


device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print("Using {} (main.py)".format(device))

def train_model(model, train_data, criterion, optimizer, num_epochs, device, alpha=0.01):
    model.to(device)
    train_losses = []

    model.train()
    for epoch in range(num_epochs):
        epoch_train_loss = 0.0

        for inputs, targets in train_data:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            # L2 regularization (?)
            l2_reg = torch.tensor(0., requires_grad=True)
            for param in model.parameters():
                l2_reg = l2_reg + torch.norm(param)
            #print("TTTTTTTTTTTTTTTTTT", loss, l2_reg, l2_reg * alpha)
            #loss = loss + 1E-4 * l2_reg

            loss.backward()
            optimizer.step()

            # Update adjacency matrix (?)
            with torch.no_grad():
                model.A += alpha*(model.A.grad - model.A)
            model.A.grad.zero_()

            epoch_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss /= len(train_data.dataset)
        train_losses.append(epoch_train_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}')
        if epoch % 5 == 0:
            alpha /= 10

    return train_losses


def loo_cv(model_class, dataset, criterion, optimizer_class, num_epochs, device, batch_size = 100, alpha=0.0001):
    loo_train_losses = []
    loo_val_losses = []

    for i in tqdm(range(len(dataset) // 59)):  # There are 59 time frames for each movie
        val_data = [dataset[j] for j in range(i * 59, (i + 1) * 59)]
        train_data = [dataset[j] for j in range(len(dataset)) if j < i * 59 or j >= (i + 1) * 59]

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        model = model_class(3, 14, 17, 7, 3)
        # I didn't quite understand what to put there but:
        # in_channels: 19 because of the frequencies of the bands (the 8 and 13 Hz overlap)
        # num_electrodes: 14 because there are 14 channels
        # k_adj: 17 because it's prime
        # out_channels: 7 but it seems it can be other things
        # num_classes: 3 because of valence, arousal, dominance

        optimizer = optimizer_class(model.parameters())

        train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs, device, alpha)
        loo_train_losses.append(train_losses[-1])

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets in val_loader:
                #print(array(inputs.cpu()))
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # L2 regularization
                l2_reg = torch.tensor(0., requires_grad=True)
                for param in model.parameters():
                    l2_reg = l2_reg + torch.norm(param)
                loss = loss + alpha * l2_reg

                val_loss += loss.item()

                correct += (outputs.round() == targets).sum().item()
                total += len(outputs)
                print("Got:", array(outputs.cpu()), "Checking first: ", array(outputs.round().cpu())[0], array(targets.cpu())[0])

            print("Accuracy: {}".format(correct / total / 3))
            loo_val_losses.append(val_loss)


    avg_train_loss = sum(loo_train_losses) / len(loo_train_losses)
    avg_val_loss = sum(loo_val_losses) / len(loo_val_losses)

    print(f'Average Train Loss: {avg_train_loss:.4f}')
    print(f'Average Validation Loss: {avg_val_loss:.4f}')

    return avg_train_loss, avg_val_loss


print("Extracting dataset...")
inputs, targets = [], []
valence, arousal, dominance = read_valence_arousal_dominance()

i = 0
for patient in tqdm(range(23)):
    for rec_type in ['stimuli']:  # No need 'baseline'
        for movie in range(18):
            raw = read_raw(patient, rec_type, movie, verbose=0)
            #for theta_psd, theta_frequencies, alpha_psd, alpha_frequencies, beta_psd, beta_frequencies in get_features(
                    #raw):
                #inputs.append(concatenate((theta_frequencies, alpha_frequencies, beta_frequencies), axis=1))
            for theta_psd, alpha_psd, beta_psd in get_features(raw):
                inputs.append(concatenate((theta_psd, alpha_psd, beta_psd), axis=1))
                #print(theta_psd)
                #inputs.append(array([theta_psd, alpha_psd, beta_psd]))
                targets.append([valence[i], arousal[i], dominance[i]])
            i += 1

print("Success")
dataset = TensorDataset(torch.tensor(array(inputs, dtype=float32), device=device),
                        torch.tensor(array(targets, dtype=float32), device=device))

criterion = nn.CrossEntropyLoss() # nn.MSELoss(reduction='sum')  # nn.L1Loss(reduction='sum')  # nn.MSELoss()  # nn.MultiLabelSoftMarginLoss()  # nn.CrossEntropyLoss()
optimizer_class = lambda params: optim.Adam(params, lr=0.001)


begin = process_time_ns()
avg_train_loss, avg_val_loss = loo_cv(DGCNN, dataset, criterion, optimizer_class, num_epochs=100,
                                      device=device, batch_size=200, alpha=0.01)
print("Training completed in {}.".format(format_time((process_time_ns() - begin) * 1E-9)))
