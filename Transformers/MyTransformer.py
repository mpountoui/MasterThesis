try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

Path = '/Users/ioannisbountouris/PycharmProjects/MasterThesis'

if IN_COLAB :
    import sys
    sys.path.append('/content/MasterThesis')
    Path = '/content/MasterThesis'

'----------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'

import matplotlib.pyplot as plt
import torch.nn          as nn
import numpy             as np
import torch

from tqdm                       import tqdm
from tqdm                       import trange
from torch.optim                import Adam
from torch.nn                   import CrossEntropyLoss
from torch.nn                   import MultiheadAttention
from torch.utils.data           import DataLoader
from torchvision.transforms     import ToTensor
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10

np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'----------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'
'---------------------------------------------------- ViT Block -------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'


class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads  = n_heads
        self.norm1    = nn.LayerNorm(hidden_d)
        self.mhsa     = MultiheadAttention(hidden_d, n_heads, batch_first=True, bias=False)
        self.norm2    = nn.LayerNorm(hidden_d)
        self.mlp      = nn.Sequential(
                            nn.Linear(hidden_d, mlp_ratio * hidden_d),
                            nn.GELU(),
                            nn.Linear(mlp_ratio * hidden_d, hidden_d)
                        )

    '------------------------------------------------------------------------------------------------------------------'

    def forward(self, x):
        Normalized_X        = self.norm1(x)
        Attention_output, _ = self.mhsa(Normalized_X, Normalized_X, Normalized_X)
        out = x + Attention_output
        out = out + self.mlp(self.norm2(out))
        return out


'----------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------ ViT -----------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'


class ViT(nn.Module):
    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        # Super constructor
        super(ViT, self).__init__()

        # Attributes
        self.chw       = chw  # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks  = n_blocks
        self.n_heads   = n_heads
        self.hidden_d  = hidden_d

        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] // n_patches, chw[2] // n_patches)

        # 1) Linear mapper
        self.input_d       = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.register_buffer('positional_embeddings', self.PositionalEmbeddings(n_patches ** 2 + 1, hidden_d), persistent=False)

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # 5) Classification MLPk
        self.mlp = nn.Sequential(
                        nn.Linear(self.hidden_d, out_d),
                        nn.Softmax(dim=-1)
                    )

    '------------------------------------------------------------------------------------------------------------------'

    def Patchify(self, images, patch_size):
        dim1 = patch_size[0]
        dim2 = patch_size[1]
        unfold = nn.Unfold(kernel_size=(dim1, dim2), stride=(dim1, dim2))
        patches = unfold(images)
        patches = patches.permute(0, 2, 1)
        return patches

    '------------------------------------------------------------------------------------------------------------------'

    def PositionalEmbeddings(self, sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result

    '------------------------------------------------------------------------------------------------------------------'

    def get_hidden_state(self, images, layers=[-1]):

        # Dividing images into patches
        n, c, h, w = images.shape
        patches = self.Patchify(images, self.patch_size).to(self.positional_embeddings.device)

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        features = [None] * len(layers)
        counter = 1
        for block in self.blocks:
            out = block(out)
            try:
                index = layers.index(counter)
                features[index] = out
            except ValueError:
                {}
            counter += 1

        return out, features

    '------------------------------------------------------------------------------------------------------------------'

    def get_features(self, images, layers=[-1]):
        _, features = self.get_hidden_state(images, layers)
        featuresReshape = []
        for f in features:
            if f is not None:
                featuresReshape.append(f.view(f.shape[0], -1))

        if len(featuresReshape):
            return featuresReshape
        else:
            raise Exception("wrong input")

    '------------------------------------------------------------------------------------------------------------------'

    def forward(self, images):
        out, _ = self.get_hidden_state(images, ())
        # Getting the classification token only
        out = out[:, 0]

        return self.mlp(out)  # Map to output dimension, output category distribution


'----------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------------------------------------'


def PerformTraining(model, criterion, LR, N_EPOCHS, train_loader, device):

    optimizer = Adam(model.parameters(), lr=LR)
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        counter = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            counter += len(x)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")
        print(f"Counter {counter}")


'----------------------------------------------------------------------------------------------------------------------'


def PerformTesting(model, criterion, test_loader, device):

    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        accuracy = correct / total * 100
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {accuracy:.2f}%")

    return accuracy


'----------------------------------------------------------------------------------------------------------------------'


def ModelTrainingAndEvaluation(TrainLoader, TestLoader, NumberOfBlocks, NumberOfHeads, NumberOfPatches, HiddenDimension, N_EPOCHS, LR):

    # Defining model and training options
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    model = ViT( (3, 32, 32),
                 n_patches=NumberOfPatches,
                 n_blocks=NumberOfBlocks,
                 hidden_d=HiddenDimension,
                 n_heads=NumberOfHeads,
                 out_d=10).to(device)

    criterion = CrossEntropyLoss()
    PerformTraining(model, criterion, LR, N_EPOCHS, TrainLoader, device)
    accuracy = PerformTesting(model, criterion, TestLoader, device)

    return accuracy


'----------------------------------------------------------------------------------------------------------------------'


def CreateDataLoaders():
    # Loading Data
    transform = ToTensor()

    train_set = CIFAR10(root='./../LoadDataset/Data', train=True , download=True, transform=transform)
    test_set  = CIFAR10(root='./../LoadDataset/Data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True , batch_size=128)
    test_loader  = DataLoader(test_set , shuffle=False, batch_size=128)

    return train_loader, test_loader


'----------------------------------------------------------------------------------------------------------------------'


def CreatePlot(X, Y, title, xlabel, ylabel):
    plt.plot(X, Y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(title  + '.png')
    plt.show()


'----------------------------------------------------------------------------------------------------------------------'


def TransformerEvaluationBasedOnPatches(train_loader, test_loader):
    NumberOfPatces = [8]
    epochs = [200]
    learning_rate = [0.001]
    hidden_d = [384]
    n_blocks = [7]
    n_heads  = [12]
    for b in n_blocks:
        for h in n_heads:
            for d in hidden_d:
                for e in epochs:
                    for lr in learning_rate:
                        AccuracyScores = []
                        for n in NumberOfPatces:
                            accuracy = ModelTrainingAndEvaluation(NumberOfBlocks=b,
                                                                  NumberOfHeads=h,
                                                                  HiddenDimension=d,
                                                                  N_EPOCHS=e,
                                                                  LR=lr,
                                                                  NumberOfPatches=n,
                                                                  TrainLoader=train_loader,
                                                                  TestLoader=test_loader)
                            AccuracyScores.append(accuracy)

                        CreatePlot(NumberOfPatces, AccuracyScores, f'Blocks={b}, Hidden_d= {d}, Heads={h}, '
                                                                   f'Epochs = {e}, LR = {lr}', 'Patches', 'Accuracy')


'----------------------------------------------------------------------------------------------------------------------'


if __name__ == '__main__':

    train_loader, test_loader = CreateDataLoaders()
    TransformerEvaluationBasedOnPatches(train_loader, test_loader)

