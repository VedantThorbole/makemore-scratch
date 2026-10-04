import torch
import torch.nn.functional as F

# Load dataset

words = open("names.txt").read().splitlines()

# Create vocabulary

chars = sorted(list(set("".join(words))))

stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0

itos = {i: s for s, i in stoi.items()}


# ==========================================
# MODEL 1: Count-based Bigram Model
# ==========================================

N = torch.zeros((27, 27), dtype=torch.int32)


for w in words:
    chs = ["."] + list(w) + ["."]

    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        N[ix1, ix2] += 1


# Laplace smoothing
P = (N + 1).float()

# Normalize rows
# P(next character | current character)
P /= P.sum(1, keepdim=True)


# Random generator
g = torch.Generator().manual_seed(2147483647)

print("Count Model Generated Names:\n")

for i in range(5):
    out = []
    ix = 0
    while True:
        probs = P[ix]
        ix = torch.multinomial(probs,num_samples=1,replacement=True,generator=g).item()

        out.append(itos[ix])

        if ix == 0:
            break

    print("".join(out))

# Count model NLL

log_likelihood = 0.0
n = 0


for w in words:
    chs = ["."] + list(w) + ["."]

    for ch1, ch2 in zip(chs, chs[1:]):

        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1

avg_nll = -log_likelihood / n

print("\nCount Model NLL:")
print(avg_nll.item())


# ==========================================
# MODEL 2: Neural Network Bigram Model
# ==========================================

xs = []
ys = []

for w in words:
    chs = ["."] + list(w) + ["."]

    for ch1, ch2 in zip(chs, chs[1:]):

        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)

num = xs.nelement()

# Initialize weights
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27),generator=g,requires_grad=True)

# One-hot encoding

xenc = F.one_hot(xs,num_classes=27).float()

# Training

for k in range(100):

    # Forward pass
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1,keepdim=True)

    # Negative log likelihood + regularization

    loss = (-probs[torch.arange(num),ys].log().mean() +0.01 * (W**2).mean())

    # Backward pass
    W.grad = None
    loss.backward()

    # Update weights

    with torch.no_grad():

        W -= 50 * W.grad


print("\nNeural Network Final Loss:")
print(loss.item())


# ==========================================
# Generate from Neural Network
# ==========================================

g = torch.Generator().manual_seed(2147483647)

print("\nNeural Network Generated Names:\n")

for i in range(5):
    out = []
    ix = 0
    while True:

        xenc = F.one_hot(torch.tensor([ix]),num_classes=27).float()

        logits = xenc @ W

        probs = F.softmax(
            logits,
            dim=1
        )

        ix = torch.multinomial(probs,num_samples=1,generator=g).item()

        out.append(itos[ix])

        if ix == 0:
            break

    print("".join(out))