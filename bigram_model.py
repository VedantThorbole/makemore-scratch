import torch

# Load dataset
words = open("names.txt").read().splitlines()

# Create count matrix
N = torch.zeros((27, 27), dtype=torch.int32)


# Character vocabulary
chars = sorted(list(set("".join(words))))

stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0

itos = {i: s for s, i in stoi.items()}


# Build bigram count matrix
for w in words:
    chs = ["."] + list(w) + ["."]

    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1


# Convert counts into probabilities
P = N.float()

# Normalize rows
# P(next_character | current_character)
P /= P.sum(1, keepdim=True)


# Random generator for reproducibility
g = torch.Generator().manual_seed(2147483647)


# Generate sample names
print("Generated names:\n")

for i in range(5):
    out = []
    ix = 0

    while True:
        probs = P[ix]

        ix = torch.multinomial(
            probs,
            num_samples=1,
            replacement=True,
            generator=g
        ).item()

        out.append(itos[ix])

        if ix == 0:
            break

    print("".join(out))


# Calculate Negative Log Likelihood
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


print("\nNegative Log Likelihood:")
print(avg_nll.item())