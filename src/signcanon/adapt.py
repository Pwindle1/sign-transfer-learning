from __future__ import annotations

import numpy as np


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed(net, inputs, batch_size: int = 128) -> np.ndarray:
    import torch

    device = _device()
    net = net.to(device).eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size].float().to(device)
            n, t, vc = batch.shape
            x = batch.view(n, t, net.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
            n, c, t, v, m = x.size()
            x = x.permute(0, 4, 3, 1, 2).contiguous().view(n, m * v * c, t)
            x = net.data_bn(x)
            x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
            for layer in range(1, 11):
                x = getattr(net, f"l{layer}")(x)
            width = x.size(1)
            feats = x.view(n, m, width, -1).mean(3).mean(1)
            chunks.append(feats.cpu().numpy())
    return np.concatenate(chunks)


def prototype_accuracy(
    support_embeddings: np.ndarray,
    support_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    def normalize(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    support = normalize(support_embeddings)
    test = normalize(test_embeddings)
    classes = sorted(set(support_labels.tolist()))
    prototypes = normalize(np.stack([support[support_labels == c].mean(0) for c in classes]))
    predictions = np.array(classes)[(test @ prototypes.T).argmax(1)]
    return float((predictions == test_labels).mean())


def full_finetune(
    donor_state: dict,
    num_classes: int,
    support_inputs,
    support_labels: np.ndarray,
    build_net,
    epochs: int = 30,
    lr: float = 0.01,
    batch_size: int = 128,
):
    import torch

    device = _device()
    net = build_net(num_classes)
    net.load_state_dict({k: v for k, v in donor_state.items() if not k.startswith("fc.")}, strict=False)
    net = net.to(device).train()
    optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=1e-4, foreach=False)
    criterion = torch.nn.CrossEntropyLoss()
    targets = torch.tensor(support_labels)
    n = len(support_labels)
    for _ in range(epochs):
        order = torch.randperm(n)
        for i in range(0, n, batch_size):
            batch = order[i : i + batch_size]
            optimizer.zero_grad()
            loss = criterion(net(support_inputs[batch].float().to(device)), targets[batch].long().to(device))
            loss.backward()
            optimizer.step()
    return net


def classify_accuracy(net, test_inputs, test_labels: np.ndarray, batch_size: int = 256) -> float:
    import torch

    device = _device()
    net = net.to(device).eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(test_inputs), batch_size):
            logits = net(test_inputs[i : i + batch_size].float().to(device))
            predictions.append(logits.argmax(1).cpu().numpy())
    return float((np.concatenate(predictions) == test_labels).mean())
