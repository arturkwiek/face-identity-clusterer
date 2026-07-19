import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def synthetic_identities():
    """Generate embeddings for 3 synthetic 'persons' + noise points.

    Each person is a random unit vector; their faces are the vector plus small
    Gaussian jitter, re-normalized — mimicking ArcFace behavior where the same
    person's embeddings have high cosine similarity.
    """
    rng = np.random.default_rng(42)
    dim, per_person = 512, 12
    persons = []
    for _ in range(3):
        base = rng.normal(size=dim)
        base /= np.linalg.norm(base)
        persons.append(base)

    embeddings, labels = [], []
    for pid, base in enumerate(persons):
        for _ in range(per_person):
            e = base + rng.normal(scale=0.02, size=dim)
            e /= np.linalg.norm(e)
            embeddings.append(e.astype(np.float32))
            labels.append(pid)

    # random noise faces (far from everyone)
    for _ in range(4):
        e = rng.normal(size=dim)
        e /= np.linalg.norm(e)
        embeddings.append(e.astype(np.float32))
        labels.append(-1)

    return np.vstack(embeddings), np.asarray(labels)
