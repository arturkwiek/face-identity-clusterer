import numpy as np

from face_clusterer.classifier import IdentityResolver, OnlineClassifier
from face_clusterer.config import ClassifierConfig, PrototypeConfig
from face_clusterer.prototypes import PrototypeStore


def _unit(rng, dim=512):
    v = rng.normal(size=dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _person_samples(rng, base, n, scale=0.02):
    out = []
    for _ in range(n):
        e = base + rng.normal(scale=scale, size=len(base))
        out.append((e / np.linalg.norm(e)).astype(np.float32))
    return out


def _labeled_set(rng, n_persons=3, per_person=8):
    bases = [_unit(rng) for _ in range(n_persons)]
    embeddings, labels = [], []
    for i, base in enumerate(bases):
        for e in _person_samples(rng, base, per_person):
            embeddings.append(e)
            labels.append(f"person_{i:04d}")
    return np.vstack(embeddings), labels, bases


def test_fit_and_predict_known_person():
    rng = np.random.default_rng(0)
    x, y, bases = _labeled_set(rng)
    clf = OnlineClassifier(ClassifierConfig())

    report = clf.fit(x, y)
    assert report.trained and report.n_persons == 3
    assert clf.is_ready

    # an unseen face of a known person is classified correctly
    probe = _person_samples(rng, bases[1], 1)[0]
    pid, confidence = clf.predict(probe)
    assert pid == "person_0001" and confidence >= 0.75


def test_abstains_below_confidence_threshold():
    rng = np.random.default_rng(1)
    x, y, _ = _labeled_set(rng)
    # threshold of 1.0 is unreachable — the classifier must always abstain
    clf = OnlineClassifier(ClassifierConfig(confidence_threshold=1.0))
    clf.fit(x, y)

    pid, confidence = clf.predict(_unit(rng))
    assert pid is None and 0.0 <= confidence <= 1.0


def test_not_trained_without_enough_data():
    rng = np.random.default_rng(2)
    # 3 samples for a single person: too few persons AND too few samples
    base = _unit(rng)
    x = np.vstack(_person_samples(rng, base, 3))
    clf = OnlineClassifier(ClassifierConfig(min_samples_per_person=5))

    report = clf.fit(x, ["person_0000"] * 3)
    assert not report.trained and not clf.is_ready
    assert "person" in report.reason
    # an untrained classifier abstains rather than guessing
    assert clf.predict(base) == (None, 0.0)


def test_underrepresented_persons_are_dropped():
    rng = np.random.default_rng(3)
    x, y, bases = _labeled_set(rng, n_persons=2, per_person=8)
    # a third person with only 2 samples must not become a class
    rare = _person_samples(rng, _unit(rng), 2)
    x = np.vstack([x, np.vstack(rare)])
    y = y + ["person_rare"] * 2

    report = OnlineClassifier(ClassifierConfig(min_samples_per_person=5)).fit(x, y)
    assert report.trained
    assert "person_rare" not in report.classes and report.n_persons == 2


def test_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(4)
    x, y, bases = _labeled_set(rng)
    path = str(tmp_path / "classifier")

    clf = OnlineClassifier(ClassifierConfig(path=path))
    clf.fit(x, y)
    clf.save()

    reloaded = OnlineClassifier(ClassifierConfig(path=path))
    assert reloaded.load()
    probe = _person_samples(rng, bases[2], 1)[0]
    assert reloaded.predict(probe)[0] == clf.predict(probe)[0]


def test_load_missing_model_returns_false(tmp_path):
    clf = OnlineClassifier(ClassifierConfig(path=str(tmp_path / "nope")))
    assert not clf.load()
    assert not clf.is_ready


def test_resolver_prefers_classifier_then_prototypes():
    rng = np.random.default_rng(5)
    x, y, bases = _labeled_set(rng)

    protos = PrototypeStore(PrototypeConfig(match_threshold=0.6))
    for i, base in enumerate(bases):
        pid = protos.add(base)
        assert pid == f"person_{i:04d}"        # ids line up with the labels

    clf = OnlineClassifier(ClassifierConfig())
    clf.fit(x, y)
    resolver = IdentityResolver(protos, clf)

    ident = resolver.resolve(_person_samples(rng, bases[0], 1)[0])
    assert ident.person_id == "person_0000" and ident.source == "classifier"

    # without a classifier the same face still resolves, via prototypes
    ident2 = IdentityResolver(protos, None).resolve(
        _person_samples(rng, bases[0], 1)[0]
    )
    assert ident2.person_id == "person_0000" and ident2.source == "prototype"


def test_resolver_reports_unknown_for_stranger():
    rng = np.random.default_rng(6)
    protos = PrototypeStore(PrototypeConfig(match_threshold=0.6))
    protos.add(_unit(rng))

    ident = IdentityResolver(protos, None).resolve(_unit(rng))
    assert ident.person_id is None and ident.source == "unknown"


def test_resolver_ignores_classifier_for_forgotten_person():
    """A stale model naming a deleted person must not win the cascade."""
    rng = np.random.default_rng(7)
    x, y, bases = _labeled_set(rng)

    clf = OnlineClassifier(ClassifierConfig())
    clf.fit(x, y)

    # prototype store knows nobody — every classifier verdict is stale
    empty = PrototypeStore(PrototypeConfig(match_threshold=0.6))
    ident = IdentityResolver(empty, clf).resolve(_person_samples(rng, bases[0], 1)[0])
    assert ident.person_id is None and ident.source == "unknown"


def test_commit_updates_prototype():
    rng = np.random.default_rng(8)
    protos = PrototypeStore(PrototypeConfig(match_threshold=0.6, ema_alpha=0.5))
    base = _unit(rng)
    pid = protos.add(base)
    resolver = IdentityResolver(protos, None)

    probe = _person_samples(rng, base, 1)[0]
    ident = resolver.resolve(probe)
    assert ident.person_id == pid

    before = protos.prototypes[pid].vector.copy()
    resolver.commit(ident, probe)
    after = protos.prototypes[pid].vector
    assert float(after @ probe) > float(before @ probe)
