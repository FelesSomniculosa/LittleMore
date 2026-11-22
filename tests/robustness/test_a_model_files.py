"""A. Model-file robustness: every broken, odd or misused model file must give a clean error, never a crash or hang.

Every load runs in a child process (`_rb.run_child`) with a timeout, and the way the child ended is classified:
a clean exit with a reported status passes; an abort (exit code 3), an access violation, any other crash code or a
timeout fails the test with the case named. When a broken file happens to load, the child also opens a session,
generates a few greedy tokens and tokenizes (`_child.exercise`): a file that loads must not crash later either.

Each load is made twice where it matters: with `devices = "none"` (the plain engine loader) and with the defaults
(auto-fit: the library first parses the file itself in `plan_fit`, then the engine loads it), because the two parsers
are different code.

Covered: zero-byte and non-GGUF files, truncation at many offsets (header, key/value pairs, tensor infos, tensor
data), a seeded fuzz of the metadata of the tiny model (random bytes and structured field values) and of the
Qwen3-0.6B LittleBit model, wrong architecture, missing tokenizer keys, huge and negative lengths, tensor offsets out
of range, duplicate tensor names, a non-file path, a missing path, paths with spaces / non-ASCII characters / more
than 260 characters / invalid UTF-8, a file deleted or locked while loaded or before loading, the same file loaded
twice and two different models loaded at once.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import random
import shutil
import struct
from pathlib import Path

import pytest

import _rb

FUZZ_SEEDS_TINY = 500       # seeded mutations of the tiny model's metadata (the brief asks for at least 200)
FUZZ_SEEDS_QWEN06 = 60      # seeded mutations of the 0.6B model's metadata (larger file: fewer, run serially)
WORKERS = 4                 # parallel child processes


def _exercise(path: Path, auto_fit: bool, timeout: float = 120) -> _rb.Outcome:
    """Load (and if it loads, use) `path` in a child process. auto_fit=False uses devices="none"."""
    opts = "None" if auto_fit else "{'devices': 'none'}"
    return _rb.run_child(f"emit(**exercise({str(path)!r}, {opts}))", timeout=timeout)


def _assert_clean(o: _rb.Outcome, what: str, expect_load: str | None = None) -> None:
    assert o.clean, f"{what}: {o.describe()}"
    if expect_load is not None:
        assert o.result.get("load") == expect_load, f"{what}: expected {expect_load}, got {o.result}"


# --------------------------------------------------------------------------------------------------------------
# Not a model file at all
# --------------------------------------------------------------------------------------------------------------
NON_GGUF = {
    "zero-byte": b"",
    "magic-only": b"GGUF",
    "magic-and-version": b"GGUF\x03\x00\x00\x00",
    "version-1": b"GGUF\x01\x00\x00\x00" + bytes(16),
    "version-99": b"GGUF\x63\x00\x00\x00" + bytes(16),
    "version-0": b"GGUF\x00\x00\x00\x00" + bytes(16),
    "text": b"hello, I am not a model\n" * 50,
    "png": b"\x89PNG\r\n\x1a\n" + bytes(200),
    "safetensors": struct.pack("<Q", 2) + b"{}" + bytes(64),
    "random-4k": random.Random(1).randbytes(4096),
    "gguf-then-random": b"GGUF\x03\x00\x00\x00" + random.Random(2).randbytes(4096),
    "big-endian-gguf": b"GGUF\x00\x00\x00\x03" + bytes(16),
}


@pytest.mark.parametrize("auto_fit", [False, True], ids=["none", "autofit"])
@pytest.mark.parametrize("case", sorted(NON_GGUF))
def test_non_gguf_files_are_a_clean_model_load_error(tiny: Path, case: str, auto_fit: bool) -> None:
    p = _rb.SCRATCH / "nongguf" / f"{case}.gguf"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(NON_GGUF[case])
    _assert_clean(_exercise(p, auto_fit), case, "LM_ERR_MODEL_LOAD")


# --------------------------------------------------------------------------------------------------------------
# Truncation
# --------------------------------------------------------------------------------------------------------------
def _truncation_offsets(L: _rb.GGUFLayout) -> list[int]:
    """Cut points: inside the header, at and inside each of the first key/value pairs, inside the token array, in the
    tensor infos, at the start of the data and inside several tensors (the start, middle and last byte)."""
    offs = {0, 1, 3, 4, 5, 7, 8, 12, 15, 16, 20, 23, 24, 25}
    for kv in L.kv:
        offs |= {kv["start"], kv["start"] + 4, kv["start"] + 8, kv["start"] + 9, (kv["start"] + kv["end"]) // 2, kv["end"] - 1}
    for t in L.tensors[::8]:
        offs |= {t["start"], t["start"] + 9, t["end"] - 1}
    offs |= {L.tensor_info_end, L.tensor_info_end + 1, L.data_start - 1, L.data_start, L.data_start + 1}
    for t in L.tensors[::5]:
        offs |= {L.data_start + t["offset"] + 1, L.data_start + t["offset"] + 100}
    offs |= {L.file_size // 2, L.file_size - 4096, L.file_size - 2, L.file_size - 1}
    return sorted(o for o in offs if 0 <= o < L.file_size)


def _run_parallel(jobs: list[tuple[str, Path, bool]]) -> list[tuple[str, _rb.Outcome]]:
    """Run (label, file, auto_fit) loads in parallel child processes; returns (label, outcome) in job order."""
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(_exercise, p, af) for _, p, af in jobs]
        return [(label, f.result()) for (label, _, _), f in zip(jobs, futs)]


def test_truncated_at_many_offsets(tiny: Path) -> None:
    L = _rb.GGUFLayout.parse(tiny)
    data = tiny.read_bytes()
    out = _rb.SCRATCH / "trunc"
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for off in _truncation_offsets(L):
        p = out / f"trunc_{off}.gguf"
        p.write_bytes(data[:off])
        jobs.append((f"truncated at {off} of {L.file_size}", p, off % 2 == 1))  # alternate the two load paths
    bad = [(label, o) for label, o in _run_parallel(jobs) if not (o.clean and o.result.get("load") == "LM_ERR_MODEL_LOAD")]
    print(f"{len(jobs)} truncation points, {len(jobs) - len(bad)} clean LM_ERR_MODEL_LOAD")
    assert not bad, "\n".join(f"{label}: {o.describe()}" for label, o in bad[:10])


# --------------------------------------------------------------------------------------------------------------
# Seeded fuzzing of the metadata
# --------------------------------------------------------------------------------------------------------------
# Values a structured mutation writes into an integer field: boundaries, sign flips, absurdly large sizes.
INTERESTING = [0, 1, 2, 3, 7, 8, 9, 31, 32, 63, 64, 127, 128, 255, 256, 4095, 4096, 65535, 65536,
               2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**62, 2**63 - 1, 2**63, 2**64 - 1, -1, -2]


def mutate(data: bytes, L: _rb.GGUFLayout, seed: int) -> tuple[bytes, str]:
    """One seeded mutation of the metadata region. Returns (new bytes, description)."""
    rng = random.Random(seed)
    buf = bytearray(data)
    kind = seed % 5
    if kind == 4:  # structured, aimed at the model's hyperparameters: an interesting value in a scalar key/value
        f = rng.choice([f for f in L.fields if f.what.endswith("].value")])
        v = rng.choice(INTERESTING + [f.value + 1, f.value - 1, f.value * 2, f.value // 2])
        _rb.patch_int(buf, f, v)
        key = L.kv[int(f.what[3:f.what.index("]")])]["key"]
        return bytes(buf), f"{key} (offset {f.offset}) {f.value} -> {v}"
    if kind == 0:  # structured: an interesting value in one integer field (length, count, type tag, dim, offset)
        f = rng.choice(L.fields)
        v = rng.choice(INTERESTING + [f.value + 1, f.value - 1, f.value * 2, f.value ^ (1 << rng.randrange(8 * f.width))])
        _rb.patch_int(buf, f, v)
        return bytes(buf), f"{f.what} (offset {f.offset}) {f.value} -> {v}"
    if kind == 1:  # random bytes anywhere in the metadata
        n = rng.choice([1, 1, 2, 4, 8])
        where = []
        for _ in range(n):
            i = rng.randrange(L.tensor_info_end)
            buf[i] = rng.randrange(256)
            where.append(i)
        return bytes(buf), f"{n} random byte(s) at {where}"
    if kind == 2:  # a bit flip in one field
        f = rng.choice(L.fields)
        bit = rng.randrange(8 * f.width)
        _rb.patch_int(buf, f, f.value ^ (1 << bit))
        return bytes(buf), f"{f.what} bit {bit} flipped"
    # kind 3: insert or delete a few bytes inside the metadata (everything after it shifts)
    i = rng.randrange(8, L.tensor_info_end)
    n = rng.choice([1, 3, 4, 8])
    if rng.random() < 0.5:
        del buf[i:i + n]
        return bytes(buf), f"{n} byte(s) deleted at {i}"
    buf[i:i] = rng.randbytes(n)
    return bytes(buf), f"{n} byte(s) inserted at {i}"


def test_fuzz_metadata_tiny(tiny: Path) -> None:
    """FUZZ_SEEDS_TINY seeded mutations of the tiny model; each loaded (and used, if it loads) in a child process."""
    L = _rb.GGUFLayout.parse(tiny)
    data = tiny.read_bytes()
    out = _rb.SCRATCH / "fuzz_tiny"
    out.mkdir(parents=True, exist_ok=True)
    jobs, desc = [], {}
    for seed in range(FUZZ_SEEDS_TINY):
        mutated, d = mutate(data, L, seed)
        p = out / f"seed_{seed}.gguf"
        p.write_bytes(mutated)
        label = f"seed {seed}"
        desc[label] = d
        jobs.append((label, p, seed % 3 != 0))  # two thirds through the auto-fit parser, one third devices="none"
    results = _run_parallel(jobs)
    tally: dict[str, int] = {}
    bad = []
    for label, o in results:
        key = o.kind if not o.clean else ("loaded+used" if o.result.get("load") == "LM_OK" else o.result.get("load"))
        tally[key] = tally.get(key, 0) + 1
        if not o.clean:
            bad.append((label, o))
        elif o.result.get("load") == "LM_OK" and o.result.get("free_model") != "LM_OK":
            bad.append((label, o))
    print(f"tiny fuzz: {len(results)} mutations: {tally}")
    assert not bad, "\n".join(f"{label} [{desc[label]}]: {o.describe()}" for label, o in bad[:15])


def test_fuzz_metadata_qwen06(qwen06: Path) -> None:
    """FUZZ_SEEDS_QWEN06 seeded mutations of the Qwen3-0.6B LittleBit model (real tokenizer, 28 layers). The file is
    copied once; each mutation is written into the copy in place, loaded in a child, and undone."""
    L = _rb.GGUFLayout.parse(qwen06)
    head = qwen06.open("rb").read(L.data_start)
    work = _rb.SCRATCH / "fuzz_qwen06.gguf"
    if not work.exists() or work.stat().st_size != qwen06.stat().st_size:
        shutil.copyfile(qwen06, work)
    tally: dict[str, int] = {}
    bad = []
    for seed in range(FUZZ_SEEDS_QWEN06):
        seed_ = 10_000 + seed
        mutated, d = mutate(head, L, seed_)
        if seed_ % 5 == 3:  # an insertion / deletion changes the size of the head: keep the file length consistent
            mutated = (mutated + bytes(L.data_start))[:L.data_start] if len(mutated) < L.data_start else mutated[:L.data_start]
        with open(work, "r+b") as f:
            f.write(mutated)
        try:
            o = _exercise(work, seed % 2 == 0, timeout=300)
        finally:
            with open(work, "r+b") as f:
                f.write(head)
        key = o.kind if not o.clean else ("loaded+used" if o.result.get("load") == "LM_OK" else o.result.get("load"))
        tally[key] = tally.get(key, 0) + 1
        if not o.clean:
            bad.append((f"seed {seed_} [{d}]", o))
    print(f"qwen06 fuzz: {FUZZ_SEEDS_QWEN06} mutations: {tally}")
    assert not bad, "\n".join(f"{label}: {o.describe()}" for label, o in bad[:10])


# --------------------------------------------------------------------------------------------------------------
# Targeted metadata damage (each case named, so a failure says exactly what broke)
# --------------------------------------------------------------------------------------------------------------
def _set_string_kv(data: bytes, L: _rb.GGUFLayout, key: str, value: bytes) -> bytes:
    """Replace the value of a string key/value pair (the entry is rewritten, everything after it shifts)."""
    kv = L.kv[L.kv_index(key)]
    klen = len(kv["key"].encode())
    entry = struct.pack("<Q", klen) + kv["key"].encode() + struct.pack("<IQ", _rb.GGUF_STRING, len(value)) + value
    return data[:kv["start"]] + entry + data[kv["end"]:]


def _realign(meta: bytes, original: bytes, L: _rb.GGUFLayout) -> bytes:
    """Join changed metadata `meta` (header, key/value pairs and tensor infos) with the ORIGINAL tensor data,
    padded so the data starts at an aligned offset again (tensor offsets are relative to the data start, so they
    stay valid)."""
    pad = (-len(meta)) % 32
    return meta + bytes(pad) + original[L.data_start:]


def _drop_kv(data: bytes, L: _rb.GGUFLayout, key: str) -> bytes:
    """Remove one key/value pair (and fix the pair count and the data alignment)."""
    kv = L.kv[L.kv_index(key)]
    buf = bytearray(data[:kv["start"]] + data[kv["end"]:L.tensor_info_end])
    _rb.patch_int(buf, L.field("header.n_kv"), len(L.kv) - 1)
    return _realign(bytes(buf), data, L)


def _targeted_cases(tiny: Path) -> dict[str, bytes]:
    L = _rb.GGUFLayout.parse(tiny)
    d = tiny.read_bytes()
    cases: dict[str, bytes] = {}

    def with_string(key: str, value: bytes) -> bytes:
        nd = _set_string_kv(d, L, key, value)
        delta = len(nd) - len(d)
        return _realign(nd[:L.tensor_info_end + delta], d, L)

    cases["arch-unknown"] = with_string("general.architecture", b"no-such-architecture")
    cases["arch-bert"] = with_string("general.architecture", b"bert")
    cases["arch-qwen3-on-llama-keys"] = with_string("general.architecture", b"qwen3")
    cases["arch-empty"] = with_string("general.architecture", b"")
    cases["tokenizer-model-unknown"] = with_string("tokenizer.ggml.model", b"no-such-tokenizer")
    for key in ("tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type", "tokenizer.ggml.model",
                "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "general.architecture",
                "llama.block_count", "llama.embedding_length", "llama.attention.head_count", "llama.context_length",
                "tokenizer.chat_template"):
        cases[f"missing-{key}"] = _drop_kv(d, L, key)

    def field_set(what: str, value: int) -> bytes:
        buf = bytearray(d)
        _rb.patch_int(buf, L.field(what), value)
        return bytes(buf)

    i_tok = L.kv_index("tokenizer.ggml.tokens")
    i_arch = L.kv_index("general.architecture")
    i_bc = L.kv_index("llama.block_count")
    cases["key-len-huge"] = field_set(f"kv[{i_arch}].key_len", 2**62)
    cases["key-len-negative"] = field_set(f"kv[{i_arch}].key_len", -1)
    cases["str-len-huge"] = field_set(f"kv[{i_arch}].str_len", 2**63 - 1)
    cases["str-len-negative"] = field_set(f"kv[{i_arch}].str_len", -1)
    cases["arr-n-huge"] = field_set(f"kv[{i_tok}].arr_n", 2**62)
    cases["arr-n-negative"] = field_set(f"kv[{i_tok}].arr_n", -1)
    cases["arr-n-zero"] = field_set(f"kv[{i_tok}].arr_n", 0)
    cases["arr-elem-len-huge"] = field_set(f"kv[{i_tok}].arr[0].str_len", 2**40)
    cases["n-kv-huge"] = field_set("header.n_kv", 2**62)
    cases["n-tensors-huge"] = field_set("header.n_tensors", 2**62)
    cases["n-tensors-negative"] = field_set("header.n_tensors", -1)
    cases["n-tensors-zero"] = field_set("header.n_tensors", 0)
    cases["kv-type-invalid"] = field_set(f"kv[{i_arch}].type", 77)
    cases["block-count-huge"] = field_set(f"kv[{i_bc}].value", 2**31 - 1)
    cases["block-count-zero"] = field_set(f"kv[{i_bc}].value", 0)
    cases["block-count-negative"] = field_set(f"kv[{i_bc}].value", 2**32 - 1)
    t0 = 0
    t_lb = L.tensor_index("blk.0.attn_q.lb_v")
    cases["tensor-offset-past-end"] = field_set(f"tensor[{t0}].offset", L.file_size * 4)
    cases["tensor-offset-misaligned"] = field_set(f"tensor[{t0}].offset", 3)
    cases["tensor-offset-huge"] = field_set(f"tensor[{t0}].offset", 2**63)
    cases["tensor-type-invalid"] = field_set(f"tensor[{t_lb}].type", 999)
    cases["tensor-type-b1-to-f32"] = field_set(f"tensor[{t_lb}].type", 0)
    cases["tensor-ndims-5"] = field_set(f"tensor[{t_lb}].n_dims", 5)
    cases["tensor-ndims-0"] = field_set(f"tensor[{t_lb}].n_dims", 0)
    cases["tensor-dim-zero"] = field_set(f"tensor[{t_lb}].dim[0]", 0)
    cases["tensor-dim-huge"] = field_set(f"tensor[{t_lb}].dim[1]", 2**62)
    cases["tensor-dim-negative"] = field_set(f"tensor[{t_lb}].dim[1]", -1)
    cases["tensor-dim-not-block-multiple"] = field_set(f"tensor[{t_lb}].dim[0]", 100)
    cases["tensor-name-len-huge"] = field_set(f"tensor[{t_lb}].name_len", 2**40)

    # Duplicate tensor names: rename blk.0.attn_k.lb_u to blk.0.attn_q.lb_u (same length, so nothing shifts).
    a, b = L.tensors[L.tensor_index("blk.0.attn_k.lb_u")], "blk.0.attn_q.lb_u"
    assert a["name_len"] == len(b)
    buf = bytearray(d)
    buf[a["name_start"]:a["name_start"] + a["name_len"]] = b.encode()
    cases["duplicate-tensor-name"] = bytes(buf)
    # A required tensor renamed away (missing), and a duplicated key/value pair.
    buf = bytearray(d)
    tq = L.tensors[L.tensor_index("token_embd.weight")]
    buf[tq["name_start"]:tq["name_start"] + tq["name_len"]] = b"token_embd.weighX"
    cases["missing-token-embd"] = bytes(buf)
    kv = L.kv[i_arch]
    dup = bytearray(d[:L.tensor_info_start] + d[kv["start"]:kv["end"]] + d[L.tensor_info_start:L.tensor_info_end])
    _rb.patch_int(dup, L.field("header.n_kv"), len(L.kv) + 1)
    cases["duplicate-key"] = _realign(bytes(dup), d, L)
    # head_count stored as an array of STRINGS: the auto-fit's per-layer reader must not misread or abort on it.
    i_hc = L.kv_index("llama.attention.head_count")
    hc = L.kv[i_hc]
    entry = (struct.pack("<Q", len(hc["key"])) + hc["key"].encode() + struct.pack("<IIQ", _rb.GGUF_ARRAY, _rb.GGUF_STRING, 2)
             + struct.pack("<Q", 1) + b"4" + struct.pack("<Q", 1) + b"4")
    nd = d[:hc["start"]] + entry + d[hc["end"]:]
    cases["head-count-string-array"] = _realign(nd[:L.tensor_info_end + len(nd) - len(d)], d, L)
    i_hkv = L.kv_index("llama.attention.head_count_kv")
    hkv = L.kv[i_hkv]
    entry = (struct.pack("<Q", len(hkv["key"])) + hkv["key"].encode() + struct.pack("<IIQ", _rb.GGUF_ARRAY, _rb.GGUF_STRING, 2)
             + struct.pack("<Q", 1) + b"2" + struct.pack("<Q", 1) + b"2")
    nd = d[:hkv["start"]] + entry + d[hkv["end"]:]
    cases["head-count-kv-string-array"] = _realign(nd[:L.tensor_info_end + len(nd) - len(d)], d, L)
    # head_count_kv as a float array, and as an array longer / shorter than the block count
    entry = (struct.pack("<Q", len(hkv["key"])) + hkv["key"].encode() + struct.pack("<IIQ", _rb.GGUF_ARRAY, 6, 2)
             + struct.pack("<ff", 2.0, 2.0))
    nd = d[:hkv["start"]] + entry + d[hkv["end"]:]
    cases["head-count-kv-float-array"] = _realign(nd[:L.tensor_info_end + len(nd) - len(d)], d, L)
    for n in (1, 5):
        entry = (struct.pack("<Q", len(hkv["key"])) + hkv["key"].encode() + struct.pack("<IIQ", _rb.GGUF_ARRAY, 4, n)
                 + struct.pack(f"<{n}I", *([2] * n)))
        nd = d[:hkv["start"]] + entry + d[hkv["end"]:]
        cases[f"head-count-kv-array-of-{n}"] = _realign(nd[:L.tensor_info_end + len(nd) - len(d)], d, L)
    # alignment key with absurd values (must be a power of two per the spec)
    for val in (0, 3, 2**31):
        extra = (struct.pack("<Q", len("general.alignment")) + b"general.alignment" + struct.pack("<II", 4, val))
        buf = bytearray(d[:L.tensor_info_start] + extra + d[L.tensor_info_start:L.tensor_info_end])
        _rb.patch_int(buf, L.field("header.n_kv"), len(L.kv) + 1)
        cases[f"alignment-{val}"] = _realign(bytes(buf), d, L)
    return cases


TARGETED = ["arch-unknown", "arch-bert", "arch-qwen3-on-llama-keys", "arch-empty", "tokenizer-model-unknown",
            *[f"missing-{k}" for k in ("tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type",
                                       "tokenizer.ggml.model", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                                       "general.architecture", "llama.block_count", "llama.embedding_length",
                                       "llama.attention.head_count", "llama.context_length", "tokenizer.chat_template")],
            "key-len-huge", "key-len-negative", "str-len-huge", "str-len-negative", "arr-n-huge", "arr-n-negative",
            "arr-n-zero", "arr-elem-len-huge", "n-kv-huge", "n-tensors-huge", "n-tensors-negative", "n-tensors-zero",
            "kv-type-invalid", "block-count-huge", "block-count-zero", "block-count-negative", "tensor-offset-past-end",
            "tensor-offset-misaligned", "tensor-offset-huge", "tensor-type-invalid", "tensor-type-b1-to-f32",
            "tensor-ndims-5", "tensor-ndims-0", "tensor-dim-zero", "tensor-dim-huge", "tensor-dim-negative",
            "tensor-dim-not-block-multiple", "tensor-name-len-huge", "duplicate-tensor-name", "missing-token-embd",
            "duplicate-key", "head-count-string-array", "head-count-kv-string-array", "head-count-kv-float-array",
            "head-count-kv-array-of-1", "head-count-kv-array-of-5", "alignment-0", "alignment-3", "alignment-2147483648"]

# Damage that leaves a usable model (the engine has a default or does not need the key): loading may succeed.
MAY_LOAD = {"missing-tokenizer.chat_template", "missing-tokenizer.ggml.scores", "missing-tokenizer.ggml.token_type",
            "missing-tokenizer.ggml.bos_token_id", "missing-tokenizer.ggml.eos_token_id", "missing-llama.context_length",
            "missing-general.architecture", "arr-n-zero", "head-count-kv-array-of-5", "head-count-kv-array-of-1",
            "head-count-kv-float-array", "head-count-string-array", "head-count-kv-string-array"}


@pytest.fixture(scope="module")
def targeted(tiny: Path) -> dict[str, Path]:
    out = _rb.SCRATCH / "targeted"
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k, v in _targeted_cases(tiny).items():
        p = out / f"{k}.gguf"
        p.write_bytes(v)
        paths[k] = p
    assert sorted(paths) == sorted(TARGETED), set(paths) ^ set(TARGETED)
    return paths


@pytest.mark.parametrize("auto_fit", [False, True], ids=["none", "autofit"])
@pytest.mark.parametrize("case", TARGETED)
def test_targeted_metadata_damage(targeted: dict[str, Path], case: str, auto_fit: bool) -> None:
    o = _exercise(targeted[case], auto_fit)
    assert o.clean, f"{case}: {o.describe()}"
    load = o.result.get("load")
    if case not in MAY_LOAD:
        assert load in ("LM_ERR_MODEL_LOAD", "LM_ERR_UNSUPPORTED"), f"{case}: loaded although broken: {o.result}"
    print(f"{case} [{'autofit' if auto_fit else 'none'}]: {o.result}")


# --------------------------------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------------------------------
def test_directory_and_missing_path(tiny: Path) -> None:
    for p, what in ((tiny.parent, "a directory"), (tiny.parent / "no-such-file.gguf", "a missing file"),
                    (Path("Z:/no/such/drive/model.gguf"), "a missing drive"), (Path(""), "an empty path")):
        for af in (False, True):
            o = _exercise(p, af)
            _assert_clean(o, what, "LM_ERR_MODEL_LOAD")
            assert o.result.get("load_error", "").startswith("lm_model_load"), o.result


def _long_dir(base: Path, total: int) -> Path:
    """A directory under `base` whose full path is longer than `total` characters."""
    d = base
    i = 0
    while len(str(d)) < total:
        d = d / f"long-directory-name-{i:02d}"
        i += 1
    return d


@pytest.mark.parametrize("kind", ["spaces", "unicode", "over-260"])
def test_unusual_paths_load(tiny: Path, kind: str) -> None:
    """A model in a folder with spaces, with non-ASCII (CJK, accents, emoji) names, or at a path over 260 characters
    must load like any other (the API takes UTF-8 paths)."""
    base = _rb.SCRATCH / "paths"
    if kind == "spaces":
        p = base / "dir with  spaces" / "model file.gguf"
    elif kind == "unicode":
        p = base / "Ünïcødé 模型 🚀" / "モデル ü.gguf"
    else:
        p = _long_dir(base, 270) / "model.gguf"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tiny, p)
    except OSError as e:
        pytest.skip(f"cannot create the test path ({e}); long paths may be disabled on this machine")
    for af in (False, True):
        o = _exercise(p, af)
        assert o.clean, f"{kind}: {o.describe()}"
        assert o.result.get("load") == "LM_OK" and o.result.get("generate") == "LM_OK", f"{kind} ({len(str(p))} chars): {o.result}"


def test_invalid_utf8_path_is_a_clean_error(tiny: Path) -> None:
    """A path that is not valid UTF-8 (the API's documented encoding) must be refused or fail cleanly."""
    o = _rb.run_child("st, m = load(b'C:/\\xff\\xfe/bad\\xc3.gguf', {'devices': 'none'})\nemit(load=st, error=err())")
    assert o.clean and o.result["load"] == "LM_ERR_MODEL_LOAD", o.describe()


# --------------------------------------------------------------------------------------------------------------
# Files that change while in use, and several models
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("use_mmap", [1, 0])
def test_file_deleted_or_replaced_while_loaded(tiny: Path, use_mmap: int) -> None:
    """Load, then try to delete and to overwrite the file, then generate: the model must keep working (with mmap the
    OS refuses the delete on Windows; without mmap the weights were read and the file is free)."""
    p = _rb.SCRATCH / f"deleted_{use_mmap}.gguf"
    shutil.copyfile(tiny, p)
    snippet = f"""
import os
st, m = load({str(p)!r}, {{'devices': 'none', 'use_mmap': {use_mmap}}})
st_s, s = session(m, {{'n_ctx': 64}})
st1, t1, _ = generate(s, b'Hello', {{'temperature': 0.0, 'max_tokens': 8}})
try:
    os.remove({str(p)!r}); removed = 'removed'
except OSError as e:
    removed = 'refused: ' + type(e).__name__
try:
    open({str(p)!r}, 'wb').write(b'garbage' * 1000); overwritten = 'overwritten'
except OSError as e:
    overwritten = 'refused: ' + type(e).__name__
st2, t2, _ = generate(s, b'Hello', {{'temperature': 0.0, 'max_tokens': 8}})
emit(load=st, gen1=st1, gen2=st2, same=(t1 == t2), removed=removed, overwritten=overwritten,
     free_s=free_session(s), free_m=free_model(m))
"""
    o = _rb.run_child(snippet)
    assert o.clean, o.describe()
    r = o.result
    print(r)
    assert r["load"] == r["gen1"] == r["gen2"] == "LM_OK" and r["free_m"] == "LM_OK", r
    if use_mmap == 0 or r["overwritten"].startswith("refused"):
        assert r["same"], f"the weights changed under a running model: {r}"


def test_file_locked_by_another_process_is_a_clean_error(tiny: Path) -> None:
    """A file opened by another process without sharing (Windows share mode 0) cannot be read: clean error."""
    if os.name != "nt":
        pytest.skip("share modes are a Windows concept")
    import ctypes
    from ctypes import wintypes
    p = _rb.SCRATCH / "locked.gguf"
    shutil.copyfile(tiny, p)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                wintypes.DWORD, wintypes.HANDLE]
    h = k32.CreateFileW(str(p), 0x80000000 | 0x40000000, 0, None, 3, 0x80, None)  # GENERIC_READ|WRITE, no sharing
    assert h not in (None, wintypes.HANDLE(-1).value), ctypes.get_last_error()
    try:
        for af in (False, True):
            o = _exercise(p, af)
            _assert_clean(o, "locked file", "LM_ERR_MODEL_LOAD")
    finally:
        k32.CloseHandle(h)
    # unlocked again: it loads
    _assert_clean(_exercise(p, False), "after unlock", "LM_OK")


def test_same_file_twice_and_two_models_at_once(tiny: Path, qwen06: Path) -> None:
    """The same file loaded twice gives two independent models with identical output; a second, different model
    loaded at the same time does not disturb the first. Generations are interleaved between the models."""
    snippet = f"""
g = {{'temperature': 0.0, 'max_tokens': 16}}
st1, a = load({str(tiny)!r}, {{'devices': 'none'}})
st2, b = load({str(tiny)!r}, {{'devices': 'none'}})
st3, q = load({str(qwen06)!r})
_, sa = session(a, {{'n_ctx': 128}}); _, sb = session(b, {{'n_ctx': 128}}); _, sq = session(q, {{'n_ctx': 128}})
ta = generate(sa, b'Hello', g)[1]; tq = generate(sq, b'The capital of France is', g)[1]
tb = generate(sb, b'Hello', g)[1]; tq2 = generate(sq, b'The capital of France is', g)[1]
ta2 = generate(sa, b'Hello', g)[1]
busy = free_model(a)
emit(loads=[st1, st2, st3], same_file_same_text=(ta == tb == ta2), qwen_repeatable=(tq == tq2), len_a=len(ta),
     len_q=len(tq), busy=busy, frees=[free_session(sa), free_session(sb), free_session(sq), free_model(a),
     free_model(b), free_model(q)])
"""
    o = _rb.run_child(snippet, timeout=300)
    assert o.clean, o.describe()
    r = o.result
    print(r)
    assert r["loads"] == ["LM_OK"] * 3 and r["same_file_same_text"] and r["qwen_repeatable"], r
    assert r["busy"] == "LM_ERR_BUSY" and r["frees"] == ["LM_OK"] * 6, r
