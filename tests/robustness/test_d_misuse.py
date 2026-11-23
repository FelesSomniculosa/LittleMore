"""D. Chat and API misuse: odd conversations, callbacks that stop or fail, threads, freed handles, NULL everywhere.

C-level cases run in child processes (`_rb.run_child`), so a crash is a classified failure. SDK-level cases (streams,
exceptions in a consumer, KeyboardInterrupt) run in a child too when they could hang or crash; each has a timeout.

What "correct" means here is checked against an independent value wherever there is one: the chat template against the
Hugging Face tokenizer's own `apply_chat_template` (identical greedy text from `lm_chat` and from `lm_generate` on the
Hugging Face rendering means identical prompt tokens), streaming against non-streaming (byte-exact), a reused session
against a fresh one.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import _rb
import gen_null_harness as G
from conftest import QWEN06_HF, ROOT

BUILD_BIN = ROOT / "build" / "bin"


# --------------------------------------------------------------------------------------------------------------
# NULL in every pointer parameter of every exported function (generated harness)
# --------------------------------------------------------------------------------------------------------------
def test_null_harness_is_generated_from_the_current_header() -> None:
    """The committed harness must be what the generator writes for today's header (a new function is covered)."""
    assert G.OUT.read_text(encoding="utf-8") == G.generate(), "run python tests/robustness/gen_null_harness.py"


def test_null_harness_parses_every_exported_function() -> None:
    names = {f for _, f, _ in G.parse_header(G.HEADER.read_text(encoding="utf-8"))}
    declared = set(__import__("re").findall(r"LM_API[^;(]*?LM_CALL\s+(lm_\w+)\s*\(", G.HEADER.read_text(encoding="utf-8")))
    assert names == declared and len(names) >= 17, (names ^ declared, len(names))


def test_null_harness_runs_clean(tiny: Path) -> None:
    exe = BUILD_BIN / ("null_harness" + (".exe" if os.name == "nt" else ""))
    if not exe.exists():
        pytest.skip("null_harness not built (cmake --build build)")
    p = subprocess.run([str(exe), str(tiny)], capture_output=True, text=True, timeout=120, env=_rb.child_env())
    print(ascii((p.stdout[-2000:])))
    assert p.returncode == 0 and "0 failures" in p.stdout, (p.returncode, p.stdout[-3000:], p.stderr[-2000:])


# --------------------------------------------------------------------------------------------------------------
# Conversations
# --------------------------------------------------------------------------------------------------------------
def test_odd_conversations(tiny: Path) -> None:
    o = _rb.run_child(f"""
st, m = load({str(tiny)!r}, {{'devices': 'none'}})
_, s = session(m, {{'n_ctx': 128}})
g = {{'temperature': 0.0, 'max_tokens': 4}}
out = {{}}
arr = (lib.lm_chat.argtypes[1]._type_ * 1)()
nocb = lib.lm_chat.argtypes[4]()  # a NULL token callback
out['empty_list'] = name(lib.lm_chat(s, arr, 0, None, nocb, None)); out['empty_list_err'] = err()
out['system_only'] = chat(s, [('system', 'You are terse.')], g)[0]
out['assistant_first'] = chat(s, [('assistant', 'Hi.'), ('user', 'Hello')], g)[0]
out['unknown_role'] = chat(s, [('wizard', 'abracadabra')], g)[0]
out['empty_role'] = chat(s, [('', 'x')], g)[0]
out['empty_content'] = chat(s, [('user', '')], g)[0]
arr2 = (lib.lm_chat.argtypes[1]._type_ * 2)(); arr2[0].role = b'user'; arr2[0].content = b'hi'; arr2[1].role = b'user'
out['null_content'] = name(lib.lm_chat(s, arr2, 2, None, nocb, None)); out['null_content_err'] = err()
out['many_messages'] = chat(s, [('user' if i % 2 == 0 else 'assistant', 'word ' * 3) for i in range(1000)], g)[0]
out['many_err'] = err()
out['huge_message'] = chat(s, [('user', 'x' * 1_000_000)], g)[0]
out['after_errors'] = chat(s, [('user', 'Hello')], g)[0]
emit(**out)
""", timeout=300)
    assert o.clean, o.describe()
    r = o.result
    print(ascii((r)))
    assert r["empty_list"] == "LM_ERR_INVALID_ARG" and r["empty_list_err"].startswith("lm_chat")
    assert r["null_content"] == "LM_ERR_INVALID_ARG" and "message 1" in r["null_content_err"]
    for k in ("system_only", "assistant_first", "unknown_role", "empty_role", "empty_content", "after_errors"):
        assert r[k] == "LM_OK", (k, r[k])
    assert r["many_messages"] == "LM_ERR_CONTEXT_FULL" and "raise lm_session_options.n_ctx" in r["many_err"], r
    assert r["huge_message"] == "LM_ERR_CONTEXT_FULL", r


def test_unrecognised_chat_template_is_a_clean_error(tiny: Path) -> None:
    """A template llama.cpp's built-in families do not recognise gives LM_ERR_TEMPLATE; lm_generate still works."""
    import struct
    L = _rb.GGUFLayout.parse(tiny)
    d = tiny.read_bytes()
    kv = L.kv[L.kv_index("tokenizer.chat_template")]
    tmpl = b"{{ this is not a template any family uses }}"
    entry = struct.pack("<Q", len(kv["key"])) + kv["key"].encode() + struct.pack("<IQ", _rb.GGUF_STRING, len(tmpl)) + tmpl
    meta = d[:kv["start"]] + entry + d[kv["end"]:L.tensor_info_end]
    p = _rb.SCRATCH / "bad_template.gguf"
    p.write_bytes(meta + bytes((-len(meta)) % 32) + d[L.data_start:])
    o = _rb.run_child(f"""
st, m = load({str(p)!r}, {{'devices': 'none'}})
_, s = session(m, {{'n_ctx': 64}})
c = chat(s, [('user', 'Hello')], {{'temperature': 0.0, 'max_tokens': 4}}); e = err()
g = generate(s, b'Hello', {{'temperature': 0.0, 'max_tokens': 4}})
emit(load=st, chat=c[0], chat_err=e, gen=g[0])
""")
    assert o.clean, o.describe()
    r = o.result
    assert r["load"] == "LM_OK" and r["chat"] == "LM_ERR_TEMPLATE" and r["chat_err"].startswith("lm_chat") and r["gen"] == "LM_OK", r


def _hf_prompt(messages: list[dict], **kw) -> str:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(QWEN06_HF))
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)


CONVERSATIONS = {
    "one-turn": [{"role": "user", "content": "What is the capital of France?"}],
    "system+user": [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "Name a colour."}],
    "multi-turn": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I help?"},
                   {"role": "user", "content": "Tell me a fact."}],
    "multi-turn-with-thinking": [{"role": "user", "content": "2+2?"},
                                 {"role": "assistant", "content": "<think>\nadd them\n</think>\n\n4"},
                                 {"role": "user", "content": "and 3+3?"}],
    "thinking-without-newlines": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "<think>greet</think>Hello!"},
                                  {"role": "user", "content": "Bye"}],
    "two-thinking-turns": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "A?"},
                           {"role": "assistant", "content": "<think>\none\n</think>\n\nB."}, {"role": "user", "content": "C?"},
                           {"role": "assistant", "content": "<think>\ntwo\n</think>\n\nD."}, {"role": "user", "content": "E?"}],
    # the last turn is the assistant's (after the last user turn): the template keeps its reasoning and so does lm_chat;
    # identical here because the reasoning is written in the template's own canonical form (docs/c-api.md lists the
    # cases after the last user turn where the template re-renders it and lm_chat does not)
    "thinking-after-last-user": [{"role": "user", "content": "Q?"},
                                 {"role": "assistant", "content": "<think>\nr\n</think>\n\nA."}],
}


@pytest.mark.parametrize("conv", sorted(CONVERSATIONS))
def test_qwen3_chat_template_matches_hugging_face(qwen06: Path, conv: str) -> None:
    """lm_chat's prompt must be the one Qwen3's own template renders. The C API does not expose lm_chat's tokens, so
    the check is on output: greedy lm_chat == greedy lm_generate of the Hugging Face rendering (CPU, deterministic,
    24 tokens). For the histories with reasoning, a control shows that this comparison tells the two candidate prompts
    apart: lm_generate of llama.cpp's plain ChatML rendering (reasoning kept) gives a different text."""
    if not (QWEN06_HF / "tokenizer_config.json").exists():
        pytest.skip("the Qwen3-0.6B tokenizer is not cached")
    msgs = CONVERSATIONS[conv]
    hf = _hf_prompt(msgs)
    # llama.cpp's ChatML formatting without any stripping (engine/src/llama-chat.cpp, LLM_CHAT_TEMPLATE_CHATML)
    plain = "".join(f"<|im_start|>{x['role']}\n{x['content']}<|im_end|>\n" for x in msgs) + "<|im_start|>assistant\n"
    o = _rb.run_child(f"""
st, m = load({str(qwen06)!r}, {{'devices': 'none'}})
_, s = session(m, {{'n_ctx': 512}})
g = {{'temperature': 0.0, 'max_tokens': 24}}
c = chat(s, {[(x['role'], x['content']) for x in msgs]!r}, g)
r = generate(s, {hf!r}.encode(), g)
u = generate(s, {plain!r}.encode(), g)
emit(chat=c[0], gen=r[0], same=(c[1] == r[1]), control_differs=(u[1] != r[1]), plain_is_hf=({plain!r} == {hf!r}),
     chat_text=c[1].decode('utf-8', 'replace')[:120], gen_text=r[1].decode('utf-8', 'replace')[:120])
""", timeout=300)
    assert o.clean, o.describe()
    r = o.result
    print(ascii((conv, r)))
    assert r["chat"] == r["gen"] == "LM_OK"
    # Before the fix (docs/c-api.md, chat templates) the histories with <think> blocks in earlier assistant turns differed: llama.cpp's built-in
    # ChatML formatting kept them, Qwen3's template removes them. lm_chat now strips them for such templates.
    assert r["same"], r
    if any("</think>" in x["content"] for x in msgs[:-1] if x["role"] == "assistant") and not r["plain_is_hf"]:
        assert r["control_differs"], f"the greedy comparison cannot tell the stripped from the unstripped prompt: {r}"


# --------------------------------------------------------------------------------------------------------------
# Streaming: byte-exact pieces, UTF-8 boundaries, stopping
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("prompt", ["The capital of France is", "日本の首都は", "🙂🙃 emoji: ", "Ünïcødé ﷺ עברית ‏ مرحبا"])
def test_stream_pieces_are_byte_exact_and_valid_utf8(qwen06: Path, prompt: str) -> None:
    """64 greedy tokens: the concatenated pieces equal the non-streaming bytes, and every piece is valid UTF-8 on its
    own whenever the whole text is (the library holds back an unfinished sequence; docs/c-api.md, Streaming)."""
    o = _rb.run_child(f"""
st, m = load({str(qwen06)!r}, {{'devices': 'none'}})
_, s = session(m, {{'n_ctx': 256}})
g = {{'temperature': 0.0, 'max_tokens': 64}}
a = generate(s, {prompt!r}.encode(), g)
b = generate(s, {prompt!r}.encode(), g, Collector())
whole_valid = True
try: a[1].decode('utf-8')
except UnicodeDecodeError: whole_valid = False
bad = []
for p in b[2].pieces:
    try: p.decode('utf-8')
    except UnicodeDecodeError: bad.append(p.hex())
emit(same=(a[1] == b''.join(b[2].pieces)), n=len(b[2].pieces), whole_valid=whole_valid, bad_pieces=bad,
     multibyte=sum(1 for p in b[2].pieces if any(c >= 0x80 for c in p)))
""", timeout=300)
    assert o.clean, o.describe()
    r = o.result
    print(ascii((prompt, r)))
    assert r["same"], r
    if r["whole_valid"]:
        assert not r["bad_pieces"], r


@pytest.mark.parametrize("stop_after", [1, 5, 16])
def test_callback_stop(tiny: Path, stop_after: int) -> None:
    """Returning non-zero stops: exactly stop_after pieces, LM_OK, and the session works afterwards (16 = the last
    of 16 tokens)."""
    o = _rb.run_child(f"""
st, m = load({str(tiny)!r}, {{'devices': 'none'}})
_, s = session(m, {{'n_ctx': 128}})
g = {{'temperature': 0.0, 'max_tokens': 16}}
full = generate(s, b'Hello', g)
c = Collector(stop_after={stop_after}); st2 = generate(s, b'Hello', g, c)[0]
again = generate(s, b'Hello', g)
emit(st=st2, n=len(c.pieces), prefix=full[1].startswith(c.text), again_same=(again[1] == full[1]), full_pieces=len(full[2].pieces))
""")
    assert o.clean, o.describe()
    r = o.result
    assert r["st"] == "LM_OK" and r["n"] == min(stop_after, r["full_pieces"]) and r["prefix"] and r["again_same"], r


def test_raw_ctypes_callback_exception_does_not_crash(tiny: Path) -> None:
    """A Python exception inside a raw ctypes callback cannot cross into C: ctypes reports it and returns 0. The
    generation then simply continues; nothing crashes, the session stays usable."""
    o = _rb.run_child(f"""
from littlemore import _lib as L
st, m = load({str(tiny)!r}, {{'devices': 'none'}})
_, s = session(m, {{'n_ctx': 128}})
calls = []
def boom(p, n, u):
    calls.append(n); raise RuntimeError('boom')
cb = L.TokenCallback(boom)
st2 = name(lib.lm_generate(s, b'Hello', sampling(temperature=0.0, max_tokens=8), cb, None))
emit(st=st2, calls=len(calls), after=generate(s, b'Hello', {{'temperature': 0.0, 'max_tokens': 4}})[0])
""")
    assert o.clean, o.describe()
    assert o.result["st"] == "LM_OK" and o.result["calls"] >= 1 and o.result["after"] == "LM_OK", o.result


SDK = """
import sys, threading, time, gc, _thread
sys.path.insert(0, {pydir!r})
import littlemore
m = littlemore.Model({path!r}, devices='none', n_ctx=256)
"""


def _sdk_child(path: Path, body: str, timeout: float = 120) -> _rb.Outcome:
    return _rb.run_child(SDK.format(pydir=str(ROOT / "python"), path=str(path)) + body, timeout=timeout)


def test_sdk_stream_equals_collect_and_survives_consumer_errors(qwen06: Path) -> None:
    o = _sdk_child(qwen06, """
g = dict(temperature=0, max_tokens=48)
whole = m.generate('日本の首都は', **g)
streamed = ''.join(m.generate('日本の首都は', stream=True, **g))
# the consumer raises in the middle of a stream: the stream is closed, the session is released
try:
    for i, piece in enumerate(m.generate('Hello', stream=True, **g)):
        if i == 3: raise ValueError('consumer error')
except ValueError:
    pass
t = time.time(); after = m.generate('日本の首都は', **g); dt = time.time() - t
# abandoned generator (never closed, just dropped) after one piece
it = m.generate('Hello', stream=True, **g); next(it); del it; gc.collect()
after2 = m.generate('日本の首都は', **g)
# chat stream == chat
c1 = m.chat([{'role': 'user', 'content': 'Hi'}], **g); c2 = ''.join(m.chat([{'role': 'user', 'content': 'Hi'}], stream=True, **g))
emit(stream_same=(whole == streamed), after_same=(after == whole), after2_same=(after2 == whole), chat_same=(c1 == c2),
     n=len(whole), dt=round(dt, 2))
m.close()
""", timeout=300)
    assert o.clean, o.describe()
    r = o.result
    assert r["stream_same"] and r["after_same"] and r["after2_same"] and r["chat_same"], r


def test_sdk_keyboard_interrupt_during_stream_and_during_c_call(qwen06: Path) -> None:
    """Ctrl+C (simulated with _thread.interrupt_main, which raises KeyboardInterrupt in the main thread as SIGINT
    does) while iterating a stream, and while the main thread is inside a non-streaming C call: KeyboardInterrupt
    reaches the caller, nothing hangs, and the model is usable afterwards.
    No timing race: the stream is interrupted after its first piece, and the non-streaming call 0.2 s after the C call
    has started (seen through a wrapper of lm_generate), while 3000 of its tokens (tens of seconds on the CPU) remain."""
    o = _sdk_child(qwen06, """
from littlemore import _lib
m.close()
m = littlemore.Model(MODEL_PATH, devices='none', n_ctx=4096)
g = dict(temperature=0, max_tokens=3000)
ref = m.generate('Hello', temperature=0, max_tokens=8)
res = {}
try:
    for i, piece in enumerate(m.generate('Hello', stream=True, **g)):
        if i == 0:
            _thread.interrupt_main()
        time.sleep(0.01)
    res['stream'] = 'finished without interrupt'
except KeyboardInterrupt:
    res['stream'] = 'interrupted'
lib = _lib.load(); real = lib.lm_generate; started = threading.Event()
def spy(*a):
    started.set()
    return real(*a)
lib.lm_generate = spy
threading.Thread(target=lambda: (started.wait(60), time.sleep(0.2), _thread.interrupt_main())).start()
t = time.time()
try:
    m.generate('Hello', **g)
    res['collect'] = 'finished without interrupt'
except KeyboardInterrupt:
    res['collect'] = 'interrupted'
res['collect_s'] = round(time.time() - t, 2)
lib.lm_generate = real""".replace("MODEL_PATH", repr(str(qwen06))) + """
t = time.time(); res['after_same'] = m.generate('Hello', temperature=0, max_tokens=8) == ref; res['after_s'] = round(time.time() - t, 2)
m.close(); res['closed'] = m.closed
emit(**res)
""", timeout=300)
    assert o.clean, o.describe()
    r = o.result
    print(ascii((r)))
    assert r["stream"] == "interrupted" and r["collect"] == "interrupted" and r["after_same"] and r["closed"], r


def test_sdk_model_cannot_be_pickled_and_threads_share_it(qwen06: Path) -> None:
    """A Model holds C pointers: pickling (multiprocessing spawn) must fail with a Python error, not crash a
    child. Eight threads generating on one Model are serialised and all get the deterministic text."""
    o = _sdk_child(qwen06, """
import pickle
try:
    pickle.dumps(m); res = {'pickle': 'pickled?!'}
except Exception as e:
    res = {'pickle': type(e).__name__}
ref = m.generate('Hello', temperature=0, max_tokens=16)
outs = []
def work(): outs.append(m.generate('Hello', temperature=0, max_tokens=16))
ts = [threading.Thread(target=work) for _ in range(8)]; [t.start() for t in ts]; [t.join(120) for t in ts]
res.update(n=len(outs), all_same=all(x == ref for x in outs))
m.close()
emit(**res)
""", timeout=300)
    assert o.clean, o.describe()
    r = o.result
    assert r["pickle"] in ("TypeError", "PicklingError", "ValueError") and r["n"] == 8 and r["all_same"], r


# --------------------------------------------------------------------------------------------------------------
# Threads and lifetimes at the C level
# --------------------------------------------------------------------------------------------------------------
def test_many_sessions_of_one_model_in_threads(qwen06: Path, has_cuda: bool) -> None:
    """Six sessions of one model generating in six threads at once (CPU model and, if present, CUDA model): every
    thread's greedy text equals the single-threaded reference."""
    devices = ["'none'"] + (["'CUDA0'"] if has_cuda else [])
    for dev in devices:
        o = _rb.run_child(f"""
import threading
st, m = load({str(qwen06)!r}, {{'devices': {dev}}})
g = {{'temperature': 0.0, 'max_tokens': 24}}
_, s0 = session(m, {{'n_ctx': 128}}); ref = generate(s0, b'The capital of France is', g)[1]; free_session(s0)
sess = [session(m, {{'n_ctx': 128}})[1] for _ in range(6)]
out = [None] * 6
def work(i): out[i] = generate(sess[i], b'The capital of France is', g)
ts = [threading.Thread(target=work, args=(i,)) for i in range(6)]; [t.start() for t in ts]; [t.join() for t in ts]
busy = free_model(m)
emit(dev={dev}, statuses=[o[0] for o in out], same=[o[1] == ref for o in out], busy=busy,
     frees=[free_session(s) for s in sess], free=free_model(m))
""", timeout=600)
        assert o.clean, o.describe()
        r = o.result
        print(ascii((r)))
        assert r["statuses"] == ["LM_OK"] * 6 and all(r["same"]) and r["busy"] == "LM_ERR_BUSY" and r["free"] == "LM_OK", r


def test_one_session_from_two_threads_and_free_during_generation(tiny: Path) -> None:
    """Two threads on ONE session are serialised (both texts equal the reference). lm_session_free from another thread
    while a generation runs waits for it to finish (the header's rule); lm_model_free meanwhile is LM_ERR_BUSY."""
    o = _rb.run_child(f"""
import threading, time
from littlemore import _lib as L
st, m = load({str(tiny)!r}, {{'devices': 'none'}})
_, s = session(m, {{'n_ctx': 128}})
g = {{'temperature': 0.0, 'max_tokens': 32}}
ref = generate(s, b'Hello', g)[1]
out = [None, None]
def work(i): out[i] = generate(s, b'Hello', g)
ts = [threading.Thread(target=work, args=(i,)) for i in range(2)]; [t.start() for t in ts]; [t.join() for t in ts]
same = [o[1] == ref for o in out]
# free during generation: a slow callback keeps the generation running for about a second
started = threading.Event()
def slow(p, n, u):
    started.set(); time.sleep(0.03); return 0
cb = L.TokenCallback(slow)
res = {{}}
def gen(): res['gen'] = name(lib.lm_generate(s, b'Hello', sampling(temperature=0.0, max_tokens=32), cb, None))
t = threading.Thread(target=gen); t.start(); started.wait(10)
res['model_free_busy'] = free_model(m)
t0 = time.time(); res['session_free'] = free_session(s); res['free_waited_s'] = round(time.time() - t0, 2)
t.join()
res['model_free'] = free_model(m)
emit(same=same, **res)
""", timeout=120)
    assert o.clean, o.describe()
    r = o.result
    print(ascii((r)))
    assert r["same"] == [True, True] and r["gen"] == "LM_OK" and r["model_free_busy"] == "LM_ERR_BUSY"
    assert r["session_free"] == "LM_OK" and r["free_waited_s"] > 0.2 and r["model_free"] == "LM_OK", r


def test_session_reuse_after_errors(tiny: Path) -> None:
    """After each kind of failed call (context full, bad sampling, reserved field, a stopped callback) the same
    session gives exactly the text of a fresh session."""
    o = _rb.run_child(f"""
st, m = load({str(tiny)!r}, {{'devices': 'none'}})
g = {{'temperature': 0.0, 'max_tokens': 16}}
_, fresh = session(m, {{'n_ctx': 64}}); ref = generate(fresh, b'Hello', g)[1]
_, s = session(m, {{'n_ctx': 64}})
errs = [generate(s, b'Hello ' * 200, g)[0], generate(s, b'Hello', {{'max_tokens': 0}})[0],
        generate(s, b'Hello', {{'reserved0': 3}})[0], generate(s, b'Hello', g, Collector(stop_after=2))[0],
        generate(s, b'', g)[0]]
emit(errs=errs, same=(generate(s, b'Hello', g)[1] == ref))
""")
    assert o.clean, o.describe()
    r = o.result
    # the empty prompt is LM_OK here: the tiny model adds a beginning-of-text token (a Qwen3 model, which adds none,
    # gives LM_ERR_TOKENIZE; tests/robustness/test_c_prompts.py)
    assert r["errs"] == ["LM_ERR_CONTEXT_FULL", "LM_ERR_INVALID_ARG", "LM_ERR_UNSUPPORTED", "LM_OK", "LM_OK"], r
    assert r["same"], r


@pytest.mark.parametrize("case", ["session-double-free", "model-double-free", "session-use-after-free"])
def test_freed_handles_are_undefined_behaviour_documented(tiny: Path, case: str) -> None:
    """Passing a freed handle is a caller error the C API cannot detect (an opaque pointer to freed memory), as for
    free(). The header says so (lm_model_free / lm_session_free). This test records what happens; it only requires
    that the documented, detectable misuse (freeing a model with live sessions) is refused, which the other tests
    check. The Python SDK never passes a freed handle (close() is idempotent; use after close is LittleMoreError)."""
    body = {
        "session-double-free": "_, s = session(m, None); a = free_session(s); b = free_session(s); emit(a=a, b=b)",
        "model-double-free": "a = free_model(m); b = free_model(m); emit(a=a, b=b)",
        "session-use-after-free": "_, s = session(m, None); a = free_session(s); b = generate(s, b'Hi', None)[0]; emit(a=a, b=b)",
    }[case]
    o = _rb.run_child(f"st, m = load({str(tiny)!r}, {{'devices': 'none'}})\n" + body)
    print(ascii((case, o.kind, o.results)))
    header = (ROOT / "include" / "littlemore.h").read_text(encoding="utf-8")
    assert "undefined behaviour" in header and "freed" in header, "the header must state the freed-handle rule"


# A Phi-4-reasoning-like template: "<|im_start|>" and "<|im_sep|>" (llama.cpp formats it as Phi-4, not ChatML) and
# "<think>...</think>" in its text. lm_chat must not strip reasoning for it (only Qwen3-style ChatML templates do).
PHI4_REASONING = ("{% for message in messages %}{{'<|im_start|>' + message['role'] + '<|im_sep|>' + message['content'] + "
                  "'<|im_end|>'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant<|im_sep|>'}}{% endif %}"
                  "{# You may reason first, inside <think> ... </think>. #}")


def _with_template(src: Path, dst: Path, template: str) -> Path:
    """Copy of a GGUF with its tokenizer.chat_template replaced (metadata rewritten, tensor data copied unchanged)."""
    import struct
    if dst.exists():
        return dst
    L = _rb.GGUFLayout.parse(src)
    kv = L.kv[L.kv_index("tokenizer.chat_template")]
    t = template.encode("utf-8")
    entry = struct.pack("<Q", len(kv["key"])) + kv["key"].encode() + struct.pack("<IQ", _rb.GGUF_STRING, len(t)) + t
    with open(src, "rb") as f:
        meta = f.read(L.tensor_info_end)
    meta = meta[:kv["start"]] + entry + meta[kv["end"]:]
    tmp = dst.with_name(dst.name + ".tmp")
    with open(src, "rb") as f, open(tmp, "wb") as out:
        out.write(meta + bytes((-len(meta)) % 32))
        f.seek(L.data_start)
        while True:
            chunk = f.read(64 << 20)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dst)
    return dst


def test_phi4_style_reasoning_template_is_not_stripped(qwen06: Path) -> None:
    """A template that llama.cpp formats as Phi-4 keeps the reasoning of earlier assistant turns: greedy lm_chat equals
    lm_generate of the Phi-4 rendering WITH the <think> block, and differs from the rendering without it (control)."""
    model = _with_template(qwen06, _rb.SCRATCH / "qwen06_phi4_template.gguf", PHI4_REASONING)
    msgs = [("user", "2+2?"), ("assistant", "<think>\nadd them\n</think>\n\n4"), ("user", "and 3+3?")]

    def phi4(ms):
        return "".join(f"<|im_start|>{r}<|im_sep|>{c}<|im_end|>" for r, c in ms) + "<|im_start|>assistant<|im_sep|>"

    kept = phi4(msgs)
    stripped = phi4([(r, c.split("</think>")[-1].lstrip("\n") if r == "assistant" else c) for r, c in msgs])
    o = _rb.run_child(f"""
st, m = load({str(model)!r}, {{'devices': 'none'}})
_, s = session(m, {{'n_ctx': 512}})
g = {{'temperature': 0.0, 'max_tokens': 24}}
c = chat(s, {msgs!r}, g)
k = generate(s, {kept!r}.encode(), g)
x = generate(s, {stripped!r}.encode(), g)
emit(load=st, chat=c[0], same_as_kept=(c[1] == k[1]), kept_differs_from_stripped=(k[1] != x[1]))
""", timeout=600)
    assert o.clean, o.describe()
    r = o.result
    print(ascii(r))
    assert r["load"] == r["chat"] == "LM_OK" and r["kept_differs_from_stripped"], r
    assert r["same_as_kept"], r
