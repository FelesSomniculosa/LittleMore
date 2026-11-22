/*
 * test_api.c - tests of the LittleMore C API (include/littlemore.h), written in plain C.
 *
 * Being C (compiled as C, not C++) is part of the test: it proves the public header is valid C with no C++
 * types. The test loads a tiny random LittleBit GGUF (built by tests/c/make_tiny_gguf.py; CTest does that
 * first) and checks:
 *   - version queries; NULL arguments and bad struct sizes return LM_ERR_* (no crash) with a message;
 *   - a missing file gives LM_ERR_MODEL_LOAD and a non-empty lm_last_error();
 *   - lm_backends_info() lists the CPU (and CUDA0 in "cuda" mode);
 *   - auto-fit: on the CPU nothing is offloaded; on CUDA everything fits the real free memory, a forced tiny
 *     budget falls back to 0 GPU layers, and a budget for exactly one block gives the head plus one block;
 *   - tokenize/detokenize round trip and the buffer-too-small protocol;
 *   - lm_chat streams at least one piece, applies the model's chat template (same text as lm_generate on the
 *     hand-formatted ChatML prompt), stops when the callback asks, respects max_tokens;
 *   - determinism: temperature 0 twice gives the same text, and so does the same seed at temperature 0.8;
 *   - a prompt longer than the context gives LM_ERR_CONTEXT_FULL;
 *   - two threads on one session are serialised and both get the single-thread answer;
 *   - repeated load/session/chat/free cycles do not grow the process's private memory (a simple leak check).
 *
 * Usage:  test_api <model.gguf> <cpu|cuda>
 *   The model path may also come from the LITTLEMORE_TEST_MODEL environment variable. Without a model the
 *   test is skipped (exit code 77, which CTest reports as "skipped"); "cuda" mode is skipped the same way when
 *   no CUDA device is present.
 */
#include "littlemore.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#    include <psapi.h>
#else
#    include <pthread.h>
#    include <unistd.h>
#endif

#define SKIP_CODE 77

static int g_failures = 0;
static int g_checks   = 0;

/* CHECK records a failure with its line and keeps going, so one run reports every broken property. */
#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond)) {                                                         \
            ++g_failures;                                                      \
            fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);    \
            fprintf(stderr, __VA_ARGS__);                                      \
            fprintf(stderr, "\n");                                             \
        }                                                                      \
    } while (0)

/* True when the last error message names the function called in `expr` ("lm_tokenize(m, ...)" -> it must
 * contain "lm_tokenize"). The library starts every message with the failing function's name; checking it
 * means a call that forgot to set its own message cannot pass on the stale message of an earlier failure. */
static int error_names_call(const char * expr) {
    char name[64];
    size_t i = 0;
    while (expr[i] && expr[i] != '(' && i + 1 < sizeof(name)) {
        name[i] = expr[i];
        ++i;
    }
    name[i] = '\0';
    return strstr(lm_last_error(), name) != NULL;
}

/* An error status must come with a message of its own: every failing call is checked through this. */
#define CHECK_ERR(expr, expected)                                                                    \
    do {                                                                                             \
        lm_status st_ = (expr);                                                                      \
        CHECK(st_ == (expected), "%s returned %s, expected %s (last error: %s)", #expr,              \
              lm_status_name(st_), lm_status_name(expected), lm_last_error());                       \
        CHECK(lm_last_error()[0] != '\0', "%s failed without a message", #expr);                      \
        CHECK(error_names_call(#expr), "message of %s does not name the call: %s", #expr,            \
              lm_last_error());                                                                      \
    } while (0)

#define CHECK_OK(expr)                                                                               \
    do {                                                                                             \
        lm_status st_ = (expr);                                                                      \
        CHECK(st_ == LM_OK, "%s returned %s: %s", #expr, lm_status_name(st_), lm_last_error());      \
    } while (0)

/* ---- streaming helpers ------------------------------------------------------------------------------------ */

/* Collects every streamed piece into one string; stops after `stop_after` pieces when that is > 0. */
typedef struct collector {
    char   text[8192];
    size_t len;
    int    pieces;
    int    stop_after;
    int    bad_piece; /* a piece whose len disagreed with its NUL terminator */
} collector;

static int collect_cb(const char * piece, size_t len, void * user) {
    collector * c = (collector *) user;
    if (strlen(piece) != len) {
        c->bad_piece = 1;
    }
    if (c->len + len < sizeof(c->text)) {
        memcpy(c->text + c->len, piece, len);
        c->len += len;
        c->text[c->len] = '\0';
    }
    c->pieces++;
    return (c->stop_after > 0 && c->pieces >= c->stop_after) ? 1 : 0;
}

static void collector_init(collector * c, int stop_after) {
    memset(c, 0, sizeof(*c));
    c->stop_after = stop_after;
}

/* Prints text with control characters escaped, so generated whitespace is visible in the log. */
static void print_escaped(const char * label, const char * text) {
    printf("%s \"", label);
    for (; *text; ++text) {
        const unsigned char c = (unsigned char) *text;
        if (c == '\n') {
            printf("\\n");
        } else if (c < 0x20) {
            printf("\\x%02x", c);
        } else {
            putchar(c);
        }
    }
    printf("\"\n");
}

static lm_sampling greedy(int32_t max_tokens) {
    lm_sampling s;
    lm_sampling_default(&s);
    s.temperature = 0.0f;
    s.max_tokens  = max_tokens;
    return s;
}

/* ---- platform helpers ------------------------------------------------------------------------------------- */

/* Private (non-shared) bytes of this process: grows with every leaked heap or engine buffer. 0 = unknown. */
static uint64_t private_bytes(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &pmc, sizeof(pmc))) {
        return (uint64_t) pmc.PrivateUsage;
    }
    return 0;
#else
    /* /proc/self/statm: total, resident, shared pages; resident - shared approximates private memory. */
    FILE * f = fopen("/proc/self/statm", "r");
    unsigned long total = 0, resident = 0, shared = 0;
    if (!f) {
        return 0;
    }
    if (fscanf(f, "%lu %lu %lu", &total, &resident, &shared) != 3) {
        resident = shared = 0;
    }
    fclose(f);
    return (uint64_t) (resident - shared) * (uint64_t) sysconf(_SC_PAGESIZE);
#endif
}

typedef struct thread_job {
    lm_session * session;
    const char * prompt;
    collector    out;
    lm_status    status;
} thread_job;

static void run_job(thread_job * job) {
    lm_sampling s = greedy(16);
    collector_init(&job->out, 0);
    job->status = lm_generate(job->session, job->prompt, &s, collect_cb, &job->out);
}

#ifdef _WIN32
static DWORD WINAPI job_main(LPVOID arg) {
    run_job((thread_job *) arg);
    return 0;
}
#else
static void * job_main(void * arg) {
    run_job((thread_job *) arg);
    return NULL;
}
#endif

/* Runs two jobs at the same time and waits for both. */
static void run_two_threads(thread_job * a, thread_job * b) {
#ifdef _WIN32
    HANDLE ta = CreateThread(NULL, 0, job_main, a, 0, NULL);
    HANDLE tb = CreateThread(NULL, 0, job_main, b, 0, NULL);
    WaitForSingleObject(ta, INFINITE);
    WaitForSingleObject(tb, INFINITE);
    CloseHandle(ta);
    CloseHandle(tb);
#else
    pthread_t ta, tb;
    pthread_create(&ta, NULL, job_main, a);
    pthread_create(&tb, NULL, job_main, b);
    pthread_join(ta, NULL);
    pthread_join(tb, NULL);
#endif
}

/* ---- the test mode ---------------------------------------------------------------------------------------- */

static int g_cuda = 0; /* 1 in "cuda" mode */

/* Options for the mode under test: "cpu" uses no GPU at all; "cuda" uses CUDA0 only (so the result does not
 * depend on whether a Vulkan device is also present). */
static lm_model_options mode_options(void) {
    lm_model_options o;
    lm_model_options_default(&o);
    o.devices = g_cuda ? "CUDA0" : "none";
    return o;
}

/* ---- tests ------------------------------------------------------------------------------------------------ */

static void test_versions(void) {
    char expect[32];
    snprintf(expect, sizeof(expect), "%d.%d.%d", LM_VERSION_MAJOR, LM_VERSION_MINOR, LM_VERSION_PATCH);
    CHECK(lm_api_version() == LM_API_VERSION, "library API %d, header %d", lm_api_version(), LM_API_VERSION);
    CHECK(strcmp(lm_version(), expect) == 0, "lm_version() = %s, header says %s", lm_version(), expect);
    CHECK(strcmp(lm_status_name(LM_ERR_MODEL_LOAD), "LM_ERR_MODEL_LOAD") == 0, "%s", lm_status_name(LM_ERR_MODEL_LOAD));
}

static void test_null_args(void) {
    lm_model *   m = (lm_model *) (uintptr_t) 1;   /* must be overwritten with NULL on failure */
    size_t       n = 0;
    int32_t      tok[4];
    char         buf[16];
    lm_message   msg = { "user", "hi" };
    lm_sampling  s;
    lm_model_info info;

    CHECK_ERR(lm_model_options_default(NULL), LM_ERR_INVALID_ARG);
    CHECK_ERR(lm_session_options_default(NULL), LM_ERR_INVALID_ARG);
    CHECK_ERR(lm_sampling_default(NULL), LM_ERR_INVALID_ARG);
    CHECK_ERR(lm_model_load(NULL, NULL, &m), LM_ERR_INVALID_ARG);
    CHECK(m == NULL, "lm_model_load must set *out = NULL on failure");
    CHECK_ERR(lm_model_load("x.gguf", NULL, NULL), LM_ERR_INVALID_ARG);
    CHECK_ERR(lm_model_get_info(NULL, &info), LM_ERR_INVALID_ARG);
    CHECK_ERR(lm_session_new(NULL, NULL, NULL), LM_ERR_INVALID_ARG);
    lm_sampling_default(&s);
    CHECK_ERR(lm_chat(NULL, &msg, 1, &s, collect_cb, NULL), LM_ERR_INVALID_ARG);
    CHECK_ERR(lm_generate(NULL, "x", &s, collect_cb, NULL), LM_ERR_INVALID_ARG);
    CHECK_ERR(lm_tokenize(NULL, "x", 1, tok, 4, 1, &n), LM_ERR_INVALID_ARG);
    CHECK_ERR(lm_detokenize(NULL, tok, 1, buf, sizeof(buf), 0, &n), LM_ERR_INVALID_ARG);
    CHECK(lm_backends_info(NULL, 10) == 0, "lm_backends_info(NULL, 10) must fail");
    CHECK(lm_last_error()[0] != '\0', "lm_backends_info(NULL, 10) failed without a message");
    lm_model_free(NULL);   /* documented no-ops */
    lm_session_free(NULL);
}

/* The struct_size rules: too small is refused; larger with zero tail is fine; larger with a non-zero unknown
 * field is refused as unsupported. Uses a missing file so a passing options check ends in LM_ERR_MODEL_LOAD. */
static void test_struct_size_rules(void) {
    struct { lm_model_options o; uint64_t future_field; } big;
    lm_model_options o;
    lm_model * m = NULL;

    lm_model_options_default(&o);
    o.struct_size = 3;
    CHECK_ERR(lm_model_load("missing.gguf", &o, &m), LM_ERR_INVALID_ARG);

    memset(&big, 0, sizeof(big));
    lm_model_options_default(&big.o);
    big.o.struct_size = sizeof(big);
    CHECK_ERR(lm_model_load("missing.gguf", &big.o, &m), LM_ERR_MODEL_LOAD);
    big.future_field = 1;
    CHECK_ERR(lm_model_load("missing.gguf", &big.o, &m), LM_ERR_UNSUPPORTED);
}

static void test_bad_path(void) {
    lm_model * m = (lm_model *) (uintptr_t) 1;
    lm_model_options o = mode_options();
    CHECK_ERR(lm_model_load("this/file/does-not-exist.gguf", &o, &m), LM_ERR_MODEL_LOAD);
    CHECK(m == NULL, "out must be NULL after a failed load");
    CHECK(strstr(lm_last_error(), "does-not-exist.gguf") != NULL, "message should name the file: %s", lm_last_error());
}

static void test_backends_info(void) {
    size_t need = lm_backends_info(NULL, 0);
    char * buf;
    char   small[8];
    CHECK(need > 0, "size query failed: %s", lm_last_error());
    if (need == 0) {
        return;
    }
    buf = (char *) malloc(need + 1);
    CHECK(lm_backends_info(buf, need + 1) == need, "second call returned another length");
    printf("backends:\n%s", buf);
    CHECK(strncmp(buf, "CPU", 3) == 0 || strstr(buf, "\nCPU") != NULL, "no CPU device line in:\n%s", buf);
    if (g_cuda) {
        CHECK(strstr(buf, "CUDA0:") != NULL, "no CUDA0 line in:\n%s", buf);
    }
    /* Truncation: snprintf semantics, always NUL-terminated, same total length returned. */
    CHECK(lm_backends_info(small, sizeof(small)) == need, "truncated call returned another length");
    CHECK(strlen(small) == sizeof(small) - 1, "truncated text is not NUL-terminated at cap - 1");
    free(buf);
}

static lm_model * load(const char * path, const lm_model_options * o) {
    lm_model * m = NULL;
    lm_status st = lm_model_load(path, o, &m);
    CHECK(st == LM_OK && m != NULL, "load %s: %s: %s", path, lm_status_name(st), lm_last_error());
    return m;
}

static lm_model_info get_info(const lm_model * m) {
    lm_model_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    CHECK_OK(lm_model_get_info(m, &info));
    return info;
}

static void test_autofit(const char * path) {
    lm_model_options o = mode_options();
    lm_model * m;
    lm_model_info info;

    /* Auto-fit with the real free memory. */
    m = load(path, &o);
    if (!m) {
        return;
    }
    info = get_info(m);
    printf("model: %s, %d layers, auto-fit -> n_gpu_layers %d on '%s' (budget %llu, reserve %llu, layer %llu, head %llu)\n",
           info.description, info.n_layer, info.n_gpu_layers, info.gpu_devices,
           (unsigned long long) info.fit_budget_bytes, (unsigned long long) info.fit_reserve_bytes,
           (unsigned long long) info.fit_layer_bytes, (unsigned long long) info.fit_head_bytes);
    CHECK(info.n_layer == 2, "tiny model has 2 layers, got %d", info.n_layer);
    CHECK(info.n_vocab > 0 && info.n_embd == 256, "n_vocab %d n_embd %d", info.n_vocab, info.n_embd);
    if (!g_cuda) {
        CHECK(info.n_gpu_layers == 0, "CPU mode offloaded %d layers", info.n_gpu_layers);
        CHECK(info.gpu_devices[0] == '\0', "CPU mode used GPU devices '%s'", info.gpu_devices);
        lm_model_free(m);
        return;
    }
    CHECK(strcmp(info.gpu_devices, "CUDA0") == 0, "gpu_devices '%s'", info.gpu_devices);
    CHECK(info.n_gpu_layers == info.n_layer + 1, "the tiny model should fit entirely: %d of %d + 1",
          info.n_gpu_layers, info.n_layer);
    CHECK(info.fit_budget_bytes > 0 && info.fit_layer_bytes > 0 && info.fit_head_bytes > 0 && info.fit_reserve_bytes > 0,
          "auto-fit details not reported");
    {
        /* The layer estimate must at least hold the layer's weights: the model's bytes minus the token
         * embedding are all in the layers and the head (a loose sanity bound, not an exact one). */
        CHECK(info.fit_layer_bytes < info.size_bytes, "layer estimate %llu >= whole model %llu",
              (unsigned long long) info.fit_layer_bytes, (unsigned long long) info.size_bytes);
    }
    {
        lm_model_info budget_info = info;
        lm_model *    mb;
        lm_session *  s = NULL;
        collector     c;
        lm_sampling   smp = greedy(4);
        lm_message    msg = { "user", "the quick brown fox" };

        /* A tiny forced budget: nothing fits, the model must fall back to the CPU and still work. */
        o.vram_budget = 1;
        mb = load(path, &o);
        if (mb) {
            budget_info = get_info(mb);
            CHECK(budget_info.n_gpu_layers == 0, "budget 1 byte still offloaded %d layers", budget_info.n_gpu_layers);
            CHECK(budget_info.fit_budget_bytes == 1, "forced budget not reported: %llu",
                  (unsigned long long) budget_info.fit_budget_bytes);
            CHECK_OK(lm_session_new(mb, NULL, &s));
            collector_init(&c, 0);
            CHECK_OK(lm_chat(s, &msg, 1, &smp, collect_cb, &c));
            CHECK(c.pieces >= 1, "CPU-fallback model streamed nothing");
            lm_session_free(s);
            lm_model_free(mb);
        }
        /* Room for the reserve, the head and one and a half blocks: the head and exactly one block fit. */
        o.vram_budget = info.fit_reserve_bytes + info.fit_head_bytes + info.fit_layer_bytes + info.fit_layer_bytes / 2;
        mb = load(path, &o);
        if (mb) {
            budget_info = get_info(mb);
            CHECK(budget_info.n_gpu_layers == 2, "head + 1 block budget gave %d layers", budget_info.n_gpu_layers);
            lm_model_free(mb);
        }
        /* Room for the reserve and less than the head: nothing fits. */
        o.vram_budget = info.fit_reserve_bytes + info.fit_head_bytes / 2;
        mb = load(path, &o);
        if (mb) {
            budget_info = get_info(mb);
            CHECK(budget_info.n_gpu_layers == 0, "budget below the head gave %d layers", budget_info.n_gpu_layers);
            lm_model_free(mb);
        }
    }
    lm_model_free(m);
}

static void test_tokens(const lm_model * m) {
    const char * text = "the quick brown fox jumps over the lazy dog";
    int32_t tok[64];
    char    back[256];
    size_t  n = 0, len = 0, n_special = 0;

    CHECK_OK(lm_tokenize(m, text, strlen(text), tok, 64, 0, &n));
    CHECK(n > 0 && n < 64, "n = %zu", n);
    CHECK_OK(lm_detokenize(m, tok, n, back, sizeof(back), 0, &len));
    CHECK(len == strlen(text) && strcmp(back, text) == 0, "round trip gave '%s'", back);

    /* add_special adds the beginning-of-text token of this vocabulary (add_bos_token is true). */
    CHECK_OK(lm_tokenize(m, text, strlen(text), tok, 64, 1, &n_special));
    CHECK(n_special == n + 1, "add_special: %zu tokens, plain: %zu", n_special, n);

    /* Too small: reports the needed size. */
    CHECK_ERR(lm_tokenize(m, text, strlen(text), tok, 1, 0, &len), LM_ERR_BUFFER_TOO_SMALL);
    CHECK(len == n, "needed %zu, reported %zu", n, len);
    CHECK_OK(lm_tokenize(m, text, strlen(text), tok, 64, 0, &n)); /* the plain tokens again (no BOS) */
    CHECK_ERR(lm_detokenize(m, tok, n, back, 4, 0, &len), LM_ERR_BUFFER_TOO_SMALL);
    CHECK(len == strlen(text), "needed %zu, reported %zu", strlen(text), len);

    /* Out-of-range token id. */
    tok[0] = 1 << 30;
    CHECK_ERR(lm_detokenize(m, tok, 1, back, sizeof(back), 0, &len), LM_ERR_TOKENIZE);
}

static void test_chat(lm_model * m) {
    lm_session * s = NULL;
    collector    chat, raw, again, stop, seeded1, seeded2;
    lm_sampling  smp = greedy(12);
    lm_message   msg = { "user", "the quick brown fox" };
    lm_session_options so;

    lm_session_options_default(&so);
    so.n_ctx = 128;
    CHECK_OK(lm_session_new(m, &so, &s));
    if (!s) {
        return;
    }

    /* Streaming: at least one piece, every piece NUL-terminated at len. */
    collector_init(&chat, 0);
    CHECK_OK(lm_chat(s, &msg, 1, &smp, collect_cb, &chat));
    printf("chat: %d pieces\n", chat.pieces);
    print_escaped("chat text:", chat.text);
    CHECK(chat.pieces >= 1, "no piece streamed");
    CHECK(chat.pieces <= smp.max_tokens, "%d pieces for max_tokens %d", chat.pieces, smp.max_tokens);
    CHECK(!chat.bad_piece, "a piece's len disagreed with its NUL terminator");

    /* The model's template (ChatML) is applied: the same answer as generating from the hand-made prompt. */
    collector_init(&raw, 0);
    CHECK_OK(lm_generate(s, "<|im_start|>user\nthe quick brown fox<|im_end|>\n<|im_start|>assistant\n", &smp, collect_cb, &raw));
    CHECK(strcmp(chat.text, raw.text) == 0, "chat \"%s\" != generate(ChatML prompt) \"%s\"", chat.text, raw.text);
    /* ... and that comparison can tell prompts apart: the bare message (no template) gives another answer.
     * (A property of this fixed tiny model, checked so that the equality above is not true for any prompt.) */
    collector_init(&again, 0);
    CHECK_OK(lm_generate(s, "the quick brown fox", &smp, collect_cb, &again));
    print_escaped("generate text (no template):", again.text);
    CHECK(strcmp(chat.text, again.text) != 0, "the untemplated prompt gave the same answer \"%s\"", again.text);

    /* Determinism at temperature 0 (the session's earlier state must not leak into the next call). */
    collector_init(&again, 0);
    CHECK_OK(lm_chat(s, &msg, 1, &smp, collect_cb, &again));
    CHECK(strcmp(chat.text, again.text) == 0, "temperature 0 gave \"%s\" then \"%s\"", chat.text, again.text);

    /* Determinism of seeded sampling. */
    smp.temperature = 0.8f;
    smp.seed = 1234;
    collector_init(&seeded1, 0);
    collector_init(&seeded2, 0);
    CHECK_OK(lm_generate(s, "the lazy dog", &smp, collect_cb, &seeded1));
    CHECK_OK(lm_generate(s, "the lazy dog", &smp, collect_cb, &seeded2));
    CHECK(seeded1.pieces >= 1 && strcmp(seeded1.text, seeded2.text) == 0, "seed 1234 gave \"%s\" then \"%s\"",
          seeded1.text, seeded2.text);

    /* The callback can stop generation; that is not an error. */
    smp = greedy(12);
    collector_init(&stop, 1);
    CHECK_OK(lm_chat(s, &msg, 1, &smp, collect_cb, &stop));
    CHECK(stop.pieces == 1, "callback asked to stop after 1 piece, got %d", stop.pieces);

    /* max_tokens bounds the output. */
    smp = greedy(2);
    collector_init(&stop, 0);
    CHECK_OK(lm_generate(s, "the lazy dog", &smp, collect_cb, &stop));
    CHECK(stop.pieces >= 1 && stop.pieces <= 2, "max_tokens 2 gave %d pieces", stop.pieces);

    /* Invalid arguments on a valid session. */
    CHECK_ERR(lm_chat(s, &msg, 0, &smp, collect_cb, NULL), LM_ERR_INVALID_ARG);
    CHECK_ERR(lm_chat(s, NULL, 1, &smp, collect_cb, NULL), LM_ERR_INVALID_ARG);
    CHECK_ERR(lm_generate(s, NULL, &smp, collect_cb, NULL), LM_ERR_INVALID_ARG);
    smp.max_tokens = 0;
    CHECK_ERR(lm_generate(s, "x", &smp, collect_cb, NULL), LM_ERR_INVALID_ARG);
    {
        lm_message bad = { NULL, "x" };
        smp = greedy(2);
        CHECK_ERR(lm_chat(s, &bad, 1, &smp, collect_cb, NULL), LM_ERR_INVALID_ARG);
    }
    lm_session_free(s);

    /* A prompt that does not fit the context. */
    so.n_ctx = 16;
    s = NULL;
    CHECK_OK(lm_session_new(m, &so, &s));
    if (s) {
        smp = greedy(4);
        CHECK_ERR(lm_generate(s, "the quick brown fox jumps over the lazy dog while a small model reads bits of signs",
                              &smp, NULL, NULL), LM_ERR_CONTEXT_FULL);
        lm_session_free(s);
    }
}

static void test_threads(lm_model * m) {
    lm_session * s = NULL;
    thread_job   ref, a, b;
    CHECK_OK(lm_session_new(m, NULL, &s));
    if (!s) {
        return;
    }
    ref.session = a.session = b.session = s;
    ref.prompt = a.prompt = b.prompt = "the quick brown fox";
    run_job(&ref);
    CHECK(ref.status == LM_OK && ref.out.pieces >= 1, "reference run: %s", lm_status_name(ref.status));
    run_two_threads(&a, &b);
    CHECK(a.status == LM_OK && b.status == LM_OK, "thread runs: %s / %s", lm_status_name(a.status), lm_status_name(b.status));
    CHECK(strcmp(a.out.text, ref.out.text) == 0 && strcmp(b.out.text, ref.out.text) == 0,
          "concurrent calls on one session differ: \"%s\" / \"%s\" vs \"%s\"", a.out.text, b.out.text, ref.out.text);
    lm_session_free(s);
}

/* One full life cycle, the unit of the leak check. use_mmap = 0 so the weights are private memory and a leaked
 * model is visible in the private-bytes counter. */
static void cycle(const char * path) {
    lm_model_options o = mode_options();
    lm_model *   m = NULL;
    lm_session * s = NULL;
    lm_message   msg = { "user", "the lazy dog" };
    lm_sampling  smp = greedy(4);
    collector    c;
    o.use_mmap = 0;
    if (lm_model_load(path, &o, &m) != LM_OK) {
        CHECK(0, "cycle load: %s", lm_last_error());
        return;
    }
    if (lm_session_new(m, NULL, &s) == LM_OK) {
        collector_init(&c, 0);
        lm_chat(s, &msg, 1, &smp, collect_cb, &c);
        lm_session_free(s);
    }
    lm_model_free(m);
}

static void test_leaks(const char * path) {
    /* Limit 4 MiB over 20 cycles. Measured (Windows, this tiny model): a correct build grows by less than
     * 0.1 MiB (sometimes shrinks); a leaked model grows by 17 MiB on the CPU and 7.5 MiB on CUDA (its weights
     * are then in GPU memory, only the host-side part shows here); a leaked session by more than 480 MiB. */
    const int      n_cycles = 20;
    const uint64_t limit    = 4ull << 20;
    uint64_t before, after;
    int i;
    cycle(path); /* warm-up: first-use allocations (backend registries, CUDA context, caches) */
    cycle(path);
    before = private_bytes();
    if (before == 0) {
        printf("leak check skipped: private memory size not available on this platform\n");
        return;
    }
    for (i = 0; i < n_cycles; ++i) {
        cycle(path);
    }
    after = private_bytes();
    printf("leak check: private bytes %llu -> %llu over %d cycles (growth %lld bytes, limit %llu)\n",
           (unsigned long long) before, (unsigned long long) after, n_cycles, (long long) (after - before),
           (unsigned long long) limit);
    CHECK(after <= before + limit, "private memory grew by %llu bytes over %d cycles",
          (unsigned long long) (after - before), n_cycles);
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : getenv("LITTLEMORE_TEST_MODEL");
    const char * mode = argc > 2 ? argv[2] : "cpu";
    lm_model *   m;

    /* Unbuffered output: if the library crashes, the lines printed before the crash are not lost. */
    setvbuf(stdout, NULL, _IONBF, 0);
    if (strcmp(mode, "cpu") != 0 && strcmp(mode, "cuda") != 0) {
        fprintf(stderr, "usage: test_api <model.gguf> <cpu|cuda>\n");
        return 2;
    }
    g_cuda = strcmp(mode, "cuda") == 0;
    if (!path || !*path) {
        printf("SKIP: no model given (argument 1 or LITTLEMORE_TEST_MODEL); build one with tests/c/make_tiny_gguf.py\n");
        return SKIP_CODE;
    }
    {
        FILE * f = fopen(path, "rb");
        if (!f) {
            printf("SKIP: model %s not found; build it with tests/c/make_tiny_gguf.py\n", path);
            return SKIP_CODE;
        }
        fclose(f);
    }
    if (g_cuda) {
        size_t need = lm_backends_info(NULL, 0);
        char * buf = (char *) malloc(need + 1);
        int has_cuda = buf && need > 0 && lm_backends_info(buf, need + 1) == need && strstr(buf, "CUDA0:") != NULL;
        free(buf);
        if (!has_cuda) {
            printf("SKIP: cuda mode but no CUDA0 device is loaded\n");
            return SKIP_CODE;
        }
    }
    printf("test_api: mode %s, model %s, littlemore %s (API %d)\n", mode, path, lm_version(), lm_api_version());

    test_versions();
    test_null_args();
    test_struct_size_rules();
    test_bad_path();
    test_backends_info();
    test_autofit(path);
    {
        lm_model_options o = mode_options();
        m = load(path, &o);
    }
    if (m) {
        test_tokens(m);
        test_chat(m);
        test_threads(m);
        lm_model_free(m);
    }
    test_leaks(path);

    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
