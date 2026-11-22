/*
 * test_api.c - tests of the LittleMore C API (include/littlemore.h), written in plain C.
 *
 * Being C (compiled as C, not C++) is part of the test: it proves the public header is valid C with no C++
 * types. The test loads a tiny random LittleBit GGUF (built by tests/c/make_tiny_gguf.py; CTest does that
 * first) and checks:
 *   - the struct layouts have no hidden padding (compile-time asserts), version queries; NULL arguments, bad
 *     struct sizes and non-zero reserved fields return LM_ERR_* (no crash) with a message;
 *   - a missing file gives LM_ERR_MODEL_LOAD and a non-empty lm_last_error();
 *   - lm_backends_info() lists the CPU (and CUDA0 in "cuda" mode);
 *   - auto-fit: on the CPU nothing is offloaded; on CUDA everything fits the real free memory, a forced tiny
 *     budget falls back to 0 GPU layers, and a budget for exactly one block gives the head plus one block;
 *   - tokenize/detokenize round trip and the buffer-too-small protocol;
 *   - lm_chat streams at least one piece, applies the model's chat template (same text as lm_generate on the
 *     hand-formatted ChatML prompt), stops when the callback asks, respects max_tokens;
 *   - determinism: temperature 0 twice gives the same text, and so does the same seed at temperature 0.8;
 *   - a prompt longer than the context gives LM_ERR_CONTEXT_FULL; a generation that fills the context stops
 *     at exactly n_ctx - n_prompt tokens;
 *   - re-entrant calls from a token callback (lm_chat, lm_generate, lm_session_free on the same session) and
 *     lm_model_free with a live session are refused with an error, the outer call completes;
 *   - an older, smaller lm_model_info is filled only up to its last field; lm_detokenize's special flag;
 *   - lm_last_error() is per thread;
 *   - two threads on one session are serialised and both get the single-thread answer;
 *   - repeated load/session/chat/free cycles do not grow the process's private memory (a simple leak check).
 *
 * Usage:  test_api <model.gguf> <cpu|cuda>
 *   The model path may also come from the LITTLEMORE_TEST_MODEL environment variable. Without a model the
 *   test is skipped (exit code 77, which CTest reports as "skipped"); "cuda" mode is skipped the same way when
 *   no CUDA device is present.
 */
#include "littlemore.h"
#include "../../src/utf8_boundary.h" /* the library's own boundary rule, to check streamed pieces against */

#include <stddef.h>
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

/* The header promises structs without hidden padding on 64-bit platforms: sizeof() equals the end of the last
 * field, and no field follows a gap. Checked here at compile time, from C, i.e. with the C compiler's layout. */
#define END_OF(T, f) (offsetof(T, f) + sizeof(((T *) 0)->f))
_Static_assert(sizeof(void *) == 8, "64-bit platforms only");
_Static_assert(sizeof(lm_model_options) == END_OF(lm_model_options, use_mmap), "tail padding in lm_model_options");
_Static_assert(offsetof(lm_model_options, devices) == END_OF(lm_model_options, reserved0), "hole in lm_model_options");
_Static_assert(offsetof(lm_model_options, vram_budget) == END_OF(lm_model_options, devices), "hole in lm_model_options");
_Static_assert(sizeof(lm_model_options) == 40, "lm_model_options size");
_Static_assert(sizeof(lm_session_options) == END_OF(lm_session_options, reserved0), "tail padding in lm_session_options");
_Static_assert(sizeof(lm_session_options) == 24, "lm_session_options size");
_Static_assert(sizeof(lm_sampling) == END_OF(lm_sampling, reserved0), "tail padding in lm_sampling");
_Static_assert(sizeof(lm_sampling) == 32, "lm_sampling size");
_Static_assert(offsetof(lm_model_info, size_bytes) == END_OF(lm_model_info, reserved0), "hole in lm_model_info");
_Static_assert(sizeof(lm_model_info) == END_OF(lm_model_info, description), "tail padding in lm_model_info");
_Static_assert(sizeof(lm_model_info) == 336, "lm_model_info size");
_Static_assert(sizeof(lm_message) == 16, "lm_message size");
_Static_assert(sizeof(lm_status) == 4, "lm_status is 32-bit");

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
    int    split_piece; /* a piece that ended inside an unfinished UTF-8 sequence (only the last may) */
} collector;

/* Token callback: records the text, flags pieces whose length disagrees with their NUL terminator or that split a UTF-8 sequence, and stops after `stop_after` pieces. */
static int collect_cb(const char * piece, size_t len, void * user) {
    collector * c = (collector *) user;
    if (strlen(piece) != len) {
        c->bad_piece = 1;
    }
    if (lm_utf8_complete_prefix(piece, len) != len) {
        c->split_piece = 1;
    }
    if (c->len + len < sizeof(c->text)) {
        memcpy(c->text + c->len, piece, len);
        c->len += len;
        c->text[c->len] = '\0';
    }
    c->pieces++;
    return (c->stop_after > 0 && c->pieces >= c->stop_after) ? 1 : 0;
}

/* Resets a collector; stop_after = 0 means never ask to stop. */
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

/* Sampling settings for deterministic output (temperature 0) of at most max_tokens pieces. */
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

/* One thread's work: generate 16 greedy tokens from the job's session and prompt. */
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

/* The header's version constants agree with what the library reports. */
static void test_versions(void) {
    char expect[32];
    snprintf(expect, sizeof(expect), "%d.%d.%d", LM_VERSION_MAJOR, LM_VERSION_MINOR, LM_VERSION_PATCH);
    CHECK(lm_api_version() == LM_API_VERSION, "library API %d, header %d", lm_api_version(), LM_API_VERSION);
    CHECK(strcmp(lm_version(), expect) == 0, "lm_version() = %s, header says %s", lm_version(), expect);
    CHECK(strcmp(lm_status_name(LM_ERR_MODEL_LOAD), "LM_ERR_MODEL_LOAD") == 0, "%s", lm_status_name(LM_ERR_MODEL_LOAD));
}

/* NULL arguments are refused with LM_ERR_INVALID_ARG, the error names the failing function and outputs are set to NULL. */
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
    CHECK(lm_model_free(NULL) == LM_OK, "lm_model_free(NULL) is a no-op");
    CHECK(lm_session_free(NULL) == LM_OK, "lm_session_free(NULL) is a no-op");
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
    /* A size that cuts a field in two cannot come from any header version. */
    o.struct_size = offsetof(lm_model_options, n_gpu_layers) + 2;
    CHECK_ERR(lm_model_load("missing.gguf", &o, &m), LM_ERR_INVALID_ARG);
    /* An older caller that ends after n_gpu_layers: accepted (the file is missing, so MODEL_LOAD). */
    o.struct_size = END_OF(lm_model_options, n_gpu_layers);
    CHECK_ERR(lm_model_load("missing.gguf", &o, &m), LM_ERR_MODEL_LOAD);
    /* Reserved fields must be 0. */
    lm_model_options_default(&o);
    o.reserved0 = 7;
    CHECK_ERR(lm_model_load("missing.gguf", &o, &m), LM_ERR_UNSUPPORTED);

    memset(&big, 0, sizeof(big));
    lm_model_options_default(&big.o);
    big.o.struct_size = sizeof(big);
    CHECK_ERR(lm_model_load("missing.gguf", &big.o, &m), LM_ERR_MODEL_LOAD);
    big.future_field = 1;
    CHECK_ERR(lm_model_load("missing.gguf", &big.o, &m), LM_ERR_UNSUPPORTED);
}

/* Loading a missing file fails with LM_ERR_MODEL_LOAD and leaves the output pointer NULL. */
static void test_bad_path(void) {
    lm_model * m = (lm_model *) (uintptr_t) 1;
    lm_model_options o = mode_options();
    CHECK_ERR(lm_model_load("this/file/does-not-exist.gguf", &o, &m), LM_ERR_MODEL_LOAD);
    CHECK(m == NULL, "out must be NULL after a failed load");
    CHECK(strstr(lm_last_error(), "does-not-exist.gguf") != NULL, "message should name the file: %s", lm_last_error());
}

/* lm_backends_info: the size query, a too-small buffer, and the device list. */
static void test_backends_info(void) {
    size_t need = lm_backends_info(NULL, 0);
    char * buf;
    char   small[8];
    CHECK(need > 0, "size query failed: %s", lm_last_error());
    if (need == 0) {
        return;
    }
    /* The text holds live free-memory figures, so its length may change between calls (documented): allocate
     * slack and accept any length that fits. */
    buf = (char *) malloc(need + 64);
    CHECK(buf != NULL, "out of memory");
    if (!buf) {
        return;
    }
    {
        const size_t got = lm_backends_info(buf, need + 64);
        CHECK(got > 0 && got < need + 64, "second call returned %u", (unsigned) got);
    }
    printf("backends:\n%s", buf);
    CHECK(strncmp(buf, "CPU", 3) == 0 || strstr(buf, "\nCPU") != NULL, "no CPU device line in:\n%s", buf);
    if (g_cuda) {
        CHECK(strstr(buf, "CUDA0:") != NULL, "no CUDA0 line in:\n%s", buf);
    }
    /* Truncation: snprintf semantics, always NUL-terminated, same total length returned. */
    CHECK(lm_backends_info(small, sizeof(small)) >= sizeof(small), "truncated call returned too small a length");
    CHECK(strlen(small) == sizeof(small) - 1, "truncated text is not NUL-terminated at cap - 1");
    free(buf);
}

/* Loads a model and fails the test with the library's message if that does not work. */
static lm_model * load(const char * path, const lm_model_options * o) {
    lm_model * m = NULL;
    lm_status st = lm_model_load(path, o, &m);
    CHECK(st == LM_OK && m != NULL, "load %s: %s: %s", path, lm_status_name(st), lm_last_error());
    return m;
}

/* Reads the model info using the current struct size. */
static lm_model_info get_info(const lm_model * m) {
    lm_model_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    CHECK_OK(lm_model_get_info(m, &info));
    return info;
}

/* Model info after a default load: layer counts, sizes and GPU placement agree with the mode under test. */
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

/* tokenize / detokenize: round trip, size queries, small buffers and special tokens. */
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

    /* special = 1 keeps the beginning-of-text token's text ("<s>" in this vocabulary); special = 0 drops it. */
    CHECK_OK(lm_tokenize(m, text, strlen(text), tok, 64, 1, &n_special));
    CHECK_OK(lm_detokenize(m, tok, n_special, back, sizeof(back), 1, &len));
    CHECK(strncmp(back, "<s>", 3) == 0, "special = 1 gave '%s'", back);
    CHECK_OK(lm_detokenize(m, tok, n_special, back, sizeof(back), 0, &len));
    /* (With the BOS token present the engine keeps the SentencePiece space before the first word, so the text
     * is " the quick ...": the exact round trip holds for tokens made with add_special = 0, checked above.) */
    CHECK(strstr(back, "<s>") == NULL && strstr(back, text) != NULL, "special = 0 gave '%s'", back);

    /* Out-of-range token id. */
    tok[0] = 1 << 30;
    CHECK_ERR(lm_detokenize(m, tok, 1, back, sizeof(back), 0, &len), LM_ERR_TOKENIZE);
}

/* Sessions: chat template, raw generation, stopping from the callback, and seeded-sampling reproducibility. */
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
    CHECK(!chat.split_piece, "a piece ended inside an unfinished UTF-8 sequence");

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

    /* A generation that runs into the end of the context: stops (LM_OK) after exactly n_ctx - n_prompt tokens.
     * This tiny model emits one-byte pieces ("x", "\n"), so pieces == tokens here. */
    so.n_ctx = 24;
    s = NULL;
    CHECK_OK(lm_session_new(m, &so, &s));
    if (s) {
        const char * p = "the lazy dog";
        size_t n_prompt = 0;
        collector full;
        lm_tokenize(m, p, strlen(p), NULL, 0, 1, &n_prompt);
        smp = greedy(1000);
        collector_init(&full, 0);
        CHECK_OK(lm_generate(s, p, &smp, collect_cb, &full));
        CHECK((size_t) full.pieces == 24 - n_prompt, "full context: %d pieces, expected %u (n_ctx 24, prompt %u)",
              full.pieces, (unsigned) (24 - n_prompt), (unsigned) n_prompt);
        lm_session_free(s);
    }

    /* LM_SEED_RANDOM runs (its output is random, so only the status is checked). */
    s = NULL;
    so.n_ctx = 128;
    CHECK_OK(lm_session_new(m, &so, &s));
    if (s) {
        smp = greedy(4);
        smp.temperature = 0.8f;
        smp.seed = LM_SEED_RANDOM;
        CHECK_OK(lm_generate(s, "the lazy dog", &smp, NULL, NULL));
        lm_session_free(s);
    }
}

/* ---- re-entrancy: what a token callback may and may not call ---------------------------------------------- */

typedef struct reentry {
    lm_session * session;
    lm_model *   model;
    lm_status    chat_status, generate_status, free_status, model_free_status, tokenize_status;
    int          calls;
} reentry;

/* Token callback that calls back into its own session, which the library must refuse. */
static int reentry_cb(const char * piece, size_t len, void * user) {
    reentry *   r   = (reentry *) user;
    lm_message  msg = { "user", "again" };
    lm_sampling s   = greedy(2);
    size_t      n   = 0;
    (void) piece;
    (void) len;
    if (r->calls++ == 0) {
        r->chat_status       = lm_chat(r->session, &msg, 1, &s, collect_cb, NULL);
        r->generate_status   = lm_generate(r->session, "again", &s, collect_cb, NULL);
        r->free_status       = lm_session_free(r->session);
        r->model_free_status = lm_model_free(r->model);
        r->tokenize_status   = lm_tokenize(r->model, "the dog", 7, NULL, 0, 0, &n); /* allowed: size query */
        if (r->tokenize_status == LM_ERR_BUFFER_TOO_SMALL && n > 0) {
            r->tokenize_status = LM_OK; /* cap 0 with tokens to return: the expected "too small" answer */
        }
    }
    return 0;
}

/* Calling the API from inside the session's own token callback is refused; the running generation continues. */
static void test_reentrancy(lm_model * m) {
    lm_session * s = NULL;
    reentry      r;
    lm_sampling  smp = greedy(4);
    lm_message   msg = { "user", "the quick brown fox" };
    CHECK_OK(lm_session_new(m, NULL, &s));
    if (!s) {
        return;
    }
    memset(&r, 0, sizeof(r));
    r.session = s;
    r.model   = m;
    CHECK_OK(lm_chat(s, &msg, 1, &smp, reentry_cb, &r));
    CHECK(r.calls == 4, "the outer chat should run to max_tokens (4 pieces), got %d", r.calls);
    CHECK(r.chat_status == LM_ERR_REENTRANT, "lm_chat from the callback: %s", lm_status_name(r.chat_status));
    CHECK(r.generate_status == LM_ERR_REENTRANT, "lm_generate from the callback: %s", lm_status_name(r.generate_status));
    CHECK(r.free_status == LM_ERR_REENTRANT, "lm_session_free from the callback: %s", lm_status_name(r.free_status));
    CHECK(r.model_free_status == LM_ERR_BUSY, "lm_model_free with a live session: %s", lm_status_name(r.model_free_status));
    CHECK(r.tokenize_status == LM_OK, "lm_tokenize from the callback: %s", lm_status_name(r.tokenize_status));
    /* The session survived the refused free and still works. */
    {
        collector c;
        collector_init(&c, 0);
        CHECK_OK(lm_chat(s, &msg, 1, &smp, collect_cb, &c));
        CHECK(c.pieces >= 1, "session unusable after a refused free");
    }
    /* Outside a callback, lm_model_free is refused while the session lives. */
    CHECK_ERR(lm_model_free(m), LM_ERR_BUSY);
    CHECK_OK(lm_session_free(s));
}

/* ---- model info from an older, smaller struct ------------------------------------------------------------- */

/* An older, smaller lm_model_info struct is filled only up to its size. */
static void test_small_info(const lm_model * m) {
    unsigned char raw[sizeof(lm_model_info)];
    lm_model_info * info = (lm_model_info *) raw;
    size_t i;
    memset(raw, 0xAB, sizeof(raw));
    info->struct_size = END_OF(lm_model_info, n_gpu_layers); /* an "old" struct of three fields */
    CHECK_OK(lm_model_get_info(m, info));
    CHECK(info->n_layer == 2, "n_layer %d", info->n_layer);
    for (i = END_OF(lm_model_info, n_gpu_layers); i < sizeof(raw); ++i) {
        if (raw[i] != 0xAB) {
            CHECK(0, "byte %u after the caller's struct was written", (unsigned) i);
            break;
        }
    }
    info->struct_size = END_OF(lm_model_info, n_gpu_layers) + 1; /* cuts n_ctx_train */
    CHECK_ERR(lm_model_get_info(m, info), LM_ERR_INVALID_ARG);
}

/* ---- lm_last_error is per thread -------------------------------------------------------------------------- */

typedef struct error_job {
    char message[256]; /* may truncate a long message; only its beginning is checked */
} error_job;

static void error_job_run(error_job * job) {
    lm_model * m = NULL;
    lm_model_load("thread-only-missing.gguf", NULL, &m); /* fails on this thread */
    snprintf(job->message, sizeof(job->message), "%s", lm_last_error());
}

#ifdef _WIN32
static DWORD WINAPI error_job_main(LPVOID arg) {
    error_job_run((error_job *) arg);
    return 0;
}
#else
static void * error_job_main(void * arg) {
    error_job_run((error_job *) arg);
    return NULL;
}
#endif

/* lm_last_error() is per thread: another thread's failure does not change this thread's message. */
static void test_error_per_thread(void) {
    error_job job;
    char      mine[256]; /* may truncate a long engine message: the checks below only rely on its beginning */
    lm_model * m = NULL;
    memset(&job, 0, sizeof(job));
    lm_model_load("main-thread-missing.gguf", NULL, &m);
    snprintf(mine, sizeof(mine), "%s", lm_last_error());
#ifdef _WIN32
    {
        HANDLE t = CreateThread(NULL, 0, error_job_main, &job, 0, NULL);
        WaitForSingleObject(t, INFINITE);
        CloseHandle(t);
    }
#else
    {
        pthread_t t;
        pthread_create(&t, NULL, error_job_main, &job);
        pthread_join(t, NULL);
    }
#endif
    CHECK(strstr(job.message, "thread-only-missing.gguf") != NULL, "other thread's error: %s", job.message);
    /* Prefix comparison: robust to `mine` being truncated (the message length depends on how many engine log lines
     * the failing path adds: 179 characters with the GPU auto-fit pre-check, 336 without a GPU). */
    CHECK(strncmp(lm_last_error(), mine, strlen(mine)) == 0 && strstr(mine, "main-thread-missing.gguf") != NULL,
          "this thread's error changed to: %s", lm_last_error());
}

/* Two threads generating on two sessions of one model give the same text as a single thread. */
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

/* Repeated load / generate / free cycles must not grow the process's private memory (limits inside). */
static void test_leaks(const char * path) {
    /* Limit 4 MiB over 20 cycles. Measured (Windows, this tiny model): a correct build changes by -0.6 to
     * +0.9 MiB (allocator noise); a leaked model grows by 17 MiB on the CPU and 7.5 MiB on CUDA (its weights
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

/* Runs every test in turn. Usage: test_api <model.gguf> <cpu|cuda>; returns SKIP_CODE (ctest: skipped) when the model or the CUDA device is missing. */
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
        char * buf = (char *) malloc(need + 64);
        int has_cuda = buf && need > 0 && lm_backends_info(buf, need + 64) > 0 && strstr(buf, "CUDA0:") != NULL;
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
    test_error_per_thread();
    test_backends_info();
    test_autofit(path);
    {
        lm_model_options o = mode_options();
        m = load(path, &o);
    }
    if (m) {
        test_tokens(m);
        test_chat(m);
        test_reentrancy(m);
        test_small_info(m);
        test_threads(m);
        CHECK_OK(lm_model_free(m));
    }
    test_leaks(path);

    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
