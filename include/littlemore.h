/*
 * littlemore.h - the LittleMore C API.
 *
 * LittleMore runs LittleBit (sub-1-bit) language models, and ordinary GGUF models, through a patched llama.cpp.
 * This header is the whole public interface of the `littlemore` shared library: load a model, open a session,
 * chat or generate text with streaming, tokenize, and ask which compute backends are available.
 *
 * Design rules (they make the library usable from C, Python ctypes, C#, Rust, ... and safe to upgrade):
 *   - Plain C only: no C++ types, no engine (llama.h / ggml.h) types. Objects are opaque pointers.
 *   - Platforms: 64-bit only (LP64: Linux, macOS; LLP64: Windows). The struct layouts below are designed for
 *     these ABIs: every struct ends exactly at the end of its last field (explicit `reserved` fields fill what
 *     would otherwise be hidden padding), so sizeof(struct) is the same as "end of the last field".
 *   - ABI stability (append-only rule): every options/info struct starts with a `struct_size` field. Fill it
 *     with sizeof() of the struct as you compiled it (the lm_*_default() initialisers do this). Fields are only
 *     ever appended at the end, never moved, resized or removed; `reserved` fields must be 0 and may become real
 *     fields later. So:
 *       - an older caller (smaller struct_size) is read up to its last field, the library uses defaults for the
 *         rest; struct_size must end on a field boundary (a size cutting a field in two is LM_ERR_INVALID_ARG);
 *       - a newer caller (larger struct_size) is accepted only when every byte after the last field this library
 *         knows is zero, because a non-zero unknown field is a request it cannot honour (LM_ERR_UNSUPPORTED
 *         instead of ignoring it silently).
 *     lm_message is the exception: it is used in arrays, so it has no struct_size and never changes.
 *     LM_API_VERSION changes only when this contract itself is broken.
 *   - Errors: every fallible function returns an lm_status. After any failure lm_last_error() returns a
 *     non-empty, human-readable message for the calling thread that starts with the failing function's name.
 *     For FFI: lm_status is a C enum, int-sized (32-bit) with every supported compiler; bind it as int32.
 *   - Threads: a model may be shared by any number of sessions and threads. One session is used by one call at
 *     a time; concurrent calls on the same session from different threads are serialised by a lock inside the
 *     session (they wait, they do not corrupt it). Different sessions run in parallel. What a token callback
 *     may call is listed at lm_token_cb.
 *   - Backends: the ggml compute backends (CPU, CUDA, Vulkan, ...) are plugins loaded at run time from the
 *     directory that contains the littlemore library, at the first call that needs them. A machine without a
 *     GPU (or without its driver) simply gets fewer backends.
 *   - Calling convention: every function and the callback type are LM_CALL (__cdecl on Windows, the platform
 *     default elsewhere), so bindings that must name a convention can rely on it.
 *
 * Minimal example:
 *
 *     lm_model * model = NULL;
 *     if (lm_model_load("model.gguf", NULL, &model) != LM_OK) { fprintf(stderr, "%s\n", lm_last_error()); return 1; }
 *     lm_session * s = NULL;
 *     lm_session_new(model, NULL, &s);
 *     lm_message msg = { "user", "Hello!" };
 *     lm_sampling smp; lm_sampling_default(&smp);
 *     lm_chat(s, &msg, 1, &smp, print_piece, NULL);   // print_piece: int (*)(const char *, size_t, void *)
 *     lm_session_free(s);
 *     lm_model_free(model);
 *
 * Licence: MIT.
 */
#ifndef LITTLEMORE_H
#define LITTLEMORE_H

#include <stddef.h>
#include <stdint.h>

/* --------------------------------------------------------------------------------------------------------------
 * Export and calling-convention macros. LM_BUILD is defined only while the library itself is compiled.
 * ------------------------------------------------------------------------------------------------------------ */
#if defined(_WIN32)
#    if defined(LM_BUILD)
#        define LM_API __declspec(dllexport)
#    else
#        define LM_API __declspec(dllimport)
#    endif
#    define LM_CALL __cdecl
#else
#    define LM_API __attribute__((visibility("default")))
#    define LM_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------------------------------------------
 * Versions
 * ------------------------------------------------------------------------------------------------------------ */

/** Library version (semantic versioning). The build reads these lines; keep them in this form. */
#define LM_VERSION_MAJOR 0
#define LM_VERSION_MINOR 1
#define LM_VERSION_PATCH 0

/**
 * Version of the ABI contract described in this header (also the shared library's SOVERSION). It is incremented
 * only by a breaking change (a removed or changed function, a field moved inside a struct). Adding functions,
 * status codes or appended struct fields does not change it. Compare it with lm_api_version() at run time to
 * detect a header/library mismatch.
 */
#define LM_API_VERSION 1

/** Returns the library version as "MAJOR.MINOR.PATCH" (a static string; never NULL, never freed). */
LM_API const char * LM_CALL lm_version(void);

/** Returns the LM_API_VERSION the library was built with. A caller should refuse to run if it differs from the
 *  LM_API_VERSION of the header it was compiled against. */
LM_API int32_t LM_CALL lm_api_version(void);

/* --------------------------------------------------------------------------------------------------------------
 * Status codes and errors
 * ------------------------------------------------------------------------------------------------------------ */

/**
 * Result of every fallible call. LM_OK is 0; every error is positive, so `if (status) { ... }` tests for failure.
 * The numeric values are part of the ABI and never change; new codes are only appended.
 */
typedef enum lm_status {
    LM_OK                   = 0,  /**< success */
    LM_ERR_INVALID_ARG      = 1,  /**< a NULL pointer, a bad value or a bad struct_size */
    LM_ERR_UNSUPPORTED      = 2,  /**< a request this library version cannot honour (e.g. a non-zero field it
                                       does not know, or a non-zero reserved field) */
    LM_ERR_BACKEND          = 3,  /**< no usable compute backend (e.g. no CPU plugin next to the library) or a
                                       requested device does not exist */
    LM_ERR_MODEL_LOAD       = 4,  /**< the model file is missing, unreadable or not a model the engine accepts */
    LM_ERR_OUT_OF_MEMORY    = 5,  /**< an allocation (host or device) failed */
    LM_ERR_SESSION          = 6,  /**< the engine could not create the context of a session */
    LM_ERR_TEMPLATE         = 7,  /**< the chat template could not be applied to the messages */
    LM_ERR_TOKENIZE         = 8,  /**< text could not be tokenized, or a token id is out of range */
    LM_ERR_CONTEXT_FULL     = 9,  /**< the prompt does not fit into the session's context size */
    LM_ERR_DECODE           = 10, /**< the engine failed while computing a batch */
    LM_ERR_BUFFER_TOO_SMALL = 11, /**< an output buffer is too small; the required size is reported */
    LM_ERR_INTERNAL         = 12, /**< an unexpected internal error (a bug: please report it) */
    LM_ERR_REENTRANT        = 13, /**< lm_chat / lm_generate / lm_session_free called on a session from inside
                                       that session's own token callback (see lm_token_cb) */
    LM_ERR_BUSY             = 14  /**< the object is still in use: lm_model_free while sessions of the model
                                       are alive */
} lm_status;

/**
 * Message describing the most recent failure on the calling thread (thread-local storage, so threads never see
 * each other's errors). It is non-empty after any call that returned an error, starts with the failing
 * function's name, and stays valid until the next failing littlemore call on the same thread. Successful calls
 * do not clear it. Never NULL; empty ("") if no call on this thread has failed yet.
 */
LM_API const char * LM_CALL lm_last_error(void);

/** Returns a short constant name for a status code, e.g. "LM_ERR_MODEL_LOAD" ("LM_ERR_UNKNOWN" for other values). */
LM_API const char * LM_CALL lm_status_name(lm_status status);

/* --------------------------------------------------------------------------------------------------------------
 * Backends
 * ------------------------------------------------------------------------------------------------------------ */

/**
 * Describes the loaded compute backends and their devices, one device per line:
 *
 *     CPU: <cpu description> [CPU]
 *     CUDA0: NVIDIA GeForce RTX 4050 Laptop GPU [GPU, free 5.1 GiB of 6.0 GiB]
 *
 * The first word of each line is the device name used by lm_model_options.devices. Loads the backend plugins
 * if no call has done so yet.
 *
 * Works like snprintf: writes at most cap - 1 bytes plus a terminating NUL (nothing when cap is 0) and returns
 * the full length of the text, without the NUL. Call it with (NULL, 0) to learn the size. The text contains
 * the devices' current free memory, so its length can change between two calls: compare the second call's
 * return value with your buffer size (or allocate some slack) instead of assuming it is unchanged. Returns 0
 * on failure (buf == NULL with cap > 0, or no backend could be loaded); lm_last_error() then says why.
 */
LM_API size_t LM_CALL lm_backends_info(char * buf, size_t cap);

/* --------------------------------------------------------------------------------------------------------------
 * Model
 * ------------------------------------------------------------------------------------------------------------ */

/** A loaded model (weights and vocabulary). Opaque; created by lm_model_load, destroyed by lm_model_free. */
typedef struct lm_model lm_model;

/** n_gpu_layers value that asks the library to fit as many layers as the free GPU memory allows. */
#define LM_GPU_LAYERS_AUTO (-1)

/**
 * Options of lm_model_load. Initialise with lm_model_options_default() and change what you need.
 */
typedef struct lm_model_options {
    /** sizeof(lm_model_options) as compiled by the caller (see the ABI rules at the top). */
    size_t struct_size;

    /**
     * Number of layers whose weights go to the GPU devices.
     *   LM_GPU_LAYERS_AUTO (-1, default): auto-fit. The library estimates the bytes of every layer from the model
     *       file (its weights plus its KV cache at fit_n_ctx tokens), keeps a reserve per GPU device for compute
     *       buffers (sized for the default 512-token batch, i.e. lm_session_options.n_batch = 0 or 512), and
     *       offloads as many layers as fit into the free memory of the GPU devices, summed over all of them (or
     *       into vram_budget). If not even one fits, the model gets 0 GPU layers; the GPU is then still used
     *       for large prompt batches (see "GPU use at 0 GPU layers" at `devices`; devices = "none" is the switch
     *       for no GPU at all). The chosen value is reported by lm_model_get_info().
     *   0: weights stay in host memory (same note: "GPU use at 0 GPU layers" at `devices`).
     *   N > 0: the last N layers go to the GPU, as llama.cpp's -ngl N. As in llama.cpp the output head counts as
     *       one extra layer: a model with L blocks is fully offloaded at N >= L + 1.
     * Which device a LittleBit (B1) weight finally lives on is decided by the engine per device capability: on a
     * device that can run the fused LittleBit op (CUDA) it follows its layer, otherwise (e.g. Vulkan) it stays on
     * the CPU.
     */
    int32_t n_gpu_layers;

    /** Reserved, must be 0. */
    int32_t reserved0;

    /**
     * Which devices to use, by the names lm_backends_info() prints.
     *   NULL or "" (default): every GPU device, without duplicates (when one GPU is seen by two backends, e.g.
     *       CUDA0 and Vulkan0, the first backend's device is used, as llama.cpp does).
     *   "none": the CPU-only switch. No GPU at all: weights, KV cache and every computation stay on the CPU,
     *       and no GPU memory is allocated (asserted by the tests from the engine's allocation log).
     *   "CUDA0" or "CUDA0,CUDA1": exactly these devices (GPU devices only; the CPU is always used).
     *
     * GPU use at 0 GPU layers (the one place this is described; intended, supported behaviour): with any GPU
     * device in this list, the engine still runs large prompt batches on the GPU when no layer is offloaded
     * (explicit n_gpu_layers = 0 or an auto-fit that fits nothing), copying the weights there for the duration
     * of the batch, as llama.cpp does. This costs GPU memory and makes reading a prompt much faster; generation
     * itself runs on the CPU either way. Measured on one RTX 4050 Laptop GPU (6 GiB), 0 GPU layers, GPU kept vs
     * devices = "none":
     *     GPU memory:              0.7 to 1.0 GiB          vs 0
     *     prompt, 512 tokens:      0.6B LittleBit 5936 vs 354 tokens/s; 1.7B LittleBit 2824 vs 123 tokens/s
     *     first token, 512 tokens: 0.6B LittleBit  120 vs 1449 ms;      1.7B LittleBit  202 vs 4192 ms
     *     generation:              unchanged (87.7 vs 85.7 and 34.7 vs 34.2 tokens/s)
     * Use devices = "none" when the GPU must not be touched (no GPU memory at all); keep the default to have the
     * GPU speed up prompts even when the weights do not fit on it. Full table: docs/c-api.md.
     */
    const char * devices;

    /**
     * Auto-fit only: GPU memory budget in bytes, the total over all devices used. 0 (default) means "the sum of
     * the free memory the devices report right now". A non-zero value replaces that measurement, e.g. to leave
     * room for other programs, or to test the CPU fallback with a tiny budget. (On Windows the reported free
     * memory does not include what other processes use; set a budget when other programs use the GPU.)
     */
    uint64_t vram_budget;

    /**
     * Auto-fit only: the context size (tokens) whose KV cache the fit reserves room for on the GPU. 0 (default)
     * means 4096, capped at the model's training context. Use the n_ctx you will give lm_session_new.
     */
    uint32_t fit_n_ctx;

    /** Non-zero (default 1): let the engine memory-map the file where it can (faster loads, pages shared between
     *  processes). 0: read the weights into memory allocated by the library. */
    int32_t use_mmap;
} lm_model_options;

/** Fills *opts with the defaults (and opts->struct_size). Returns LM_ERR_INVALID_ARG for NULL. */
LM_API lm_status LM_CALL lm_model_options_default(lm_model_options * opts);

/**
 * Loads a GGUF model.
 *
 * @param path  UTF-8 path of the .gguf file.
 * @param opts  options, or NULL for the defaults (auto-fit on every GPU).
 * @param out   receives the model on success; set to NULL on failure.
 * @return LM_OK; LM_ERR_INVALID_ARG (NULL path/out, bad options), LM_ERR_BACKEND (no backend, unknown device),
 *         LM_ERR_MODEL_LOAD (missing, unreadable or unsupported file), LM_ERR_UNSUPPORTED, LM_ERR_OUT_OF_MEMORY.
 */
LM_API lm_status LM_CALL lm_model_load(const char * path, const lm_model_options * opts, lm_model ** out);

/**
 * Frees a model. NULL is ignored (LM_OK). Sessions borrow the model: while any session of it is alive the model
 * is not freed and LM_ERR_BUSY is returned (free the sessions first, then call again).
 */
LM_API lm_status LM_CALL lm_model_free(lm_model * model);

/**
 * Facts about a loaded model, including what the auto-fit decided. Filled by lm_model_get_info().
 */
typedef struct lm_model_info {
    /** sizeof(lm_model_info) as compiled by the caller; the library writes only the fields that fit in it. */
    size_t struct_size;

    int32_t  n_layer;        /**< number of repeating blocks (layers) of the model */
    int32_t  n_gpu_layers;   /**< layers actually requested from the engine after auto-fit (0 = none;
                                  n_layer + 1 = everything including the output head) */
    int32_t  n_ctx_train;    /**< context size the model was trained with */
    int32_t  n_vocab;        /**< vocabulary size */
    int32_t  n_embd;         /**< embedding (hidden) size */
    int32_t  reserved0;      /**< reserved, written as 0 */
    uint64_t size_bytes;     /**< total bytes of all weights */
    uint64_t n_params;       /**< number of parameters */

    /** Auto-fit details (all 0 when n_gpu_layers was not LM_GPU_LAYERS_AUTO or no GPU device was used). */
    uint64_t fit_budget_bytes;   /**< GPU memory the fit worked with, summed over the devices (measured free
                                      memory or vram_budget) */
    uint64_t fit_reserve_bytes;  /**< part of the budget kept back for compute buffers and runtime overhead
                                      (per-device reserve times the number of devices) */
    uint64_t fit_layer_bytes;    /**< estimated bytes of the largest layer (weights + KV cache at fit_n_ctx) */
    uint64_t fit_head_bytes;     /**< estimated bytes of the output head (the "extra" layer) */

    /** Comma-separated names of the GPU devices given to the engine ("" = none). NUL-terminated, truncated. */
    char gpu_devices[128];
    /** Short engine description of the model, e.g. "qwen3 0.6B F16". NUL-terminated, truncated. */
    char description[128];
} lm_model_info;

/**
 * Reports facts about a model. Set info->struct_size = sizeof(lm_model_info) first; an older, smaller struct
 * (ending on one of its field boundaries) is filled up to its last field and nothing after it is written.
 * @return LM_OK, or LM_ERR_INVALID_ARG for NULL arguments, or a struct_size that does not end on a field boundary
 *         or ends before the field n_gpu_layers (the oldest layout accepted covers every field before it).
 */
LM_API lm_status LM_CALL lm_model_get_info(const lm_model * model, lm_model_info * info);

/* --------------------------------------------------------------------------------------------------------------
 * Session
 * ------------------------------------------------------------------------------------------------------------ */

/** A conversation state (the engine context and its KV cache) on one model. Opaque. */
typedef struct lm_session lm_session;

/** Options of lm_session_new. Initialise with lm_session_options_default(). */
typedef struct lm_session_options {
    /** sizeof(lm_session_options) as compiled by the caller. */
    size_t struct_size;
    /** Context size in tokens: prompt plus generated text must fit. 0 (default) = 4096, capped at the model's
     *  training context. The engine may round its allocation up (to a multiple of 256 tokens);
     *  the limit stays the size asked for. */
    uint32_t n_ctx;
    /** CPU threads, for both prompt reading and generation. 0 (default) = llama.cpp's default (4 threads). */
    int32_t n_threads;
    /** Most tokens processed in one engine call while reading a prompt (also the engine's physical batch,
     *  n_ubatch). 0 (default) = 512, the size the auto-fit reserve is computed for. */
    uint32_t n_batch;
    /** Reserved, must be 0. */
    uint32_t reserved0;
} lm_session_options;

/** Fills *opts with the defaults (and opts->struct_size). Returns LM_ERR_INVALID_ARG for NULL. */
LM_API lm_status LM_CALL lm_session_options_default(lm_session_options * opts);

/**
 * Creates a session on a model. The model must outlive the session (lm_model_free refuses while it is alive).
 * @param opts  options, or NULL for the defaults.
 * @param out   receives the session on success; set to NULL on failure.
 * @return LM_OK; LM_ERR_INVALID_ARG; LM_ERR_UNSUPPORTED; LM_ERR_SESSION or LM_ERR_OUT_OF_MEMORY when the engine
 *         context (KV cache, compute buffers) cannot be created. The engine reports no error code here, so the
 *         choice between these two is a guess from its log text (an "alloc"/"memory" message means
 *         LM_ERR_OUT_OF_MEMORY); the message in lm_last_error() carries the engine's own words.
 */
LM_API lm_status LM_CALL lm_session_new(lm_model * model, const lm_session_options * opts, lm_session ** out);

/**
 * Frees a session. NULL is ignored (LM_OK). Called from inside the session's own token callback it does
 * nothing and returns LM_ERR_REENTRANT. It must not run while another thread is inside a call on the session
 * (it waits for a running call to finish, but calls that start afterwards would use freed memory).
 */
LM_API lm_status LM_CALL lm_session_free(lm_session * session);

/* --------------------------------------------------------------------------------------------------------------
 * Generation
 * ------------------------------------------------------------------------------------------------------------ */

/**
 * Called once per piece of generated text.
 * @param piece  generated bytes, NUL-terminated, valid only during the call. Bytes are passed on unchanged;
 *               the only thing the library does is hold back an incomplete UTF-8 sequence at the end of a piece
 *               until the next token completes it, so whenever the model produces valid UTF-8 every piece is
 *               valid UTF-8 on its own. Invalid byte sequences the model produces are passed through as they
 *               are, and at the end of the generation whatever is still held back is delivered.
 * @param len    length of piece in bytes (without the NUL).
 * @param user   the pointer given to lm_chat / lm_generate.
 * @return 0 to continue, any other value to stop generating (the call then returns LM_OK).
 *
 * What a callback may call (it runs on the thread that called lm_chat / lm_generate, inside that call):
 *   - allowed: lm_tokenize, lm_detokenize, lm_model_get_info, lm_backends_info, lm_last_error, lm_status_name,
 *     lm_version, and calls on OTHER sessions;
 *   - refused with LM_ERR_REENTRANT (nothing happens, the running generation continues): lm_chat, lm_generate
 *     and lm_session_free on the session that is generating;
 *   - refused with LM_ERR_BUSY: lm_model_free of the session's model.
 */
typedef int (LM_CALL * lm_token_cb)(const char * piece, size_t len, void * user);

/** lm_sampling.seed value that asks for a different random seed on every call. */
#define LM_SEED_RANDOM 0xFFFFFFFFu

/** Sampling settings. Initialise with lm_sampling_default(). */
typedef struct lm_sampling {
    /** sizeof(lm_sampling) as compiled by the caller. */
    size_t struct_size;
    /** Softmax temperature. <= 0 means greedy decoding (always the most likely token): fully deterministic.
     *  Default 0.8. */
    float temperature;
    /** Keep only the k most likely tokens before sampling; 0 = off. Default 40. */
    int32_t top_k;
    /** Keep the smallest set of tokens whose probability sums to at least top_p; 1.0 = off. Default 0.95. */
    float top_p;
    /** Seed of the random sampler; the same seed, prompt and settings give the same text. Default 0, so with
     *  the defaults every call gives the SAME text even at temperature > 0 (reproducible on purpose); use
     *  LM_SEED_RANDOM, or a seed of your own per call, for varied answers. */
    uint32_t seed;
    /** Most tokens to generate; generation also ends at an end-of-generation token or when the context is
     *  full. Must be > 0. Default 256. */
    int32_t max_tokens;
    /** Reserved, must be 0. */
    int32_t reserved0;
} lm_sampling;

/** Fills *s with the defaults (and s->struct_size). Returns LM_ERR_INVALID_ARG for NULL. */
LM_API lm_status LM_CALL lm_sampling_default(lm_sampling * s);

/** One chat message: a role ("system", "user", "assistant") and its UTF-8 text. Fixed layout (used in arrays). */
typedef struct lm_message {
    const char * role;
    const char * content;
} lm_message;

/**
 * Formats a conversation with the model's chat template (stored in the GGUF; llama.cpp's "chatml" when the
 * model has none), ends it with the opening of an assistant turn, and generates the assistant's answer,
 * streaming it to cb.
 *
 * Every call is complete in itself: the session's previous state is discarded and the whole conversation in
 * msgs is read again. For a multi-turn chat pass the full history, including earlier assistant answers.
 *
 * Special tokens: the formatted conversation is tokenized as a whole with special-token parsing on (as
 * llama.cpp's own chat programs do), so special-token text inside a message, e.g. "<|im_end|>" in user input,
 * becomes the real control token and can end the turn early or start a forged one. Remove or escape such text
 * from untrusted content before passing it in.
 *
 * @param s     sampling settings, or NULL for the defaults.
 * @param cb    piece callback, or NULL to generate without streaming (useful only for timing).
 * @return LM_OK (also when cb asked to stop); LM_ERR_INVALID_ARG (NULL session/msgs, n == 0, a NULL role or
 *         content, bad sampling, text too long); LM_ERR_TEMPLATE; LM_ERR_TOKENIZE; LM_ERR_CONTEXT_FULL (the
 *         formatted prompt leaves no room in the context); LM_ERR_DECODE; LM_ERR_REENTRANT.
 */
LM_API lm_status LM_CALL lm_chat(lm_session * session, const lm_message * msgs, size_t n, const lm_sampling * s,
                                 lm_token_cb cb, void * user);

/**
 * Generates a continuation of a raw prompt (no chat template; the model's beginning-of-text token is added if
 * the model asks for one, and special-token text such as "<|im_start|>" in the prompt is parsed as tokens).
 * Same streaming, stopping and state rules as lm_chat.
 */
LM_API lm_status LM_CALL lm_generate(lm_session * session, const char * prompt, const lm_sampling * s,
                                     lm_token_cb cb, void * user);

/* --------------------------------------------------------------------------------------------------------------
 * Tokens
 * ------------------------------------------------------------------------------------------------------------ */

/**
 * Converts UTF-8 text to token ids.
 * @param text         the text; need not be NUL-terminated.
 * @param len          its length in bytes.
 * @param tokens       output array (may be NULL when cap is 0).
 * @param cap          capacity of tokens, in elements.
 * @param add_special  non-zero: add the model's beginning-of-text token (if it uses one) and parse special-token
 *                     text as tokens; 0: plain text only.
 * @param n_out        receives the number of tokens; also on LM_ERR_BUFFER_TOO_SMALL, where it is the
 *                     capacity needed.
 * @return LM_OK; LM_ERR_INVALID_ARG; LM_ERR_BUFFER_TOO_SMALL; LM_ERR_TOKENIZE.
 */
LM_API lm_status LM_CALL lm_tokenize(const lm_model * model, const char * text, size_t len, int32_t * tokens,
                                     size_t cap, int add_special, size_t * n_out);

/**
 * Converts token ids back to UTF-8 text (the exact inverse of lm_tokenize with add_special = 0 for text the
 * vocabulary can encode; with a beginning-of-text token in front, SentencePiece vocabularies keep the space
 * they put before the first word, as llama.cpp does).
 * @param buf      output buffer (may be NULL when cap is 0); NUL-terminated when the text fits.
 * @param cap      its size in bytes, including room for the NUL.
 * @param special  non-zero: render special tokens (e.g. "<|im_end|>") as text and keep the beginning-of-text
 *                 token's text; 0: leave them out.
 * @param len_out  receives the text length in bytes without the NUL; also on LM_ERR_BUFFER_TOO_SMALL, where
 *                 the needed buffer size is *len_out + 1.
 * @return LM_OK; LM_ERR_INVALID_ARG; LM_ERR_BUFFER_TOO_SMALL; LM_ERR_TOKENIZE (a token id out of range).
 */
LM_API lm_status LM_CALL lm_detokenize(const lm_model * model, const int32_t * tokens, size_t n, char * buf,
                                       size_t cap, int special, size_t * len_out);

#ifdef __cplusplus
}
#endif

#endif /* LITTLEMORE_H */
