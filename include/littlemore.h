/*
 * littlemore.h - the LittleMore C API.
 *
 * LittleMore runs LittleBit (sub-1-bit) language models, and ordinary GGUF models, through a patched llama.cpp.
 * This header is the whole public interface of the `littlemore` shared library: load a model, open a session,
 * chat or generate text with streaming, tokenize, and ask which compute backends are available.
 *
 * Design rules (they make the library usable from C, Python ctypes, C#, Rust, ... and safe to upgrade):
 *   - Plain C only: no C++ types, no engine (llama.h / ggml.h) types. Objects are opaque pointers.
 *   - ABI stability: every options/info struct starts with a `struct_size` field. Fill it with sizeof() of the
 *     struct as you compiled it (the lm_*_default() initialisers do this). A newer library reads only the
 *     fields an older caller knew about and uses defaults for the rest; an older library accepts a larger
 *     struct from a newer caller only when the fields it does not know are all zero (so a request it cannot
 *     honour is refused with LM_ERR_UNSUPPORTED instead of being ignored silently). New fields are only ever
 *     appended. LM_API_VERSION changes only when this contract itself is broken.
 *   - Errors: every fallible function returns an lm_status. After any failure lm_last_error() returns a
 *     non-empty, human-readable message for the calling thread.
 *   - Threads: a model may be shared by any number of sessions and threads. One session is used by one call at
 *     a time; concurrent calls on the same session are serialised by a lock inside the session (they do not
 *     corrupt it, they wait). Different sessions run in parallel.
 *   - Backends: the ggml compute backends (CPU, CUDA, Vulkan, ...) are plugins loaded at run time from the
 *     directory that contains the littlemore library, at the first call that needs them. A machine without a
 *     GPU (or without its driver) simply gets fewer backends.
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
 * Export macro. LM_BUILD is defined only while the library itself is compiled.
 * ------------------------------------------------------------------------------------------------------------ */
#if defined(_WIN32)
#    if defined(LM_BUILD)
#        define LM_API __declspec(dllexport)
#    else
#        define LM_API __declspec(dllimport)
#    endif
#else
#    define LM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------------------------------------------
 * Versions
 * ------------------------------------------------------------------------------------------------------------ */

/** Library version (semantic versioning). The build reads these three lines; keep them in this form. */
#define LM_VERSION_MAJOR 0
#define LM_VERSION_MINOR 1
#define LM_VERSION_PATCH 0

/**
 * Version of the ABI contract described in this header. It is incremented only by a breaking change (a removed
 * or changed function, a field moved inside a struct). Adding functions or appending struct fields does not
 * change it. Compare it with lm_api_version() at run time to detect a header/library mismatch.
 */
#define LM_API_VERSION 1

/** Returns the library version as "MAJOR.MINOR.PATCH" (a static string; never NULL, never freed). */
LM_API const char * lm_version(void);

/** Returns the LM_API_VERSION the library was built with. A caller should refuse to run if it differs from the
 *  LM_API_VERSION of the header it was compiled against. */
LM_API int32_t lm_api_version(void);

/* --------------------------------------------------------------------------------------------------------------
 * Status codes and errors
 * ------------------------------------------------------------------------------------------------------------ */

/**
 * Result of every fallible call. LM_OK is 0; every error is positive, so `if (status) { ... }` tests for failure.
 * The numeric values are part of the ABI and never change.
 */
typedef enum lm_status {
    LM_OK                   = 0,  /**< success */
    LM_ERR_INVALID_ARG      = 1,  /**< a NULL pointer, a bad value or a struct_size that is too small */
    LM_ERR_UNSUPPORTED      = 2,  /**< a request this library version cannot honour (e.g. a non-zero field it
                                       does not know in a struct from a newer header) */
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
    LM_ERR_INTERNAL         = 12  /**< an unexpected internal error (a bug: please report it) */
} lm_status;

/**
 * Message describing the most recent failure on the calling thread (thread-local storage, so threads never see
 * each other's errors). It is non-empty after any call that returned an error and stays valid until the next
 * failing littlemore call on the same thread. Successful calls do not clear it. Never NULL; empty ("") if no
 * call on this thread has failed yet.
 */
LM_API const char * lm_last_error(void);

/** Returns a short constant name for a status code, e.g. "LM_ERR_MODEL_LOAD" ("LM_ERR_UNKNOWN" for other values). */
LM_API const char * lm_status_name(lm_status status);

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
 * the full length of the text, without the NUL. Call it with (NULL, 0) to learn the size. Returns 0 on failure
 * (buf == NULL with cap > 0, or no backend could be loaded); lm_last_error() then says why.
 */
LM_API size_t lm_backends_info(char * buf, size_t cap);

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
     *       file (its weights plus its KV cache at fit_n_ctx tokens), reserves room for the compute buffers, and
     *       offloads as many layers as fit into the free memory of the GPU devices (or into vram_budget). If not
     *       even one fits, the model runs on the CPU (0 layers). The chosen value is reported by
     *       lm_model_get_info().
     *   0: weights stay in host memory.
     *   N > 0: the last N layers go to the GPU, as llama.cpp's -ngl N. As in llama.cpp the output head counts as
     *       one extra layer: a model with L blocks is fully offloaded at N >= L + 1.
     * Which device a LittleBit (B1) weight finally lives on is decided by the engine per device capability: on a
     * device that can run the fused LittleBit op (CUDA) it follows its layer, otherwise (e.g. Vulkan) it stays on
     * the CPU.
     */
    int32_t n_gpu_layers;

    /**
     * Which devices to use, by the names lm_backends_info() prints.
     *   NULL or "" (default): every GPU device, without duplicates (when one GPU is seen by two backends, e.g.
     *       CUDA0 and Vulkan0, the first backend's device is used, as llama.cpp does).
     *   "none": no GPU at all. Weights, KV cache and every computation stay on the CPU. (Unlike n_gpu_layers = 0,
     *       which keeps the weights on the CPU but still lets the engine send large prompt batches to a GPU.)
     *   "CUDA0" or "CUDA0,CUDA1": exactly these devices.
     */
    const char * devices;

    /**
     * Auto-fit only: GPU memory budget in bytes, summed over the devices used. 0 (default) means "the free memory
     * the devices report right now". A non-zero value replaces that measurement, e.g. to leave room for other
     * programs, or to test the CPU fallback with a tiny budget.
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
LM_API lm_status lm_model_options_default(lm_model_options * opts);

/**
 * Loads a GGUF model.
 *
 * @param path  UTF-8 path of the .gguf file.
 * @param opts  options, or NULL for the defaults (auto-fit on every GPU).
 * @param out   receives the model on success; set to NULL on failure.
 * @return LM_OK; LM_ERR_INVALID_ARG (NULL path/out, bad options), LM_ERR_BACKEND (no backend, unknown device),
 *         LM_ERR_MODEL_LOAD (missing, unreadable or unsupported file), LM_ERR_UNSUPPORTED.
 */
LM_API lm_status lm_model_load(const char * path, const lm_model_options * opts, lm_model ** out);

/** Frees a model. NULL is ignored. Every session of the model must be freed first (sessions borrow it). */
LM_API void lm_model_free(lm_model * model);

/**
 * Facts about a loaded model, including what the auto-fit decided. Filled by lm_model_get_info().
 */
typedef struct lm_model_info {
    /** sizeof(lm_model_info) as compiled by the caller; the library writes only the fields that fit in it. */
    size_t struct_size;

    int32_t  n_layer;        /**< number of repeating blocks (layers) of the model */
    int32_t  n_gpu_layers;   /**< layers actually requested from the engine after auto-fit (0 = CPU only;
                                  n_layer + 1 = everything including the output head) */
    int32_t  n_ctx_train;    /**< context size the model was trained with */
    int32_t  n_vocab;        /**< vocabulary size */
    int32_t  n_embd;         /**< embedding (hidden) size */
    uint64_t size_bytes;     /**< total bytes of all weights */
    uint64_t n_params;       /**< number of parameters */

    /** Auto-fit details (all 0 when n_gpu_layers was not LM_GPU_LAYERS_AUTO or no GPU device was used). */
    uint64_t fit_budget_bytes;   /**< GPU memory the fit worked with (measured free memory or vram_budget) */
    uint64_t fit_reserve_bytes;  /**< part of the budget kept back for compute buffers and runtime overhead */
    uint64_t fit_layer_bytes;    /**< estimated bytes of the largest layer (weights + KV cache at fit_n_ctx) */
    uint64_t fit_head_bytes;     /**< estimated bytes of the output head (the "extra" layer) */

    /** Comma-separated names of the GPU devices given to the engine ("" = none). NUL-terminated, truncated. */
    char gpu_devices[128];
    /** Short engine description of the model, e.g. "qwen3 0.6B F16". NUL-terminated, truncated. */
    char description[128];
} lm_model_info;

/**
 * Reports facts about a model. Set info->struct_size = sizeof(lm_model_info) first.
 * @return LM_OK, or LM_ERR_INVALID_ARG for NULL arguments or a struct_size smaller than the first two fields.
 */
LM_API lm_status lm_model_get_info(const lm_model * model, lm_model_info * info);

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
    /** Most tokens processed in one engine call while reading a prompt. 0 (default) = 512. */
    uint32_t n_batch;
} lm_session_options;

/** Fills *opts with the defaults (and opts->struct_size). Returns LM_ERR_INVALID_ARG for NULL. */
LM_API lm_status lm_session_options_default(lm_session_options * opts);

/**
 * Creates a session on a model. The model must outlive the session.
 * @param opts  options, or NULL for the defaults.
 * @param out   receives the session on success; set to NULL on failure.
 * @return LM_OK; LM_ERR_INVALID_ARG; LM_ERR_SESSION or LM_ERR_OUT_OF_MEMORY when the engine context (KV cache,
 *         compute buffers) cannot be created.
 */
LM_API lm_status lm_session_new(lm_model * model, const lm_session_options * opts, lm_session ** out);

/** Frees a session. NULL is ignored. Must not be called while another thread uses the session. */
LM_API void lm_session_free(lm_session * session);

/* --------------------------------------------------------------------------------------------------------------
 * Generation
 * ------------------------------------------------------------------------------------------------------------ */

/**
 * Called once per piece of generated text.
 * @param piece  UTF-8 text, NUL-terminated, valid only during the call. Pieces always end on a whole UTF-8
 *               character (bytes of a character split over several tokens are held back until it is complete).
 * @param len    length of piece in bytes (without the NUL).
 * @param user   the pointer given to lm_chat / lm_generate.
 * @return 0 to continue, any other value to stop generating (the call then returns LM_OK).
 */
typedef int (*lm_token_cb)(const char * piece, size_t len, void * user);

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
    /** Seed of the random sampler; the same seed, prompt and settings give the same text. Default 0. */
    uint32_t seed;
    /** Most tokens to generate; generation also ends at an end-of-generation token or when the context is
     *  full. Must be > 0. Default 256. */
    int32_t max_tokens;
} lm_sampling;

/** Fills *s with the defaults (and s->struct_size). Returns LM_ERR_INVALID_ARG for NULL. */
LM_API lm_status lm_sampling_default(lm_sampling * s);

/** One chat message: a role ("system", "user", "assistant") and its UTF-8 text. */
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
 * @param s     sampling settings, or NULL for the defaults.
 * @param cb    piece callback, or NULL to generate without streaming (useful only for timing).
 * @return LM_OK (also when cb asked to stop); LM_ERR_INVALID_ARG (NULL session/msgs, n == 0, a NULL role or
 *         content, bad sampling); LM_ERR_TEMPLATE; LM_ERR_TOKENIZE; LM_ERR_CONTEXT_FULL (the formatted prompt
 *         leaves no room in the context); LM_ERR_DECODE.
 */
LM_API lm_status lm_chat(lm_session * session, const lm_message * msgs, size_t n, const lm_sampling * s,
                         lm_token_cb cb, void * user);

/**
 * Generates a continuation of a raw prompt (no chat template; the model's beginning-of-text token is added if
 * the model asks for one, and special-token text such as "<|im_start|>" in the prompt is parsed as tokens).
 * Same streaming, stopping and state rules as lm_chat.
 */
LM_API lm_status lm_generate(lm_session * session, const char * prompt, const lm_sampling * s,
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
LM_API lm_status lm_tokenize(const lm_model * model, const char * text, size_t len, int32_t * tokens, size_t cap,
                             int add_special, size_t * n_out);

/**
 * Converts token ids back to UTF-8 text (the exact inverse of lm_tokenize for text the vocabulary can encode).
 * @param buf      output buffer (may be NULL when cap is 0); NUL-terminated when the text fits.
 * @param cap      its size in bytes, including room for the NUL.
 * @param special  non-zero: render special tokens (e.g. "<|im_end|>") as text and keep the beginning-of-text
 *                 token's text; 0: leave them out.
 * @param len_out  receives the text length in bytes without the NUL; also on LM_ERR_BUFFER_TOO_SMALL, where
 *                 the needed buffer size is *len_out + 1.
 * @return LM_OK; LM_ERR_INVALID_ARG; LM_ERR_BUFFER_TOO_SMALL; LM_ERR_TOKENIZE (a token id out of range).
 */
LM_API lm_status lm_detokenize(const lm_model * model, const int32_t * tokens, size_t n, char * buf, size_t cap,
                               int special, size_t * len_out);

#ifdef __cplusplus
}
#endif

#endif /* LITTLEMORE_H */
