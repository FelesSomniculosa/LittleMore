/*
 * example_chat.c - the minimal C API example of include/littlemore.h, as a program: load a model (auto-fit), open a
 * session, answer one chat message with streaming, free everything. Used by tests/robustness/test_h_8b_end_to_end.py
 * to run the C API end to end on a real model.
 *
 * Usage: example_chat <model.gguf> [message] [max_tokens]
 * Exit code: 0 success (the answer is printed to stdout), 1 failure (the error is printed to stderr), 2 usage.
 */
#include "littlemore.h"

#include <stdio.h>
#include <stdlib.h>

/* Token callback: print each piece as it arrives (pieces are raw UTF-8 bytes, valid on their own whenever the model
 * produced valid UTF-8) and count them. Returns 0 to continue. */
static int LM_CALL print_piece(const char * piece, size_t len, void * user) {
    fwrite(piece, 1, len, stdout);
    fflush(stdout);
    ++*(int *) user;
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: example_chat <model.gguf> [message] [max_tokens]\n");
        return 2;
    }
    if (lm_api_version() != LM_API_VERSION) {
        fprintf(stderr, "library API version %d, header %d\n", (int) lm_api_version(), LM_API_VERSION);
        return 1;
    }
    lm_model * model = NULL;
    if (lm_model_load(argv[1], NULL, &model) != LM_OK) {   /* NULL options: auto-fit on every GPU */
        fprintf(stderr, "%s\n", lm_last_error());
        return 1;
    }
    lm_model_info info;
    info.struct_size = sizeof(info);
    lm_model_get_info(model, &info);
    fprintf(stderr, "model: %s, %d of %d layers on the GPU (%s)\n", info.description, (int) info.n_gpu_layers,
            (int) info.n_layer + 1, info.gpu_devices[0] ? info.gpu_devices : "none");

    lm_session * session = NULL;
    if (lm_session_new(model, NULL, &session) != LM_OK) {
        fprintf(stderr, "%s\n", lm_last_error());
        lm_model_free(model);
        return 1;
    }
    lm_message msg = { "user", argc > 2 ? argv[2] : "Hello!" };
    lm_sampling smp;
    lm_sampling_default(&smp);
    smp.temperature = 0.0f;                                  /* greedy: the same answer every time */
    smp.max_tokens  = argc > 3 ? atoi(argv[3]) : 32;
    int pieces = 0;
    const lm_status st = lm_chat(session, &msg, 1, &smp, print_piece, &pieces);
    printf("\n");
    if (st != LM_OK) {
        fprintf(stderr, "%s\n", lm_last_error());
    }
    fprintf(stderr, "%d pieces, status %s\n", pieces, lm_status_name(st));
    lm_session_free(session);
    lm_model_free(model);
    return st == LM_OK && pieces > 0 ? 0 : 1;
}
