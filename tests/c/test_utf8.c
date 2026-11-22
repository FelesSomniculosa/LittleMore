/*
 * test_utf8.c - unit test of the UTF-8 boundary rule used when streaming generated text (src/utf8_boundary.h).
 *
 * The rule: hold back only an unfinished multi-byte sequence at the end; never change, drop or hold back
 * anything else (invalid bytes pass through). Each case gives the bytes and the expected length of the part
 * that may be delivered now. Plain C, no model or library needed.
 */
#include "../../src/utf8_boundary.h"

#include <stdio.h>
#include <string.h>

typedef struct utf8_case {
    const char * name;
    const char * bytes;
    size_t       len;
    size_t       expect;
} utf8_case;

/* Checks the UTF-8 boundary rule of the streaming code against a table of complete and split sequences. */
int main(void) {
    static const utf8_case cases[] = {
        { "empty", "", 0, 0 },
        { "ascii", "abc", 3, 3 },
        { "2-byte complete (e acute)", "a\xC3\xA9", 3, 3 },
        { "2-byte split after lead", "a\xC3", 2, 1 },
        { "3-byte complete (euro)", "\xE2\x82\xAC", 3, 3 },
        { "3-byte split after lead", "x\xE2", 2, 1 },
        { "3-byte split after 2 bytes", "x\xE2\x82", 3, 1 },
        { "4-byte complete (emoji)", "\xF0\x9F\x98\x80", 4, 4 },
        { "4-byte split after 1", "\xF0", 1, 0 },
        { "4-byte split after 2", "\xF0\x9F", 2, 0 },
        { "4-byte split after 3", "\xF0\x9F\x98", 3, 0 },
        { "complete then split", "\xE2\x82\xAC\xF0\x9F", 5, 3 },
        { "lone continuation passes", "a\x80", 2, 2 },
        { "three continuations pass", "\x80\x80\x80", 3, 3 },
        { "0xF8 passes", "a\xF8", 2, 2 },
        { "0xFF passes", "\xFF", 1, 1 },
        { "lead then ASCII passes", "\xC3" "A", 2, 2 },
        { "lead then lead: only the last is held", "\xE2\xC3", 2, 1 },
    };
    int failures = 0;
    size_t i;
    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const size_t got = lm_utf8_complete_prefix(cases[i].bytes, cases[i].len);
        if (got != cases[i].expect) {
            fprintf(stderr, "FAIL %s: got %u, expected %u\n", cases[i].name, (unsigned) got, (unsigned) cases[i].expect);
            ++failures;
        }
    }
    /* Streaming simulation: a 4-byte character delivered one byte per token must come out whole, exactly once. */
    {
        const char * text = "ok \xF0\x9F\x98\x80!";
        char pending[16], out[32];
        size_t np = 0, no = 0, k;
        for (k = 0; k < strlen(text); ++k) {
            size_t ready;
            pending[np++] = text[k];
            ready = lm_utf8_complete_prefix(pending, np);
            memcpy(out + no, pending, ready);
            no += ready;
            memmove(pending, pending + ready, np - ready);
            np -= ready;
            if (ready > 0 && (unsigned char) out[no - 1] >= 0xC0 && (unsigned char) out[no - 1] <= 0xF7) {
                fprintf(stderr, "FAIL streaming: a piece ended on a lead byte\n");
                ++failures;
            }
        }
        memcpy(out + no, pending, np); /* end of generation: flush what is held back */
        no += np;
        if (no != strlen(text) || memcmp(out, text, no) != 0) {
            fprintf(stderr, "FAIL streaming: bytes changed\n");
            ++failures;
        }
    }
    printf("%u cases, %d failures\n", (unsigned) (sizeof(cases) / sizeof(cases[0])), failures);
    return failures == 0 ? 0 : 1;
}
