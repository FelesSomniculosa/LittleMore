"""Generate tests/robustness/c/null_harness.c: NULL passed to every pointer parameter of every exported function.

The C API promises that bad arguments give an error status, never a crash. This script reads every `LM_API`
declaration of include/littlemore.h and writes a plain-C program that calls each function once per pointer parameter
with that parameter NULL (all others valid), and once with every pointer NULL. For each call the program checks:

  * the process does not crash (it reaches its summary line; CTest and the pytest wrapper check the exit code);
  * a NULL that the header does not allow gives LM_ERR_INVALID_ARG, and lm_last_error() then starts with the name of
    the function (the header's error rule);
  * a NULL that the header allows (OPTIONAL below, e.g. opts = NULL for the defaults, a NULL callback, buf = NULL with
    cap = 0) gives LM_OK.

The generated file is committed; tests/robustness/test_d_misuse.py regenerates it in memory and fails when the committed
copy is stale (a new function in the header must get its line here: an unknown parameter type is a generator error).

Usage:  python tests/robustness/gen_null_harness.py   (rewrites tests/robustness/c/null_harness.c)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "include" / "littlemore.h"
OUT = Path(__file__).resolve().parent / "c" / "null_harness.c"

# A valid value (C expression) for each parameter type, in the context the harness sets up (see PRELUDE).
VALID = {
    "const char *": None,               # depends on the parameter name, see VALID_BY_NAME
    "const lm_model_options *": "&mopts",
    "lm_model_options *": "&mopts_out",
    "lm_model **": "&model_out",
    "lm_model *": "model",
    "const lm_model *": "model",
    "lm_model_info *": "&info",
    "const lm_session_options *": "&sopts",
    "lm_session_options *": "&sopts_out",
    "lm_session **": "&session_out",
    "lm_session *": "session",
    "lm_sampling *": "&smp_out",
    "const lm_sampling *": "&smp",
    "const lm_message *": "msgs",
    "lm_token_cb": "count_cb",
    "void *": "NULL",
    "char *": "buf",
    "int32_t *": "ids",
    "const int32_t *": "ids",
    "size_t *": "&n_out",
    "size_t": None,                     # by name
    "int": "0",
    "lm_status": "LM_ERR_MODEL_LOAD",
}
VALID_BY_NAME = {"path": "model_path", "text": "\"Hello\"", "prompt": "\"Hello\"", "cap": "sizeof(buf) / sizeof(buf[0])",
                 "len": "5", "n": "1"}
# Size arguments that go with a buffer: `cap` of tokenize is in elements (ids), of detokenize / backends_info in bytes.
CAP_BY_FUNC = {"lm_tokenize": "64", "lm_detokenize": "sizeof(buf)", "lm_backends_info": "sizeof(buf)"}

# Pointer parameters the header allows to be NULL (with the other arguments valid), per function.
OPTIONAL = {
    "lm_model_load": {"opts"},
    "lm_model_free": {"model"},
    "lm_session_new": {"opts"},
    "lm_session_free": {"session"},
    "lm_chat": {"s", "cb", "user"},
    "lm_generate": {"s", "cb", "user"},
}
# NULL that is allowed only together with a zero size (the size is then set to 0 for that call).
NULL_WITH_ZERO = {("lm_tokenize", "tokens"): "cap", ("lm_detokenize", "buf"): "cap", ("lm_detokenize", "tokens"): "n",
                  ("lm_backends_info", "buf"): "cap"}
# Functions that free their argument: the harness calls them last, on objects of their own.
FREEING = {"lm_model_free", "lm_session_free"}

DECL = re.compile(r"LM_API\s+(?P<ret>[\w\s\*]+?)\s*LM_CALL\s+(?P<name>lm_\w+)\s*\((?P<args>[^;]*?)\)\s*;", re.S)


def parse_header(text: str) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """[(return type, function name, [(param type, param name), ...]), ...] of every LM_API function."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    out = []
    for m in DECL.finditer(text):
        args = " ".join(m["args"].split())
        params = []
        if args != "void":
            for a in args.split(","):
                a = a.strip()
                pm = re.match(r"(?P<type>.*?[\s\*])(?P<pname>\w+)$", a)
                if not pm:
                    raise SystemExit(f"cannot parse parameter {a!r} of {m['name']}")
                ptype = " ".join(pm["type"].replace("*", " * ").split()).replace("* *", "**")
                params.append((ptype, pm["pname"]))
        out.append((" ".join(m["ret"].split()), m["name"], params))
    return out


def valid(func: str, ptype: str, pname: str) -> str:
    if ptype == "const char *" or ptype == "size_t":
        if pname == "cap" and func in CAP_BY_FUNC:
            return CAP_BY_FUNC[func]
        if pname not in VALID_BY_NAME:
            raise SystemExit(f"no valid value for {ptype} {pname} of {func}: extend gen_null_harness.py")
        return VALID_BY_NAME[pname]
    if ptype not in VALID:
        raise SystemExit(f"unknown parameter type {ptype!r} ({pname}) of {func}: extend gen_null_harness.py")
    return VALID[ptype]


def is_pointer(ptype: str) -> bool:
    return "*" in ptype or ptype == "lm_token_cb"


PRELUDE = r'''/*
 * null_harness.c - GENERATED by tests/robustness/gen_null_harness.py from include/littlemore.h. Do not edit by hand:
 * change the generator and run it again (tests/robustness/test_d_misuse.py fails when this file is stale).
 *
 * Calls every exported function with NULL in each pointer parameter (others valid) and with every pointer NULL.
 * A NULL the header forbids must give LM_ERR_INVALID_ARG with an lm_last_error() that names the function; a NULL it
 * allows must give LM_OK. Nothing may crash. Usage: null_harness <tiny model .gguf>. Exit code: 0 = all checks
 * passed, 1 = a check failed, 77 = no model given (skip).
 */
#include "littlemore.h"

#include <stdio.h>
#include <string.h>

static int g_fail = 0, g_checks = 0;

/* One check: `ok` must hold; prints the case on failure. */
static void check(int ok, const char * what, const char * detail) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        printf("FAIL %s: %s (last error: \"%s\")\n", what, detail, lm_last_error());
    }
}

/* A NULL the header forbids: the call must fail with LM_ERR_INVALID_ARG and an error message naming the function. */
static void expect_invalid(lm_status st, const char * func, const char * what) {
    check(st == LM_ERR_INVALID_ARG, what, lm_status_name(st));
    check(strncmp(lm_last_error(), func, strlen(func)) == 0, what, "lm_last_error() does not start with the function name");
}

/* A NULL the header allows: the call must succeed. */
static void expect_ok(lm_status st, const char * what) {
    check(st == LM_OK, what, lm_status_name(st));
}

/* Token callback of the valid calls: counts pieces, stops after 2. */
static int LM_CALL count_cb(const char * piece, size_t len, void * user) {
    (void) piece; (void) len; (void) user;
    static int n = 0;
    return ++n % 2 == 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: null_harness <model.gguf> (skipped)\n");
        return 77;
    }
    const char * model_path = argv[1];
    lm_model_options mopts, mopts_out;
    lm_model_options_default(&mopts);
    mopts.devices = "none";
    lm_session_options sopts, sopts_out;
    lm_session_options_default(&sopts);
    sopts.n_ctx = 64;
    lm_sampling smp, smp_out;
    lm_sampling_default(&smp);
    smp.temperature = 0.0f;
    smp.max_tokens = 4;
    lm_model_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    lm_message msgs[1] = { { "user", "Hello" } };
    char buf[4096];
    int32_t ids[64] = { 1, 2, 3 };
    size_t n_out = 0;
    lm_model * model_out = NULL;
    lm_session * session_out = NULL;
    lm_model * model = NULL;
    lm_session * session = NULL;
    if (lm_model_load(model_path, &mopts, &model) != LM_OK || lm_session_new(model, &sopts, &session) != LM_OK) {
        printf("cannot load %s: %s\n", model_path, lm_last_error());
        return 1;
    }
    (void) mopts_out; (void) sopts_out; (void) smp_out; (void) model_out; (void) session_out; (void) n_out;

    /* functions without pointer parameters, and lm_status_name on values outside the enum */
    check(lm_version() != NULL && lm_version()[0] != 0, "lm_version", "empty");
    check(lm_api_version() == LM_API_VERSION, "lm_api_version", "differs from the header");
    check(lm_last_error() != NULL, "lm_last_error", "NULL");
    check(strcmp(lm_status_name((lm_status) -1), "LM_ERR_UNKNOWN") == 0, "lm_status_name(-1)", lm_status_name((lm_status) -1));
    check(strcmp(lm_status_name((lm_status) 9999), "LM_ERR_UNKNOWN") == 0, "lm_status_name(9999)", lm_status_name((lm_status) 9999));
    check(strcmp(lm_status_name((lm_status) 0x7fffffff), "LM_ERR_UNKNOWN") == 0, "lm_status_name(INT_MAX)", "not unknown");
'''

EPILOGUE = r'''
    /* the freeing functions, last: NULL is allowed and does nothing */
    expect_ok(lm_session_free(NULL), "lm_session_free(NULL)");
    expect_ok(lm_model_free(NULL), "lm_model_free(NULL)");
    expect_ok(lm_session_free(session), "lm_session_free(session)");
    expect_ok(lm_model_free(model), "lm_model_free(model)");
    printf("null_harness: %d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
'''


def call(func: str, ret: str, args: list[str]) -> str:
    return f"{func}({', '.join(args)})"


def generate() -> str:
    funcs = parse_header(HEADER.read_text(encoding="utf-8"))
    lines = [PRELUDE]
    names = [f for _, f, _ in funcs]
    for required in ("lm_model_load", "lm_chat", "lm_tokenize", "lm_detokenize", "lm_model_get_info"):
        if required not in names:
            raise SystemExit(f"{required} not found in the header: the parser is broken")
    for ret, func, params in funcs:
        ptrs = [(t, n) for t, n in params if is_pointer(t)]
        if not ptrs or func in FREEING:
            continue
        lines.append(f"\n    /* {func} */")
        cases = [[n] for _, n in ptrs] + ([[n for _, n in ptrs]] if len(ptrs) > 1 else [])
        for nulls in cases:
            args = []
            for t, n in params:
                if n in nulls:
                    args.append("NULL")
                else:
                    args.append(valid(func, t, n))
            # a buffer NULL that the header allows only with a zero size: first the call with a non-zero size (must
            # fail), then with that size 0 (allowed)
            allowed = len(nulls) == 1 and nulls[0] in OPTIONAL.get(func, set())
            what = f"{func}({', '.join('NULL' if n in nulls else n for _, n in params)})"
            if len(nulls) == 1 and (func, nulls[0]) in NULL_WITH_ZERO:
                expr = call(func, ret, args)
                if func == "lm_backends_info":
                    lines.append(f'    check({expr} == 0, "{what} with a non-zero size", "did not fail");')
                else:
                    lines.append(f'    expect_invalid({expr}, "{func}", "{what} with a non-zero size");')
                zero = NULL_WITH_ZERO[(func, nulls[0])]
                args = ["0" if n == zero else a for (t, n), a in zip(params, args)]
                allowed = True
                what += f" with {zero} = 0"
            expr = call(func, ret, args)
            if func == "lm_backends_info":  # returns a size: 0 is the failure value
                if allowed:
                    lines.append(f'    check({expr} > 0, "{what}", "returned 0");')
                else:
                    lines.append(f'    check({expr} == 0, "{what}", "did not fail");')
                continue
            if ret != "lm_status":
                raise SystemExit(f"{func} returns {ret}: extend gen_null_harness.py")
            if func in ("lm_tokenize", "lm_detokenize") and allowed:
                # a NULL buffer with a zero capacity is a size query: LM_OK or LM_ERR_BUFFER_TOO_SMALL are both right
                lines.append(f"    {{ lm_status st = {expr}; check(st == LM_OK || st == LM_ERR_BUFFER_TOO_SMALL, \"{what}\", lm_status_name(st)); }}")
            elif allowed:
                lines.append(f'    expect_ok({expr}, "{what}");')
            else:
                lines.append(f'    expect_invalid({expr}, "{func}", "{what}");')
            if func == "lm_model_load" and allowed:
                lines.append("    lm_model_free(model_out); model_out = NULL;")
            if func == "lm_session_new" and allowed:
                lines.append("    lm_session_free(session_out); session_out = NULL;")
    lines.append(EPILOGUE)
    return "\n".join(lines)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(generate(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
