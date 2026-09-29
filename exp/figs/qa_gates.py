"""QA gates.

Every figure script recomputes the numbers it draws straight from the result
files and asserts them against the values printed in the manuscript.  A figure
that silently drifts from its table is the failure mode these gates exist to
prevent, so a mismatch aborts the build instead of writing a stale panel.
"""
import sys

_FAILS = []


def check(label, ok, detail=""):
    print(("  OK   " if ok else "  FAIL ") + label + ("  " + detail if detail else ""))
    if not ok:
        _FAILS.append(label)
    return ok


def close(label, got, want, tol, unit=""):
    ok = abs(got - want) <= tol
    return check(label, ok, "got %.4g%s, manuscript %.4g%s (tol %.3g)"
                 % (got, unit, want, unit, tol))


def within(label, got, lo, hi, unit=""):
    ok = lo <= got <= hi
    return check(label, ok, "got %.4g%s, manuscript range [%.4g, %.4g]"
                 % (got, unit, lo, hi))


def finish(name):
    if _FAILS:
        print("\nABORT: %s failed %d gate(s): %s" % (name, len(_FAILS), ", ".join(_FAILS)))
        sys.exit(1)
    print("  all gates passed\n")
