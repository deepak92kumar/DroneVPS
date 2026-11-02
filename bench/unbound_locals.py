#!/usr/bin/env python3
"""Find locals that can be READ on a path where they were never ASSIGNED.

This is the bug that has now killed the node twice in flight: `_use_pp` (76
crashes) and `_fe` (4 crashes, flight of 2 Nov, ALT_HOLD at 30 m). Both times
a variable was assigned inside ONE branch of an if/else and read by code
BELOW the branch that runs on every frame. On the ground the other branch
never runs, so no test and no bench ever took that path.

pylint 4.1.3 does not catch it. Checked: its E0601/E0606 report nothing on
either the real pre-fix live_node.py or on an eight-line reproduction of the
same shape. pyflakes and ruff are not installed and pyflakes does not
implement this check at all. So the analysis is here.

What it does is a definite-assignment pass over each function body:

  * a name is "bound" only if every path reaching this statement assigned it
  * if/else contributes the INTERSECTION of the two branches
  * a branch that ends in return/raise/break/continue does not fall through,
    so it contributes nothing and the other branch's bindings stand alone
  * for/while bodies may run zero times, so they contribute nothing
  * with bodies always run, so they contribute
  * try bodies may stop part way, so they contribute nothing; finally does

Reading a name that is assigned SOMEWHERE in the function but is not bound
here is reported. A name never assigned in the function is a global or a
builtin and is not our business.
"""
import ast
import sys

# Branch-enders: after one of these the rest of the branch is unreachable.
_JUMP = (ast.Return, ast.Raise, ast.Break, ast.Continue)
# ast.TryStar is 3.11+; this runs on 3.10 too.
_TRYSTAR = ((ast.TryStar,) if hasattr(ast, "TryStar") else ())


def _terminates(body):
    """Does this block definitely leave the enclosing flow?"""
    for st in body:
        if isinstance(st, _JUMP):
            return True
        if isinstance(st, ast.If):
            if st.orelse and _terminates(st.body) and _terminates(st.orelse):
                return True
        if isinstance(st, ast.With) and _terminates(st.body):
            return True
        if isinstance(st, (ast.Try,) + _TRYSTAR):
            # Only if the body and EVERY handler leave.
            if (_terminates(st.body)
                    and st.handlers
                    and all(_terminates(h.body) for h in st.handlers)):
                return True
            if st.finalbody and _terminates(st.finalbody):
                return True
    return False


def _targets(node, out):
    """Names this assignment target binds."""
    if isinstance(node, ast.Name):
        out.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for e in node.elts:
            _targets(e, out)
    elif isinstance(node, ast.Starred):
        _targets(node.value, out)
    # Attribute and Subscript targets bind nothing new.


class _Func:
    """One function's locals, and where each one is read."""

    def __init__(self, fn, path):
        self.fn, self.path = fn, path
        self.assigned = set()       # every name assigned anywhere inside
        self.declared = set()       # global / nonlocal: not ours
        self.params = set()
        self.findings = []
        for a in (list(fn.args.posonlyargs) + list(fn.args.args)
                  + list(fn.args.kwonlyargs)):
            self.params.add(a.arg)
        for a in (fn.args.vararg, fn.args.kwarg):
            if a is not None:
                self.params.add(a.arg)
        self._collect(fn)

    def _collect(self, node):
        """Every name assigned anywhere in this function, nested scopes out."""
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.Lambda, ast.ClassDef)):
                # A nested scope of its own; its name IS bound here though.
                if not isinstance(ch, ast.Lambda):
                    self.assigned.add(ch.name)
                continue
            if isinstance(ch, (ast.Global, ast.Nonlocal)):
                self.declared.update(ch.names)
            elif isinstance(ch, ast.Assign):
                for t in ch.targets:
                    _targets(t, self.assigned)
            elif isinstance(ch, (ast.AugAssign, ast.AnnAssign)):
                _targets(ch.target, self.assigned)
            elif isinstance(ch, ast.NamedExpr):
                _targets(ch.target, self.assigned)
            elif isinstance(ch, (ast.For, ast.AsyncFor)):
                _targets(ch.target, self.assigned)
            elif isinstance(ch, (ast.With, ast.AsyncWith)):
                for it in ch.items:
                    if it.optional_vars is not None:
                        _targets(it.optional_vars, self.assigned)
            elif isinstance(ch, ast.ExceptHandler):
                if ch.name:
                    self.assigned.add(ch.name)
            elif isinstance(ch, (ast.Import, ast.ImportFrom)):
                for al in ch.names:
                    self.assigned.add((al.asname or al.name).split(".")[0])
            self._collect(ch)

    # -- the pass itself ----------------------------------------------------

    def run(self):
        self._block(self.fn.body, set(self.params))
        return self.findings

    _COMP = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

    def _reads(self, node, bound):
        """Report every Load of an unbound local inside this expression.

        Comprehensions get their own scope in Python 3, so `[g for g in xs]`
        does not read the function's `g` - it binds its own. Walking flat
        reported that as a finding; it is not one.
        """
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.Lambda, ast.ClassDef)):
            # A closure can be CALLED later, after the name is bound.
            return
        if isinstance(node, self._COMP):
            inner = set(bound)
            for c in node.generators:
                # The FIRST iterable is evaluated in the enclosing scope; the
                # rest see the targets bound before them.
                self._reads(c.iter, inner)
                _targets(c.target, inner)
                for f in c.ifs:
                    self._reads(f, inner)
            if isinstance(node, ast.DictComp):
                self._reads(node.key, inner)
                self._reads(node.value, inner)
            else:
                self._reads(node.elt, inner)
            return
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                and node.id in self.assigned
                and node.id not in self.declared
                and node.id not in bound):
            self.findings.append((node.lineno, node.id, self.fn.name))
            return
        for ch in ast.iter_child_nodes(node):
            self._reads(ch, bound)

    def _block(self, body, bound):
        """Walk a block; returns the set bound at its end."""
        for st in body:
            bound = self._stmt(st, bound)
        return bound

    def _stmt(self, st, bound):
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef,
                           ast.ClassDef)):
            return bound | {st.name}

        if isinstance(st, ast.If):
            self._reads(st.test, bound)
            # The test's walrus binds for both branches.
            tb = set()
            for n in ast.walk(st.test):
                if isinstance(n, ast.NamedExpr):
                    _targets(n.target, tb)
            b0 = bound | tb
            after_body = self._block(st.body, set(b0))
            after_else = self._block(st.orelse, set(b0)) if st.orelse else b0
            body_out = _terminates(st.body)
            else_out = _terminates(st.orelse) if st.orelse else False
            if body_out and else_out:
                return b0            # nothing below runs anyway
            if body_out:
                return after_else
            if else_out:
                return after_body
            if not st.orelse:
                return b0            # the if may not have run
            return after_body & after_else

        if isinstance(st, (ast.For, ast.AsyncFor)):
            self._reads(st.iter, bound)
            tgt = set()
            _targets(st.target, tgt)
            self._block(st.body, bound | tgt)
            self._block(st.orelse, set(bound))
            return bound             # zero iterations is a path

        if isinstance(st, ast.While):
            self._reads(st.test, bound)
            self._block(st.body, set(bound))
            self._block(st.orelse, set(bound))
            return bound

        if isinstance(st, (ast.With, ast.AsyncWith)):
            b = set(bound)
            for it in st.items:
                self._reads(it.context_expr, b)
                if it.optional_vars is not None:
                    _targets(it.optional_vars, b)
            return self._block(st.body, b)   # the body always runs

        if isinstance(st, (ast.Try,) + _TRYSTAR):
            # The body can stop at any statement, so a handler sees only what
            # was bound BEFORE the try. Nothing from the body is definite TO
            # A HANDLER.
            body_end = self._block(st.body, set(bound))
            for h in st.handlers:
                hb = set(bound)
                if h.name:
                    hb.add(h.name)
                self._block(h.body, hb)
            # But if every handler returns/raises/breaks/continues - and the
            # project's handlers nearly all do - then the ONLY way to reach
            # the statement after the try is the body running to its end, so
            # the body's bindings are definite there. Without this the checker
            # calls `try: x = f() / except: return` a finding, which it is
            # not, and 20 of the 23 first reports in the flight-critical
            # modules were exactly that shape.
            #
            # No handlers at all (try/finally) is the same case: an exception
            # in the body propagates instead of falling through.
            # Every way of reaching the statement after the try, and what is
            # definitely bound on each: the body ran to its end, or a handler
            # that falls through ran. A handler can be entered after the body
            # stopped part way, so it is credited only with what was bound
            # BEFORE the try plus what the handler itself binds. The answer is
            # the intersection, which is how `try: t = int(x) / except
            # ValueError: t = 0` comes out bound - it is bound on both paths.
            ends = [body_end]
            for h in st.handlers:
                if _terminates(h.body):
                    continue
                hb = set(bound)
                if h.name:
                    hb.add(h.name)
                ends.append(self._block(h.body, hb))
            if st.orelse:
                # else runs only when the body completed, so it replaces the
                # body's end as that path's answer.
                ends[0] = self._block(st.orelse, set(body_end))
            out = set(ends[0])
            for e in ends[1:]:
                out &= e
            if st.finalbody:
                out = self._block(st.finalbody, set(out))
            return out

        if isinstance(st, ast.Assign):
            self._reads(st.value, bound)
            b = set(bound)
            for t in st.targets:
                self._reads_target(t, b)
                _targets(t, b)
            return b

        if isinstance(st, (ast.AugAssign, ast.AnnAssign)):
            if st.value is not None:
                self._reads(st.value, bound)
            if isinstance(st, ast.AugAssign):
                self._reads(st.target, bound)   # x += 1 READS x
            b = set(bound)
            _targets(st.target, b)
            return b

        if isinstance(st, (ast.Global, ast.Nonlocal)):
            return bound | set(st.names)

        if isinstance(st, ast.Delete):
            b = set(bound)
            for t in st.targets:
                self._reads_target(t, b)
                if isinstance(t, ast.Name):
                    b.discard(t.id)
            return b

        # Everything else: expressions, return, raise, assert, pass, import.
        b = set(bound)
        for ch in ast.iter_child_nodes(st):
            self._reads(ch, b)
        for n in ast.walk(st):
            if isinstance(n, ast.NamedExpr):
                _targets(n.target, b)
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for al in n.names:
                    b.add((al.asname or al.name).split(".")[0])
        return b

    def _reads_target(self, t, bound):
        """a[i] = v and a.b = v READ a."""
        if isinstance(t, (ast.Subscript, ast.Attribute)):
            self._reads(t, bound)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                self._reads_target(e, bound)


def check(path):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src, filename=path)
    out = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for lineno, name, fn in _Func(n, path).run():
                out.append((path, lineno, name, fn))
    # One report per (name, line).
    seen, uniq = set(), []
    for r in out:
        k = (r[0], r[1], r[2])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


def main(argv):
    import glob
    paths = []
    for a in argv or ["vps_vio"]:
        paths.extend(sorted(glob.glob(a + "/**/*.py", recursive=True))
                     if not a.endswith(".py") else [a])
    bad = []
    for p in paths:
        bad.extend(check(p))
    for path, lineno, name, fn in bad:
        print(f"{path}:{lineno}: '{name}' can be read unbound in {fn}()")
    print(f"\n{len(bad)} in {len(paths)} files")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
