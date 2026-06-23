"""Outgroup-based merge: candidate generation.

Dependency-free. Reads a Newick subtree, prunes a given set of outgroup leaves
(suppressing the degree-2 nodes left behind and summing their branch lengths,
which is likelihood-exact because P(t1)*P(t2)=P(t1+t2)), records where each
removed leaf was attached as a *split* over the surviving leaves (stable under
further edits, unlike a node id), and enumerates the N nearest edges to each
anchor for the merge search.

Output of prune_and_record:
  clean : Tree              -- pruned tree, ready for likelihood
  anchors : {leaf -> Split} -- where each removed leaf attached, as a split
The scorer then takes `clean` + an MSA + `candidate_edges(anchor, N)`.
"""

from collections import deque

Split = frozenset  # a split is the frozenset of leaf names on the smaller side


class Tree:
    """Unrooted tree: integer nodes, adjacency with branch lengths, leaf names."""

    def __init__(self):
        self.adj = {}          # node -> {neighbor: branch_length}
        self.name = {}         # node -> leaf name (leaves only)
        self._next = 0

    # ---- construction ----------------------------------------------------
    def _new_node(self, name=None):
        n = self._next
        self._next += 1
        self.adj[n] = {}
        if name is not None:
            self.name[n] = name
        return n

    def _link(self, u, v, bl):
        self.adj[u][v] = bl
        self.adj[v][u] = bl

    def _unlink(self, u, v):
        self.adj[u].pop(v, None)
        self.adj[v].pop(u, None)

    # ---- Newick ----------------------------------------------------------
    @classmethod
    def parse_newick(cls, s):
        t = cls()
        s = s.strip()
        if s.endswith(';'):
            s = s[:-1]
        pos = 0

        def parse_clade():
            nonlocal pos
            node = t._new_node()
            if s[pos] == '(':
                pos += 1  # consume '('
                while True:
                    child, bl = parse_clade()
                    t._link(node, child, bl)
                    if s[pos] == ',':
                        pos += 1
                        continue
                    if s[pos] == ')':
                        pos += 1
                        break
            # label
            start = pos
            while pos < len(s) and s[pos] not in '(),':
                pos += 1
            label = s[start:pos]
            name, bl = label, 0.0
            if ':' in label:
                name, blstr = label.split(':', 1)
                bl = float(blstr)
            if name:
                t.name[node] = name
            return node, bl

        root, _ = parse_clade()
        # Suppress a bifurcating (degree-2) root into a proper unrooted tree.
        if len(t.adj[root]) == 2 and root not in t.name:
            (a, ba), (b, bb) = list(t.adj[root].items())
            t._unlink(root, a); t._unlink(root, b)
            t._link(a, b, ba + bb)
            del t.adj[root]
        return t

    def to_newick(self):
        # pick any leaf's neighbor as a traversal root
        start = next(iter(self.name))
        root = next(iter(self.adj[start]))

        def emit(node, parent):
            kids = [(nb, bl) for nb, bl in self.adj[node].items() if nb != parent]
            if not kids:  # leaf
                return self.name.get(node, '')
            inner = ','.join(f"{emit(nb, node)}:{bl:.6g}" for nb, bl in kids)
            return f"({inner})" + self.name.get(node, '')
        # root is an internal-ish anchor; emit start as a pendant of root
        body = ','.join(
            f"{emit(nb, root)}:{bl:.6g}" for nb, bl in self.adj[root].items())
        rootname = self.name.get(root, '')
        return f"({body}){rootname};"

    # ---- queries ---------------------------------------------------------
    def leaves(self):
        return {n for n in self.adj if len(self.adj[n]) == 1}

    def leaf_names(self):
        return {self.name[n] for n in self.leaves()}

    def _leaf_by_name(self, name):
        for n in self.leaves():
            if self.name.get(n) == name:
                return n
        raise KeyError(name)

    def _real_through(self, start, blocked, real_set):
        """Names in `real_set` reachable from `start` without crossing back
        through `blocked`. Outgroups (incl. the kept bridge) are not in
        `real_set`, so they neither count toward nor subdivide a split."""
        seen = {blocked, start}
        out = set()
        stack = [start]
        while stack:
            x = stack.pop()
            if x in self.name and self.name[x] in real_set:
                out.add(self.name[x])
            for nb in self.adj[x]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return out

    # ---- anchor of a removed leaf (computed on the FULL tree) -------------
    def anchor_split(self, leaf_name, real_set):
        """Split over REAL leaves where `leaf_name` was attached. Real leaves
        exclude every outgroup -- including the kept bridge -- so an outgroup
        sitting between two groups doesn't push the anchor onto one side; the
        whole connecting branch reads as one split. Walks outward past any
        all-outgroup region until two sides each hold a real leaf."""
        L = self._leaf_by_name(leaf_name)
        v = next(iter(self.adj[L]))          # L's only neighbor
        prev = L
        while True:
            sides = []
            for nb in self.adj[v]:
                if nb == prev:
                    continue
                s = self._real_through(nb, v, real_set)
                sides.append((nb, s))
            nonempty = [(nb, s) for nb, s in sides if s]
            if len(nonempty) >= 2:
                side = nonempty[0][1]
                other = set(real_set) - side
                # canonical: smaller side (ties -> lexicographically smallest)
                pick = min((side, other), key=lambda z: (len(z), sorted(z)))
                return Split(pick)
            if len(nonempty) == 1:           # all-outgroup on the other branches
                prev, v = v, nonempty[0][0]
                continue
            return Split(frozenset())        # degenerate: no real leaves

    # ---- pruning ---------------------------------------------------------
    def prune_leaves(self, names_to_remove):
        for nm in names_to_remove:
            L = self._leaf_by_name(nm)
            (v, _), = self.adj[L].items()
            self._unlink(L, v)
            del self.adj[L]
            self.name.pop(L, None)
            self._cleanup(v)

    def _cleanup(self, node):
        """Repair node after a neighbor was removed: drop degree-1 internal
        stubs (cascading), suppress degree-2 nodes summing branch lengths."""
        while node in self.adj:
            deg = len(self.adj[node])
            if deg == 1 and node not in self.name:        # internal stub
                (nb, _), = self.adj[node].items()
                self._unlink(node, nb)
                del self.adj[node]
                node = nb
            elif deg == 2 and node not in self.name:      # suppress
                (a, ba), (b, bb) = list(self.adj[node].items())
                self._unlink(node, a); self._unlink(node, b)
                del self.adj[node]
                self._link(a, b, ba + bb)
                break
            else:
                break

    # ---- merge candidates ------------------------------------------------
    def edges(self):
        seen = set()
        out = []
        for u in self.adj:
            for v in self.adj[u]:
                if (v, u) not in seen:
                    seen.add((u, v))
                    out.append((u, v))
        return out

    def split_of_edge(self, u, v, real_set=None):
        if real_set is None:
            real_set = self.leaf_names()
        return Split(self._real_through(u, v, real_set))

    def find_anchor_edges(self, anchor, real_set):
        """Edges in THIS (clean) tree whose real-leaf split equals the anchor.
        With a kept bridge between two groups this is a *region* (the edges on
        either side of the bridge), so we return all of them."""
        out = []
        for (u, v) in self.edges():
            s = self.split_of_edge(u, v, real_set)
            if s == anchor or (set(real_set) - s) == anchor:
                out.append((u, v))
        return out

    def candidate_edges(self, anchor_edges, n=14):
        """The `n` edges closest to the anchor region, by edge-graph BFS
        distance (going 'away'). `anchor_edges` may be one edge or a list."""
        if anchor_edges and isinstance(anchor_edges[0], int):
            anchor_edges = [anchor_edges]            # a single (u,v)
        adj_e = self._edge_adjacency()
        def norm(e):
            return e if e in adj_e else (e[1], e[0])
        starts = [norm(e) for e in anchor_edges]
        order, seen = [], set(starts)
        q = deque(starts)
        while q and len(order) < n:
            e = q.popleft()
            order.append(e)
            for ne in adj_e[e]:
                if ne not in seen:
                    seen.add(ne)
                    q.append(ne)
        return order

    def _edge_adjacency(self):
        # two edges are adjacent if they share a node
        es = []
        for (u, v) in self.edges():
            es.append((u, v))
        adj_e = {e: [] for e in es}
        by_node = {}
        for e in es:
            for endp in e:
                by_node.setdefault(endp, []).append(e)
        for e in es:
            nbrs = set()
            for endp in e:
                for f in by_node[endp]:
                    if f != e:
                        nbrs.add(f)
            adj_e[e] = list(nbrs)
        return adj_e


def prune_and_record(newick, remove_names, real_leaves=None, keep_outgroups=()):
    """Read Newick, record each removed leaf's anchor as a split over the REAL
    leaves (so the kept bridge and other outgroups don't subdivide it), then
    prune the removed leaves.

      remove_names   : outgroup leaves to delete from the tree
      keep_outgroups : outgroup leaves to KEEP in the tree (e.g. the bridge M)
                       but still exclude from anchor splits
      real_leaves    : the relevant leaves; if None, inferred as
                       all_leaves - remove_names - keep_outgroups

    Returns (clean_tree, {removed_leaf -> anchor_split_over_real_leaves}).
    """
    t = Tree.parse_newick(newick)
    all_leaves = t.leaf_names()
    remove_names = [n for n in remove_names if n in all_leaves]
    if real_leaves is None:
        real_leaves = all_leaves - set(remove_names) - set(keep_outgroups)
    real_leaves = set(real_leaves)
    anchors = {nm: t.anchor_split(nm, real_leaves) for nm in remove_names}
    t.prune_leaves(remove_names)
    return t, anchors
