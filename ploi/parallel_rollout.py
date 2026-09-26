"""Parallel rollout workers for the batched evaluator (opt-in).

The batched harness spends ~65% of its time in `state_to_graph_wrapper`,
building one graph per active problem per round in a serial Python loop
(measured on visitall, 20 problems: 101s of a 155s run). Those builds are
independent of one another, so they parallelise cleanly - but only if the
per-problem state never has to cross a process boundary, because a pddlgym
state pickles in ~4.6ms, comparable to the ~5.7ms it costs to featurize one.

So workers own problems, not tasks. Worker `w` holds every problem with
`index % n_workers == w` together with all of its state: the env, the
current state, the revisit monitor, the PlanningResult, the node->object
table and the per-problem fallback RNG. None of that is ever sent anywhere.
Per round the parent exchanges two small messages with each worker:

    parent -> worker   FEATURIZE  [ids still active]
    worker -> parent   {id: graph dict}                 ~0.16 MB, ~0.05 ms
    parent -> worker   STEP  {id: beam candidates}      tens of bytes
    worker -> parent   {id: (finished, success, ...)}   tens of bytes

Featurization and env stepping run in parallel; the parent keeps the model
and the batched forward pass. Serialisation costs about 1% of the work it
replaces (0.05ms to round-trip a graph against 5.7ms to build one).

Only NUMPY crosses the pipe, never torch tensors. Converting to HeteroData
inside the worker looks like an obvious extra win (it would move ~9% of the
run into the parallel region) and is a trap: torch installs a ForkingPickler
that relocates every tensor into its own shared-memory segment, so each
graph costs ~14 file descriptors and mmaps instead of one buffer copy.
Measured on visitall, 50 problems, 16 workers: graph build went from 65s to
631s. The PyG conversion stays in the parent.

Workers are forked *after* the envs exist, so they inherit the tester, the
graph metadata and the envs with no pickling at startup, and they never
touch CUDA - which is what makes forking a CUDA-initialised parent safe.
It is the same arrangement PyTorch's DataLoader uses with num_workers > 0.

Off unless GABAR_FEATURIZE_WORKERS=N is set. The serial path is untouched
and stays the reference for parity checks (tools/parity_matrix.sh).
"""

import multiprocessing as mp
import os

# Wire protocol. Payloads are chosen so that only numpy arrays and plain
# ints ever cross the boundary.
FEATURIZE, STEP, COLLECT, STOP = range(4)


def available_cores():
    """Cores this process may actually run on, not the machine's total.

    GPU nodes commonly allocate 2-8 CPUs per GPU, so os.cpu_count() (which
    reports the whole machine) overstates it by an order of magnitude.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:            # not Linux
        return os.cpu_count() or 1


def configured_workers(verbose=True):
    """Worker count from GABAR_FEATURIZE_WORKERS; 0 means "stay serial".

    Capped at the number of available cores. Not cores-1: the parent is
    blocked in recv() for the whole featurize phase, so it is not competing
    with the workers for CPU. Measured on a 2-core allocation, graph build
    over 20 visitall problems: 72.5s serial, 66.1s at 2 workers, 70.0s at 4,
    72.4s at 8, 80.7s at 16. Oversubscription is not merely useless but
    actively harmful, hence the hard cap.
    """
    try:
        requested = max(0, int(os.environ.get("GABAR_FEATURIZE_WORKERS", "0")))
    except ValueError:
        return 0
    if requested == 0:
        return 0
    allowed = max(1, available_cores())
    if requested > allowed and verbose:
        print(f"[batch] GABAR_FEATURIZE_WORKERS={requested} capped to "
              f"{allowed} ({available_cores()} cores available to this "
              f"process); oversubscription slows this harness down",
              flush=True)
    return min(requested, allowed)


def compact_beam(beam_results):
    """One graph's beam output as plain Python, safe to send to a worker.

    Beam entries hold device tensors; `decode_beam_results` only ever indexes
    them and calls int(), so a list of (score, [token ids]) is an exact
    stand-in and costs bytes instead of a CUDA-tensor pickle.
    """
    return [(float(entry[0]), [int(t) for t in entry[1]])
            for entry in beam_results]


class RolloutWorkerPool:
    """Forked workers, each owning a disjoint subset of the problems.

    Construct AFTER the per-problem states exist; the fork hands them over
    for free. `featurize_one` and `step_one` are callables (in practice
    closures over the tester) that the workers inherit, so the code they run
    is literally the code the serial path runs.
    """

    def __init__(self, n_workers, problem_ids, build_state,
                 featurize_one, step_one):
        """`build_state(p)` returns the per-problem state dict. It runs IN
        the worker that will own p, so constructing the envs (a deepcopy, a
        reset and a full regrounding, ~0.9s each) parallelises too instead
        of running serially in the parent."""
        ids = sorted(problem_ids)
        n_workers = max(1, min(n_workers, len(ids)))
        # Round-robin over sorted ids: deterministic, and it spreads the
        # long-running problems (which cluster by size in the test sets)
        # across workers instead of piling them on one.
        self.owner = {p: i % n_workers for i, p in enumerate(ids)}

        ctx = mp.get_context("fork")
        self._conns, self._procs = [], []
        for w in range(n_workers):
            mine = [p for p in ids if self.owner[p] == w]
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker_loop,
                args=(child_conn, mine, build_state, featurize_one, step_one),
                daemon=True)
            proc.start()
            child_conn.close()          # parent keeps only its end
            self._conns.append(parent_conn)
            self._procs.append(proc)

        # Block until every worker has built its envs. Keeps setup inside the
        # caller's setup timing, and surfaces a build failure here rather
        # than as a mystifying hang on the first featurize round.
        for conn in self._conns:
            status = conn.recv()
            if isinstance(status, BaseException):
                self.close()
                raise RuntimeError("rollout worker failed to build its "
                                   "problems") from status

    @property
    def n_workers(self):
        return len(self._procs)

    def featurize(self, active_ids):
        """{problem id: graph dict} for `active_ids`, built in parallel."""
        return self._round_trip(FEATURIZE, self._split_list(active_ids))

    def step(self, beams_by_problem):
        """Decode and apply one greedy step per problem, in parallel.

        Returns {problem id: (finished, success, plan_length, time_taken)}.
        """
        return self._round_trip(STEP, self._split_dict(beams_by_problem))

    def collect_results(self):
        """Final PlanningResult per problem. Called once, when all are done."""
        return self._round_trip(COLLECT, [None] * self.n_workers)

    def close(self):
        for conn in self._conns:
            try:
                conn.send((STOP, None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        for conn in self._conns:
            conn.close()
        self._conns, self._procs = [], []

    # -- internals ---------------------------------------------------------

    def _round_trip(self, cmd, payload_per_worker):
        """Send to every worker, then gather. Sends are issued first so the
        workers overlap instead of running one at a time."""
        for conn, payload in zip(self._conns, payload_per_worker):
            conn.send((cmd, payload))
        merged = {}
        for conn in self._conns:
            merged.update(conn.recv())
        return merged

    def _split_list(self, ids):
        buckets = [[] for _ in self._conns]
        for p in ids:
            buckets[self.owner[p]].append(p)
        return buckets

    def _split_dict(self, by_problem):
        buckets = [{} for _ in self._conns]
        for p, value in by_problem.items():
            buckets[self.owner[p]][p] = value
        return buckets

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _worker_loop(conn, my_ids, build_state, featurize_one, step_one):
    """Builds and then owns the problems in `my_ids` for the life of the
    rollout; replies to one command at a time. Groundings are kept between
    the FEATURIZE and STEP halves of a round so they are enumerated exactly
    once, as in the serial path."""
    groundings = {}
    try:
        try:
            states = {p: build_state(p) for p in my_ids}
        except BaseException as exc:        # report, do not hang the parent
            conn.send(exc)
            return
        conn.send({})                       # ready
        while True:
            cmd, payload = conn.recv()
            if cmd == STOP:
                return

            if cmd == FEATURIZE:
                out = {}
                for p in payload:
                    graph, grounding = featurize_one(states[p])
                    groundings[p] = grounding
                    out[p] = graph

            elif cmd == STEP:
                out = {}
                for p, beam in payload.items():
                    st = states[p]
                    finished = step_one(st, beam, groundings[p])
                    result = st["result"]
                    out[p] = (finished, result.success,
                              result.plan_length, result.time_taken)

            elif cmd == COLLECT:
                out = {p: st["result"] for p, st in states.items()}

            else:
                out = {}

            conn.send(out)
    except (EOFError, KeyboardInterrupt):
        return
    finally:
        conn.close()
