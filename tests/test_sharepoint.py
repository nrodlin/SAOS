"""
T1 validation of Sharepoint (ZMQ publisher) and ReceptionPoint (REQ client).

Sharepoint is fed with documented stubs that expose exactly the attributes shareData reads (light path buffers,
atmosphere layers, DM command vectors) filled with random data. A real SUB socket receives everything. Expectations are
independent of the implementation: the payloads must be bit-equal to the published arrays and every topic must appear
exactly at the iterations whose (iteration + 1) is a multiple of its decimation. ReceptionPoint is tested against a
REP server thread. Every test uses its own random free port and short timeouts; sockets and contexts are always closed.
"""
import pickle
import socket
import threading
import types

import numpy as np
import pytest
import zmq

from citools import known_issue
from SAOS.ReceptionPoint import ReceptionPoint
from SAOS.Sharepoint import Sharepoint

pytestmark = pytest.mark.fast

N = 8  # shape of the stub maps


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------

def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class Link:
    """Sharepoint + SUB subscriber on a free port, synchronised (slow joiner) and always closed."""

    def __init__(self, logger, **decim):
        self.port = free_port()
        self.sp = Sharepoint(logger, port=self.port, ip='127.0.0.1', **decim)
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b'')
        self.sub.connect(f'tcp://127.0.0.1:{self.port}')
        for _ in range(100):  # handshake with a private topic until the subscription is active
            self.sp.socket.send_multipart([b'sync', b''])
            if self.sub.poll(50):
                break
        else:
            raise RuntimeError('subscriber never connected')
        while self.sub.poll(100):
            self.sub.recv_multipart()

    def iteration(self, it, lps, atm_list=(), dm_list=()):
        """Call shareData and return ({topic: payload}, catalogue of the 'topics' message)."""
        assert self.sp.shareData(lps, it, atm_list=list(atm_list), dm_list=list(dm_list)) is True
        got, catalogue = {}, None
        while True:
            assert self.sub.poll(2000), 'timeout waiting for the end-of-iteration "topics" message'
            topic, payload = self.sub.recv_multipart()
            if topic == b'topics':
                catalogue = payload.decode().split(',') if payload else []
                break
            assert topic.decode() not in got, f'topic {topic} published twice in one iteration'
            got[topic.decode()] = pickle.loads(payload)
        assert not self.sub.poll(50), 'extra messages after the "topics" message'
        return got, catalogue

    def close(self):
        self.sub.close(linger=0)
        self.ctx.term()
        self.sp.socket.close(linger=0)
        self.sp.context.term()


@pytest.fixture
def link(logger):
    made = []

    def make(**decim):
        lk = Link(logger, **decim)
        made.append(lk)
        return lk
    yield make
    for lk in made:
        lk.close()


def make_lp(rng, n_dirs=1):
    """Stub of a LightPath with all the buffers shareData may read (random content; n_dirs > 1 -> 3D dm buffers)."""
    shape = (n_dirs, N, N) if n_dirs > 1 else (N, N)
    r = lambda *s: rng.standard_normal(s)
    return types.SimpleNamespace(
        atm=object(), dm=object(), wfs=object(), sci=object(),
        atmosphere_opd=r(N, N), atmosphere_phase=r(N, N),
        dm_opd=r(*shape), dm_phase=r(*shape),
        slopes_1D=r(2 * N), slopes_2D=r(2 * N, N),
        wfs_opd=r(N, N), wfs_phase=r(N, N), wfs_frame=r(2 * N, 2 * N),
        sci_opd=r(N, N), sci_phase=r(N, N), sci_frame=r(N, N))


def make_atm(rng, n_layers):
    layers = {f'layer_{j + 1}': types.SimpleNamespace(screen=types.SimpleNamespace(scrn=rng.standard_normal((N, N))))
              for j in range(n_layers)}
    return types.SimpleNamespace(nLayer=n_layers, **layers)


def make_dm(rng, n_act=10):
    valid = np.zeros(n_act + 4, dtype=bool)
    valid[2:2 + n_act] = True
    return types.SimpleNamespace(dm_layer=types.SimpleNamespace(cmd_1D=rng.standard_normal(n_act + 4)), validAct=valid)


def due(it, dec):
    return dec > 0 and (it + 1) % dec == 0


def assert_equal_payload(got, exp, topic):
    got, exp = np.asarray(got), np.asarray(exp)
    assert got.shape == exp.shape and got.dtype == exp.dtype and np.array_equal(got, exp), f'payload of {topic} differs'


def check_schedule(link_, n_iter, expected_fn, lps, atm_list=(), dm_list=()):
    """
    Run n_iter iterations; expected_fn(it) -> {topic: array} that must be published exactly at that iteration.
    Returns {topic: [iterations]} of the observed publications.
    """
    seen = {}
    for it in range(n_iter):
        got, _ = link_.iteration(it, lps, atm_list, dm_list)
        exp = expected_fn(it)
        assert set(got) == set(exp), f'iteration {it}: published {sorted(got)}, expected {sorted(exp)}'
        for t in exp:
            assert_equal_payload(got[t], exp[t], t)
            seen.setdefault(t, []).append(it)
    return seen


# ----------------------------------------------------------------------------------------------------------------------
# Sharepoint: global objects
# ----------------------------------------------------------------------------------------------------------------------

def test_atmosphere_layers_decimation(link, rng, report):
    """Atmosphere layers: one topic per atmosphere/layer, published exactly when (it+1) % atm == 0, bit-equal."""
    lk = link(atm=3)
    atms = [make_atm(rng, 2), make_atm(rng, 1)]

    def expected(it):
        if not due(it, 3):
            return {}
        return {f'atmosphere_{i + 1}/layer_{j + 1}': getattr(a, f'layer_{j + 1}').screen.scrn
                for i, a in enumerate(atms) for j in range(a.nLayer)}

    seen = check_schedule(lk, 10, expected, [], atm_list=atms)
    assert sorted(seen) == ['atmosphere_1/layer_1', 'atmosphere_1/layer_2', 'atmosphere_2/layer_1']
    for t, its in seen.items():
        assert its == [2, 5, 8], (t, its)
    report.check('atm publications per topic', len(seen['atmosphere_1/layer_1']), expected=3)


@known_issue('SHARE-DM-DECIM', 'DM publish sits outside the decimation if and uses topics[-1]', 'SAOS/Sharepoint.py:306-308')
def test_dm_command_decimation(link, rng, report):
    """DM 1D command (valid actuators only): published only when (it+1) % dm == 0, topic dm_<i>/1D_command."""
    lk = link(dm=2)
    dms = [make_dm(rng), make_dm(rng)]

    def expected(it):
        if not due(it, 2):
            return {}
        return {f'dm_{i + 1}/1D_command': d.dm_layer.cmd_1D[d.validAct] for i, d in enumerate(dms)}

    seen = check_schedule(lk, 6, expected, [], dm_list=dms)
    assert all(its == [1, 3, 5] for its in seen.values())


@known_issue('SHARE-DM-DECIM', 'DM publish sits outside the decimation if and uses topics[-1]', 'SAOS/Sharepoint.py:306-308')
def test_dm_command_does_not_hijack_atmosphere_topic(link, rng):
    """With atm and dm active, a non-DM iteration must not publish the DM command under an atmosphere topic."""
    lk = link(atm=1, dm=2)
    atms, dms = [make_atm(rng, 1)], [make_dm(rng)]
    got, _ = lk.iteration(0, [], atms, dms)  # it+1 = 1: atm due, dm not due
    assert set(got) == {'atmosphere_1/layer_1'}
    assert_equal_payload(got['atmosphere_1/layer_1'], atms[0].layer_1.screen.scrn, 'atmosphere_1/layer_1')


# ----------------------------------------------------------------------------------------------------------------------
# Sharepoint: per light path
# ----------------------------------------------------------------------------------------------------------------------

def test_lightpath_topics_decimation(link, rng, report):
    """Per-direction atmosphere, slopes, WFS, WFS frame, science and science frame: topic set, bit-equal payloads
    and publication iterations for two light paths with different decimations."""
    dec = dict(atm_per_dir=2, slopes=3, wfs=4, wfs_frame=5, sci=6, sci_frame=2)
    lk = link(**dec)
    lps = [make_lp(rng), make_lp(rng)]
    n_iter = 12

    def expected(it):
        out = {}
        for i, lp in enumerate(lps):
            p = f'lightPath{i}/'
            if due(it, dec['atm_per_dir']):
                out[p + 'atmosphere_opd'], out[p + 'atmosphere_phase'] = lp.atmosphere_opd, lp.atmosphere_phase
            if due(it, dec['slopes']):
                out[p + 'slopes_1D'], out[p + 'slopes_2D'] = lp.slopes_1D, lp.slopes_2D
            if due(it, dec['wfs']):
                out[p + 'wfs_opd'], out[p + 'wfs_phase'] = lp.wfs_opd, lp.wfs_phase
            if due(it, dec['wfs_frame']):
                out[p + 'wfs_frame'] = lp.wfs_frame
            if due(it, dec['sci']):
                out[p + 'sci_opd'], out[p + 'sci_phase'] = lp.sci_opd, lp.sci_phase
            if due(it, dec['sci_frame']):
                out[p + 'sci_frame'] = lp.sci_frame
        return out

    seen = check_schedule(lk, n_iter, expected, lps)
    assert seen['lightPath1/sci_frame'] == [1, 3, 5, 7, 9, 11]
    assert seen['lightPath0/wfs_frame'] == [4, 9]
    report.metric('topics observed', len(seen))


def test_lightpath_payload_types_and_decimation_one(link, rng):
    """Decimation 1 publishes every iteration; the payload content follows the current buffer values."""
    lk = link(slopes=1, sci_frame=1)
    lp = make_lp(rng)
    for it in range(4):
        lp.slopes_1D = rng.standard_normal(2 * N)
        lp.sci_frame = rng.standard_normal((N, N))
        got, _ = lk.iteration(it, [lp])
        assert set(got) == {'lightPath0/slopes_1D', 'lightPath0/slopes_2D', 'lightPath0/sci_frame'}
        assert_equal_payload(got['lightPath0/slopes_1D'], lp.slopes_1D, 'slopes_1D')
        assert_equal_payload(got['lightPath0/sci_frame'], lp.sci_frame, 'sci_frame')


def test_sections_absent_in_light_path_are_not_published(link, rng):
    """A light path without wfs/sci/atm/dm sections publishes nothing for them, even with the decimations enabled."""
    lk = link(atm_per_dir=1, dm_per_dir=1, slopes=1, wfs=1, wfs_frame=1, sci=1, sci_frame=1)
    lp = make_lp(rng)
    lp.atm = lp.dm = lp.wfs = lp.sci = None
    got, catalogue = lk.iteration(0, [lp])
    assert got == {} and catalogue == []


def test_disabled_decimations_publish_nothing(link, rng):
    """All decimations at 0: no data topic, only the (empty) 'topics' message."""
    lk = link()
    lps = [make_lp(rng)]
    for it in range(3):
        got, catalogue = lk.iteration(it, lps, [make_atm(rng, 1)], [make_dm(rng)])
        assert got == {} and catalogue == []


def test_topics_catalogue(link, rng):
    """The final 'topics' message lists the topics configured for the iteration, in publication order, each time."""
    dec = dict(atm=2, slopes=3, wfs=1, sci_frame=4)
    lk = link(**dec)
    lps = [make_lp(rng), make_lp(rng)]
    atms = [make_atm(rng, 2)]
    exp = ['atmosphere_1/layer_1', 'atmosphere_1/layer_2']
    for i in range(2):
        p = f'lightPath{i}/'
        exp += [p + 'slopes_1D', p + 'slopes_2D', p + 'wfs_opd', p + 'wfs_phase', p + 'sci_frame']
    for it in range(5):
        _, catalogue = lk.iteration(it, lps, atms)
        assert catalogue == exp, f'iteration {it}'


@known_issue('SHARE-DM-TOPIC', 'the dm_phase topic sends dm_opd', 'SAOS/Sharepoint.py:334, 340')
def test_dm_per_dir_phase_topic_sends_phase(link, rng):
    """dm_per_dir with several directions: dm_opd_<j> carries dm_opd[j] and dm_phase_<j> carries dm_phase[j]."""
    lk = link(dm_per_dir=2)
    lp = make_lp(rng, n_dirs=3)
    for it in range(4):
        got, _ = lk.iteration(it, [lp])
        if not due(it, 2):
            assert got == {}, f'iteration {it} published {sorted(got)}'
            continue
        assert set(got) == {f'lightPath0/dm_{k}_{j}' for k in ('opd', 'phase') for j in (1, 2, 3)}
        for j in range(3):
            assert_equal_payload(got[f'lightPath0/dm_opd_{j + 1}'], lp.dm_opd[j], 'dm_opd')
            assert_equal_payload(got[f'lightPath0/dm_phase_{j + 1}'], lp.dm_phase[j], 'dm_phase')


@known_issue('SHARE-DM-TOPIC', 'the dm_phase topic sends dm_opd', 'SAOS/Sharepoint.py:334, 340')
def test_dm_per_dir_single_direction_phase_topic(link, rng):
    """dm_per_dir with one direction (2D buffers) at iterations where it is due: dm_phase topic carries dm_phase."""
    lk = link(dm_per_dir=1)
    lp = make_lp(rng)
    got, _ = lk.iteration(0, [lp])
    assert set(got) == {'lightPath0/dm_opd', 'lightPath0/dm_phase'}
    assert_equal_payload(got['lightPath0/dm_opd'], lp.dm_opd, 'dm_opd')
    assert_equal_payload(got['lightPath0/dm_phase'], lp.dm_phase, 'dm_phase')


@known_issue('SHARE-DMDIR-DECIM', 'single-direction dm_opd/dm_phase publish ignores dm_per_dir decimation '
             '(new, same pattern as SHARE-DM-DECIM)', 'SAOS/Sharepoint.py:336-341')
def test_dm_per_dir_single_direction_decimation(link, rng):
    """dm_per_dir=3 with one direction: nothing may be published at iterations 0, 1, 3, 4; both topics at 2, 5."""
    lk = link(dm_per_dir=3)
    lp = make_lp(rng)
    seen = {}
    for it in range(6):
        got, _ = lk.iteration(it, [lp])
        for t in got:
            seen.setdefault(t, []).append(it)
    assert seen == {'lightPath0/dm_opd': [2, 5], 'lightPath0/dm_phase': [2, 5]}


# ----------------------------------------------------------------------------------------------------------------------
# ReceptionPoint
# ----------------------------------------------------------------------------------------------------------------------

class RepServer:
    """REP server thread answering with answer_fn(request); records the requests. Closed on exit."""

    def __init__(self, answer_fn):
        self.port = free_port()
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.bind(f'tcp://127.0.0.1:{self.port}')
        self.requests, self.answer_fn, self.stop = [], answer_fn, False
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop:
            if self.sock.poll(50):
                req = pickle.loads(self.sock.recv())
                self.requests.append(req)
                self.sock.send(pickle.dumps(self.answer_fn(req)))

    def close(self):
        self.stop = True
        self.thread.join(2)
        self.sock.close(linger=0)
        self.ctx.term()


def close_rp(rp):
    rp.socket.close(linger=0)
    rp.context.term()


def test_reception_actuator_cmd_roundtrip(logger, rng):
    """actuator_cmd: the request is the pickled string and the answer comes back bit-equal."""
    cmd = rng.standard_normal(37)
    srv = RepServer(lambda req: cmd)
    rp = ReceptionPoint(logger, port=srv.port, ip='127.0.0.1', timeout=2000)
    try:
        out = rp.sendRequest('actuator_cmd')
        assert srv.requests == ['actuator_cmd']
        assert_equal_payload(out, cmd, 'actuator_cmd')
    finally:
        close_rp(rp)
        srv.close()


def test_reception_loop_cmd_state_tracking(logger):
    """loop_cmd: payload carries the buffer type and the previous state (-1 at first); result is int(desired_state)."""
    states = iter([1, 0, 1])
    srv = RepServer(lambda req: {'desired_state': next(states)})
    rp = ReceptionPoint(logger, port=srv.port, ip='127.0.0.1', timeout=2000)
    try:
        outs = [rp.sendRequest('loop_cmd') for _ in range(3)]
        assert outs == [1, 0, 1]
        assert srv.requests == [{'buffer_type': 'loop_cmd', 'actual_state': s} for s in (-1, 1, 0)]
    finally:
        close_rp(rp)
        srv.close()


def test_reception_unsupported_type_returns_none(logger):
    """An unknown buffer type returns None without sending anything."""
    srv = RepServer(lambda req: 'x')
    rp = ReceptionPoint(logger, port=srv.port, ip='127.0.0.1', timeout=300)
    try:
        assert rp.sendRequest('foo') is None
        assert rp.sendRequest() is None
        assert not srv.sock.poll(200) and srv.requests == []
    finally:
        close_rp(rp)
        srv.close()


@pytest.mark.parametrize('buffer_type', ['actuator_cmd', 'loop_cmd'])
def test_reception_timeout_returns_none_and_recovers(logger, buffer_type):
    """No server: None after the timeout (bounded time); once a server is up the next request succeeds."""
    port = free_port()
    rp = ReceptionPoint(logger, port=port, ip='127.0.0.1', timeout=300)
    srv = None
    try:
        assert rp.sendRequest(buffer_type) is None
        srv = RepServer.__new__(RepServer)
        RepServer.__init__(srv, lambda req: {'desired_state': 1} if buffer_type == 'loop_cmd' else 'ok')
        rp.port = srv.port  # reconnect target changes: rebuild the socket on the new port
        rp.socket.close(linger=0)
        rp.socket = rp.context.socket(zmq.REQ)
        rp.socket.connect(f'tcp://127.0.0.1:{srv.port}')
        rp.poller = zmq.Poller()
        rp.poller.register(rp.socket, zmq.POLLIN)
        out = rp.sendRequest(buffer_type)
        assert out == (1 if buffer_type == 'loop_cmd' else 'ok')
    finally:
        close_rp(rp)
        if srv:
            srv.close()


def test_reception_poller_only_holds_live_socket(logger):
    """After a timeout the socket is rebuilt: the poller must hold exactly the current socket, none of the closed ones."""
    rp = ReceptionPoint(logger, port=free_port(), ip='127.0.0.1', timeout=200)
    try:
        assert rp.sendRequest('actuator_cmd') is None
        registered = [s for s, _ in rp.poller.sockets]
        assert registered == [rp.socket], f'{len(registered)} sockets registered, {sum(s.closed for s in registered)} closed'
    finally:
        close_rp(rp)


def test_reception_second_timeout_returns_none_in_bounded_time(logger):
    """Two consecutive timeouts must each return None after about `timeout` ms (guarded by a thread deadline so that a
    hang is reported as a failure instead of blocking the suite)."""
    rp = ReceptionPoint(logger, port=free_port(), ip='127.0.0.1', timeout=200)
    results = []

    def work():
        for _ in range(3):
            results.append(rp.sendRequest('actuator_cmd'))

    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(4)
    hung = th.is_alive()
    if not hung:
        close_rp(rp)
    assert not hung, f'sendRequest did not return after {len(results)} completed timeouts'
    assert results == [None, None, None]
