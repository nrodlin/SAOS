# `test_sharepoint.py` — ZMQ publisher and request client

Source: `tests/test_sharepoint.py`

Covers `SAOS.Sharepoint` (ZMQ PUB publisher, `shareData`) and `SAOS.ReceptionPoint` (ZMQ REQ client, `sendRequest`). These tests concern communication and bookkeeping, not physics: there is no theory or numerical tolerance. `Sharepoint` is fed with documented stubs (`SimpleNamespace`) exposing exactly the attributes `shareData` reads (light-path buffers, atmosphere layers, DM command vectors) filled with random data; a real SUB socket receives everything (slow-joiner handled by a handshake on a private `sync` topic). Expectations are independent of the implementation: payloads must be bit-equal (shape, dtype and values) to the published arrays and every topic must appear exactly at the iterations whose $(it+1)$ is a multiple of its decimation. `ReceptionPoint` is tested against a REP server thread. Each test uses its own random free local port and short timeouts, and always closes its sockets and contexts. All tests are marked `fast` (module-level). No tolerance yaml is used (exact comparisons).

Scheduling rule checked everywhere: a topic with decimation $k>0$ is published at iteration $it$ iff $(it+1)\bmod k=0$; a decimation of 0 disables it. Each call of `shareData` ends with a `topics` message that lists the topics configured for the iteration.

## `test_atmosphere_layers_decimation`
- **Module:** `Sharepoint.shareData` (atmosphere layers).
- **Tier:** fast.
- **What it checks:** with `atm=3` and two atmospheres (2 and 1 layers), one topic per atmosphere/layer (`atmosphere_<i>/layer_<j>`) is published exactly at iterations 2, 5, 8 of 10, with payload bit-equal to the layer screen.
- **Theory / assumptions:** none (scheduling and payload identity).
- **Tolerance:** exact.
- **References:** none.

## `test_dm_command_decimation`
- **Module:** `Sharepoint.shareData` (DM 1D command).
- **Tier:** fast.
- **What it checks:** with `dm=2` the topic `dm_<i>/1D_command` (valid actuators only, `cmd_1D[validAct]`) is published only at iterations 1, 3, 5 of 6.
- **Theory / assumptions:** none.
- **Tolerance:** exact.
- **References:** none.

## `test_dm_command_does_not_hijack_atmosphere_topic`
- **Module:** `Sharepoint.shareData`.
- **Tier:** fast.
- **What it checks:** regression for a topic mix-up: with `atm=1` and `dm=2`, at iteration 0 only the atmosphere topic is due, so only `atmosphere_1/layer_1` is published with the atmosphere screen, and the DM command is not published under an atmosphere topic.
- **Theory / assumptions:** none.
- **Tolerance:** exact.
- **References:** none.

## `test_lightpath_topics_decimation`
- **Module:** `Sharepoint.shareData` (per light path topics).
- **Tier:** fast.
- **What it checks:** for two light paths and decimations (`atm_per_dir`=2, `slopes`=3, `wfs`=4, `wfs_frame`=5, `sci`=6, `sci_frame`=2) over 12 iterations, the set of topics (`lightPath<i>/atmosphere_opd`, `atmosphere_phase`, `slopes_1D`, `slopes_2D`, `wfs_opd`, `wfs_phase`, `wfs_frame`, `sci_opd`, `sci_phase`, `sci_frame`), their bit-equal payloads and publication iterations match the schedule rule at every iteration; `sci_frame` of light path 1 appears at 1, 3, 5, 7, 9, 11 and `wfs_frame` of light path 0 at 4, 9.
- **Theory / assumptions:** none.
- **Tolerance:** exact.
- **References:** none.

## `test_lightpath_payload_types_and_decimation_one`
- **Module:** `Sharepoint.shareData`.
- **Tier:** fast.
- **What it checks:** with decimation 1 (`slopes`, `sci_frame`) topics are published every iteration (4 iterations) and the payload follows the current buffer values (the buffers are refreshed before each call).
- **Theory / assumptions:** none.
- **Tolerance:** exact.
- **References:** none.

## `test_sections_absent_in_light_path_are_not_published`
- **Module:** `Sharepoint.shareData`.
- **Tier:** fast.
- **What it checks:** a light path with `atm`, `dm`, `wfs`, `sci` set to None publishes nothing for those sections even with all the decimations enabled; the catalogue is empty.
- **Theory / assumptions:** none.
- **Tolerance:** exact.
- **References:** none.

## `test_disabled_decimations_publish_nothing`
- **Module:** `Sharepoint.shareData`.
- **Tier:** fast.
- **What it checks:** with all decimations at 0 no data topic is published over 3 iterations (only the empty `topics` message), even with atmosphere and DM objects supplied.
- **Theory / assumptions:** none.
- **Tolerance:** exact.
- **References:** none.

## `test_topics_catalogue`
- **Module:** `Sharepoint.shareData` (final `topics` message).
- **Tier:** fast.
- **What it checks:** the `topics` message lists the configured topics in publication order at every iteration (5 iterations) with two light paths and an atmosphere with two layers, regardless of whether each topic is due at that iteration (decimations atm=2, slopes=3, wfs=1, sci_frame=4).
- **Theory / assumptions:** catalogue lists configured, not just due, topics.
- **Tolerance:** exact list equality.
- **References:** none.

## `test_dm_per_dir_phase_topic_sends_phase`
- **Module:** `Sharepoint.shareData` (`dm_per_dir` with several directions).
- **Tier:** fast.
- **What it checks:** with `dm_per_dir=2` and 3 directions (3D DM buffers), `dm_opd_<j>` carries `dm_opd[j]` and `dm_phase_<j>` carries `dm_phase[j]` for $j=1..3$ at the due iterations, nothing otherwise. Regression for the phase topic sending the OPD buffer.
- **Theory / assumptions:** none.
- **Tolerance:** exact.
- **References:** none.

## `test_dm_per_dir_single_direction_phase_topic`
- **Module:** `Sharepoint.shareData` (`dm_per_dir`, one direction).
- **Tier:** fast.
- **What it checks:** with one direction (2D buffers) and `dm_per_dir=1`, topics `lightPath0/dm_opd` and `lightPath0/dm_phase` are published, each carrying its own buffer.
- **Theory / assumptions:** none.
- **Tolerance:** exact.
- **References:** none.

## `test_dm_per_dir_single_direction_decimation`
- **Module:** `Sharepoint.shareData` (`dm_per_dir`, one direction).
- **Tier:** fast.
- **What it checks:** with `dm_per_dir=3` and one direction, nothing is published at iterations 0, 1, 3, 4, and both topics at 2 and 5.
- **Theory / assumptions:** none (the decimation must also apply in the single-direction branch).
- **Tolerance:** exact.
- **References:** none.

## `test_reception_actuator_cmd_roundtrip`
- **Module:** `ReceptionPoint.sendRequest('actuator_cmd')`.
- **Tier:** fast.
- **What it checks:** the request sent is the pickled string `'actuator_cmd'` and the 37-element answer comes back bit-equal.
- **Theory / assumptions:** none.
- **Tolerance:** exact.
- **References:** none.

## `test_reception_loop_cmd_state_tracking`
- **Module:** `ReceptionPoint.sendRequest('loop_cmd')`.
- **Tier:** fast.
- **What it checks:** the request payload carries `buffer_type` and the previous state (`-1` at the first request, then the last desired state), and the result is `int(desired_state)`; three requests answered 1, 0, 1 produce requests with `actual_state` -1, 1, 0.
- **Theory / assumptions:** none (protocol state tracking).
- **Tolerance:** exact.
- **References:** none.

## `test_reception_unsupported_type_returns_none`
- **Module:** `ReceptionPoint.sendRequest`.
- **Tier:** fast.
- **What it checks:** an unknown buffer type, or no argument, returns None and nothing is sent to the server.
- **Theory / assumptions:** none.
- **Tolerance:** exact.
- **References:** none.

## `test_reception_timeout_returns_none_and_recovers`
- **Module:** `ReceptionPoint.sendRequest` (timeout and recovery).
- **Tier:** fast.
- **What it checks:** parametrised over `buffer_type` = `actuator_cmd`, `loop_cmd`. With no server the request returns None after the timeout (300 ms, bounded time); once a REP server is started on a new port and the socket rebuilt, the next request succeeds with the expected answer.
- **Theory / assumptions:** none (ZMQ REQ/REP lock-step must recover from a timeout by rebuilding the socket).
- **Tolerance:** exact.
- **References:** none.

## `test_reception_poller_only_holds_live_socket`
- **Module:** `ReceptionPoint` (poller and socket rebuild).
- **Tier:** fast.
- **What it checks:** after a timeout the socket is rebuilt and the poller holds exactly the current socket and none of the closed ones. Regression for stale closed sockets in the poller.
- **Theory / assumptions:** none.
- **Tolerance:** exact.
- **References:** none.

## `test_reception_second_timeout_returns_none_in_bounded_time`
- **Module:** `ReceptionPoint.sendRequest`.
- **Tier:** fast.
- **What it checks:** three consecutive timeouts (200 ms each) each return None within a thread deadline of 4 s, so a hang is reported as a failure instead of blocking the suite. Regression for a hang at the second timeout.
- **Theory / assumptions:** none.
- **Tolerance:** thread deadline 4 s; results `[None, None, None]`.
- **References:** none.
