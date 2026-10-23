#!/usr/bin/env python3
"""Generate the real KRAS eight-leaf joined-v2 native-run input, never run CUDA.

The retained original run supplies protocols, wavelength overrides, seeds, NMA
and mass identities. The operator's successor request supplies 40,000
accepted steps per leaf, 500-step windows, a 960-tet budget, common-minimum
adaptive time bounded to0.0005..0.002ps, a64^3 grid at1Angstrom spacing,
held terminal baselines and live ASC. Historical execution is
not counted as successor execution.

Five scientific selections have no production default. --science-settings is
a SHA256-pinned JSON object with exactly center_atom (eight integer IDs),
kappa_foliated, lif_decay, lif_threshold and lif_kick. Test/harness constants
are deliberately not consulted. --inspect reads authorities and lists missing
selections without creating a file, starting CUDA or invoking the executable.

Writing additionally requires a successful *actual cargo build* record, the
actual prism-gpu build-script output, all 95 matching OUT_DIR/mirror modules,
and the governed trust manifest. The completed executable then validates the
temporary input with its existing --validate-only path before publication.
This is input/build binding, not numerical, native-runtime or restart approval.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
import tempfile

def _env_root(name):
    """Absolute root from the environment; there is no compiled-in host default.
    When unset, a non-existent sentinel naming the variable is returned so every
    read fails closed; entry points call _require_env() first."""
    value = os.environ.get(name, '')
    return Path(value) if os.path.isabs(value) else Path('/nonexistent') / (name + '_UNSET')


def _require_env(*names):
    for name in names:
        if not os.path.isabs(os.environ.get(name, '')):
            raise SystemExit(f'{name} must name an absolute path (no compiled-in host default)')


ENGINE = Path(__file__).resolve().parents[1]
TASK = _env_root('PRISM_BIPRISM_TASK_ROOT')
BASELINE = TASK / 'NATIVE_ACCEPTANCE/KRAS_ORIGINAL_WHILE_003'
MATH = TASK / 'INTERNAL_QC/v10_engine_integration/math'
SELECTION_KEYS = {'center_atom', 'kappa_foliated', 'lif_decay', 'lif_threshold', 'lif_kick'}
FULL_STEPS, CHUNK_STEPS, TARGET_TETS = 40000, 500, 960
GRID_DIMENSION, GRID_SPACING_ANGSTROM = 64, 1.0
DT_MIN_PS, DT_MAX_PS = 0.0005, 0.002
MAX_SEED_PEEL_PASSES = 256
SEED_SOURCE = 'crates/prism-gpu/src/kernels/nhs_biprism_bowyer_watson_seed.cuh'
REQUESTED_SEED_RADIUS = 8.5
# At most one multiplicity-bearing packet per core vertex per observation.
QUEUE_CAPACITY = 8 * 192 * CHUNK_STEPS
BASELINE_HASHES = {
    'launch.json': '299c85e4b5ed148f13072931b964809a9d4a6f50fc0913506e0f2f3ff945e0c5',
    'NATIVE_OUTPUT/native_differential_protocol_plan.json': 'd42ecf390def44da508960739f0775a61f17b449b5d736c3ea2a4ae7ff13b530',
    'NATIVE_OUTPUT/multistream_capacity_admission.json': '15882380ef1d74260d2cad34c64ad6ed0ca518ddf9432a1ae21e3f7bfeef5c4e',
    'NATIVE_OUTPUT/kras_stream0_protocol_state.json': '06b10c2601cd20d74a5552b2f025819292b85ad8cab99b83cbcf14e19e4a27f2',
    'NATIVE_OUTPUT/native_may_warmup_stream0.json': '68bbe2fea01731c6c99917b03a012116d7c2fa1aa5213ea93e16c01cb5632695',
    'NATIVE_OUTPUT/kras_stream0_graph_owner_admission.json': '444bd55b663377b7ea53dbd202f82d64398033a74ecd14ad85c40b675c114d8a',
    'NATIVE_OUTPUT/kras_stream1_protocol_state.json': '7d620d37a196e61c991e1780f0246028cd5e51d46eb9f66d420852edf7c82678',
    'NATIVE_OUTPUT/native_may_warmup_stream1.json': 'b1f510f859416e80218f786649f770e6c1f10a6a6438dba646e6ba30d17aa002',
    'NATIVE_OUTPUT/kras_stream1_graph_owner_admission.json': 'c2191e7e8fda2c0d7f648b50dac222980bf62b99d602f536eca0a370a76755d8',
    'NATIVE_OUTPUT/kras_stream2_protocol_state.json': '05355ced070512fd0ed659be4d4cdb75e5a06375f5e7ddc9953e14851b16f750',
    'NATIVE_OUTPUT/native_may_warmup_stream2.json': '91635f47bb0d713d0f41de12ba46214a8c3ffb8b085f73c37ed47dee35254471',
    'NATIVE_OUTPUT/kras_stream2_graph_owner_admission.json': 'b7365bb21ade1cd471289321cdacf5c59432f0585b98363380ff15a4301f2895',
    'NATIVE_OUTPUT/kras_stream3_protocol_state.json': 'b0b0c1917fec446c69b6721bc67d434279ef6697fe10817df38a12559ce9da0e',
    'NATIVE_OUTPUT/native_may_warmup_stream3.json': '29fd6ff0d2a402b0a258ab30467c90681f47bd71e37fe497159bfab6d9ef5b5b',
    'NATIVE_OUTPUT/kras_stream3_graph_owner_admission.json': '9a8a21359d913e44f328a50fb916caae0802cf4fef41e34c733bf1161f1b9190',
    'NATIVE_OUTPUT/kras_stream4_protocol_state.json': '4bf65409e6642c26009ac5b1ebeb4004fa95cf1e4e3de119533eca70e2f2a1d1',
    'NATIVE_OUTPUT/native_may_warmup_stream4.json': '83c13684834cd67ae9cdbc40b5e2bd92eb901c8ee710061de86886a10c579d04',
    'NATIVE_OUTPUT/kras_stream4_graph_owner_admission.json': '2ef3a11a7bc4765c34124dbeb903b4f29f5def30e2ea91eed8caaed7849e340f',
    'NATIVE_OUTPUT/kras_stream5_protocol_state.json': '2cdfbc4f2fab5ca3107531d9415117d0833f27983b22399f97368741400accf9',
    'NATIVE_OUTPUT/native_may_warmup_stream5.json': '0dd2302d0834f57a5be48f63ba315269dbfd1ad35fb7cc8e79a67dbbddbfcb62',
    'NATIVE_OUTPUT/kras_stream5_graph_owner_admission.json': '4502514d467202c662ca5dfbcb8cae0869850fc0926f9263b6a078e526eb2fc7',
    'NATIVE_OUTPUT/kras_stream6_protocol_state.json': 'a993eebf8b3bdee60f39d33569aec98d7a33797f8a1088976c7fa6fd60866b61',
    'NATIVE_OUTPUT/native_may_warmup_stream6.json': '10323a3f5b77eaff98b2907e1829295480ea86e87884259e6c37f4ca38fa0744',
    'NATIVE_OUTPUT/kras_stream6_graph_owner_admission.json': 'fdf7982bd974e3ff3e401d53fcab6b1b41998ad834c8b852bf37269da6664a7c',
    'NATIVE_OUTPUT/kras_stream7_protocol_state.json': '8414a54cd0870e92c08c2f55f10f2c71411ce7469dc06f5008e9cafe119afe69',
    'NATIVE_OUTPUT/native_may_warmup_stream7.json': '2dd0b9839d7248b3ba978d7f6fe1ff8ca6694530b2958f2e45f6cbd53a87d30a',
    'NATIVE_OUTPUT/kras_stream7_graph_owner_admission.json': '599f04aa1e049377a48e04722531184a98f26f51d7227d04f1df6d99d8372c58',
}
MAPPING_SHA = 'b9d1fa3e0fe8c14e614e3420408deafbfc2bb306596e880500a377eee26bf241'
INVERSE_SHA = '37e070f483d07f186cddc1d8ac2cdf2a8339b57d0bf6884742864b54bfedc772'
CENTER_INPUT = 'INTERNAL_QC/v10_multiplicity_revision/native/host_controls/KRAS_INPUT.bpqc'
CENTER_SELECTION = 'INTERNAL_QC/v10_bowyer_native/implementation/direct_qualification_001/seeded_tiles.bin.selection.tsv'
CENTER_INPUT_SHA = 'eff6c982d8cdbd9f77e6e4ecf37852fb33ca486712747079e3a0e42efaf5bd54'
CENTER_SELECTION_SHA = 'b32acda57f813d8b24b013826a3f8cc635917e1c02a95c6e1004a63ca40d69f9'
CENTER_AUTHORITY_HASHES = {
    'INTERNAL_QC/v10_bowyer_native/implementation/direct_qualification_001/EXECUTION.json':
        '258fe5ef9cd2cde10bbfb3bf72e257bd30a70a73d287f6dc4ca9408288306a17',
    'INTERNAL_QC/v10_engine_integration/dec_bounded/native_qualification_001/EXECUTION.json':
        '46c84d51f0023dd781c651daafdac0752caf462202a57a1b9364ba3feaa1354c',
    'INTERNAL_QC/v10_engine_integration/native_stage_trace/native_qualification_001/EXECUTION.json':
        'f1660d184ee4d3007454526950f68bed111923b42d34132cc96e39a70ea4ef3a',
    CENTER_INPUT + '.json': '33cb045e34f4e2e80cba6bf6a8d1c89a5cd9a114b67baf8415dfac756fb1da3e',
}


class Refusal(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Refusal(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f'duplicate JSON key: {key}')
            result[key] = value
        return result
    def constant(value):
        raise Refusal(f'nonfinite JSON constant: {value}')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def read_file(path, expected=None):
    path = Path(path)
    require(path.is_absolute(), f'path is not absolute: {path}')
    require(stat.S_ISREG(path.lstat().st_mode), f'not a regular, nonsymlink file: {path}')
    data = path.read_bytes()
    if expected is not None:
        require(re.fullmatch(r'[0-9a-f]{64}', expected) is not None, f'invalid SHA256 for {path}')
        require(sha(data) == expected, f'SHA256 mismatch: {path}')
    return data


def pin(path, expected=None):
    path = Path(path)
    return {'path': str(path), 'sha256': sha(read_file(path, expected))}


def flag_value(argv, flag):
    require(argv.count(flag) == 1, f'original launch must contain exactly one {flag}')
    index = argv.index(flag)
    require(index + 1 < len(argv), f'original launch lacks value for {flag}')
    return argv[index + 1]


def protocol_from_records(state, row, leaf):
    require(state['stream_index'] == leaf and row['stream_id'] == leaf, 'protocol leaf identity differs')
    names = ['cold_hold', 'ramp', 'warm_hold', 'ramp_down', 'cold_return']
    duration = [row['phase_steps'][key] for key in names]
    require(all(type(v) is int and v >= 0 for v in duration), 'invalid recorded protocol duration')
    ends, total = [], 0
    for value in duration:
        total += value
        ends.append(total)
    actual = [state[k] for k in ['cold_hold_end', 'ramp_end', 'warm_hold_end', 'ramp_down_end', 'total_steps']]
    require(ends == actual and total == row['nominal_protocol_steps'], 'saved phase boundaries disagree')
    require(state['uv_burst_mode'] == 'protocol_schedule', 'unsupported recorded UV mode')
    return {
        'start_temp': state['start_temp_K'], 'end_temp': state['end_temp_K'],
        'cold_hold_steps': duration[0], 'ramp_steps': duration[1], 'warm_hold_steps': duration[2],
        'current_step': 0,
        'uv_burst_energy': state['uv_burst_energy_kcal_mol'],
        'uv_burst_interval': state['uv_burst_interval_steps'],
        'uv_burst_duration': state['uv_burst_duration_steps'],
        'scan_wavelengths': state['scan_wavelengths_nm'],
        'wavelength_dwell_steps': state['wavelength_dwell_steps'],
        'ramp_down_steps': duration[3], 'cold_return_steps': duration[4], 'stepped_holds': [],
    }


def centers_from_selection(raw, center, topology):
    """Read the actual eight-leaf selection; never derive a new pocket or mesh."""
    rows = list(csv.DictReader(io.StringIO(raw.decode()), delimiter='\t'))
    require(len(rows) == 8 * 256, 'retained selection must contain all eight256-vertex populations')
    populations = [[] for _ in range(8)]
    for row in rows:
        leaf, vertex, atom, core, depth = (int(row[k]) for k in ['leaf', 'vertex', 'atom', 'core', 'depth'])
        require(0 <= leaf < 8 and 0 <= vertex < 256 and 0 <= atom < topology['n_atoms'],
                'retained selection identity is outside the real topology')
        require(core == int(vertex < 192) and depth >= 0, 'retained selection core/depth contract differs')
        populations[leaf].append((vertex, atom, depth))
    result = []
    for leaf, population in enumerate(populations):
        population.sort()
        require([p[0] for p in population] == list(range(256)) and len({p[1] for p in population}) == 256,
                f'leaf{leaf} selection is missing or duplicates a vertex/atom')
        require(population[0][1:] == (center, 0), f'leaf{leaf} center differs from its actual native input')
        core = {p[1] for p in population[:192]}
        for group in topology['h_clusters']:
            members = {group['central_atom'], *group['hydrogen_atoms'][:group['n_hydrogens']]}
            require(members <= core or members.isdisjoint(core), f'leaf{leaf} core splits a native constraint group')
        result.append(population[0][1])
    return result


def load_center_authority(topology_pin, task=TASK):
    records = {name: strict_json(read_file(task / name, digest)) for name, digest in CENTER_AUTHORITY_HASHES.items()}
    meta = records[CENTER_INPUT + '.json']
    require(meta['source'] == topology_pin, 'diagnostic centers target a different topology')
    raw = read_file(task / CENTER_INPUT, CENTER_INPUT_SHA)
    require(raw[:8] == b'BPCASE02' and len(raw) >= 32, 'native center transport ABI differs')
    atoms, center, exclusions, groups, hmr, residues = struct.unpack('<6I', raw[8:32])
    require((atoms, center, exclusions, groups, hmr, residues) ==
            (meta['n_atoms'], meta['center_atom_id'], meta['exclusion_entries'], meta['constraint_groups'],
             int(meta['hmr_applied']), meta['n_residues']), 'native center transport/metadata disagree')
    require(meta['transport'] == {'path': str(task / CENTER_INPUT), 'sha256': CENTER_INPUT_SHA, 'bytes': len(raw)},
            'native center transport metadata is stale')
    topology = strict_json(read_file(Path(topology_pin['path']), topology_pin['sha256']))
    require(atoms == topology['n_atoms'] and groups == len(topology['h_clusters'])
            and 0 <= center < atoms, 'center topology/constraint population differs')
    for name, record in records.items():
        if not name.endswith('/EXECUTION.json'):
            continue
        require(record['input_hashes'].get(str(task / CENTER_INPUT)) == CENTER_INPUT_SHA
                and str(task / CENTER_INPUT) in record['command']
                and record['source_binary_inputs_unchanged'] is True,
                'native center execution does not bind the retained transport')
        if '/v10_engine_integration/' in name:
            require(record['input_hashes'].get(str(task / CENTER_SELECTION)) == CENTER_SELECTION_SHA
                    and str(task / CENTER_SELECTION) in record['command'],
                    'later native execution does not bind the same selection')
    centers = centers_from_selection(read_file(task / CENTER_SELECTION, CENTER_SELECTION_SHA), center, topology)
    return {
        'center_atom': centers,
        'scope': meta['center_scope'],
        'atom_identity': {'atom': center, 'name': topology['atom_names'][center],
                          'residue_id': topology['residue_ids'][center],
                          'residue_name': topology['residue_names'][center]},
        'pins': {str(task / name): digest for name, digest in {
            **CENTER_AUTHORITY_HASHES, CENTER_INPUT: CENTER_INPUT_SHA, CENTER_SELECTION: CENTER_SELECTION_SHA}.items()},
        'historical_exit_codes': {name: record['exit_code'] for name, record in records.items() if name.endswith('/EXECUTION.json')},
        'production_geometry_requalification_required': True,
    }


def load_authorities(engine=ENGINE, baseline=BASELINE, mass_root=MATH):
    records = {name: strict_json(read_file(baseline / name, digest)) for name, digest in BASELINE_HASHES.items()}
    launch = records['launch.json']
    argv = launch['native_argv']
    require('--multi-differential' in argv and flag_value(argv, '--multi-stream') == '8', 'not the recorded eight-leaf plan')
    require('--multi-scale' not in argv, 'multiscale requires its actual consumer; no scalar substitution')
    require('--clustering-backend' not in argv, 'explicit original backend needs a matching consumer')
    require('--adaptive-dt' in argv and '--asymmetric-steering' in argv, 'requested adaptive/ASC authority absent')
    plan = records['NATIVE_OUTPUT/native_differential_protocol_plan.json']
    require(plan['shared_horizon_steps'] == FULL_STEPS and len(plan['streams']) == 8, 'original protocol population differs')
    # Existing production auto->GPU hash semantics on sm_120. Do not use the
    # factory's test epsilon or substitute a multiscale radius.
    persistent = read_file(engine / 'crates/prism-nhs/src/persistent_engine.rs').decode()
    cluster_body = persistent.split('pub fn cluster_spikes(', 1)[1].split('pub fn cluster_ghost_phase_lattice(', 1)[0]
    require('let epsilon = 5.0_f32;' in cluster_body and 'gcb::resolve_auto' in cluster_body,
            'native automatic clustering contract changed; review its real owner')
    fused = read_file(engine / 'crates/prism-nhs/src/fused_engine.rs').decode()
    for method in ['twin_thermal_shock', 'twin_equilibrium_observer', 'twin_uv_aromatic_probe', 'twin_hysteresis_probe']:
        body = fused.split(f'pub fn {method}() -> Self', 1)[1].split('\n    }', 1)[0]
        require('stepped_holds: vec![]' in body, 'recorded empty stepped-hold authority changed')
    rows = sorted(plan['streams'], key=lambda r: r['stream_id'])
    protocols, dt_bits, alphas = [], [], []
    for leaf in range(8):
        state = records[f'NATIVE_OUTPUT/kras_stream{leaf}_protocol_state.json']
        protocols.append(protocol_from_records(state, rows[leaf], leaf))
        warmup = records[f'NATIVE_OUTPUT/native_may_warmup_stream{leaf}.json']
        require(warmup['stream_id'] == leaf, 'base timestep leaf identity differs')
        dt_bits.append(warmup['adaptive_base_dt_ps_bits'])
        owner = records[f'NATIVE_OUTPUT/kras_stream{leaf}_graph_owner_admission.json']
        require(owner['source_stream_id'] == leaf, 'ASC source leaf identity differs')
        alphas.append(owner['asc_control_alpha'])
    require(len(set(dt_bits)) == 1 and len(set(alphas)) == 1, 'recorded base dt or ASC gain differs across leaves')
    native_cli = read_file(engine / 'crates/prism-nhs/src/bin/nhs_rt_full.rs').decode()
    require('default_value="legacy-whole-system"' in native_cli and '--asc-support-mode' not in argv,
            'original default ASC support authority changed')
    capacity = records['NATIVE_OUTPUT/multistream_capacity_admission.json']
    grids = capacity['owners']
    require(len(grids) == 8 and sorted(g['stream_id'] for g in grids) == list(range(8)), 'recorded grid owners differ')
    require(len({(g['grid_dim'], g['grid_spacing']) for g in grids}) == 1, 'eight recorded grids differ')
    top_path = Path(flag_value(argv, '--topology'))
    nma_path = Path(flag_value(argv, '--nma-perturb'))
    topology = pin(top_path, launch['runtime_file_hashes'][str(top_path)])
    nma = pin(nma_path, launch['runtime_file_hashes'][str(nma_path)])
    mapping_path = mass_root / 'KRAS_PHYSICAL_MASS_MAPPING.json'
    mapping = strict_json(read_file(mapping_path, MAPPING_SHA))
    require(mapping['target_topology']['sha256'] == topology['sha256'], 'mass map targets a different topology')
    authority = mapping['mass_authority']
    return {
        'topology': topology, 'atom_count': mapping['n_atoms'],
        'center_authority': load_center_authority(topology),
        'masses': {'authority': pin(Path(authority['path']), authority['sha256']),
                   'mapping': pin(mapping_path, MAPPING_SHA),
                   'true_inverse_masses_au': pin(mass_root / 'kras_true_inv_masses_atomic.f32le', INVERSE_SHA)},
        'protocols': protocols, 'base_seed': int(flag_value(argv, '--replica-seed')),
        'nma': {'modes': nma, 'amplification': float(flag_value(argv, '--nma-amplification')),
                'scan_fraction': float(flag_value(argv, '--nma-scan-fraction'))},
        'dt_ps': struct.unpack('<f', struct.pack('<I', dt_bits[0]))[0],
        'asc': {'kind': 'explicit_legacy_whole_system', 'alpha': alphas[0]},
        'grid_dimension': grids[0]['grid_dim'], 'grid_spacing_angstrom': grids[0]['grid_spacing'],
        'chronicle_max_bytes': int(flag_value(argv, '--md-chronicle-max-bytes')),
        'source_percentile': int(flag_value(argv, '--spike-percentile')),
        'baseline_pins': {str(baseline / k): v for k, v in BASELINE_HASHES.items()},
    }


def load_harmonic_binding(a, path, expected):
    binding_pin = pin(path, expected)
    source = ENGINE / 'scripts/bind_biprism_harmonic_source.py'
    spec = importlib.util.spec_from_file_location('biprism_harmonic_source', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.verify_binding(binding_pin, a['topology'])
    return dict(a, harmonic_source_binding=binding_pin, harmonic_binding_authority=manifest)


def validate_science(values, atoms):
    require(isinstance(values, dict), 'scientific selections must be a JSON object')
    missing, extra = SELECTION_KEYS - values.keys(), values.keys() - SELECTION_KEYS
    require(not missing and not extra, f'missing scientific keys={sorted(missing)}; unknown keys={sorted(extra)}')
    centers = values['center_atom']
    require(isinstance(centers, list) and len(centers) == 8 and
            all(type(v) is int and 0 <= v < atoms for v in centers), 'center_atom requires eight valid physical atom IDs')
    represented = {}
    for key in SELECTION_KEYS - {'center_atom'}:
        value = values[key]
        require(type(value) in (int, float), f'{key} must be a finite number')
        try:
            require(math.isfinite(value), f'{key} must be finite')
            rounded = struct.unpack('<f', struct.pack('<f', value))[0]
        except OverflowError as error:
            raise Refusal(f'{key} is outside FP32 range') from error
        require(math.isfinite(rounded), f'{key} is outside FP32 range')
        represented[key] = rounded
    require(values['kappa_foliated'] != 0 and represented['kappa_foliated'] != 0,
            'physical foliation must not be silently disabled by value or FP32 underflow')
    require(0 <= values['lif_decay'] <= 1, 'lif_decay is outside [0,1]')
    require(all(values[key] > 0 and represented[key] > 0 for key in ['lif_threshold', 'lif_kick']),
            'threshold and live feedback kick must remain positive in FP32')


def operator_authority(a, science):
    """Keep requested and superseded values outside the strict runnable schema."""
    return {
        'harmonic_source_binding': a.get('harmonic_source_binding'),
        'harmonic_execution': {'historical_producer_verification': 'UNKNOWN',
            'current_contract': 'OPENMM_FULL_K_HALF_ENERGY',
            'target_contract': 'AMBER_NO_HALF_ENERGY',
            'bond_scale': 0.5, 'angle_scale': 0.5,
            'binding_status': 'BOUND' if a.get('harmonic_source_binding') else 'MISSING',
            'scope': 'current exact-row provider normalization; historical literal-upload equivalence is not asserted'},
        'operator_request_date': '2026-10-18',
        'scientific_selections': science,
        'center_authority': a['center_authority'],
        'accepted_steps': {'per_leaf': FULL_STEPS, 'leaves': 8, 'chunk_steps': CHUNK_STEPS,
                           'chunks': FULL_STEPS // CHUNK_STEPS,
                           'full_step_budget_field': FULL_STEPS,
                           'meaning_of_requested_960': 'seeded tetrahedron budget, not accepted physical steps'},
        'grid': {'dimension': GRID_DIMENSION, 'spacing_angstrom': GRID_SPACING_ANGSTROM,
                 'superseded_retained_dimension': a['grid_dimension'],
                 'superseded_retained_spacing_angstrom': a['grid_spacing_angstrom']},
        'time': {'governor': 'armed_adaptive', 'initial_request_ps': DT_MAX_PS,
                 'dt_min_ps': DT_MIN_PS, 'dt_max_ps': DT_MAX_PS,
                 'superseded_retained_initial_request_ps': a['dt_ps'],
                 'selection': 'minimum of all eight raw Director requests capped at maximum; any request below minimum refuses'},
        'mesh': {'target_tetrahedra': TARGET_TETS,
                 'requested_peel_passes': MAX_SEED_PEEL_PASSES,
                 'peel_binding': 'kernel pre-scan limit BPBW_MAX_PEEL_PASSES=256 plus post-seed admission of actual measured peel_passes',
                 'peel_kernel_source': pin(ENGINE / SEED_SOURCE),
                 'requested_radius': REQUESTED_SEED_RADIUS,
                 'radius_binding': 'UNRESOLVED: cited native source/receipts define no radius cutoff or unit; no runtime field emitted',
                 'clustering_epsilon_angstrom': 5.0,
                 'clustering_scope': 'separate retained native clustering parameter; not the requested seeder radius'},
        'tolerances': {'operator_ceiling': 1e-6, 'picard_residual': 1e-7,
                       'dec_residual': 1e-6, 'absolute_energy_drift': 1e-6,
                       'unitarity_defect': 1e-7,
                       'policy': 'existing stricter Picard and unitarity gates retained'},
    }


def build_config(a, science, run_id, output_root, generation, device, modules, engine=ENGINE):
    validate_science(science, a['atom_count'])
    require(science['center_atom'] == a['center_authority']['center_atom'],
            'selected centers differ from the retained actual eight-leaf center authority')
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,128}', run_id) is not None, 'invalid run_id')
    require(output_root.is_absolute() and not output_root.exists(), 'run output root must be absolute and unused')
    require(type(generation) is int and 0 < generation < 2**64, 'physical_time_generation must be a nonzero u64')
    require(type(device) is int and device >= 0, 'invalid device ordinal')
    require(a.get('harmonic_source_binding') is not None,
            'original topology requires an explicit exact-row harmonic source binding')
    leaves = []
    for leaf in range(8):
        seed = a['base_seed'] + leaf * 12345  # actual native Chronicle/replica seed law
        require(seed < 2**64, 'native replica seed overflow')
        leaves.append({'seed': seed, 'stream_id': leaf, 'center_atom': science['center_atom'][leaf],
                       'protocol': a['protocols'][leaf], 'uv_enabled': True, 'nma': a['nma'], 'asc': a['asc']})
    return {
        'schema': 'prism.biprism.native_run.v1', 'run_id': run_id, 'output_root': str(output_root),
        'device_ordinal': device, 'topology': a['topology'], 'masses': a['masses'],
        'harmonic_source_binding': a['harmonic_source_binding'],
        'unit_bridge_source': pin(engine / 'crates/prism-gpu/src/kernels/nhs_biprism_unit_bridge.cuh'),
        'modules': modules, 'leaves': leaves,
        'numerical': {'dt_ps': DT_MAX_PS, 'kappa_foliated': science['kappa_foliated'],
                      'constraint_policy': 'ConstrainedCore', 'gonzalez_exact_energy': True,
                      'lif_decay': science['lif_decay'], 'lif_threshold': science['lif_threshold'],
                      'lif_kick': science['lif_kick'], 'picard_residual': 1e-7, 'dec_residual': 1e-6,
                      'absolute_energy_drift': 1e-6, 'unitarity_defect': 1e-7, 'dec_iteration_cap': 192},
        'governor': 'armed_adaptive', 'physical_time_generation': generation,
        'adaptive_dt_bounds': {'dt_min_ps': DT_MIN_PS, 'dt_max_ps': DT_MAX_PS},
        'director_adaptive_dt': True, 'hold_terminal_controls': True,
        'required_surfaces': ['amber', 'protocol', 'uv', 'constraints', 'work', 'v10', 'voronoi_lif',
                              'native_observation', 'arrow_v5', 'full_ledger', 'chronicle', 'four_grids',
                              'pathb', 'clustering', 'nma', 'asc', 'governor'],
        'grid_dimension': GRID_DIMENSION, 'grid_spacing_angstrom': GRID_SPACING_ANGSTROM,
        'target_tetrahedra': TARGET_TETS, 'max_seed_peel_passes': MAX_SEED_PEEL_PASSES,
        'queue_capacity': QUEUE_CAPACITY,
        'chronicle_max_bytes': a['chronicle_max_bytes'], 'chunk_steps': CHUNK_STEPS,
        'full_step_budget': FULL_STEPS, 'clustering_epsilon_angstrom': 5.0,
        'source_percentile': a['source_percentile'], 'wavefront_max_dt_steps': None,
    }


def validate_build_record(record, engine):
    argv = record.get('command', record.get('argv'))
    require(isinstance(argv, list) and all(isinstance(x, str) for x in argv), 'build record requires actual argv')
    require(len(argv) > 1 and Path(argv[0]).name == 'cargo' and argv[1] == 'build', 'a check/test/proxy is not a production cargo build')
    require(type(record.get('exit_code')) is int and record['exit_code'] == 0 and
            record.get('sources_unchanged', True) is True, 'production build failed or source changed')
    require('--release' in argv and flag_value(argv, '--bin') == 'nhs_rt_full' and
            flag_value(argv, '-p') == 'prism-nhs', 'wrong production executable/profile')
    require('--no-default-features' in argv and set(flag_value(argv, '--features').split(',')) ==
            {'gpu', 'ensemble', 'v2_ignition'}, 'different production feature set')
    require(Path(record['cwd']).resolve() == engine.resolve(), 'build ran in a different workspace')
    start = dt.datetime.fromisoformat(record['started_utc'].replace('Z', '+00:00'))
    end = dt.datetime.fromisoformat(record['ended_utc'].replace('Z', '+00:00'))
    require(start.tzinfo is not None and end.tzinfo is not None and end >= start, 'invalid build interval')
    return start.timestamp(), end.timestamp()


def executable_identity(engine):
    binary = engine / 'target/release/nhs_rt_full'
    require(binary.is_absolute(), 'production executable path must be absolute')
    fields = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    with os.fdopen(os.open(binary, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size > 0 and before.st_mode & 0o111,
                'production executable must be a nonempty regular executable')
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        after = os.fstat(stream.fileno())
        current = binary.lstat()
        require(all(getattr(before, key) == getattr(after, key) == getattr(current, key)
                    for key in fields), 'production executable changed while reading its identity')
    return {'path': str(binary), 'sha256': digest.hexdigest(), 'size': before.st_size,
            'mtime_ns': before.st_mtime_ns, 'device': before.st_dev, 'inode': before.st_ino,
            'mode': before.st_mode, 'ctime_ns': before.st_ctime_ns}


def verify_build_binary(engine, record):
    expected = record.get('binary_after')
    require(isinstance(expected, dict), 'production build record lacks binary_after')
    require(isinstance(expected.get('path'), str) and
            isinstance(expected.get('sha256'), str) and
            re.fullmatch(r'[0-9a-f]{64}', expected['sha256']) is not None and
            type(expected.get('size')) is int and expected['size'] > 0 and
            type(expected.get('mtime_ns')) is int and expected['mtime_ns'] >= 0,
            'invalid production build binary_after identity')
    actual = executable_identity(engine)
    for key in ('path', 'sha256', 'size', 'mtime_ns'):
        require(actual[key] == expected[key],
                f'production executable {key} differs from build binary_after')
    return actual


def require_unchanged_executable(engine, expected, boundary):
    require(executable_identity(engine) == expected,
            f'production executable changed {boundary}')


def verify_build(engine, record_path, output_path, trust_path, trust_class):
    record = strict_json(read_file(record_path))
    start, end = validate_build_record(record, engine)
    factory = read_file(engine / 'crates/prism-nhs/src/biprism_run.rs').decode()
    require(all(name in factory for name in ['pub physical_time_generation:', 'pub director_adaptive_dt:',
                                             'pub hold_terminal_controls:', 'pub adaptive_dt_bounds:',
                                             'pub max_seed_peel_passes:', 'ArmedAdaptive']),
            'joined bounded-adaptive shared-time/terminal factory is not installed')
    seed = read_file(engine / SEED_SOURCE).decode()
    peel_limit = re.findall(r'\bstatic\s+constexpr\s+uint32_t\s+BPBW_MAX_PEEL_PASSES\s*=\s*(\d+)u\s*;', seed)
    require(peel_limit == [str(MAX_SEED_PEEL_PASSES)], 'configured peel bound differs from the compiled seeder source')
    source_manifest_path = engine / 'crates/prism-gpu/GOVERNED_PTX_MEMBERS.json'
    members = strict_json(read_file(source_manifest_path))
    rows = members['members']
    names = {r['ptx'] for r in rows}
    require(members['contract_member_count'] == len(rows) == len(names) == 95 and
            sum(r['kind'] == 'standard' for r in rows) == 92 and
            sum(r['kind'] == 'cooperative' for r in rows) == 3, 'governed95 membership differs')
    output = read_file(output_path).decode()
    require('cargo:info=PTX compilation completed successfully' in output and
            'cargo:info=Using CUDA target arch: sm_120' in output, 'actual95 sm120 compilation did not finish')
    compiled = re.findall(r'^cargo:info=PTX compiled(?: \(cooperative\))?: (.+)$', output, re.MULTILINE)
    mirror = (engine / 'target/ptx').resolve()
    require(len(compiled) == 95 and {Path(p).name for p in compiled} == names and
            all(Path(p).resolve().parent == mirror for p in compiled), 'build output is not the actual canonical95 mirror')
    out_dir = output_path.parent / 'out/ptx'
    require(output_path.resolve().is_relative_to(engine / 'target/release/build'), 'OUT_DIR build evidence is outside canonical release tree')
    trust = strict_json(read_file(trust_path))
    membership = trust['governed_ptx_membership']
    require(membership['count'] == 95 and membership['source_owned_manifest'] == pin(source_manifest_path), 'trust membership source differs')
    require(set(membership['per_kernel_sha256']) == names, 'trust member population differs')
    trust_rows = trust['ptx_hashes']
    require(len(trust_rows) == 95 and {r['name']: r['sha256'] for r in trust_rows} == membership['per_kernel_sha256'], 'trust PTX rows differ')
    for directory in [mirror, out_dir]:
        require({p.name for p in directory.glob('*.ptx')} == names, f'extra or missing PTX members: {directory}')
        for name in sorted(names):
            path = directory / name
            data = read_file(path, membership['per_kernel_sha256'][name])
            require(len(data) > 0 and b'.version' in data[:8192] and b'.target sm_120' in data[:8192], f'invalid sm120 PTX: {path}')
            require(read_file(Path(str(path) + '.sha256')).decode().strip() == sha(data), f'PTX sidecar differs: {path}')
            require(start <= path.stat().st_mtime <= end + 2, f'PTX is outside this completed build interval: {path}')
    digest = sha(''.join(f'{n}:{membership["per_kernel_sha256"][n]}\n' for n in sorted(names)).encode())
    require(membership['fused_sha256'] == digest, 'governed aggregate digest differs')
    # This task requires a complete fresh source build. mtime is only a stale
    # input rejection; actual output bytes are separately pinned above.
    for crate in ['prism-gpu', 'prism-nhs', 'prismcode']:
        root = engine / 'crates' / crate
        for path in root.rglob('*'):
            if path.is_file() and (path.suffix in {'.rs', '.cu', '.cuh'} or path.name == 'Cargo.toml'):
                require(path.stat().st_mtime <= start, f'source changed during/after this build: {path}')
    for path in [engine / 'Cargo.toml', engine / 'Cargo.lock', source_manifest_path]:
        require(path.stat().st_mtime <= start, f'build authority changed: {path}')
    binary_identity = verify_build_binary(engine, record)
    require(start <= binary_identity['mtime_ns'] / 1e9 <= end + 2,
            'executable is outside the completed build interval')
    return {'trust_manifest': pin(trust_path), 'trust_class': trust_class, 'ptx_directory': str(mirror),
            'biprism_sha256': membership['per_kernel_sha256']['nhs_biprism_native.ptx'],
            'fused_sha256': membership['per_kernel_sha256']['nhs_amber_fused.ptx']}, binary_identity


def publish_validated(config, destination, engine, binary_identity):
    require(destination.is_absolute() and not destination.exists() and not destination.is_symlink(), 'configuration destination must be absolute and unused')
    require(destination.parent.is_dir(), 'configuration parent directory must already exist')
    require('PRISMOS_QNANL_FORCE_KERNEL_PACKET' not in os.environ, 'different QNANL packet authority is selected')
    for key in ['PRISM4D_PTX_DIR', 'PRISM_PTX_DIR']:
        if key in os.environ:
            require(Path(os.environ[key]).resolve() == (engine / 'target/ptx').resolve(), f'{key} conflicts with governed mirror')
    raw = (json.dumps(config, indent=2, allow_nan=False) + '\n').encode()
    fd, temporary = tempfile.mkstemp(prefix='.biprism-config-', suffix='.json', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        require_unchanged_executable(engine, binary_identity, 'before --validate-only')
        completed = subprocess.run([str(engine / 'target/release/nhs_rt_full'), 'biprism-run',
                                    '--config', temporary, '--validate-only'],
                                   cwd=engine, capture_output=True, text=True, timeout=120, check=False)
        require_unchanged_executable(engine, binary_identity, 'during --validate-only')
        require(completed.returncode == 0, f'actual input validator refused: {completed.stderr[-8000:]}')
        result = strict_json(completed.stdout.strip())
        require(result.get('status') == 'INPUTS_VERIFIED' and result.get('device_initialized') is False and
                result.get('configuration_sha256') == sha(raw) and result.get('physical_steps_per_leaf') == FULL_STEPS,
                'actual input validator result does not bind this configuration')
        require(not Path(config['output_root']).exists(), 'validation unexpectedly created the run root')
        os.link(temporary, destination)  # exclusive publication; no overwrite
        dirfd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
        return result
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--inspect', action='store_true', help='Read authorities and list missing inputs; write nothing')
    parser.add_argument('--science-settings', type=Path)
    parser.add_argument('--science-settings-sha256')
    parser.add_argument('--harmonic-source-binding', type=Path)
    parser.add_argument('--harmonic-source-binding-sha256')
    parser.add_argument('--run-id')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--physical-time-generation', type=int)
    parser.add_argument('--device-ordinal', type=int, default=0)
    parser.add_argument('--build-record', type=Path, help='Actual successful production cargo build execution JSON')
    parser.add_argument('--gpu-build-output', type=Path, help='Actual target/release/build/prism-gpu-*/output')
    parser.add_argument('--trust-manifest', type=Path)
    parser.add_argument('--trust-class', choices=['signed', 'source_owned_dev_hash_bound'])
    parser.add_argument('--configuration-output', type=Path)
    args = parser.parse_args()
    try:
        a = load_authorities()
        require((args.harmonic_source_binding is None) == (args.harmonic_source_binding_sha256 is None),
                'harmonic source binding requires both path and SHA256')
        if args.harmonic_source_binding is not None:
            a = load_harmonic_binding(a, args.harmonic_source_binding, args.harmonic_source_binding_sha256)
        science = None
        if args.science_settings is not None:
            require(args.science_settings_sha256 is not None, '--science-settings requires its exact SHA256')
            science = strict_json(read_file(args.science_settings, args.science_settings_sha256))
            validate_science(science, a['atom_count'])
        if args.inspect:
            print(json.dumps({'status': 'INPUT_BINDING_ONLY', 'device_initialized': False,
                'missing_scientific_keys': sorted(SELECTION_KEYS) if science is None else [],
                'missing_input_bindings': [] if a.get('harmonic_source_binding') else ['harmonic_source_binding'],
                'protocol_steps': [sum(p[k] for k in ['cold_hold_steps', 'ramp_steps', 'warm_hold_steps', 'ramp_down_steps', 'cold_return_steps']) for p in a['protocols']],
                'protocols': a['protocols'], 'grid': [GRID_DIMENSION, GRID_SPACING_ANGSTROM],
                'initial_native_dt_request_ps': DT_MAX_PS, 'native_auto_gpu_hash_radius_angstrom': 5.0,
                'target_tetrahedra': TARGET_TETS, 'accepted_steps_per_leaf': FULL_STEPS,
                'chunk_steps': CHUNK_STEPS, 'chunks': FULL_STEPS // CHUNK_STEPS,
                'queue_capacity': QUEUE_CAPACITY, 'queue_capacity_basis': '8 leaves *192 core vertices *500 observations, one packet per vertex/observation',
                'constraint_policy': 'ConstrainedCore', 'mass_scope': 'v10 ConstrainedShakeSafe uses serialized HMR masses; feedback uses verified physical masses',
                'asc': a['asc'], 'asc_scope': 'new requested live owner; historical ASC_attached=false is not relabelled',
                'topology': a['topology'], 'masses': a['masses'], 'baseline_pins': a['baseline_pins'],
                'operator_authority': operator_authority(a, science)}, indent=2))
            return 0 if science is not None and a.get('harmonic_source_binding') else 2
        require(science is not None, f'missing scientific keys={sorted(SELECTION_KEYS)}')
        require(a.get('harmonic_source_binding') is not None, 'missing exact-row harmonic source binding')
        required = ['run_id', 'output_root', 'physical_time_generation', 'build_record', 'gpu_build_output',
                    'trust_manifest', 'trust_class', 'configuration_output']
        require(all(getattr(args, key) is not None for key in required),
                f'missing command arguments={[key for key in required if getattr(args, key) is None]}')
        modules, binary_identity = verify_build(ENGINE, args.build_record, args.gpu_build_output,
                                              args.trust_manifest, args.trust_class)
        config = build_config(a, science, args.run_id, args.output_root, args.physical_time_generation,
                              args.device_ordinal, modules)
        result = publish_validated(config, args.configuration_output, ENGINE, binary_identity)
        print(json.dumps({'status': 'CONFIGURATION_WRITTEN_INPUTS_VERIFIED', 'configuration': pin(args.configuration_output),
                          'science_settings': pin(args.science_settings, args.science_settings_sha256),
                          'build_record': pin(args.build_record), 'gpu_build_output': pin(args.gpu_build_output),
                          'binary': binary_identity, 'validation': result,
                          'operator_authority': operator_authority(a, science),
                          'native_run_executed': False, 'scientific_acceptance_claimed': False}, indent=2))
        return 0
    except (Refusal, OSError, KeyError, IndexError, TypeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f'configuration refused: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
