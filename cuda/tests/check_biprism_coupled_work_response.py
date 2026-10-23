"""CPU reconstruction of a bound C1--C4 candidate and its electronic response.

This is independent numerical diagnosis, never native qualification. Unlike a
frozen force comparison, ``solve_electronic`` rebuilds all eight endpoint
densities, charges, Dirichlet DEC solutions and foliated Hamiltonians together.
Every retained input is checked against the immutable native receipt.
"""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np

import biprism_exchange_layout as EX
import check_biprism_frozen_force_terms as F
import check_biprism_nuclear_budget as NB
import check_biprism_spinor_owner as SO
import check_biprism_v10_dec_readback as D


# Precision is selected from the immutable compiled source, never the current
# working file. Historical captures retain their original FP32 node arithmetic.
CONTINUOUS_COORDINATE_PROBE_SOURCES = {"7b7baf5540eb2bf2d8197034fd2f479f36f4f92ca418ff2d70d2a4f4ac6b611c", "e3eae2399b75792464110fd0d4953cef9c8ee8bab9f74c2a1cf2dd5c334774d2", "73743fabf4d3fc8d45fdfbb1c21dfc83c089f900114ce41a922eac30cae8c5b1", "c31e207662fa17f73eac110657baf2042430358ab00179eede04fd8a11135412", "a8883f8e28bd38abf40b181109f73a758a496b5678d6d58ebf08920dd267bcb7", "48e2dd62bea9c9617ca409bb526be415daa2c656b6aad5c48e80244cf4d88e91", "1ba28b83c107ef9fc1c0f1154ddd988de87fa0b9e5a873467e8c94b4f515fd6f"}
ARC_SOURCES = {
    "7b7baf5540eb2bf2d8197034fd2f479f36f4f92ca418ff2d70d2a4f4ac6b611c", #173 base-branch derivative model only; physical map unchanged.
    "e3eae2399b75792464110fd0d4953cef9c8ee8bab9f74c2a1cf2dd5c334774d2", #172 diagnostic-only retained tangent basis; physical equations unchanged.
    "73743fabf4d3fc8d45fdfbb1c21dfc83c089f900114ce41a922eac30cae8c5b1", #170 bounded signed tangent actions; physical map unchanged.
    "c31e207662fa17f73eac110657baf2042430358ab00179eede04fd8a11135412", #169 widen only after prior continuous tangent probes fail; physical map unchanged.
    "a8883f8e28bd38abf40b181109f73a758a496b5678d6d58ebf08920dd267bcb7", #168 measured continuous construction window; physical map unchanged.
    "48e2dd62bea9c9617ca409bb526be415daa2c656b6aad5c48e80244cf4d88e91", #167 bounded represented action-window checks; physical and derivative maps unchanged.
    "1ba28b83c107ef9fc1c0f1154ddd988de87fa0b9e5a873467e8c94b4f515fd6f", #165 continuous-coordinate tangent model; physical/action maps retain pairs.
    "db32d445588beb5a0aa873d009679a48d92d0cae2477aecf56566d4a36d7e441", #164 paired tangent radius above measured coordinate noise floor.
    "a354810f818be13dc8a04f0f5886c737b1e0aedcf650416b858dc9339b6a5b24", #163 tangent probe lifetime/refinement; physical map unchanged.
    "c1a5258d7fa5fd08bb3490b5640f5b66a305f92dffe378b495649d754eb4ea04", #162 paired tangent fallback; physical map unchanged.
    "825fb84bd40ad713f592483a127ae0d641d8c197b0b66b87fc7a45946e488fb5", #161 paired original-h Jacobian probes; physical map unchanged.
    "b5a080ca5d9225c15012b4014e959fddec4c9e261d3117bd663af491baf6e3fa", #160 private work uses endpoint-metric raw kinetic conversion.
    "ee8102e3091b01b93f8eb7608b60899292a232ea4a732e385d47acae3adb45a9", #159 same private work gate; endpoint pass owns conversion reduction.
    "1aced30e3d59a64182c2dae8f1cc879f661bf0c96be26957a2be3601456b924e", #158 private anchor checks the same continuous work as its corrector.
    "7bb852fc8486d59986b738cd6ec4e2795be33899520cc171a7d246916e2a6b00", #157 bounded tangent action probe lengths; physical map unchanged.
    "8db85754e5799f8f54528fa2d5675ca7ed12a8a18c8798dc1536cba4146381b4", #156 measured work bound only; physical map unchanged.
    "8d7513cb27f0e4111d310ee7eb44dda4ce0c887899131ef2a433ad25af3bfa60", #155 bounded derivative rebuilding; physical map unchanged.
    "eeb1230fe626c522628627199fabcd43d4e5aba684a92d69b77336959863da7f", #154 retry only failed private segments; physical map unchanged.
    "d38c54c7d2f28ed4b1a970dc4c72c3f16d881b06079933fa74d8566133281602", #153 measured current-J tangent; original physical equations unchanged.
    "708359de492b8538fd3b3bda19780c58650fce1988cc4d0dfb14357b932d7b6b", # native152: exact original-h predictor intersection
    "0d619b729d18f01a50758b53d43cab0b84c4b338320c72ba7515c4d52847ea56", # native151: measured private-corrector retry, unchanged physical gates
"7f3120a4015571b87b9c37f040df1817a1ee0b9535babfbc2fcedbc7a09c09f2", "a4050d3d16111501f04dbf8221481dfc1854361e9da8f146041161eb54e993a1", "e377ace0ec6bd3307e8e1e53545200bf317de257c4eba0d31a734eb9195bc9f4", "16a08fd580cab96ccd1d7f9f08dbb71969975f7692ef74b5013bef304a786c98", "d8983a2881d9fbffd76dc9ad854e3054a22280d2358e3f8715750d0033fddb53", "060e32c8dd1c1825cbcccba265533e4319c4a97a0b3a33aa978a80ea10c99932", "0f04799b01a06b20426b73f1459c5e2f5f54980efd2289f5138e20b80a132bbf", "9f1743e5105e8fdfda36937bfd4faf6e168eb4b2c21d0ac9bd8e6e8930adbcef", "4e2cd6b8ff951d471ada008c2f9f347d59d21656f41a53c137b099a56bb4af78", "e806fc51c7078b3472ea34aa0ddd5fa60450e5120f57b1bf17181c7f7b7c9230", "b3cfd4a7e39cf025e187b32db17cdbeabf27c069249f98bd58c170f3fd3a8575"}
CONTINUOUS_NEWTON_SOURCES = ARC_SOURCES | {"7c7234e43c031f79afab0ba87d58e55af5cdae79528913aa12c9929b1950e74c"}
MAGNETIC_PHASE64_SOURCES = {"377781e4837a7b96b7ef04367df5460d42a444379189ffb096841f3d9e2852f3"} | CONTINUOUS_NEWTON_SOURCES
MAGNETIC_NODE64_SOURCES = {"5b5ac25dc5deb1ced1c24fda532335bf7772156552e188929f92dbfbe46a2fef"}
MAGNETIC_NODE32_SOURCES = {
    "07c2b26dd255deffe7e43dd29e611f743370c5d7218d752d935e4ed5c8143d54",
    "9d1e64eaa153493938d875a3f673dd92d2c58f2e3d6076ccf6ac1b9c3320765e",
    "527a8328816dfaf3a521e2f1f914f7cb099b2105975b84308f626027e0567f54",
    "a4a939de53956412bf17f3b5cae909ac9dbdb09aebdd45c0ecdc7b3c6abc257b",
    "fd96f71de77b74dab7d7a4160071438d7fc7efbef776d1880c02ad9f0ec4d3c0",
    "69bdfbac4e8767c0f421a10421ba6fd431efeb102cbf1a9988a56fb01fc60dcb",
    "853ae36aaeba0ab423ad0ade2df8cbfc6e8ad98deac80bb83c0c05d4a38498b2",
    "edb40d1a749c9de019742112b7611ea25fe5b84869aced538b2bdd886cd87cf6",
}


def magnetic_nodes_f64(build):
    sources = [v for k, v in build['input_hashes'].items()
               if k.endswith('/nhs_biprism_variational_so3g_v10.cuh')]
    if len(sources) != 1:
        raise ValueError('missing or ambiguous magnetic source owner')
    if sources[0] in MAGNETIC_NODE64_SOURCES | MAGNETIC_PHASE64_SOURCES:
        return True
    if sources[0] in MAGNETIC_NODE32_SOURCES:
        return False
    raise ValueError('unreviewed magnetic arithmetic source')


def density(z):
    cross = z[..., 0] * z[..., 1].conj()
    return np.stack((abs(z[..., 0])**2, abs(z[..., 1])**2,
                     cross.real, cross.imag), axis=-1)


def evaluated_identity(header, trace):
    """Name the evaluated C4 row, never the next proposal in mutable QR state.

    A trial/evaluation limit causes the final nonconvergence fault. Private
    continuation progress is distinct from returning to the original problem.
    This function describes a refusal; it grants no physical acceptance.
    """
    if not trace or len(trace) != header[2]:
        raise ValueError('complete evaluated history required')
    stage, phase = divmod(int(trace[-1][7]), 1000)
    kind, column = divmod(phase, 100)
    labels = {0: 'base', 1: 'derivative_probe', 2: 'correction_trial',
              4: 'action_check', 5: 'action_check', 6: 'tangent_base',
              7: 'tangent_derivative_probe', 8: 'tangent_action_check', 9: 'private_transition'}
    if kind not in labels or stage != header[11] or kind == 9:
        raise ValueError('require an evaluated candidate at the retained live stage')
    return dict(substep=header[1], evaluation=header[2]-1,
                candidate=labels[kind], column=column, stage=stage,
                original_h_reached=NB.original_target_reached(header),
                first_return_to_original_h=next((i for i, row in enumerate(trace)
                                                if int(row[7])//1000 == 16), None),
                exhausted_limits=([name for name, used, limit in
                                   [('nonlinear_trials', header[3], NB.limits(header[0])[1]),
                                    ('full_evaluations', header[2], NB.limits(header[0])[0])]
                                   if used == limit]))


def evaluated_velocity(base, direction, row, pending_mode, pending_columns):
    """Recover the input consumed by C1, before the next proposal overwrote QR.

    On an accepted final correction the producer copies that evaluated input
    into nuclear_base_velocity_f64, resets the columns, and proposes probe0.
    Adding the old trial weight to that new base would apply a second step.
    Refused correction trials retain the old base and correction direction.
    """
    mode = (int(row[7]) % 1000) // 100
    if mode in (0,6):
        if (pending_mode, pending_columns) != (1, 0) or row[6] != 0.:
            raise ValueError('evaluated base lacks the published probe-zero identity')
        return base.copy()
    accepted_trial = mode == 2 and row[3] < (1.-1e-4*row[6])*row[5]
    if accepted_trial:
        if (pending_mode, pending_columns) != (1, 0):
            raise ValueError('accepted correction lacks the published new-base identity')
        return base.copy()
    return base + row[6]*direction


def physical_candidate_gates(scalars, raw, original_h_reached):
    """Evaluate the retained C4 conjuncts, without inventing a rejection.

    This is the scalar part of publication only: earlier validity/DEC/norm
    checks retain their own evidence. Small private h cannot qualify the
    requested original-h problem even when its dimensional residual is tiny.
    """
    energy_tol = float(np.float32(1e-6))
    picard_tol = float(np.float32(1e-7))
    energy = np.sum(scalars[:, 16]+scalars[:, 17])
    checks = [
        ('nuclear_equation', raw, 5e-9),
        ('position_reconstruction', np.max(scalars[:, 8]), picard_tol),
        ('represented_equation', np.max(scalars[:, 13]), picard_tol),
        ('foliated_chain', abs(np.sum(scalars[:, 17]-scalars[:, 20])), energy_tol),
        ('foliated_transfer', abs(np.sum(scalars[:, 19])), energy_tol),
        ('total_energy', abs(energy), energy_tol),
        ('outer_energy', np.max(abs(scalars[:, 22]+energy)), energy_tol),
        ('electronic_berry_foliated_work', abs(np.sum(scalars[:, 12]+scalars[:, 3]+scalars[:, 20])), energy_tol),
        ('coordinate_midpoint_work',
         np.max(abs(scalars[:, 10:12])) if np.all(scalars[:, 4] >= 0.) else float('inf'), 2e-7),
    ]
    gates = [dict(predicate=name, value=float(value), limit=limit,
                  satisfied=bool(np.isfinite(value) and value <= limit))
             for name, value, limit in checks]
    gates.append(dict(predicate='original_timestep_return', satisfied=bool(original_h_reached)))
    return gates


class CoupledResponse:
    def __init__(self, execution, epoch, *, forensic_attempt129=False, terminal_tangent_reference=False):
        self.terminal_tangent_reference=terminal_tangent_reference
        self.receipt = json.loads(Path(execution).read_text())
        build_pin = self.receipt['build_receipt']
        build_raw = Path(build_pin['path']).read_bytes()
        assert hashlib.sha256(build_raw).hexdigest() == build_pin['sha256']
        bound_build = json.loads(build_raw)
        self.magnetic_nodes_f64 = magnetic_nodes_f64(bound_build)
        self.continuous_coordinate_source = any(v in CONTINUOUS_COORDINATE_PROBE_SOURCES for k,v in bound_build["input_hashes"].items() if k.endswith("/nhs_biprism_variational_so3g_v10.cuh"))
        self.continuous_coordinate_probe = False
        self.magnetic_phase_f64 = any(v in MAGNETIC_PHASE64_SOURCES for k,v in
            bound_build['input_hashes'].items() if k.endswith('/nhs_biprism_variational_so3g_v10.cuh'))
        self.continuous_newton = any(v in CONTINUOUS_NEWTON_SOURCES for k,v in
            bound_build['input_hashes'].items() if k.endswith('/nhs_biprism_variational_so3g_v10.cuh'))
        self.prefix = self.receipt['command'][4] + f'.epoch{epoch}'
        self.raw = self.read('numerical.bin')
        self.stride = len(self.raw)//8
        assert self.stride in (1680768,1697920,1784448,1837696,1997440)
        bank = self.read('exchange.bin')
        bank_stride, _, fmt, _ = EX.layout(bank)
        assert fmt == 'd', 'This control requires the live FP64 operator.'
        self.omega = np.array([np.frombuffer(bank, '<f4', 192,
                              l*bank_stride+3072) for l in range(8)], float)
        self.berry_mass = np.maximum(self.omega, float(np.float32(1e-4)))
        owners = self.read('spinor.owners.bps1')
        states = [SO.read(owners, l) for l in range(8)]
        self.z0 = np.array([F.density(np.array(s['start']))[1] for s in states])
        self.z_native = np.array([F.density(np.array(s['candidate']))[1] for s in states])
        self.rho0 = density(self.z0)
        self.pops0 = self.rho0[..., 0]-self.rho0[..., 1]
        tiles = self.read('tiles.bin')
        source = (F.K/'nhs_biprism_xtb_lut.cuh').read_text().split('c_gfn_xtb_lut[8] = {')[1].split('};')[0]
        self.lut = np.array([[np.float32(x) for x in re.findall(r'([-+\d.]+)f', s)]
                            for s in source.splitlines() if '{' in s], float)
        assert hashlib.sha256((F.K/'nhs_biprism_electronic_discrete.cuh').read_bytes()).hexdigest() == D.EXPECTED_ELECTRONIC
        refs = (np.frombuffer(self.read('au_reference_high.bin'), '<f4').astype(float)
                + np.frombuffer(self.read('au_reference_low.bin'), '<f4')).reshape(8, -1, 3)
        mass_path = self.receipt['command'][3]
        mass_raw = Path(mass_path).read_bytes()
        assert hashlib.sha256(mass_raw).hexdigest() == self.receipt['input_hashes'][mass_path]
        masses = (np.float32(1.)/np.maximum(np.frombuffer(mass_raw, '<f4'), np.float32(1e-6))).astype(float)
        self.boundary_phi = np.frombuffer(self.read('potential.prepared.f64le'), '<f8').reshape(8, 256).copy()
        self.edge, self.elem, self.reference, self.mass, self.amber, self.basew = [], [], [], [], [], []
        self.velocity, self.alpha, self.start, self.endpoint = [], [], [], []
        self.headers, self.traces = [], []
        for l in range(8):
            tile = tiles[l*D.TILE_BYTES:(l+1)*D.TILE_BYTES]
            ids, edge, _, _ = D.parse_tile(tile)
            self.edge.append(np.array(edge))
            self.elem.append((np.frombuffer(tile, '<u4', 256)>>24)&7)
            self.reference.append(refs[l, ids])
            self.mass.append(masses[ids])
            self.amber.append(np.frombuffer(tile, '<f4', 1024, 1024).reshape(256, 4)[:, 3].astype(float))
            self.basew.append(np.frombuffer(tile, '<f4', len(edge), 96256).astype(float))
            if forensic_attempt129:
                # This exact rejected build wrote counters into NHE8 reserved
                # words without versioning. Do not weaken the canonical reader
                # or silently call its records conforming. Decode the known
                # historical producer only for explicitly requested forensics.
                source = str(F.K/'nhs_biprism_variational_so3g_v10.cuh')
                assert self.receipt['input_hashes'][source] == '527a8328816dfaf3a521e2f1f914f7cb099b2105975b84308f626027e0567f54'
                assert self.receipt['exit_code'] == 4 and not self.receipt['coupled_diagnostic_accepted']
                base = l*self.stride+NB.OFFSET
                head = struct.unpack_from('<32I', self.raw, base)
                assert head[0] == NB.KRYLOV64_MAGIC and head[1] < 4 and head[2] <= 192 and head[3] <= 48
                assert head[2] == sum(head[3:8]) and head[3] == sum(head[8:10])
                assert head[11] <= 16 and head[12] <= 16 and not any(head[16:])
                assert head[13] <= head[3] and head[14] <= head[13] and head[15] in (0, 1)
                trace = [struct.unpack_from('<9d', self.raw, base+128+72*i) for i in range(head[2])]
                assert np.isfinite(trace).all()
                print('NONCANONICAL_ATTEMPT129_TRACE', l, 'reuse_counters', head[13:16],
                      'canonical_reader_accepts', False, 'native_qualification', False)
            else:
                head, trace = NB.read_trace(self.raw, l)
            self.headers.append(head); self.traces.append(trace)
            if terminal_tangent_reference:
                # A failed tangent model can set the collective fault after
                # C4; the existing loop then retains one unclassified row.
                # Load only the bound raw owners for independent reference
                # experiments. Never relabel this row as a C4 publication.
                assert self.receipt['exit_code']==4 and np.isnan(trace[-1]).all()
                count,_,_,iteration=struct.unpack_from('<4I',self.raw,l*self.stride+106752+605248)
                mode=struct.unpack_from('<I',self.raw,l*self.stride+106752+607632)[0]
                tangent=struct.unpack_from('<I',self.raw,l*self.stride+1680768+17048)[0]
                NB.terminal_tangent_refusal(head,trace,count,mode,tangent,iteration,
                                            self.arr(l,106752+605232,1)[0],self.arr(l,106752+605208,1)[0])
                self.velocity.append(self.arr(l,83712,576).reshape(3,192).T)
                self.alpha.append(float('nan')) # No current evaluated alpha publication.
                self.start.append(np.frombuffer(self.raw,'<f4',192*6,l*self.stride+72960).astype(float).reshape(192,3,2).sum(axis=2))
                self.endpoint.append(np.frombuffer(self.raw,'<f4',256*6,l*self.stride+77568).astype(float).reshape(256,3,2).sum(axis=2))
                continue
            stage, phase = divmod(int(trace[-1][7]), 1000)
            mode, j = divmod(phase, 100)
            if self.continuous_coordinate_source and mode==7:
                refinements=struct.unpack_from("<I",self.raw,l*self.stride+1680768+17052)[0]
                self.continuous_coordinate_probe |= refinements>=2
            assert mode in (0, 1, 2, 5, 6, 7, 8), 'Require matching evaluated base, probe, action check, or actual trial.'
            off = 673768+j*576*8 if mode in (1,7) else 600064
            direction = self.arr(l, 106752+off, 576).reshape(3, 192).T
            pending_columns = struct.unpack_from('<I', self.raw, l*self.stride+106752+605248)[0]
            pending_mode = struct.unpack_from('<I', self.raw, l*self.stride+106752+607632)[0]
            self.velocity.append(evaluated_velocity(
                self.arr(l, 88320, 576).reshape(3, 192).T, direction,
                trace[-1], pending_mode, pending_columns))
            self.alpha.append(trace[-1][2])
            self.start.append(np.frombuffer(self.raw, '<f4', 192*6, l*self.stride+72960).astype(float).reshape(192, 3, 2).sum(axis=2))
            self.endpoint.append(np.frombuffer(self.raw, '<f4', 256*6, l*self.stride+77568).astype(float).reshape(256, 3, 2).sum(axis=2))
        self.velocity, self.alpha, self.start, self.endpoint = map(np.array, (self.velocity, self.alpha, self.start, self.endpoint))
        self.mass, self.amber, self.elem, self.reference = map(np.array, (self.mass, self.amber, self.elem, self.reference))
        self.h = NB.live_h(self.headers[0])
        assert self.h and all(NB.live_h(h) == self.h for h in self.headers)
        command = self.receipt['command']
        self.kappa = next((float(x.split('=')[1]) for x in command if x.startswith('--asymmetric-kappa=')), .1)
        self.kappa = float(np.float32(self.kappa))
        self.charge_factor = float(np.float32(.15))

    def read(self, suffix):
        p = self.prefix+'.'+suffix
        raw = Path(p).read_bytes()
        assert self.receipt['artifacts'][p] == dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return raw

    def arr(self, leaf, offset, n):
        return np.frombuffer(self.raw, '<f8', n, leaf*self.stride+offset).copy()

    def geometry(self, velocity):
        x = self.endpoint.copy()
        x[:, :192] = self.start + .5*self.h*velocity
        return self.geometry_at(x)

    def geometry_at(self, x):
        result = []
        for l, edge in enumerate(self.edge):
            u, v = edge.T
            e, E = x[l, v]-x[l, u], self.reference[l, v]-self.reference[l, u]
            l2 = np.sum(e*e, axis=1)+float(np.float32(1e-12))
            l20 = np.sum(E*E, axis=1)+float(np.float32(1e-12))
            w = np.maximum(self.basew[l], float(np.float32(1e-4)))*np.sqrt(l20/l2)
            A = np.zeros((256, 256))
            np.add.at(A, (u, u), w); np.add.at(A, (v, v), w)
            np.add.at(A, (u, v), -w); np.add.at(A, (v, u), -w)
            boundary = -A[:192, 192:]@self.boundary_phi[l, 192:]
            result.append((x[l], e, l2, w, A[:192, :192], boundary))
        return result

    def potential(self, z, geometry):
        return self.potential_parts(z, geometry).sum(axis=1)

    def potential_parts(self, z, geometry):
        """Valence, stationary Dirichlet DEC, and canonical edge energy."""
        rho = np.zeros((8, 256, 4)); rho[..., 0] = 1.
        rho[:, :192] = density(z)
        q = self.amber.copy(); q[:, :192] += self.charge_factor*rho[:, :192, 1]
        out = np.zeros((8, 3))
        for l, (x, e, l2, w, A, boundary) in enumerate(geometry):
            edge = self.edge[l]; u, v = edge.T
            phi = self.boundary_phi[l].copy()
            phi[:192] = np.linalg.solve(A, q[l, :192]+boundary)
            p = F.provider(x, edge, self.elem[l], q[l], self.lut)
            edge_energy, _ = F.edge_energy_force(p, rho[l, u])
            valence = self.omega[l]*(rho[l, :192, 0]*self.lut[self.elem[l, :192], 1]
                       + rho[l, :192, 1]*self.lut[self.elem[l, :192], 2])
            out[l] = (np.sum(valence),
                      np.sum(q[l, :192]*phi[:192])-.5*np.sum(w*(phi[u]-phi[v])**2),
                      np.sum(edge_energy))
        return out

    def nuclear_map(self, velocity, alpha, *, initial=None, represented_velocity=True,
                    diagnostics=False):
        z, geometry, H, connection, data, _, _ = self.solve_electronic(velocity, initial)
        fb = self.reciprocal(z, H, data)
        mapped = np.zeros_like(velocity); coordinate = np.zeros(8)
        endpoint = self.endpoint.copy(); dtm = np.zeros(8); work = np.zeros(8)
        kinetic_delta = np.zeros(8); midpoint_defect = np.zeros(8)
        metric_minimum = np.zeros((8, 2))
        for l, (x, e, l2, w, _, _) in enumerate(geometry):
            edge = self.edge[l]; u, v = edge.T
            vv = np.zeros((256, 3)); vv[:192] = velocity[l]
            rho, charge, phi, p, _ = data[l]
            fm, g = F.metric_force(x, self.reference[l], vv, self.mass[l], edge)
            fd = (.5*(phi[u]-phi[v])**2*w/l2)[:, None]*e
            _, fx = F.edge_energy_force(p, rho[u])
            fc = F.accumulate(edge, fm+fd+fx)[:192]
            used = fc+fb[l]-alpha[l]*self.h*velocity[l]
            p0 = self.arr(l, 43008, 576).reshape(3, 192).T
            m = self.mass[l, :192]
            vm = np.linalg.solve(g, ((p0+.5*self.h*used)/m[:, None])[..., None])[..., 0]
            tau_full, tau = F.magnetic_rotation(x, vv, edge, self.elem[l], charge, self.lut, self.mass[l], self.h,
                                         field_nodes_f64=self.magnetic_nodes_f64)
            if self.magnetic_phase_f64:
                tau = tau_full
            C = np.zeros((192, 3, 3)); tx, ty, tz = tau[:192].T
            C[:, 0, 1]=tz; C[:, 0, 2]=-ty; C[:, 1, 0]=-tz
            C[:, 1, 2]=tx; C[:, 2, 0]=ty; C[:, 2, 1]=-tx
            vp = np.linalg.solve(g-C, np.einsum('nij,nj->ni', g+C, vm)[..., None])[..., 0]
            mapped[l] = .5*(vm+vp)
            p1 = m[:, None]*np.einsum('nij,nj->ni', g, vp)+.5*self.h*used
            endpoint[l, :192] = self.start[l]+self.h*mapped[l]
            _, ge = F.metric_force(endpoint[l], self.reference[l], vv, self.mass[l], edge)
            ve = np.linalg.solve(ge, (p1/m[:, None])[..., None])[..., 0]
            if represented_velocity:
                ve = ve.astype(np.float32).astype(float)
            te = .5*np.sum(m[:, None]*ve*np.einsum('nij,nj->ni', ge, ve))
            v0 = np.frombuffer(self.raw, '<f4', 576, l*self.stride+2304).astype(float).reshape(3, 192).T
            tn = .5*np.sum(p0*v0)
            tm0 = .5*np.sum(p0*np.linalg.solve(g, (p0/m[:, None])[..., None])[..., 0])
            tm1 = .5*np.sum(p1*np.linalg.solve(g, (p1/m[:, None])[..., None])[..., 0])
            dtm[l] = (te-tm1)-(tn-tm0)
            work[l] = self.h*np.sum((fc-alpha[l]*self.h*velocity[l])*mapped[l])
            if diagnostics:
                kinetic_delta[l] = te-tn
                midpoint_defect[l] = tm1-tm0-self.h*np.sum(used*mapped[l])
                metric_minimum[l] = (np.linalg.eigvalsh(g).min(),
                                     np.linalg.eigvalsh(ge).min())
        initial_x = self.endpoint.copy(); initial_x[:, :192] = self.start
        if not hasattr(self, '_potential_start'):
            self._potential_start = self.potential(self.z0, self.geometry_at(initial_x))
        delta_e = self.potential(z, geometry)-self.potential(self.z0, geometry)
        potential_delta = self.potential(z, self.geometry_at(endpoint))-self._potential_start
        coordinate = dtm+potential_delta-delta_e+work
        if diagnostics:
            def foliated_energy(endpoint_spinor):
                rho = density(endpoint_spinor)
                pop = rho[..., 0]-rho[..., 1]
                return -.5*self.kappa*np.sum((self.omega+np.roll(self.omega, -1, axis=0))
                    *pop*np.roll(pop, -1, axis=0), axis=1)
            foliated_delta = foliated_energy(z)-foliated_energy(self.z0)
            reciprocal_work = self.h*np.sum(fb*mapped, axis=(1, 2))
            joint = delta_e+reciprocal_work+foliated_delta
            direct_energy = kinetic_delta+potential_delta+foliated_delta
            return mapped-velocity, coordinate, z, dict(
                kinetic_delta=kinetic_delta, potential_delta=potential_delta,
                foliated_delta=foliated_delta, midpoint_defect=midpoint_defect,
                joint_electronic_berry_foliated=joint,
                velocity_lag=self.h*np.sum(fb*(mapped-velocity), axis=(1, 2)),
                direct_energy=direct_energy,
                partition_error=direct_energy-(coordinate+midpoint_defect+joint),
                metric_minimum=metric_minimum)
        return mapped-velocity, coordinate, z

    def assemble(self, z, velocity, geometry):
        rho = np.zeros((8, 256, 4)); rho[..., 0] = 1.
        rho[:, :192] = .5*(self.rho0+density(z))
        q = self.amber.copy(); q[:, :192] += self.charge_factor*rho[:, :192, 1]
        pop = rho[:, :192, 0]-rho[:, :192, 1]
        fol = -.5*self.kappa*((np.roll(self.omega, 1, axis=0)+self.omega)*np.roll(pop, 1, axis=0)
                +(self.omega+np.roll(self.omega, -1, axis=0))*np.roll(pop, -1, axis=0))
        H = np.zeros((8, 192, 2, 2), complex)
        connection = np.zeros((8, 192))
        data = []
        for l, (x, e, l2, w, factor, boundary) in enumerate(geometry):
            edge = self.edge[l]; u, v = edge.T
            phi = self.boundary_phi[l].copy()
            phi[:192] = np.linalg.solve(factor, q[l, :192]+boundary)
            p = F.provider(x, edge, self.elem[l], q[l], self.lut)
            S, E0, E1, V, gamma, invD, ds, k, de, n = p
            kap = (.5*w*invD/np.sqrt((E1-E0)**2+4*V*V+float(np.float32(5e-5))**2)*(-k*ds))[:, None]*n
            vv = np.zeros((256, 3)); vv[:192] = velocity[l]
            lam = np.sum(kap*(vv[v]-vv[u]), axis=1)
            conn = np.zeros(256); np.add.at(conn, u, -lam); np.add.at(conn, v, lam)
            connection[l] = conn[:192]
            H[l, :, 0, 0] = self.omega[l]*self.lut[self.elem[l, :192], 1]+fol[l]
            H[l, :, 1, 1] = self.omega[l]*self.lut[self.elem[l, :192], 2]+self.charge_factor*phi[:192]-fol[l]
            chain = .5*gamma*(rho[l, u, 1]-rho[l, u, 0])*invD
            hu, hv = u<192, v<192
            np.add.at(H[l, :, 0, 0], u[hu], E0[hu]*invD[hu])
            np.add.at(H[l, :, 1, 1], u[hu], E1[hu]*invD[hu]-self.charge_factor*chain[hu])
            np.add.at(H[l, :, 1, 1], v[hv], self.charge_factor*chain[hv])
            np.add.at(H[l, :, 0, 1], u[hu], V[hu]*invD[hu])
            H[l, :, 1, 0] = H[l, :, 0, 1]
            data.append((rho[l], q[l], phi, p, kap))
        return H, connection, data

    def cayley(self, H, connection):
        h = H.copy(); h[..., 0, 1] += 1j*connection; h[..., 1, 0] -= 1j*connection
        eye = self.berry_mass[..., None, None]*np.eye(2)
        return 2*np.linalg.solve(eye+.5j*self.h*h, (self.berry_mass[..., None]*self.z0)[..., None])[..., 0]-self.z0

    def solve_electronic(self, velocity, initial=None):
        geometry = self.geometry(velocity)
        z = self.z0.copy() if initial is None else initial.copy()
        for it in range(24):
            H, conn, data = self.assemble(z, velocity, geometry)
            zp = self.cayley(H, conn)
            error = np.max(np.maximum(abs((zp-z).real), abs((zp-z).imag)))
            if error <= 1e-12:
                return z, geometry, H, conn, data, it+1, error
            z = zp
        raise ValueError(f'CPU C1 did not contract: {it+1} {error}')

    def reciprocal(self, z, H, data):
        mid = .5*(self.z0+z)
        hp = np.einsum('lvij,lvj->lvi', H, mid)
        sy = np.stack((-1j*mid[..., 1], 1j*mid[..., 0]), axis=-1)
        pairing = np.zeros((8, 256))
        pairing[:, :192] = np.sum(hp.conj()*sy, axis=-1).imag/self.berry_mass
        return np.array([F.accumulate(self.edge[l],
             (2*(pairing[l, self.edge[l][:, 0]]-pairing[l, self.edge[l][:, 1]]))[:, None]*data[l][4])[:192]
             for l in range(8)])

    def inspect(self):
        if getattr(self,"continuous_coordinate_probe",False):
            raise ValueError("continuous-coordinate tangent model probe is not a represented physical C4 map")
        if self.terminal_tangent_reference:
            raise ValueError('Terminal tangent reference inputs are not a matched C4 publication.')
        geometry = self.geometry(self.velocity)
        H, conn, data = self.assemble(self.z_native, self.velocity, geometry)
        next_native = self.cayley(H, conn)
        equation_distance = np.max(abs(next_native-self.z_native))
        reciprocal = self.reciprocal(self.z_native, H, data)
        mapped = np.array([self.arr(l, 61440, 576).reshape(3, 192).T for l in range(8)])
        lag = self.h*np.sum(reciprocal*(mapped-self.velocity), axis=(1, 2))
        c1_work = self.h*np.sum(reciprocal*self.velocity, axis=(1, 2))
        c3_work = self.h*np.sum(reciprocal*mapped, axis=(1, 2))
        # Discover/verify the scalar plane from the real declaration rather
        # than accepting a guessed offset in a forensic reconstruction.
        source = (F.K/'nhs_biprism_variational_so3g_v10.cuh').read_text()
        match = re.search(r'offsetof\(BiprismV10NumericalScratchFfi,scalar\)==(\d+)', source)
        assert match, 'Scalar offset assertion must be present.'
        scalars = np.array([self.arr(l, int(match[1]), 32) for l in range(8)])
        joint = scalars[:, 12]+scalars[:, 3]+scalars[:, 20]
        print('BOUND_MAGNETIC_PRECISION', 'nodes_f64', self.magnetic_nodes_f64,
              'phase_f64', self.magnetic_phase_f64)
        print('CANDIDATE_ID', 'substep', self.headers[0][1], 'evaluation', self.headers[0][2]-1,
              'encoded_stage', self.traces[0][-1][7], 'h', self.h)
        identity = evaluated_identity(self.headers[0], self.traces[0])
        assert all(evaluated_identity(h, t) == identity
                   for h, t in zip(self.headers, self.traces)), 'all8 candidate identity mismatch'
        print('EVALUATED_CANDIDATE', json.dumps(identity, sort_keys=True))
        lag_index = np.unravel_index(np.argmax(abs(mapped-self.velocity)), mapped.shape)
        raw_dx = .5*self.h*np.max(abs(mapped-self.velocity))
        print('NUCLEAR_EQUATION_CHECK', 'leaf_vertex_component', lag_index,
              'raw_residual', raw_dx, 'gate', 5e-9, 'satisfied', bool(raw_dx <= 5e-9),
              'original_timestep_reached', NB.original_target_reached(self.headers[0]))
        gates = physical_candidate_gates(scalars, raw_dx, identity['original_h_reached'])
        print('C4_PUBLICATION_GATES', json.dumps(gates, sort_keys=True))
        print('FIRST_REJECTING_C4_CONJUNCT', next((g['predicate'] for g in gates if not g['satisfied']), None))
        original_h = struct.unpack('<f', struct.pack('<I', self.headers[0][10]))[0]
        print('ORIGINAL_H_SCALED_NUCLEAR_RESIDUAL', raw_dx*original_h/self.h,
              'original_h', original_h, 'evaluated_h', self.h)
        print('C1_REFRESHED_H_CAYLEY_DISTANCE', equation_distance)
        for l in range(8):
            print('WORK_RESPONSE', l, 'C1_input_velocity_work', c1_work[l],
                  'C3_mapped_velocity_work', c3_work[l], 'retained_C3_work', scalars[l, 3],
                  'lag_work', lag[l], 'joint_work', joint[l], 'joint_minus_lag', joint[l]-lag[l])
        print('ALL8_WORK_RESPONSE', 'joint', joint.sum(), 'velocity_lag', lag.sum(),
              'joint_minus_lag', (joint-lag).sum(), 'C3_reconstruction_error', np.max(abs(c3_work-scalars[:, 3])))
        z, _, hc, cc, dc, it, err = self.solve_electronic(self.velocity)
        print('CPU_SELF_CONSISTENT_C1', 'iterations', it, 'last_change', err,
              'native_endpoint_max_difference', np.max(abs(z-self.z_native)),
              'norm_change', np.max(abs(np.sum(abs(z)**2, axis=-1)-np.sum(abs(self.z0)**2, axis=-1))),
              'reciprocal_force_difference', np.max(abs(self.reciprocal(z, hc, dc)-reciprocal)))
        # Repeat with a changed velocity and solve ALL eight electronic leaves
        # afresh: identifies actual response, not a frozen-force extrapolation.
        z2, _, h2, c2, d2, it2, err2 = self.solve_electronic(mapped, initial=z)
        f2 = self.reciprocal(z2, h2, d2)
        print('MAPPED_VELOCITY_SELF_CONSISTENT_RESPONSE', 'iterations', it2, 'last_change', err2,
              'spinor_change', np.max(abs(z2-z)), 'reciprocal_force_change', np.max(abs(f2-reciprocal)),
              'work_change', self.h*np.sum((f2-reciprocal)*mapped))
        r, work, _ = self.nuclear_map(self.velocity, self.alpha, initial=z)
        print('CPU_FULL_MAP_RECONSTRUCTION', 'raw_dx', .5*self.h*np.max(abs(r)),
              'native_velocity_max_error', np.max(abs(r-(mapped-self.velocity))),
              'coordinate_work', work.tolist(), 'native_coordinate_work', scalars[:, 11].tolist(),
              'max_coordinate_work_error', np.max(abs(work-scalars[:, 11])))
        _, continuous_work, _ = self.nuclear_map(self.velocity, self.alpha, initial=z,
                                                represented_velocity=False)
        row = 5e-9/(.5*self.h*2e-7)
        arc_norm2 = 0.
        if self.headers[0][0] in NB.ARC_MAGICS:
            arc_rows = np.array([self.arr(l, NB.arc_history_offset(h[2]-1), 4)
                                 for l,h in enumerate(self.headers)])
            assert np.all(arc_rows[:, 0] == self.h)
            assert np.all(arc_rows == arc_rows[0])
            if arc_rows[0, 3]:
                row = self.arr(0, 1680768+10872, 1)[0]
                arc_norm2 = arc_rows[0, 1]**2
            print('EVALUATED_ARCLENGTH', 'h', self.h, 'defect', arc_rows[0, 1],
                  'step', arc_rows[0, 2], 'active', bool(arc_rows[0, 3]),
                  'work_row', row, 'physical_timestep_acceptance', False)
        represented_norm2 = np.sum(r*r)+row*row*np.sum(work*work)+arc_norm2
        continuous_norm2 = np.sum(r*r)+row*row*np.sum(continuous_work*continuous_work)+arc_norm2
        native_norm2 = self.arr(0, 106752+605240, 1)[0]
        expected_norm2 = continuous_norm2 if self.continuous_newton else represented_norm2
        print('SOLVER_PRIVATE_WORK_ROW', 'continuous', self.continuous_newton,
              'native_norm2', native_norm2, 'CPU_bound_norm2', expected_norm2,
              'reconstruction_error', native_norm2-expected_norm2,
              'represented_norm2', represented_norm2, 'continuous_norm2', continuous_norm2,
              'physical_gates_use_represented_endpoint', True)
        # Reconstruct the private row from its actual native momentum/metric
        # owners as well. This tests the arithmetic handoff independently of
        # the separate CPU re-solve of the entire electronic endpoint above.
        rounding = []
        for l in range(8):
            xx, yy, zz, xy, xz, yz = self.arr(l, 97536, 6*192).reshape(6, 192)
            g = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]).transpose(2, 0, 1)
            p = self.arr(l, 56832, 576).reshape(3, 192).T
            v = np.frombuffer(self.raw, '<f4', 576, l*self.stride+11520).astype(float).reshape(3, 192).T
            raw_v = np.linalg.solve(g, (p/self.mass[l, :192, None])[..., None])[..., 0]
            te = .5*np.sum(self.mass[l, :192, None]*v*np.einsum('nij,nj->ni', g, v))
            tr = .5*np.sum(p*raw_v)
            rounding.append(te-tr)
        retained_work = scalars[:, 11]-np.array(rounding) if self.continuous_newton else scalars[:, 11]
        retained_norm2 = np.sum((mapped-self.velocity)**2)+row*row*np.sum(retained_work**2)+arc_norm2
        print('RETAINED_NEWTON_ROW', 'native_norm2', native_norm2,
              'independent_native_owner_norm2', retained_norm2,
              'difference', native_norm2-retained_norm2, 'rounding_work', rounding)
        assert abs(native_norm2-retained_norm2) <= 1e-10*max(1., abs(native_norm2)), 'native solver row ownership mismatch'
        self.work_components(geometry, data, reciprocal, scalars)
        print('SCOPE independent_CPU_self_consistent_response; native_physical_acceptance=false')

    def work_components(self, geometry, data, reciprocal, scalars):
        """Separate work terms at the SAME retained candidate.

        Metric energy is kinetic in this implementation. There is no separate
        spring potential to invent. The endpoint kinetic transport is retained
        directly; CPU component-force and stationary-field reconstruction has
        its own measured closure error, never a relaxed native gate.
        """
        start = self.endpoint.copy(); start[:, :192] = self.start
        p_start = self.potential_parts(self.z0, self.geometry_at(start))
        p_end = self.potential_parts(self.z_native, self.geometry_at(self.endpoint))
        p_mid0 = self.potential_parts(self.z0, geometry)
        p_mid1 = self.potential_parts(self.z_native, geometry)
        displacement = self.endpoint[:, :192]-self.start
        totals = np.zeros(7)
        for l, (x, e, l2, weight, _, _) in enumerate(geometry):
            edges = self.edge[l]; u, v = edges.T
            vv = np.zeros((256, 3)); vv[:192] = self.velocity[l]
            rho, _, phi, provider, _ = data[l]
            fm, _ = F.metric_force(x, self.reference[l], vv, self.mass[l], edges)
            fd = (.5*(phi[u]-phi[v])**2*weight/l2)[:, None]*e
            _, fx = F.edge_energy_force(provider, rho[u])
            works = [np.sum(F.accumulate(edges, force)[:192]*displacement[l])
                     for force in (fm, fd, fx)]
            coordinate_energy = p_end[l]-p_start[l]-(p_mid1[l]-p_mid0[l])
            components = np.array([
                scalars[l, 1]+works[0],
                coordinate_energy[1]+works[1],
                coordinate_energy[0]+coordinate_energy[2]+works[2],
                -self.alpha[l]*scalars[l, 4],
                scalars[l, 10],
                scalars[l, 12]+np.sum(reciprocal[l]*displacement[l])+scalars[l, 20],
                scalars[l, 17]-scalars[l, 20],
            ])
            totals += components
            print('CPU_MATCHED_WORK_COMPONENTS', l,
                  'metric_kinetic_transport', components[0],
                  'dec_coordinate', components[1], 'xtb_valence_coordinate', components[2],
                  'Gonzalez_projection', components[3], 'midpoint_kinetic_defect', components[4],
                  'joint_electronic_Berry_foliated', components[5],
                  'foliated_owned_minus_discrete_work', components[6],
                  'component_sum', components.sum(),
                  'native_total', scalars[l, 16]+scalars[l, 17],
                  'reconstruction_difference', components.sum()-scalars[l, 16]-scalars[l, 17])
        print('CPU_MATCHED_ALL8_COMPONENTS', totals.tolist(), 'sum', totals.sum())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution', type=Path, required=True)
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--forensic-attempt129', action='store_true',
                   help='Decode only the hash-pinned, rejected, noncanonical Attempt129 trace; no admission authority.')
    a = p.parse_args()
    CoupledResponse(a.execution, a.epoch, forensic_attempt129=a.forensic_attempt129).inspect()
