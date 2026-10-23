"""Independent algebra controls for the coupled response reconstruction.

These CPU controls do not admit a native timestep or qualify a trajectory.
"""
import unittest
import contextlib
import io
import numpy as np
from check_biprism_coupled_work_response import CoupledResponse, density, evaluated_identity, evaluated_velocity, physical_candidate_gates, magnetic_nodes_f64, MAGNETIC_NODE64_SOURCES, MAGNETIC_NODE32_SOURCES, MAGNETIC_PHASE64_SOURCES
from check_biprism_nuclear_budget import KRYLOV64_MAGIC


class CoupledWorkResponseTests(unittest.TestCase):
    def test_continuous_coordinate_probe_cannot_claim_a_represented_map(self):
        c=object.__new__(CoupledResponse);c.continuous_coordinate_probe=True
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(ValueError,'not a represented physical C4 map'):
                c.inspect()
        self.assertEqual(output.getvalue(),'')

    def test_signed_derivative_probe_retains_evaluated_input(self):
        from check_biprism_nuclear_budget import validate_paired_probe_weights
        base=np.array([1.,2.,3.]);direction=np.array([.5,-1.,2.])
        plus=[1e-8,0.,.7,1.,.1,2.,3e-7,16104.,0.]
        minus=plus[:];minus[6]=-plus[6]
        validate_paired_probe_weights([plus,minus])
        got=evaluated_velocity(base,direction,minus,1,5)
        np.testing.assert_array_equal(got,base-3e-7*direction)
        for slot,value in [(5,3.),(6,-1e-7),(7,16204.),(7,17104.),(7,16105.)]:
            bad=minus[:];bad[slot]=value
            with self.assertRaises(ValueError):validate_paired_probe_weights([plus,bad])
        with self.assertRaises(ValueError):validate_paired_probe_weights([minus])
        for column in (0,63):
            plus[7]=minus[7]=16100.+column
            validate_paired_probe_weights([plus,minus])
            validate_paired_probe_weights([plus]) # Unfinished pair at budget exhaustion.
        for phase in (16000.,16164.,16199.,16200.,16700.,17100.,1764.,1799.,1800.,1600.):
            plus[7]=minus[7]=phase
            with self.assertRaises(ValueError):validate_paired_probe_weights([plus,minus])
        for stage in (1,2):
            for column in (0,63):
                plus[7]=minus[7]=1000.*stage+700.+column
                validate_paired_probe_weights([plus,minus])
                np.testing.assert_array_equal(evaluated_velocity(base,direction,minus,1,column+1),base-3e-7*direction)

    def test_terminal_tangent_reference_cannot_publish_a_c4_result(self):
        c=object.__new__(CoupledResponse);c.terminal_tangent_reference=True
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(ValueError,'not a matched C4 publication'):
                c.inspect()
        self.assertEqual(output.getvalue(),'')
    def test_native142_fixed_final_base_does_not_consume_pending_probe(self):
        base = np.array([[.3, -.1, .2]])
        pending_probe = np.array([[.9, -.7, .6]])
        row = [8.512354436814181e-20, 9.45594406e-13, -5.,
               1.7221231797338237e-6, 0., 0., 0., 1000., 0.]
        np.testing.assert_array_equal(evaluated_velocity(base, pending_probe, row, 1, 0), base)
        with self.assertRaisesRegex(ValueError, 'probe-zero identity'):
            evaluated_velocity(base, pending_probe, row, 1, 1)
        row[6] = 1e-4
        with self.assertRaisesRegex(ValueError, 'probe-zero identity'):
            evaluated_velocity(base, pending_probe, row, 1, 0)

    def test_tiny_private_h_cannot_be_reported_as_nuclear_rejection_or_acceptance(self):
        gates = physical_candidate_gates(np.zeros((8, 32)), 8.512354436814181e-20, False)
        self.assertTrue(gates[0]['satisfied'])
        self.assertEqual([g['predicate'] for g in gates if not g['satisfied']], ['original_timestep_return'])
        self.assertEqual([g['predicate'] for g in physical_candidate_gates(np.zeros((8,32)), .001, True)
                          if not g['satisfied']], ['nuclear_equation'])

    def test_magnetic_precision_is_bound_to_compiled_source(self):
        def build(h):
            return {'input_hashes': {'/kernels/nhs_biprism_variational_so3g_v10.cuh': h}}
        for h in MAGNETIC_NODE64_SOURCES | MAGNETIC_PHASE64_SOURCES:
            self.assertTrue(magnetic_nodes_f64(build(h)))
        for h in MAGNETIC_NODE32_SOURCES:
            self.assertFalse(magnetic_nodes_f64(build(h)))
        with self.assertRaisesRegex(ValueError, 'unreviewed'):
            magnetic_nodes_f64(build('0'*64))
        with self.assertRaisesRegex(ValueError, 'source owner'):
            magnetic_nodes_f64({'input_hashes': {}})

    def test_final_accepted_trial_has_already_replaced_the_solver_base(self):
        base = np.array([[.3, -.1, .2]])
        next_probe = np.array([[.9, -.7, .6]])
        # Retained diagnostic006, substep2 evaluation182. The next probe
        # direction is not the already consumed correction direction.
        row = [0.,0.,0.,.00955729580628555,.008473853784814146,
               .017660819344357068,1.,10207.,0.]
        np.testing.assert_array_equal(evaluated_velocity(base,next_probe,row,1,0),base)
        with self.assertRaisesRegex(ValueError,'new-base identity'):
            evaluated_velocity(base,next_probe,row,2,7)
        row[3] = .02  # rejected correction: old base/direction remain live
        np.testing.assert_array_equal(evaluated_velocity(base,next_probe,row,2,7),base+next_probe)
        row[7] = 16107.  # measured derivative probe, regardless of its merit
        row[3] = .001
        np.testing.assert_array_equal(evaluated_velocity(base,next_probe,row,1,8),base+next_probe)

    def test_evaluated_candidate_is_distinct_from_private_continuation(self):
        # Attempt130's two distinct refusal classes. Pending QR mode is not
        # an input: it already describes the next unevaluated proposal.
        h = [KRYLOV64_MAGIC, 1, 192, 40, 118, 24, 0, 10, 35, 5,
             0, 16, 0] + [0]*19
        trace = [[0.]*9 for _ in range(192)]
        for row in trace[101:]:
            row[7] = 16100.
        trace[-1][7] = 16112.
        got = evaluated_identity(h, trace)
        self.assertEqual((got['candidate'], got['column'], got['stage']),
                         ('derivative_probe', 12, 16))
        self.assertEqual(got['first_return_to_original_h'], 101)
        self.assertEqual(got['exhausted_limits'], ['full_evaluations'])
        h[1:10] = [0, 176, 48, 87, 36, 0, 5, 29, 19]
        h[11] = 6
        trace = [[0.]*9 for _ in range(176)]
        trace[-1][7] = 6202.
        got = evaluated_identity(h, trace)
        self.assertEqual((got['candidate'], got['column']), ('correction_trial', 2))
        self.assertFalse(got['original_h_reached'])
        self.assertIsNone(got['first_return_to_original_h'])
        self.assertEqual(got['exhausted_limits'], ['nonlinear_trials'])

    def test_candidate_stage_must_match_retained_live_h(self):
        h = [KRYLOV64_MAGIC, 0, 1, 0, 0, 0, 0, 1, 0, 0,
             0, 6, 0] + [0]*19
        with self.assertRaisesRegex(ValueError, 'retained live stage'):
            evaluated_identity(h, [[0., 0., 0., 0., 0., 0., 0., 16100., 0.]])

    def test_cayley_connection_work_uses_its_own_velocity(self):
        rng = np.random.default_rng(128)
        c = object.__new__(CoupledResponse)
        c.h = .037
        c.berry_mass = .4 + rng.random((8, 192))
        c.z0 = rng.normal(size=(8, 192, 2)) + 1j*rng.normal(size=(8, 192, 2))
        c.z0 /= np.linalg.norm(c.z0, axis=-1)[..., None]
        H = rng.normal(size=(8, 192, 2, 2)) + 1j*rng.normal(size=(8, 192, 2, 2))
        H = .5*(H+H.conj().swapaxes(-1, -2))
        connection = rng.normal(size=(8, 192))
        endpoint = c.cayley(H, connection)
        midpoint = .5*(c.z0+endpoint)
        hp = np.einsum('...ij,...j->...i', H, midpoint)
        # H_conn has entries (+i lambda, -i lambda), so its exact
        # contribution to dU/dt is -2 lambda Im <H psi, sigma_y psi>/m.
        sigma_y = np.stack((-1j*midpoint[..., 1], 1j*midpoint[..., 0]), axis=-1)
        rate = -2*connection*np.sum(hp.conj()*sigma_y, axis=-1).imag/c.berry_mass
        energy = lambda z: np.sum(z.conj()*np.einsum('...ij,...j->...i', H, z), axis=-1).real
        np.testing.assert_allclose(energy(endpoint)-energy(c.z0), c.h*rate, atol=4e-15, rtol=2e-12)
        roundoff_bound = 32*np.finfo(float).eps
        np.testing.assert_allclose(np.sum(abs(endpoint)**2, axis=-1),
                                   np.sum(abs(c.z0)**2, axis=-1), atol=roundoff_bound, rtol=0)
        # A force paired with a different nuclear velocity has an actual lag
        # work. It cannot be credited as an independent energy correction.
        changed_connection = connection + .1
        wrong_rate = rate * np.divide(changed_connection, connection)
        self.assertGreater(np.max(abs(c.h*(wrong_rate-rate))), 1e-4)

    def test_endpoint_density_mean_is_not_normalized_spinor_mean(self):
        z0 = np.array([1., 0.], complex)
        z1 = np.array([0., 1.], complex)
        endpoint_mean = .5*(density(z0)+density(z1))
        coherent_mean = density((z0+z1)/np.sqrt(2))
        np.testing.assert_array_equal(endpoint_mean, [.5, .5, 0., 0.])
        self.assertGreater(abs(coherent_mean[2]-endpoint_mean[2]), .49)


if __name__ == '__main__':
    unittest.main()
