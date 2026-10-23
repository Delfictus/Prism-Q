#!/usr/bin/env python3
"""Provenance-boundary controls; constructed records are not native evidence."""
import unittest
import check_biprism_v10_accepted_prefix as C


def record():
    return {'native_execution': True, 'native_exit_code': 4, 'exit_code': 4,
            'sources_binary_inputs_unchanged': True, 'gpu_recovery_none_after': True,
            'new_kernel_fault_records': [], 'expected_epochs': 3,
            'full_trajectory_accepted': False,
            'boot_id': '5fabfea2-4a4c-45b0-9b61-72a05257c4cd',
            'main_compiler_resources': {'stack': 0, 'cumulative_stack': 0,
                                        'spill_stores': 0, 'spill_loads': 0},
            'command': ['binary', 'physical_input', 'selection', 'masses', 'prefix']}


class Controls(unittest.TestCase):
    def test_retained_controlled_refusal_can_have_an_accepted_prefix(self):
        C.dispatch_domain(record(), 1)

    def test_success_is_not_required_but_actual_execution_is(self):
        r = record()
        r['native_execution'] = False
        with self.assertRaisesRegex(ValueError, 'actual native'):
            C.dispatch_domain(r, 1)

    def test_other_exit_classes_are_not_silently_promoted(self):
        for status in (-9, 1, 2, 5, 124, 127):
            with self.subTest(status=status):
                r = record()
                r['exit_code'] = r['native_exit_code'] = status
                with self.assertRaisesRegex(ValueError, 'controlled scientific refusal'):
                    C.dispatch_domain(r, 1)

    def test_refusal_whole_trajectory_claim_is_rejected(self):
        r = record()
        r['full_trajectory_accepted'] = True
        with self.assertRaisesRegex(ValueError, 'cannot claim full trajectory'):
            C.dispatch_domain(r, 1)

    def test_resource_keys_are_mandatory_including_callee_stack(self):
        for key in record()['main_compiler_resources']:
            for value in (None, 1, False):
                with self.subTest(key=key, value=value):
                    r = record()
                    if value is None:
                        del r['main_compiler_resources'][key]
                    else:
                        r['main_compiler_resources'][key] = value
                    with self.assertRaisesRegex(ValueError, 'resource gate'):
                        C.dispatch_domain(r, 1)

    def test_mutation_recovery_and_faults_are_not_excused_by_prefix(self):
        for key, value in (('sources_binary_inputs_unchanged', False),
                           ('gpu_recovery_none_after', False),
                           ('new_kernel_fault_records', ['actual new fault']),
                           ('new_kernel_fault_records', None)):
            with self.subTest(key=key):
                r = record()
                r[key] = value
                with self.assertRaises(ValueError):
                    C.dispatch_domain(r, 1)

    def test_epoch_domain_not_inferred_from_filenames(self):
        for epoch in (0, 4, -1, True, '1'):
            with self.subTest(epoch=epoch):
                with self.assertRaisesRegex(ValueError, 'epoch outside'):
                    C.dispatch_domain(record(), epoch)

    def test_incomplete_command_or_boot_identity_refused(self):
        for key, value in (('command', ['binary']), ('boot_id', None), ('boot_id', 'unknown')):
            r = record()
            r[key] = value
            with self.assertRaises(ValueError):
                C.dispatch_domain(r, 1)

    def test_exit_identity_is_typed_and_consistent(self):
        for value in (0, False, '4', None):
            r = record()
            r['exit_code'] = value
            with self.assertRaisesRegex(ValueError, 'exit disagreement'):
                C.dispatch_domain(r, 1)


if __name__ == '__main__':
    print('CONTROL_SCOPE=prefix provenance rules only; no native execution or trajectory claim')
    unittest.main(verbosity=2)
