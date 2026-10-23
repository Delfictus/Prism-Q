#!/usr/bin/env python3
"""CPU rejection controls for real compiler/SASS parsing; no runtime evidence."""
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'run_biprism_v10_production.py'
SPEC = importlib.util.spec_from_file_location('production_resources_under_test', SOURCE)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


class ResourceControls(unittest.TestCase):
    def setUp(self):
        self.helpers = ['_Z12bp_v10_firstPv', '_Z13bp_v10_secondPv']
        self.resources = {'functions': [{'symbol': x} for x in [R.MAIN, *self.helpers]]}
        self.sass = f'Function : {R.MAIN}\n /*0000*/ MOV R0, R1;\n /*0010*/ RET;\n'
        self.elf = f'STT_FUNC STB_GLOBAL STO_ENTRY {R.MAIN}\n' + ''.join(
            f'STT_FUNC STB_LOCAL STV_DEFAULT ${R.MAIN}${x}\n' for x in self.helpers)

    def test_embedded_callees_require_no_separate_sass_headings(self):
        self.assertEqual(R.validate_sass(self.sass, self.resources, self.elf), [R.MAIN, *self.helpers])

    def test_callee_population_is_exact_and_owned(self):
        variants = [self.elf.replace(self.helpers[0], '_Z_other'),
                    self.elf.replace(f'${R.MAIN}$', '$different_entry$', 1),
                    self.elf + f'STT_FUNC STB_LOCAL STV_DEFAULT ${R.MAIN}$_Z_extra\n',
                    self.elf + self.elf.splitlines(True)[1],
                    self.elf.splitlines(True)[0]]
        for elf in variants:
            with self.subTest(elf=elf), self.assertRaises(R.Refusal):
                R.validate_sass(self.sass, self.resources, elf)

    def test_main_entry_and_decoded_instructions_required(self):
        for sass, elf in [(self.sass, self.elf.split('\n', 1)[1]),
                          (self.sass, self.elf + self.elf.splitlines(True)[0]),
                          (f'Function : {R.MAIN}\n', self.elf),
                          ('Function : unrelated\n /*0000*/ MOV R0, R1;\n', self.elf),
                          (self.sass + self.sass, self.elf)]:
            with self.subTest(sass=sass, elf=elf), self.assertRaises(R.Refusal):
                R.validate_sass(sass, self.resources, elf)

    def test_local_traffic_in_any_part_of_entry_is_refused(self):
        for opcode in ('LDL R0, [R1];', '@P0 STL.64 [R1], R2;', '@!P1 LDL.U8 R2, [R3];'):
            with self.subTest(opcode=opcode), self.assertRaises(R.Refusal):
                R.validate_sass(self.sass + ' /*0020*/ ' + opcode, self.resources, self.elf)
        with self.assertRaises(R.Refusal):
            R.validate_sass(self.sass + f'Function : {self.helpers[0]}\n /*0000*/ LDL R0, [R1];', self.resources, self.elf)

    def test_actual_resource_block_refuses_stack_spills_and_omissions(self):
        head = f"ptxas info : Compiling entry function '{R.MAIN}' for 'sm_120'\n"
        def block(name):
            return f'ptxas info : Function properties for {name}\n    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n'
        raw = head + block(R.MAIN) + 'ptxas info : Used 128 registers\n' + ''.join(block(x) for x in self.helpers)
        parsed = R.parse_resources(raw)
        self.assertEqual(parsed['registers'], 128)
        self.assertEqual(len(parsed['functions']), 3)
        for bad in [raw.replace('0 bytes stack frame', '8 bytes stack frame', 1),
                    raw.replace('0 bytes spill stores', '4 bytes spill stores', 1),
                    raw.replace('0 bytes spill loads', '4 bytes spill loads', 1),
                    raw.replace('sm_120', 'sm_90'), raw + block(self.helpers[0]),
                    head + block(R.MAIN) + 'ptxas info : Used 128 registers\n',
                    raw + 'ptxas info : Function properties for _Z_missing\n']:
            with self.subTest(raw=bad), self.assertRaises(R.Refusal): R.parse_resources(bad)

    def test_shared_memory_uses_actual_host_and_device_size(self):
        host = 'pub const BIPRISM_DYNAMIC_SMEM_BYTES: u32 = 31_616;'
        device = 'static_assert(sizeof(VariationalClusterDsmemV10) == 31616, "layout");'
        self.assertEqual(R.launch_shared_bytes(host, device), 31616)
        for h, d in [(host.replace('31_616', '32_640'), device), (host, ''), (host+host, device)]:
            with self.subTest(host=h, device=d), self.assertRaises(R.Refusal): R.launch_shared_bytes(h, d)


if __name__ == '__main__': unittest.main()
