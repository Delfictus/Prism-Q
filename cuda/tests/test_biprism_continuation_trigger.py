"""Compile the actual restart predicate against retained candidate identities."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROLE = Path(__file__).resolve().parents[2]
SOURCE = ROLE / 'crates/prism-gpu/src/kernels/nhs_biprism_variational_so3g_v10.cuh'


class ContinuationTrigger(unittest.TestCase):
    def test_probes_cannot_discard_a_converging_original_timestep_base(self):
        predicate = re.search(r'const bool slow_trial=([^;]+);', SOURCE.read_text()).group(1)
        code = '''#include <cassert>
struct Evaluation { unsigned count; };
struct Correction { unsigned mode; double accepted_trial_ratio; };
bool restart(Evaluation e, Correction q, double raw) { return ''' + predicate + '''; }
int main() {
    // Native initial-guess diagnostic005: trial82 improved the merit;
    // derivative probe89 deliberately moved away from that accepted base.
    const double ratio=8.6522894093813313e-13/9.4948583877912137e-13;
    for (unsigned mode : {0u,1u,4u,5u})
        assert(!restart({90u},{mode,ratio},0.0018950749288251245));
    assert(!restart({83u},{2u,ratio},1.0271650836557455e-6));
    // A truly slow correction still enables the existing continuation path.
    assert(restart({90u},{2u,ratio},0.0018950749288251245));
    assert(!restart({23u},{2u,ratio},0.0018950749288251245));
    assert(!restart({90u},{2u,0.9},0.0018950749288251245));
    assert(!restart({90u},{2u,ratio},1e-3));
}
'''
        code = '#include <initializer_list>\n' + code
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p/'control.cpp').write_text(code)
            subprocess.run(['g++','-std=c++17','-O2',str(p/'control.cpp'),
                            '-o',str(p/'control')],check=True,capture_output=True)
            subprocess.run([str(p/'control')],check=True,capture_output=True)


if __name__ == '__main__':
    unittest.main()
