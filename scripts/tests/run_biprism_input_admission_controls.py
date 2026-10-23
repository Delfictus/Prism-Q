#!/usr/bin/env python3
"""Run the exact Biprism input precheck against the actual built input owner.

CPU only: this does not construct a CUDA context or qualify a physical run.
The two retained input identities are checked before compiling the helper;
negative cases are in-memory mutations and never alter either source file.
"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess

def _require_root(name):
    """Absolute root from the environment; there is no compiled-in host default."""
    value = os.environ.get(name, '')
    if not os.path.isabs(value):
        raise SystemExit(f'{name} must name an absolute path (no compiled-in host default)')
    return Path(value)


ROLE = Path(__file__).resolve().parents[2]
BASE = _require_root('PRISM_SINGLE_CHAIN10_ROOT')
INPUTS = (
    (BASE / 'INPUTS_MAY_FOUR_FRESH_1010_V1/KRAS_A/KRAS_A.topology.json',
     '4592ef511dc34174e409baebe96e9885de290dde48ce29c96ce6aad502def9bc'),
    (BASE / 'BASELINES/1010_MAY_FROZEN_20261016/INPUTS/KRAS/kras.topology.json',
     'bc19e5f455552b6619b34e23fff55425c8c60f040168e89f21bbd028d527ff19'),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(text, name):
    start = re.search(r'\bfn ' + re.escape(name) + r'(?=[<(])', text).start()
    brace = text.index('{', start)
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError('incomplete source function ' + name)


DRIVER = r'''
fn main() {
    let args:Vec<String>=std::env::args().collect();
    let fresh=PrismPrepTopology::load(&args[1]).unwrap();
    let old=PrismPrepTopology::load(&args[2]).unwrap();
    let before=format!("{fresh:?}");
    validate_native_constructor_inputs(&fresh).unwrap();
    assert_eq!(before,format!("{fresh:?}"));
    println!("PASS actual prepared source accepted without mutation");
    old.validate().unwrap();
    let error=validate_native_constructor_inputs(&old).unwrap_err();
    assert!(format!("{error:#}").contains("harmonic source convention is absent"));
    println!("PASS actual bc19 formerly generic-valid input refused before construction");
    let mut count=2;
    let reject=|t:&PrismPrepTopology,label:&str| {
        let error=validate_native_constructor_inputs(t).unwrap_err();
        println!("PASS {label}: {error:#}");
    };
    let mut t=fresh.clone();t.harmonic_parameter_contract=None;reject(&t,"missing harmonic contract");count+=1;
    let mut t=fresh.clone();t.harmonic_parameter_contract.as_mut().unwrap().contract_sha256="0".repeat(64);
    reject(&t,"harmonic digest corruption");count+=1;
    let mut t=fresh.clone();t.harmonic_parameter_contract.as_mut().unwrap().bond_count-=1;
    reject(&t,"incomplete bond population");count+=1;
    let mut t=fresh.clone();t.harmonic_parameter_contract.as_mut().unwrap().angle_count-=1;
    reject(&t,"incomplete angle population");count+=1;
    let mut t=fresh.clone();t.harmonic_parameter_contract.as_mut().unwrap().bond_uses_periodic_boundary_conditions=true;
    reject(&t,"unsupported harmonic periodicity");count+=1;
    let mut t=fresh.clone();t.harmonic_parameter_contract.as_mut().unwrap().angle_coefficient_unit="kJ/mol/radian^2".into();
    reject(&t,"wrong coefficient unit");count+=1;
    let mut t=fresh.clone();t.bonds[0].k=f64::MAX;reject(&t,"bond upload overflow");count+=1;
    let mut t=fresh.clone();t.angles[0].force_k=f64::MAX;reject(&t,"angle upload overflow");count+=1;
    let mut t=fresh.clone();t.nonbonded_exceptions.clear();t.nonbonded_exception_contract=None;
    reject(&t,"missing exact nonbonded contract");count+=1;
    let mut t=fresh.clone();t.nonbonded_exception_contract.as_mut().unwrap().pair_digest_sha256="0".repeat(64);
    reject(&t,"nonbonded population digest corruption");count+=1;
    let mut t=fresh.clone();let cell=t.periodic_cell.as_mut().unwrap();
    cell.reference_implementation=Some("OpenMM".into());
    reject(&t,"partial recorded periodic PME authority");count+=1;
    println!("{count} exact input-owner controls passed; no CUDA context or engine execution");
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    prefix = args.output_prefix
    if not prefix.is_absolute() or not prefix.parent.is_dir():
        parser.error('output prefix needs an existing absolute parent')
    outputs = [prefix, prefix.with_suffix('.rs'), prefix.with_suffix('.txt')]
    if any(path.exists() for path in outputs):
        parser.error('output prefix already exists')
    for path, digest in INPUTS:
        if sha(path) != digest:
            raise ValueError('retained input identity changed: ' + str(path))
    source = ROLE / 'crates/prism-nhs/src/biprism_run.rs'
    input_owner = ROLE / 'crates/prism-nhs/src/input.rs'
    text = source.read_text()
    assert 'validate_native_constructor_inputs(&topology)?;' in extract(text, 'resolve_inputs')
    driver = (
        'use anyhow::{anyhow,ensure,Context,Result};\n'
        'use prism_nhs::input::{PrismPrepTopology,HarmonicEnergyConvention,NonbondedExecutionPolicy};\n'
        + extract(text, 'serr') + '\n'
        + extract(text, 'validate_native_constructor_inputs') + '\n' + DRIVER
    )
    prefix.with_suffix('.rs').write_text(driver)
    deps = ROLE / 'target/release/deps'
    libs = {name: max(deps.glob('lib' + name + '-*.rlib'), key=lambda p: p.stat().st_mtime_ns)
            for name in ('anyhow', 'prism_nhs')}
    command = ['rustc', '--edition=2021', str(prefix.with_suffix('.rs')),
               '-L', 'dependency=' + str(deps), '-o', str(prefix)]
    for name, path in libs.items():
        command += ['--extern', name + '=' + str(path)]
    for path in sorted((ROLE / 'target/release/build').glob('*/out')):
        command += ['-L', 'native=' + str(path)]
    for path in ['/usr/local/cuda/lib64', '/usr/local/cuda/lib64/stubs',
                 str(ROLE / 'BUILD_INPUTS/sdst/lib')]:
        command += ['-L', 'native=' + path]
    initial = {str(path): sha(path) for path in (source, input_owner, *libs.values())}
    log = ['scope: exact new CPU precheck linked to actual built prism_nhs input owner; no CUDA/Cargo execution',
           'source_pins ' + repr(initial), 'inputs ' + repr(INPUTS), 'compile ' + repr(command)]
    result = subprocess.run(command, text=True, capture_output=True)
    log += ['compile_exit ' + str(result.returncode), result.stdout, result.stderr]
    if result.returncode == 0:
        result = subprocess.run([str(prefix), *(str(p) for p, _ in INPUTS)], text=True, capture_output=True)
        log += ['controls_exit ' + str(result.returncode), result.stdout, result.stderr]
    assert initial == {path: sha(Path(path)) for path in initial}, 'source/library changed during controls'
    for path, digest in INPUTS:
        assert sha(path) == digest, 'retained input changed during controls'
    prefix.with_suffix('.txt').write_text('\n'.join(log))
    print(result.stdout)
    print(result.stderr)
    print('retained log:', prefix.with_suffix('.txt'))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
