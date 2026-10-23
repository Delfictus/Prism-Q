#!/usr/bin/env python3
"""Compile the actual harmonic binding against the retained Rust input owner.

CPU only. This invokes neither Cargo nor a CUDA context. Original inputs and
preparation evidence are read-only; negative files exist only in a temporary
directory. The emitted Rust driver and plain log make every case reproducible.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
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


ROLE = Path(__file__).resolve().parents[2]
DEFAULT_BINDING = _env_root('PRISM_BIPRISM_RESOLUTION_ROOT') / 'failure_7_harmonic_source_binding.json'
DEFAULT_BINDING_SHA = 'febe0f98dc250f6e86fde953065ff66d1b51cbdb2329a40f20b4345d06f3a15d'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def item(source, pattern):
    match = re.search(pattern, source)
    if match is None:
        raise ValueError('missing source item: ' + pattern)
    brace = source.index('{', match.start())
    depth = 0
    for end in range(brace, len(source)):
        depth += (source[end] == '{') - (source[end] == '}')
        if depth == 0:
            return source[match.start():end + 1]
    raise ValueError('incomplete source item: ' + pattern)


def matching_libraries(prism_rlib):
    """Use Cargo's exact dependency fingerprint, never newest-mtime serde."""
    suffix = prism_rlib.stem.removeprefix('libprism_nhs-')
    fingerprints = ROLE / 'target/release/.fingerprint'
    fingerprint = fingerprints / ('prism-nhs-' + suffix) / 'lib-prism_nhs.json'
    document = json.loads(fingerprint.read_text())
    wanted = {dep[1]: dep[3] for dep in document['deps']}
    libs = {'prism_nhs': prism_rlib}
    for name in ('anyhow', 'serde', 'serde_json', 'sha2'):
        candidates = []
        for stamp in fingerprints.glob(name.replace('_', '-') + '-*/lib-' + name):
            if int.from_bytes(bytes.fromhex(stamp.read_text().strip()), 'little') == wanted[name]:
                candidate = prism_rlib.parent / ('lib' + name + '-' + stamp.parent.name.rsplit('-', 1)[1] + '.rlib')
                if candidate.is_file():
                    candidates.append(candidate)
        # Cargo package names serde_json and sha2 keep their underscores.
        for stamp in fingerprints.glob(name + '-*/lib-' + name):
            if int.from_bytes(bytes.fromhex(stamp.read_text().strip()), 'little') == wanted[name]:
                candidate = prism_rlib.parent / ('lib' + name + '-' + stamp.parent.name.rsplit('-', 1)[1] + '.rlib')
                if candidate.is_file() and candidate not in candidates:
                    candidates.append(candidate)
        if len(candidates) != 1:
            raise ValueError('ambiguous/missing actual Cargo dependency: ' + name)
        libs[name] = candidates[0]
    return libs, fingerprint


DRIVER = r'''
fn rejected(t:&mut PrismPrepTopology, target:&Value, pin:&PinnedFile, binding:&Value,
            label:&str, reason:&str) {
    let before=serde_json::to_value(&*t).unwrap();
    let error=harmonic_binding::bind(t,target,pin,&serde_json::to_vec(binding).unwrap()).unwrap_err();
    let description=format!("{error:#}");
    assert!(description.contains(reason),"{label}: wrong refusal: {description}");
    assert!(before==serde_json::to_value(&*t).unwrap(),"{label}: refusal mutated caller");
    println!("PASS {label}: {description}");
}
fn changed_file<F:FnOnce(&mut Value)>(binding:&mut Value,key:&str,dir:&Path,number:usize,change:F) {
    let path=PathBuf::from(binding[key]["path"].as_str().unwrap());
    let mut doc:Value=serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    change(&mut doc);
    let bytes=serde_json::to_vec(&doc).unwrap();
    let path=dir.join(format!("negative_{number}.json"));
    fs::write(&path,&bytes).unwrap();
    binding[key]=json!({"path":path,"sha256":hash(&bytes)});
}
fn main() {
    let args:Vec<String>=std::env::args().collect();
    let binding_bytes=fs::read(&args[1]).unwrap();
    let binding:Value=serde_json::from_slice(&binding_bytes).unwrap();
    let pin:PinnedFile=serde_json::from_value(binding["target_topology"].clone()).unwrap();
    let raw=pin.read().unwrap();
    let target:Value=serde_json::from_slice(&raw).unwrap();
    let original:PrismPrepTopology=serde_json::from_slice(&raw).unwrap();
    let temporary=Path::new(&args[2]);
    assert!(original.harmonic_parameter_contract.is_none());
    let mut topology=original.clone();
    let result=harmonic_binding::bind(&mut topology,&target,&pin,&binding_bytes).unwrap();
    validate_native_constructor_inputs(&topology).unwrap();
    let mut after=serde_json::to_value(&topology).unwrap();
    let mut before=serde_json::to_value(&original).unwrap();
    after.as_object_mut().unwrap().remove("harmonic_parameter_contract");
    before.as_object_mut().unwrap().remove("harmonic_parameter_contract");
    assert!(after==before,"binding changed physical or unrelated metadata fields");
    assert_eq!(raw,pin.read().unwrap());
    assert_eq!(result["historical_producer_verification"],"UNKNOWN");
    assert_eq!(result["physical_arrays_imported"],json!([]));
    assert_eq!(result["imported_fields"],json!(["harmonic_parameter_contract"]));
    println!("PASS actual bc19 binds and passes constructor preconditions; all physical arrays and original bytes unchanged");
    let lower=topology.lower_harmonic_parameters(HarmonicEnergyConvention::AmberNoHalfEnergy).unwrap();
    println!("actual_lowering {}",lower.accounting);
    assert_eq!(lower.accounting["bond_scale_f64"],0.5);
    assert_eq!(lower.accounting["angle_scale_f64"],0.5);
    assert_eq!(topology.bonds[0].k.to_bits(),original.bonds[0].k.to_bits());
    assert_eq!(topology.angles[0].force_k.to_bits(),original.angles[0].force_k.to_bits());
    println!("PASS current native half-scale is applied by the existing lowerer, never to serialized/raw coefficients");
    let mut count=2;
    rejected(&mut topology,&target,&pin,&binding,"embedded contract cannot be replaced/double-bound","cannot replace");count+=1;
    macro_rules! reject_binding { ($label:expr,$reason:expr,$change:expr) => {{
        let mut b=binding.clone();($change)(&mut b);
        rejected(&mut original.clone(),&target,&pin,&b,$label,$reason);count+=1;
    }}; }
    reject_binding!("unknown schema","unsupported",|b:&mut Value|b["schema"]=json!("other"));
    reject_binding!("fabricated historical verification","unsupported",|b:&mut Value|b["historical_producer_verification"]=json!("VERIFIED"));
    reject_binding!("unknown manifest field","unknown field",|b:&mut Value|b["ignored_contract"]=json!(true));
    reject_binding!("missing authority","missing field",|b:&mut Value|{b.as_object_mut().unwrap().remove("authority_topology");});
    reject_binding!("wrong target pin","different input",|b:&mut Value|b["target_topology"]["sha256"]=json!("0".repeat(64)));
    reject_binding!("wrong target path","different input",|b:&mut Value|b["target_topology"]["path"]=json!("/nonexistent/target.json"));
    reject_binding!("wrong bond population hash","ordered bond",|b:&mut Value|b["bonds_f64_rows_sha256"]=json!("0".repeat(64)));
    reject_binding!("wrong angle population hash","ordered angle",|b:&mut Value|b["angles_f64_rows_sha256"]=json!("0".repeat(64)));
    for key in ["authority_topology","preparation_receipt","preparation_start","preparation_execution","producer","source_pdb"] {
        let mut b=binding.clone();b[key]["sha256"]=json!("0".repeat(64));
        rejected(&mut original.clone(),&target,&pin,&b,&format!("changed {key} bytes/pin"),"SHA256 mismatch");count+=1;
    }
    reject_binding!("missing forcefield population","forcefield population",|b:&mut Value|{b["forcefield_resources"].as_array_mut().unwrap().pop();});
    reject_binding!("wrong forcefield file pin","resource identity",|b:&mut Value|b["forcefield_resources"][0]["file"]["sha256"]=json!("0".repeat(64)));
    reject_binding!("reordered forcefield closure","resource identity",|b:&mut Value|b["forcefield_resources"].as_array_mut().unwrap().swap(0,1));
    let mut t=original.clone();t.bonds[0].k=f64::from_bits(t.bonds[0].k.to_bits()+1);
    rejected(&mut t,&target,&pin,&binding,"one-bit bond coefficient change","ordered bond");count+=1;
    let mut t=original.clone();t.angles[0].force_k=f64::from_bits(t.angles[0].force_k.to_bits()+1);
    rejected(&mut t,&target,&pin,&binding,"one-bit angle coefficient change","ordered angle");count+=1;
    let mut t=original.clone();t.bonds.swap(0,1);
    rejected(&mut t,&target,&pin,&binding,"reordered bond rows","ordered bond");count+=1;
    let mut t=original.clone();t.angles[0].i+=1;
    rejected(&mut t,&target,&pin,&binding,"changed angle atom identity","ordered angle");count+=1;
    let mut changed_target=target.clone();changed_target["atom_names"][0]=json!("changed");
    rejected(&mut original.clone(),&changed_target,&pin,&binding,"target JSON/parsed input disagreement","target changed");count+=1;
    for (case,key,change,reason) in [
        ("fresh atom identity mismatch","authority_topology",0,"atom identity"),
        ("actual preparation failure","preparation_execution",1,"execution does not bind"),
        ("wrong actual preparation PID","preparation_start",2,"execution does not bind"),
        ("wrong actual output bytes","preparation_execution",3,"execution does not bind"),
        ("wrong actual provider command","preparation_start",4,"command lineage"),
    ] {
        let mut b=binding.clone();
        changed_file(&mut b,key,temporary,count,|doc|match change {
            0=>doc["atom_names"][0]=json!("changed"),
            1=>doc["exit_code"]=json!(1),
            2=>doc["pid"]=json!(0),
            3=>doc["outputs"]=json!([]),
            4=>doc["argv"]=json!(["wrong-provider"]),
            _=>unreachable!(),
        });
        rejected(&mut original.clone(),&target,&pin,&b,case,reason);count+=1;
    }
    assert_eq!(raw,pin.read().unwrap());
    println!("{count} actual harmonic-binding CPU controls passed; no CUDA, Cargo or physical execution");
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--binding', type=Path, default=DEFAULT_BINDING)
    parser.add_argument('--binding-sha256', default=DEFAULT_BINDING_SHA)
    parser.add_argument('--prism-rlib', type=Path, default=ROLE / 'target/release/deps/libprism_nhs-a1611f191df22bfc.rlib')
    args = parser.parse_args()
    if str(args.binding).startswith('/nonexistent/'):
        _require_env('PRISM_BIPRISM_RESOLUTION_ROOT')
    prefix = args.output_prefix
    if not prefix.is_absolute() or not prefix.parent.is_dir():
        parser.error('output prefix requires an existing absolute parent')
    if any(path.exists() for path in (prefix, prefix.with_suffix('.rs'), prefix.with_suffix('.txt'))):
        parser.error('output prefix already exists')
    if sha(args.binding) != args.binding_sha256:
        raise ValueError('binding identity changed')
    binding = json.loads(args.binding.read_text())
    source = ROLE / 'crates/prism-nhs/src/biprism_run.rs'
    owner = ROLE / 'crates/prism-nhs/src/biprism_harmonic_binding.rs'
    input_source = ROLE / 'crates/prism-nhs/src/input.rs'
    text = source.read_text()
    driver = ('use anyhow::{anyhow,ensure,Context,Result};\n'
              'use serde::{Deserialize,Serialize};use serde_json::{Value,json};\n'
              'use sha2::{Digest,Sha256};use std::{fs,path::{Path,PathBuf}};\n'
              'mod input { pub use prism_nhs::input::*; }\n'
              'use input::{PrismPrepTopology,HarmonicEnergyConvention,NonbondedExecutionPolicy};\n'
              '#[derive(Clone,Debug,Serialize,Deserialize)]\n#[serde(deny_unknown_fields)]\n'
              + item(text, r'pub struct PinnedFile\b') + '\n'
              + item(text, r'impl PinnedFile\b') + '\n'
              + '\n'.join(item(text, r'fn ' + name + r'(?=[<(])') for name in ('hash', 'serr', 'validate_native_constructor_inputs'))
              + '\n#[path=' + json.dumps(str(owner)) + '] mod harmonic_binding;\n' + DRIVER)
    prefix.with_suffix('.rs').write_text(driver)
    libs, fingerprint = matching_libraries(args.prism_rlib)
    command = ['rustc', '--edition=2021', str(prefix.with_suffix('.rs')),
               '-L', 'dependency=' + str(args.prism_rlib.parent), '-o', str(prefix)]
    for name, path in libs.items():
        command += ['--extern', name + '=' + str(path)]
    for path in sorted((ROLE / 'target/release/build').glob('*/out')):
        command += ['-L', 'native=' + str(path)]
    for path in ['/usr/local/cuda/lib64', '/usr/local/cuda/lib64/stubs', str(ROLE / 'BUILD_INPUTS/sdst/lib')]:
        command += ['-L', 'native=' + path]
    pinned = [source, owner, input_source, Path(__file__), args.binding, fingerprint, *libs.values()]
    for key in ('target_topology', 'authority_topology', 'preparation_receipt', 'preparation_start',
                'preparation_execution', 'producer', 'source_pdb'):
        pin = binding[key]
        assert sha(Path(pin['path'])) == pin['sha256']
        pinned.append(Path(pin['path']))
    for row in binding['forcefield_resources']:
        pin = row['file']
        assert sha(Path(pin['path'])) == pin['sha256']
        pinned.append(Path(pin['path']))
    before = {str(path): sha(path) for path in pinned}
    log = ['scope: exact current binding/helpers linked to retained actual input owner; CPU only',
           'source_and_input_pins ' + json.dumps(before, sort_keys=True), 'compile ' + repr(command)]
    result = subprocess.run(command, text=True, capture_output=True)
    log += ['compile_exit ' + str(result.returncode), result.stdout, result.stderr]
    if result.returncode == 0:
        with tempfile.TemporaryDirectory(prefix='biprism_harmonic_controls_') as temporary:
            result = subprocess.run([str(prefix), str(args.binding), temporary], text=True, capture_output=True)
            log += ['controls_exit ' + str(result.returncode), result.stdout, result.stderr]
    unchanged = before == {path: sha(Path(path)) for path in before}
    log += ['sources_and_inputs_unchanged ' + str(unchanged)]
    prefix.with_suffix('.txt').write_text('\n'.join(log))
    print(result.stdout[:12000])
    print(result.stderr[:4000])
    print('retained log:', prefix.with_suffix('.txt'))
    assert unchanged, 'source/library/input changed during controls'
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
