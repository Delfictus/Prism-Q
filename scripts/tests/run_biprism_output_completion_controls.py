#!/usr/bin/env python3
"""Compile exact pure completion predicates/tests from the canonical host owner.
This is a CPU validator control, not native execution or production completion.
"""
from pathlib import Path
import argparse, hashlib, subprocess
ROLE=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-prefix',type=Path,required=True)
prefix=parser.parse_args().output_prefix
if not prefix.is_absolute() or not prefix.parent.is_dir():
    parser.error('output prefix must be absolute with an existing parent directory')
source=ROLE/'crates/prism-nhs/src/biprism_run.rs'
text=source.read_text()
def extract(name):
    at=text.index('fn '+name+'(')
    brace=text.index('{',at)
    depth=0
    for i in range(brace,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return text[at:i+1]
    raise AssertionError(name)
helpers=['retain','retain_json','validate_native_shared_clock_rows','validate_pathb_population',
 'retain_complete_pathb_state','validate_completed_output_extent','require_complete_terminal_outputs','retain_terminal_failure']
tests=['output_clock_join_rejects_single_leaf_or_uniform_wrong_native_time',
 'output_completion_requires_all_eighty_chunks_and_eight_leaves',
 'output_terminal_error_cannot_retain_success_or_hide_prior_failure',
 'output_pathb_population_must_equal_selected_source',
 'output_pathb_serialized_count_and_extent_are_checked_before_acknowledgement']
driver=prefix.with_suffix('.rs')
driver.write_text('use anyhow::{anyhow,ensure,Context,Result};\nuse serde_json::{json,Value};\n'
 'use std::{fs::{self,File,OpenOptions},path::Path,io::{Read,Seek,SeekFrom,Write}};\nconst LEAVES:usize=8;\n'
 +'\n'.join(extract(n) for n in helpers)+'\n'
 +'\n'.join('#[test]\n'+extract(n) for n in tests)+'\n')
deps=ROLE/'target/release/deps'
libs={name:max(deps.glob('lib'+name+'-*.rlib'),key=lambda p:p.stat().st_mtime_ns) for name in ('anyhow','serde_json')}
binary=prefix
cmd=['rustc','--edition=2021','--test',str(driver),'-L','dependency='+str(deps)]
for name,p in libs.items():cmd+=['--extern',name+'='+str(p)]
cmd+=['-o',str(binary)]
log=['scope: exact canonical CPU completion predicates; no CUDA, Cargo or engine execution',
 'source_sha256 '+hashlib.sha256(source.read_bytes()).hexdigest(), 'compile '+repr(cmd)]
p=subprocess.run(cmd,text=True,capture_output=True);log += ['compile_exit '+str(p.returncode),p.stdout,p.stderr]
if p.returncode==0:
    p=subprocess.run([str(binary),'--nocapture'],text=True,capture_output=True)
    log += ['tests_exit '+str(p.returncode),p.stdout,p.stderr]
prefix.with_suffix('.txt').write_text('\n'.join(log))
print('\n'.join(log));raise SystemExit(p.returncode)
