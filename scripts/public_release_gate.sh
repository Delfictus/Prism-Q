#!/usr/bin/env bash
# Prism-Q public-release gate. Run on the FULL history of the branch to be pushed, before every push.
#
# Usage:
#   scripts/public_release_gate.sh --repo <git dir or worktree> --ref <branch>   # history mode (required before push)
#   scripts/public_release_gate.sh --dir <tree>                                  # tree-only pre-scan, no history
#
# Options:
#   --gitleaks <path>   gitleaks binary (default: $GITLEAKS, else `gitleaks` on PATH)
#   --out <dir>         report directory (default: a new temp dir)
#   --deny-file <file>  extra forbidden literals, one per line (usernames, host names, share names).
#                       Keep this file OUTSIDE the repository: its contents are what must never be published.
#   --max-file-mb <n>   per-file size limit (default 50)
#   --max-total-mb <n>  total size limit of the tree at --ref / of --dir (default 1024)
#   --strict            treat WARN categories (host-specific absolute paths) as failures
#
# Gates (all must pass):
#   G1 secrets      gitleaks over every commit (history mode) or the tree (dir mode), archives depth 2, redacted
#   G2 size         no file > max-file-mb (in the tree and in every blob reachable from --ref); tree < max-total-mb
#   G3 patterns     private keys / tokens / private-network IPs / home-directory paths / emails / deny-file literals
#   G4 file policy  no tracked build outputs, archives, generated GPU code, dumps or secrets files (tree and history)
#   G5 third party  manifests/THIRD_PARTY.json exists and has no REVIEW_REQUIRED verdicts
# Exit 0 only if every gate passes. Findings are redacted; the reports name file:line and the rule only.
set -uo pipefail

REPO="" REF="" DIR="" OUT="" DENY="" STRICT=0 MAXF=50 MAXT=1024
GL="${GITLEAKS:-gitleaks}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2;;
    --ref) REF="$2"; shift 2;;
    --dir) DIR="$2"; shift 2;;
    --gitleaks) GL="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --deny-file) DENY="$2"; shift 2;;
    --max-file-mb) MAXF="$2"; shift 2;;
    --max-total-mb) MAXT="$2"; shift 2;;
    --strict) STRICT=1; shift;;
    -h|--help) sed -n '2,24p' "$0"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
if [[ -n "$REPO" && -z "$REF" ]] || [[ -z "$REPO" && -z "$DIR" ]] || [[ -n "$REPO" && -n "$DIR" ]]; then
  echo "need exactly one of: --repo <path> --ref <branch>  |  --dir <tree>" >&2; exit 2
fi
command -v "$GL" >/dev/null 2>&1 || [[ -x "$GL" ]] || { echo "gitleaks not found: $GL" >&2; exit 2; }
command -v rg >/dev/null 2>&1 || { echo "ripgrep (rg) is required" >&2; exit 2; }
OUT="${OUT:-$(mktemp -d)}"; mkdir -p "$OUT"
FAIL=0
declare -A RESULT

note() { printf '%s\n' "$*" | tee -a "$OUT/summary.txt"; }
gate() { RESULT[$1]="$2"; [[ "$2" == PASS* || "$2" == WARN* ]] || FAIL=1; note "[$1] $2"; }

# Redact a match line: keep file:line, mask the matched text.
redact() { sed -E 's/^([^:]+:[0-9]+):.*/\1: <redacted>/'; }

note "prism-q public release gate  mode=$([[ -n $REPO ]] && echo "history ref=$REF" || echo dir)  gitleaks=$("$GL" version 2>/dev/null)"

# ------------------------------------------------------------------ G1 secrets
GLCFG=()
if [[ -n "$REPO" ]]; then
  git -C "$REPO" show "$REF:.gitleaks.toml" >"$OUT/gitleaks_config.toml" 2>/dev/null && GLCFG=(--config "$OUT/gitleaks_config.toml")
elif [[ -f "$DIR/.gitleaks.toml" ]]; then GLCFG=(--config "$DIR/.gitleaks.toml"); fi
if [[ -n "$REPO" ]]; then
  "$GL" git "$REPO" "${GLCFG[@]}" --log-opts="$REF" --redact --no-banner --max-archive-depth 2 \
      --report-format json --report-path "$OUT/gitleaks.json" --exit-code 3 >"$OUT/gitleaks.log" 2>&1
else
  "$GL" dir "$DIR" "${GLCFG[@]}" --redact --no-banner --max-archive-depth 2 \
      --report-format json --report-path "$OUT/gitleaks.json" --exit-code 3 >"$OUT/gitleaks.log" 2>&1
fi
rc=$?
n=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$OUT/gitleaks.json" 2>/dev/null || echo "?")
case $rc in
  0) gate G1_secrets "PASS (0 findings)";;
  3) gate G1_secrets "FAIL ($n findings; redacted report: $OUT/gitleaks.json)";;
  *) gate G1_secrets "ERROR (gitleaks exit $rc; see $OUT/gitleaks.log)";;
esac

# ------------------------------------------------------------------ G2 size
maxb=$((MAXF * 1024 * 1024)); maxt=$((MAXT * 1024 * 1024))
if [[ -n "$REPO" ]]; then
  git -C "$REPO" rev-list --objects "$REF" \
    | git -C "$REPO" cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
    | awk -v m="$maxb" '$1=="blob" && $3>m {print $3, $4}' >"$OUT/oversize.txt"
  total=$(git -C "$REPO" ls-tree -r -l "$REF" | awk '{s+=$4} END {print s+0}')
else
  find "$DIR" -type f -size +"${MAXF}"M -not -path '*/.git/*' -printf '%s %P\n' >"$OUT/oversize.txt"
  total=$(find "$DIR" -type f -not -path '*/.git/*' -printf '%s\n' | awk '{s+=$1} END {print s+0}')
fi
if [[ -s "$OUT/oversize.txt" ]]; then
  gate G2_size "FAIL ($(wc -l <"$OUT/oversize.txt") file(s) > ${MAXF} MB; see $OUT/oversize.txt)"
elif (( total > maxt )); then
  gate G2_size "FAIL (total $((total / 1048576)) MB > ${MAXT} MB)"
else
  gate G2_size "PASS (total $((total / 1048576)) MB, no file > ${MAXF} MB)"
fi

# ------------------------------------------------------------------ G3 patterns
# Patterns are written so this script does not match itself.
declare -A PAT=(
  [private_key]='-----BEGIN ([A-Z]+ )?PRIVATE KEY-----'
  [token_github]='\b(ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}'
  [token_aws]='\b(AKIA|ASIA)[0-9A-Z]{16}\b'
  [token_slack]='\bxox[abprs]-[A-Za-z0-9-]{10,}'
  [token_openai_anthropic]='\bsk-(ant-)?[A-Za-z0-9_-]{20,}'
  [token_generic_assign]='(?i)\b(api[_-]?key|secret|token|passwd|password)\b\s*[:=]\s*["'"'"'][^"'"'"'\s]{12,}["'"'"']'
  [ip_private]='\b(192\.168\.[0-9]{1,3}\.[0-9]{1,3}|10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})\b'
  [path_home]='(/home/[A-Za-z0-9._-]+/|/Users/[A-Za-z0-9._-]+/|C:\\Users\\)'
  [email]='\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.(com|org|net|io|edu|gov|ai|dev|co|app|me|info|biz|us|uk|de|fr|jp|cn|ch|ca|au|eu|ru|in)\b'
  [codex_home]='codex-home'
  [auth_files]='(auth\.json|\.git-credentials|rclone\.conf|cookies\.txt)'
)
declare -A WARNPAT=(
  [path_host_specific]='(/mnt/[A-Za-z0-9._-]+/|/var/lib/prismos|/opt/prismos|/run/prismos)'
)
EMAIL_ALLOW='(noreply@|no-reply@|example\.(com|org)|@users\.noreply\.github\.com|@(np|jnp|torch|cp)\.)'

# Files that list forbidden names by design (ignore rules, scanner config, this gate) are not pattern-scanned.
SELF_EXCLUDE=(.gitignore .gitleaks.toml scripts/public_release_gate.sh manifests/release_gate_allow.tsv)
# Reviewed per-file exceptions: manifests/release_gate_allow.tsv  (category<TAB>path<TAB>reason). Each needs a reason.
ALLOWTSV="$OUT/release_gate_allow.tsv"; : >"$ALLOWTSV"
if [[ -n "$REPO" ]]; then git -C "$REPO" show "$REF:manifests/release_gate_allow.tsv" >"$ALLOWTSV" 2>/dev/null || : >"$ALLOWTSV"
elif [[ -f "$DIR/manifests/release_gate_allow.tsv" ]]; then cp "$DIR/manifests/release_gate_allow.tsv" "$ALLOWTSV"; fi

scan() {  # $1=name $2=regex -> writes $OUT/pat_<name>.txt as redacted "path:line", echoes count
  local f="$OUT/pat_$1.txt" ex=()
  if [[ -n "$REPO" ]]; then
    for x in "${SELF_EXCLUDE[@]}"; do ex+=(":(exclude)$x"); done
    # every commit on the ref; keep the first (newest) occurrence of each path:line, drop the commit prefix
    git -C "$REPO" rev-list "$REF" | while read -r c; do
      git -C "$REPO" grep -I -n -P -e "$2" "$c" -- . "${ex[@]}" 2>/dev/null
    done | cut -d: -f2- | awk -F: '!seen[$1":"$2]++' >"$f.raw"
  else
    for x in "${SELF_EXCLUDE[@]}"; do ex+=(-g "!$x"); done
    (cd "$DIR" && rg -n -P --no-heading --hidden -g '!.git' "${ex[@]}" -- "$2" . 2>/dev/null) | sed 's#^\./##' >"$f.raw"
  fi
  [[ "$1" == email ]] && { grep -v -E "$EMAIL_ALLOW" "$f.raw" >"$f.raw2" || true; mv "$f.raw2" "$f.raw"; }
  python3 - "$f.raw" "$ALLOWTSV" "$1" >"$f.raw2" <<'PYA'
import sys
raw, tsv, cat = sys.argv[1:4]
allow = set()
for ln in open(tsv, encoding="utf-8", errors="replace"):
    parts = ln.rstrip("\n").split("\t")
    if len(parts) >= 3 and not ln.startswith("#") and parts[2].strip():
        allow.add((parts[0], parts[1]))
for ln in open(raw, encoding="utf-8", errors="replace"):
    path = ln.split(":", 1)[0]
    if (cat, path) not in allow and ("*", path) not in allow:
        sys.stdout.write(ln)
PYA
  mv "$f.raw2" "$f.raw"
  redact <"$f.raw" >"$f"; rm -f "$f.raw"
  wc -l <"$f"
}
pfail=0; summary=""
for k in "${!PAT[@]}"; do c=$(scan "$k" "${PAT[$k]}"); summary+=" $k=$c"; (( c > 0 )) && pfail=1; done
if [[ -n "$DENY" ]]; then
  [[ -r "$DENY" ]] || { gate G3_patterns "ERROR (deny file unreadable: $DENY)"; DENY=""; }
fi
if [[ -n "$DENY" ]]; then
  re=$(grep -v -E '^\s*(#|$)' "$DENY" | python3 -c 'import re,sys; print("|".join(re.escape(l.strip()) for l in sys.stdin if l.strip()))')
  if [[ -n "$re" ]]; then c=$(scan deny_literals "($re)"); summary+=" deny_literals=$c"; (( c > 0 )) && pfail=1; fi
fi
wsum=""; wfail=0
for k in "${!WARNPAT[@]}"; do c=$(scan "$k" "${WARNPAT[$k]}"); wsum+=" $k=$c"; (( c > 0 )) && wfail=1; done
if (( pfail )); then gate G3_patterns "FAIL (${summary# }; redacted lists: $OUT/pat_*.txt)"
else gate G3_patterns "PASS (${summary# })"; fi
if (( wfail )); then
  if (( STRICT )); then gate G3w_host_paths "FAIL (${wsum# })"; else gate G3w_host_paths "WARN (${wsum# }; scripts must take roots from config)"; fi
else gate G3w_host_paths "PASS"; fi

# ------------------------------------------------------------------ G4 file policy
POLICY='(^|/)(target|vendor)/|\.(ptx|cubin|fatbin|optixir|o|a|so|dylib|dll|exe|zip|tar|tgz|7z|zst|npz|npy|parquet|arrow|feather|h5|hdf5|nc|dcd|xtc|nsys-rep|ncu-rep|qdrep|pem|key|p12|pfx)$|\.bak(\.|$)|\.pre_[a-z_]+$|(^|/)(\.env|auth\.json|\.git-credentials|core|frames\.bin)$|_core_dump\.|(^|/)\.serena/'
ALLOW='^(cuda/tests/fixtures/[^/]+\.bin|crates/prism-nhs/assets/[^/]+\.bin)$'
if [[ -n "$REPO" ]]; then
  git -C "$REPO" log --format= --name-only "$REF" | sort -u >"$OUT/all_paths.txt"
else
  (cd "$DIR" && find . -type f -not -path './.git/*' -printf '%P\n' | sort -u) >"$OUT/all_paths.txt"
fi
grep -E "$POLICY" "$OUT/all_paths.txt" | grep -v -E "$ALLOW" >"$OUT/policy_violations.txt" || true
if [[ -s "$OUT/policy_violations.txt" ]]; then
  gate G4_file_policy "FAIL ($(wc -l <"$OUT/policy_violations.txt") path(s) ever present; see $OUT/policy_violations.txt)"
else gate G4_file_policy "PASS"; fi

# ------------------------------------------------------------------ G5 third party
tp=""
if [[ -n "$REPO" ]]; then tp=$(git -C "$REPO" show "$REF:manifests/THIRD_PARTY.json" 2>/dev/null)
elif [[ -f "$DIR/manifests/THIRD_PARTY.json" ]]; then tp=$(cat "$DIR/manifests/THIRD_PARTY.json"); fi
if [[ -z "$tp" ]]; then gate G5_third_party "FAIL (manifests/THIRD_PARTY.json missing)"
else
  r=$(printf '%s' "$tp" | python3 -c '
import json,sys
d=json.load(sys.stdin)
rows=d.get("entries",[])+d.get("excluded_build_inputs",[])
bad=[r.get("path") for r in rows if r.get("verdict")=="REVIEW_REQUIRED"]
print(len(bad)); [print(" -",b) for b in bad]' 2>&1)
  cnt=$(head -1 <<<"$r")
  printf '%s\n' "$r" >"$OUT/third_party_review.txt"
  if [[ "$cnt" == 0 ]]; then gate G5_third_party "PASS"
  else gate G5_third_party "FAIL ($cnt REVIEW_REQUIRED entr(y/ies); see $OUT/third_party_review.txt)"; fi
fi

python3 - "$OUT" <<'PY'
import json,sys,pathlib
out=pathlib.Path(sys.argv[1]); lines=(out/"summary.txt").read_text().splitlines()
json.dump({"summary":lines},open(out/"gate_result.json","w"),indent=2)
PY
note "reports: $OUT"
if (( FAIL )); then note "PUBLIC RELEASE GATE: FAIL"; exit 1; fi
note "PUBLIC RELEASE GATE: PASS"; exit 0
