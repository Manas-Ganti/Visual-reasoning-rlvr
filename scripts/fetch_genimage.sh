#!/usr/bin/env bash
# Download the GenImage generator folders from Google Drive onto ARC.
#
# Fill in the LINKS table below, then:
#
#     tmux new -s genimage
#     bash scripts/fetch_genimage.sh
#
# Run it on a LOGIN or DATA-TRANSFER node — compute nodes generally have no
# outbound route. Downloads take hours, so use tmux (Ctrl-b d to detach,
# `tmux attach -t genimage` to come back).
#
# Each link may point at either a generator folder (ADM/) or just its val/
# subfolder. Both work: the manifest builder finds ai/nature class folders at
# any depth, and it derives the generator name from the top-level directory
# this script creates — not from anything inside the download. You only need a
# few hundred images per class, and GenImage's val split alone has thousands,
# so linking val/ cuts the download by roughly 25x for identical results.
#
# Re-running skips generators that already have files, so an interrupted run
# resumes by just running it again. Use --force to re-download anyway.
#
#     bash scripts/fetch_genimage.sh --dry-run          # show the plan, download nothing
#     bash scripts/fetch_genimage.sh --dest /scratch/$USER/GenImage
#     METHOD=rclone RCLONE_REMOTE=gdrive bash scripts/fetch_genimage.sh
#
# gdown is the default and needs no setup. Switch to rclone if Drive starts
# returning "quota exceeded for this file" (that is the SHARE being rate
# limited, not your account — retrying gdown usually just burns time), or if a
# generator folder holds raw image trees rather than zips, where rclone's
# parallel small-file transfers are far faster.

set -euo pipefail

# --------------------------------------------------------------------------- #
# EDIT THIS: one "Name|URL" per generator. Leave a URL empty to skip that one.
# Get each link from the Drive folder listing: right-click the generator folder
# -> Copy link. Downloading the four you need separately (rather than the whole
# parent folder) gives restartable progress instead of one long job that can
# fail near the end.
#
# Parent folder: https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS
# --------------------------------------------------------------------------- #
LINKS=(
  "ADM|"
  "BigGAN|"
  "Midjourney|"
  "stable_diffusion_v_1_4|"
  # Optional extras — fill in only if you want more than the four-generator mix.
  "stable_diffusion_v_1_5|"
  "glide|"
  "VQDM|"
  "wukong|"
)

# --------------------------------------------------------------------------- #
DEST="${DEST:-/projects/${USER}/GenImage}"
METHOD="${METHOD:-gdown}"                 # gdown | rclone
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"  # rclone remote name, for METHOD=rclone
TRANSFERS="${TRANSFERS:-8}"
FORCE=0
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

LOG_DIR="${LOG_DIR:-$DEST/_logs}"
say() { printf '[fetch] %s\n' "$*"; }
die() { printf '[fetch] ERROR: %s\n' "$*" >&2; exit 1; }

# Drive folder -> the canonical name data/build_manifest_genimage.py resolves it
# to, so the script can print the exact --generators value at the end.
canonical() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    *stable_diffusion_v_1_4*|*sdv1.4*|*sdv4*) echo "sdv1.4" ;;
    *stable_diffusion_v_1_5*|*sdv1.5*|*sdv5*) echo "sdv1.5" ;;
    *midjourney*) echo "midjourney" ;;
    *biggan*)     echo "biggan" ;;
    *glide*)      echo "glide" ;;
    *wukong*)     echo "wukong" ;;
    *vqdm*)       echo "vqdm" ;;
    *adm*)        echo "adm" ;;
    *) printf '%s' "$1" | tr '[:upper:]' '[:lower:]' ;;
  esac
}

count_files() { find "$1" -type f ! -name '*.zip' 2>/dev/null | wc -l | tr -d ' '; }

# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
case "$METHOD" in
  gdown|rclone) ;;
  *) die "METHOD must be gdown or rclone (got '$METHOD')" ;;
esac

# Checked lazily, on the first generator that actually needs downloading — so a
# re-run that only reprints the verification table works on a machine with
# neither tool installed.
_tool_checked=0
GDOWN=""

# `pip install --user` puts the gdown launcher in ~/.local/bin, which is not on
# PATH on a default ARC login shell — so an installed gdown looks missing to
# `command -v`. Fall back to the module form, which always works if the package
# is importable by the active python.
resolve_gdown() {
  # Explicit override wins: GDOWN="/path/to/gdown" or GDOWN="python3.11 -m gdown"
  if [ -n "${GDOWN:-}" ]; then
    say "gdown       : $GDOWN (from \$GDOWN)"
    return 0
  fi
  local cand
  for cand in "gdown" "$HOME/.local/bin/gdown" "python -m gdown" "python3 -m gdown" \
              "$(command -v python3 || echo python3) -m gdown"; do
    if $cand --version >/dev/null 2>&1; then
      GDOWN="$cand"
      say "gdown       : $GDOWN"
      return 0
    fi
  done

  # Nothing worked. Print what we actually inspected — "not found" alone sends
  # people to reinstall a gdown that is already installed, when the real cause
  # is almost always a python/PATH mismatch (pip --user installs into
  # ~/.local/lib/pythonX.Y, which a DIFFERENT python cannot import).
  printf '\n[fetch] gdown could not be resolved. Diagnostics:\n' >&2
  printf '  PATH                : %s\n' "$PATH" >&2
  printf '  which python        : %s (%s)\n' \
    "$(command -v python || echo none)" "$(python -V 2>&1 || true)" >&2
  printf '  which python3       : %s (%s)\n' \
    "$(command -v python3 || echo none)" "$(python3 -V 2>&1 || true)" >&2
  printf '  ~/.local/bin/gdown  : %s\n' \
    "$([ -e "$HOME/.local/bin/gdown" ] && ls -l "$HOME/.local/bin/gdown" || echo missing)" >&2
  printf '  CONDA_DEFAULT_ENV   : %s\n' "${CONDA_DEFAULT_ENV:-none}" >&2
  printf '  import gdown        : %s\n' \
    "$(python -c 'import gdown;print(gdown.__version__)' 2>&1 | tail -1)" >&2
  die "Fix, in order of likelihood:
       1. Install into the env you actually run:   conda activate vrr && pip install --upgrade gdown
       2. Put the user bin on PATH:                export PATH=\"\$HOME/.local/bin:\$PATH\"
       3. Point at it explicitly:                  GDOWN=\"/full/path/to/gdown\" bash \$0
       4. Skip gdown entirely:                     METHOD=rclone RCLONE_REMOTE=gdrive bash \$0"
}

require_tool() {
  [ "$_tool_checked" -eq 1 ] && return 0
  if [ "$METHOD" = "gdown" ]; then
    resolve_gdown
  else
    command -v rclone >/dev/null 2>&1 || die "rclone not found. Load a module or install it."
    rclone listremotes 2>/dev/null | grep -q "^${RCLONE_REMOTE}:" || die \
      "rclone remote '${RCLONE_REMOTE}:' is not configured. Run: rclone config
       (new remote, type 'drive', scope 'drive.readonly')"
  fi
  _tool_checked=1
}

[ "$DRY_RUN" -eq 0 ] && mkdir -p "$DEST" "$LOG_DIR"

say "destination : $DEST"
say "method      : $METHOD"
[ "$DRY_RUN" -eq 1 ] && say "DRY RUN — nothing will be downloaded"

# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
planned=(); skipped=(); missing=()
for entry in "${LINKS[@]}"; do
  name="${entry%%|*}"
  url="${entry#*|}"
  target="$DEST/$name"

  if [ -z "$url" ]; then
    missing+=("$name")
    continue
  fi
  if [ "$FORCE" -eq 0 ] && [ -d "$target" ] && [ "$(count_files "$target")" -gt 0 ]; then
    say "skip $name — already has $(count_files "$target") files (--force to redo)"
    skipped+=("$name")
    continue
  fi
  planned+=("$name")

  if [ "$DRY_RUN" -eq 1 ]; then
    say "would download $name -> $target"
    continue
  fi

  require_tool
  say "downloading $name -> $target"
  mkdir -p "$target"
  log="$LOG_DIR/${name}.log"
  # Never let one failed generator abort the others — record and carry on, so an
  # overnight run does not die at 3am on a single Drive quota error.
  if [ "$METHOD" = "gdown" ]; then
    # --remaining-ok lifts gdown's 50-files-per-folder cap.
    $GDOWN --folder --remaining-ok -O "$target" "$url" 2>&1 | tee "$log" || \
      say "WARNING: $name failed — see $log (try METHOD=rclone)"
  else
    rclone copy "${RCLONE_REMOTE}:${url}" "$target" -P --transfers "$TRANSFERS" 2>&1 | tee "$log" || \
      say "WARNING: $name failed — see $log"
  fi
done

if [ "$DRY_RUN" -eq 1 ]; then
  say "plan: ${#planned[@]} to download, ${#skipped[@]} already present, ${#missing[@]} without a link"
  [ ${#missing[@]} -gt 0 ] && say "no link yet for: ${missing[*]}"
  exit 0
fi

# --------------------------------------------------------------------------- #
# Extract any archives
# --------------------------------------------------------------------------- #
zips="$(find "$DEST" -name '*.zip' -type f 2>/dev/null | wc -l | tr -d ' ')"
if [ "$zips" -gt 0 ]; then
  say "extracting $zips archive(s)"
  find "$DEST" -name '*.zip' -type f -print0 | while IFS= read -r -d '' z; do
    say "  unzip $(basename "$z")"
    unzip -q -o "$z" -d "$(dirname "$z")" || say "  WARNING: failed to unzip $z"
  done
fi

# --------------------------------------------------------------------------- #
# Verify — the check that catches a half-downloaded generator
# --------------------------------------------------------------------------- #
say ""
say "=== verification ==="
printf '  %-28s %10s %10s\n' "generator" "ai" "nature"
ok=1; found=()
for entry in "${LINKS[@]}"; do
  name="${entry%%|*}"
  target="$DEST/$name"
  [ -d "$target" ] || continue
  n_ai=$(find "$target" \( -path '*/ai/*' -o -path '*/fake/*' \) -type f 2>/dev/null | wc -l | tr -d ' ')
  n_re=$(find "$target" \( -path '*/nature/*' -o -path '*/real/*' \) -type f 2>/dev/null | wc -l | tr -d ' ')
  printf '  %-28s %10s %10s\n' "$name" "$n_ai" "$n_re"
  if [ "$n_ai" -eq 0 ] || [ "$n_re" -eq 0 ]; then
    ok=0
  else
    found+=("$(canonical "$name")")
  fi
done

say ""
if [ "$ok" -eq 0 ]; then
  say "WARNING: a generator has 0 in one class. The manifest builder SKIPS an empty"
  say "         class with only a '! <gen> label=N: no images' line rather than failing,"
  say "         so fix this now — otherwise you train on a one-sided generator."
  say "         Re-run with --force for that generator, or check the folder layout."
fi

if [ ${#found[@]} -gt 0 ]; then
  gens=$(printf '%s,' "${found[@]}"); gens="${gens%,}"
  say "Next — build the manifest on the login node (CPU only):"
  say ""
  say "  export VRR_DATASET=genimage"
  say "  python data/build_manifest_genimage.py \\"
  say "      --src $DEST \\"
  say "      --generators $gens \\"
  say "      --per-generator 400 --copy"
  say ""
  say "Read the per-(generator,class) pool sizes it prints BEFORE trusting the result:"
  say "if the real pools are much smaller than the ai pools, --min-edge 512 is"
  say "rejecting ImageNet photos (often ~500x375) while keeping every 512/1024-native"
  say "generation — which makes resolution itself a shortcut feature. Rebuild with"
  say "--min-edge 384 if you see it."
else
  die "nothing usable was downloaded — check $LOG_DIR"
fi
