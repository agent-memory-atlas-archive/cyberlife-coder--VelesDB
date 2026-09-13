#!/usr/bin/env bash
# Shared helpers for the VelesDB agent-hooks scripts.
# Sourced by all five event hooks — not meant to be run directly.

umask 077

# Resolve the nearest existing directory physically (`pwd -P`). This makes an
# edit through an intermediate directory symlink inherit the policy of the
# repository whose bytes are actually targeted, including for a not-yet-
# existing file below that directory.
physical_policy_start() {
  local candidate="$1"
  local depth=0
  while [ ! -d "$candidate" ] && [ "$depth" -lt 40 ]; do
    [ "$candidate" = "/" ] && break
    candidate="$(dirname "$candidate")"
    depth=$((depth + 1))
  done
  [ -d "$candidate" ] || return 1
  (cd "$candidate" 2>/dev/null && pwd -P)
}

resolve_final_symlink() {
  local current="$1"
  local link
  local depth=0
  while [ -L "$current" ] && [ "$depth" -lt 20 ]; do
    link="$(readlink "$current")" || return 1
    case "$link" in
      /*) current="$link" ;;
      *) current="$(dirname "$current")/$link" ;;
    esac
    depth=$((depth + 1))
  done
  [ ! -L "$current" ] || return 1
  printf '%s' "$current"
}

# require_jq: fail loudly (not silently) if jq is missing, since every hook
# builds its JSON output through jq to get escaping right.
require_jq() {
  if ! command -v jq >/dev/null 2>&1; then
    echo "velesdb agent-hooks: 'jq' is required but was not found on PATH." >&2
    exit 1
  fi
}

# resolve_config CWD
# Walks up from CWD looking for a .velesdb-hooks.json file (project root
# convention). Sets PROJECT, SESSION, CONFIG_ROOT and ENFORCE_LEARNING_LOOP
# globals. Falls back to project=basename(cwd) and session="rolling" when no config file is found
# or a field is missing — so the hooks work with zero setup, but a project
# can pin stable identifiers via the config file.
resolve_config() {
  local start_dir="$1"
  local physical_start
  physical_start="$(physical_policy_start "$start_dir")" || return 1
  start_dir="$physical_start"
  local dir="$start_dir"
  local config=""
  local depth=0

  while [ "$depth" -lt 20 ]; do
    if [ -f "$dir/.velesdb-hooks.json" ]; then
      config="$dir/.velesdb-hooks.json"
      break
    fi
    if [ "$dir" = "/" ] || [ -z "$dir" ]; then
      break
    fi
    dir="$(dirname "$dir")"
    depth=$((depth + 1))
  done

  PROJECT=""
  SESSION=""
  CONFIG_ROOT=""
  ENFORCE_LEARNING_LOOP="false"
  if [ -n "$config" ] && jq -e . "$config" >/dev/null 2>&1; then
    PROJECT="$(jq -r '.project // empty' "$config")"
    SESSION="$(jq -r '.session // empty' "$config")"
    ENFORCE_LEARNING_LOOP="$(jq -r 'if .enforce_learning_loop == true then "true" else "false" end' "$config")"
    CONFIG_ROOT="$(cd "$dir" 2>/dev/null && pwd -P)" || CONFIG_ROOT="$dir"
  fi

  if [ -z "$PROJECT" ]; then
    PROJECT="$(basename "$start_dir")"
  fi
  if [ -z "$SESSION" ]; then
    SESSION="rolling"
  fi
}

# learning_loop_enabled: true only for a project that explicitly opted in.
# The hooks are installed user-wide, so a missing or malformed project config
# must fail open instead of blocking edits in unrelated repositories.
learning_loop_enabled() {
  [ "${ENFORCE_LEARNING_LOOP:-false}" = "true" ]
}

# learning_marker_identity SESSION_ID: scope mechanical learning markers to
# both the host session and the opted-in repository. One Claude session can
# change cwd, so a recall in repo A must never unlock an edit in repo B.
learning_marker_identity() {
  printf '%s\n%s' "$1" "${CONFIG_ROOT:-$PWD}"
}

# successful_memory_recall PAYLOAD
# A recall counts only after a VelesDB MCP tool returned a successful result.
# `compile_context` counts only when it actually requested memory_scope.
successful_memory_recall() {
  local payload="$1"
  local tool_name
  tool_name="$(printf '%s' "$payload" | jq -r '.tool_name // empty')"

  case "$tool_name" in
    mcp__velesdb-memory__recall|mcp__velesdb_memory__recall|\
    mcp__velesdb-memory__recall_fused|mcp__velesdb_memory__recall_fused|\
    mcp__velesdb-memory__recall_where|mcp__velesdb_memory__recall_where|\
    mcp__velesdb-memory__entity|mcp__velesdb_memory__entity|\
    mcp__velesdb-memory__why|mcp__velesdb_memory__why)
      ;;
    mcp__velesdb-memory__compile_context|mcp__velesdb_memory__compile_context)
      printf '%s' "$payload" | jq -e '.tool_input.memory_scope != null' >/dev/null 2>&1 || return 1
      ;;
    *)
      return 1
      ;;
  esac

  successful_tool_response "$payload"
}

# successful_tool_response PAYLOAD: the MCP call behind a PostToolUse payload
# returned a non-empty text result and no error.
successful_tool_response() {
  local payload="$1"

  # Claude Code's MCP PostToolUse payload exposes the successful result's
  # `content` array directly. Some older/synthetic harnesses use the complete
  # CallToolResult envelope; keep that form strict so `{}` cannot unlock.
  printf '%s' "$payload" | jq -e '
    if ((.tool_response | type) == "array") then
      any(
        .tool_response[];
        (type == "object")
        and (.type == "text")
        and ((.text? | type) == "string")
        and ((.text | length) > 0)
      )
    else
      ((.tool_response | type) == "object")
      and ((.tool_response.content? | type) == "array")
      and any(
        .tool_response.content[];
        (type == "object")
        and (.type == "text")
        and ((.text? | type) == "string")
        and ((.text | length) > 0)
      )
      and ((.tool_response.isError // .tool_response.is_error // false) == false)
      and ((.tool_response.error? // null) == null)
    end
  ' >/dev/null 2>&1
}

# read_stdin_payload: read the hook's JSON payload from stdin exactly once.
read_stdin_payload() {
  cat
}

# run_with_watchdog SECONDS OUTFILE CMD...
# Run CMD with stdout captured into OUTFILE, killing it after SECONDS.
# Returns CMD's exit status, or 124 if it had to be killed.
#
# Why not `timeout`: it is GNU coreutils, absent from a stock macOS (where it
# is `gtimeout`, if installed at all). A hook runs on every tool call and must
# never hang the agent, so the watchdog is pure bash and always available.
#
# This matters most for post-tool-use.sh: a velesdb-memory binary predating
# `compile-stdin` ignores the subcommand and starts the MCP *server* on the
# stdin we just piped it. Without this bound, that is a hung hook.
run_with_watchdog() {
  local secs="$1"
  local outfile="$2"
  shift 2

  "$@" >"$outfile" 2>/dev/null &
  local pid=$!
  local waited=0
  local limit=$((secs * 10))

  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$limit" ]; then
      kill -9 "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      return 124
    fi
    sleep 0.1
    waited=$((waited + 1))
  done

  wait "$pid"
}

# safe_marker_key VALUE: a bounded, filename-safe identity for host-provided
# session/tool ids. Hook payload ids are normally UUIDs, but treating raw
# values as path components would make slashes or an overlong id break every
# later hook in that session.
safe_marker_key() {
  local value="$1"
  local checksum
  checksum="$(printf '%s' "$value" | cksum)"
  printf '%s' "${checksum// /-}"
}

# marker_base_dir: private, per-UID storage for all hook state. Refuse links,
# foreign ownership, and non-directories before returning a writable path.
marker_base_dir() {
  local parent="${TMPDIR:-/tmp}"
  local dir="${parent}/velesdb-agent-hooks-${UID}"
  if [ -L "$dir" ]; then
    return 1
  fi
  if [ ! -e "$dir" ]; then
    mkdir -m 700 "$dir" 2>/dev/null || [ -d "$dir" ] || return 1
  fi
  [ -d "$dir" ] && [ ! -L "$dir" ] && [ -O "$dir" ] || return 1
  chmod 700 "$dir" || return 1
  printf '%s' "$dir"
}

# sentinel_path KIND SESSION_ID: path to a session-scoped marker used by Stop,
# PreCompact, the successful-recall gate, and edit-dirty checkpoints.
sentinel_path() {
  local kind="$1"
  local session_id="$2"
  local dir
  local key
  dir="$(marker_base_dir)" || return 1
  key="$(safe_marker_key "$session_id")"
  printf '%s/%s-%s.marker' "$dir" "$kind" "$key"
}

# write_private_marker PATH CONTENT: atomically replace a private regular file
# without following a pre-existing final-component symlink.
write_private_marker() {
  local path="$1"
  local value="$2"
  local tmp
  if [ -L "$path" ] || { [ -e "$path" ] && [ ! -f "$path" ]; }; then
    return 1
  fi
  tmp="$(mktemp "${path}.tmp.XXXXXX")" || return 1
  if ! printf '%s\n' "$value" > "$tmp"; then
    rm -f "$tmp"
    return 1
  fi
  mv -f "$tmp" "$path"
}

# Marker readers must apply the same final-component rule as writers. `-f`
# alone follows a symlink and would let linked/corrupt state masquerade as a
# completed recall or continuation marker.
valid_private_marker() {
  local path="$1"
  [ -f "$path" ] && [ ! -L "$path" ] && [ -O "$path" ]
}

# private_temp_file PREFIX: reserve an unpredictable mode-0600 regular file in
# the owned mode-0700 state directory before a subprocess writes its result.
# This avoids predictable redirection targets and final-symlink truncation.
private_temp_file() {
  local prefix="$1"
  local dir
  dir="$(marker_base_dir)" || return 1
  mktemp "${dir}/${prefix}.XXXXXX"
}

touch_private_marker() {
  write_private_marker "$1" ""
}

# Alias named for record callers.
write_json_atomically() {
  write_private_marker "$1" "$2"
}

# project_record: serialize the currently-resolved opted-in memory identity.
project_record() {
  jq -cn \
    --arg project "$PROJECT" \
    --arg session "$SESSION" \
    --arg root "$CONFIG_ROOT" \
    '{project: $project, session: $session, root: $root}'
}

valid_project_record() {
  jq -e '
    type == "object"
    and ((keys | sort) == ["project", "root", "session"])
    and ((.project | type) == "string")
    and ((.session | type) == "string")
    and ((.root | type) == "string" and (.root | length) > 0)
  ' "$1" >/dev/null 2>&1
}

# record_dir_path KIND SESSION_ID: a session-specific directory whose records
# are independently atomically replaced. One file per project identity avoids
# lost updates when multiple PreToolUse hooks run concurrently.
record_dir_path() {
  local marker
  marker="$(sentinel_path "$1" "$2")" || return 1
  printf '%s.records' "${marker%.marker}"
}

# record_project_json DIR RECORD: retain one exact identity without overwriting
# a malformed/colliding record. The content-derived bounded key makes parallel
# writes for different repositories independent and identical writes harmless.
record_project_json() {
  local dir="$1"
  local record="$2"
  local canonical
  local existing
  local key
  local path

  canonical="$(printf '%s' "$record" | jq -ce '
    select(type == "object")
    | select((keys | sort) == ["project", "root", "session"])
    | select((.project | type) == "string")
    | select((.session | type) == "string")
    | select((.root | type) == "string" and (.root | length) > 0)
    | {project, session, root}
  ' 2>/dev/null)" || return 1

  if [ -L "$dir" ]; then
    return 1
  fi
  mkdir -p "$dir" || return 1
  [ -d "$dir" ] && [ ! -L "$dir" ] || return 1

  key="$(safe_marker_key "$canonical")" || return 1
  path="$dir/$key.json"
  if [ -e "$path" ] || [ -L "$path" ]; then
    [ -f "$path" ] && [ ! -L "$path" ] || return 1
    existing="$(jq -c '{project, session, root}' "$path" 2>/dev/null)" || return 1
    [ "$existing" = "$canonical" ] || return 1
    return 0
  fi
  write_json_atomically "$path" "$canonical"
}

record_current_project() {
  record_project_json "$1" "$(project_record)"
}

recall_scope_present() {
  printf '%s' "$1" | jq -e '
    ((.tool_input.filter | type) == "object" and (.tool_input.filter | has("project")))
    or
    ((.tool_input.memory_scope | type) == "object" and (.tool_input.memory_scope | has("project")))
  ' >/dev/null 2>&1
}

recall_scope_project() {
  printf '%s' "$1" | jq -er '
    if ((.tool_input.filter | type) == "object" and (.tool_input.filter | has("project"))) then
      .tool_input.filter.project
    elif ((.tool_input.memory_scope | type) == "object" and (.tool_input.memory_scope | has("project"))) then
      .tool_input.memory_scope.project
    else
      empty
    end
    | select(type == "string" and length > 0)
  ' 2>/dev/null
}

recall_targets_current_project() {
  local payload="$1"
  local scoped_project
  if ! recall_scope_present "$payload"; then
    return 0
  fi
  scoped_project="$(recall_scope_project "$payload")" || return 1
  [ "$scoped_project" = "$PROJECT" ]
}

# --- The working context this conversation uses -------------------------------
# The configured `session` (`.velesdb-hooks.json`, else "rolling") is only a
# default. A conversation that keeps its state under another session — one per
# campaign, say — must be reminded of THAT one at SessionStart, PreCompact and
# Stop: naming the default after a compaction makes it load a stale context, or
# save over one another conversation owns. PostToolUse records the session of
# each successful save_working_context, and of each load_working_context that
# found one, per host session and for the current project only; the reminders
# adopt it.

# valid_working_session NAME: a session name safe to quote inside a reminder.
# Bash's =~ anchors the whole string; jq's test("^…$") would anchor one line
# and let a newline smuggle text into the model's context.
valid_working_session() {
  local pattern='^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
  [[ "$1" =~ $pattern ]]
}

# working_context_found PAYLOAD: the load_working_context result says found.
working_context_found() {
  printf '%s' "$1" | jq -e '
    [ (if ((.tool_response | type) == "array") then .tool_response[] else .tool_response.content[]? end)
      | select(type == "object" and .type == "text")
      | .text | fromjson? | select(type == "object") | .found ]
    | any(. == true)
  ' >/dev/null 2>&1
}

# working_context_call PAYLOAD: print the session a successful working-context
# call used for the current project; fail for any other call.
working_context_call() {
  local payload="$1"
  local tool_name
  local session
  tool_name="$(printf '%s' "$payload" | jq -r '.tool_name // empty')"
  case "$tool_name" in
    mcp__velesdb-memory__save_working_context|mcp__velesdb_memory__save_working_context)
      ;;
    mcp__velesdb-memory__load_working_context|mcp__velesdb_memory__load_working_context)
      # A load that found nothing names a session nobody keeps: a typo must
      # not redirect every later reminder.
      working_context_found "$payload" || return 1
      ;;
    *)
      return 1
      ;;
  esac
  successful_tool_response "$payload" || return 1
  [ "$(printf '%s' "$payload" | jq -r '.tool_input.project // empty')" = "$PROJECT" ] || return 1
  session="$(printf '%s' "$payload" | jq -r '.tool_input.session // empty')"
  valid_working_session "$session" || return 1
  printf '%s' "$session"
}

# remember_working_session SESSION_ID PAYLOAD: record the session of a
# successful working-context call for the current project.
remember_working_session() {
  local session
  local marker
  session="$(working_context_call "$2")" || return 1
  marker="$(sentinel_path "working-session" "$1")" || return 1
  write_private_marker "$marker" \
    "$(jq -cn --arg project "$PROJECT" --arg session "$session" '{project: $project, session: $session}')"
}

# adopt_working_session SESSION_ID: set SESSION to the working context this host
# session last saved or loaded for the current project. Fails, leaving SESSION
# as configured, when there is none.
adopt_working_session() {
  local marker
  local session
  [ -n "$1" ] || return 1
  marker="$(sentinel_path "working-session" "$1")" || return 1
  valid_private_marker "$marker" || return 1
  session="$(jq -r --arg project "$PROJECT" '
    if type == "object" and .project == $project and (.session | type) == "string"
    then .session else empty end
  ' "$marker" 2>/dev/null)" || return 1
  valid_working_session "$session" || return 1
  SESSION="$session"
}

# promote_pending_recall DIR RECALL_KIND HOST_SESSION PAYLOAD
#
# An explicit project scope takes precedence. Without one, prefer the pending
# record for the current opted-in root; from an unconfigured cwd, a sole target
# is unambiguous. Return 0 when promoted, 1 when no pending record exists, 2 on
# malformed state/I/O, and 3 when valid pending state does not match the recall.
promote_pending_recall() {
  local dir="$1"
  local recall_kind="$2"
  local host_session="$3"
  local payload="$4"
  local file
  local root
  local project
  local scoped_project
  local selected_root=""
  local ambiguous="false"
  local scope_present="false"
  local canonical
  local expected
  local marker_id
  local marker_path
  local -a files=()

  [ -L "$dir" ] && return 2
  if [ ! -d "$dir" ]; then
    [ -e "$dir" ] && return 2
    return 1
  fi
  for file in "$dir"/*.json; do
    [ -e "$file" ] || [ -L "$file" ] || continue
    [ -f "$file" ] && [ ! -L "$file" ] && valid_project_record "$file" || return 2
    canonical="$(jq -c '{project, session, root}' "$file")" || return 2
    expected="$(safe_marker_key "$canonical")" || return 2
    expected="${expected}.json"
    [ "$(basename "$file")" = "$expected" ] || return 2
    files+=("$file")
  done
  if [ "${#files[@]}" -eq 0 ]; then
    rmdir "$dir" 2>/dev/null && return 1
    return 2
  fi

  if recall_scope_present "$payload"; then
    scope_present="true"
    scoped_project="$(recall_scope_project "$payload")" || return 3
  fi
  if [ "$scope_present" = "true" ]; then
    for file in "${files[@]}"; do
      project="$(jq -r '.project' "$file")" || return 2
      [ "$project" = "$scoped_project" ] || continue
      root="$(jq -r '.root' "$file")" || return 2
      if [ -z "$selected_root" ]; then
        selected_root="$root"
      elif [ "$selected_root" != "$root" ]; then
        ambiguous="true"
      fi
    done
    [ "$ambiguous" = "false" ] || selected_root=""
  elif learning_loop_enabled; then
    for file in "${files[@]}"; do
      root="$(jq -r '.root' "$file")" || return 2
      if [ "$root" = "$CONFIG_ROOT" ]; then
        selected_root="$root"
        break
      fi
    done
  fi

  if [ -z "$selected_root" ] && ! learning_loop_enabled \
    && [ "$scope_present" = "false" ] && [ "${#files[@]}" -eq 1 ]; then
    selected_root="$(jq -r '.root' "${files[0]}")" || return 2
  fi
  [ -n "$selected_root" ] || return 3

  marker_id="$(printf '%s\n%s' "$host_session" "$selected_root")"
  marker_path="$(sentinel_path "$recall_kind" "$marker_id")" || return 2
  touch_private_marker "$marker_path" || return 2
  for file in "${files[@]}"; do
    root="$(jq -r '.root' "$file")" || return 2
    if [ "$root" = "$selected_root" ]; then
      rm -f "$file" || return 2
    fi
  done
  rmdir "$dir" 2>/dev/null || true
}
