# CCM Recovery System Redesign

**Date**: 2026-03-27
**Status**: Design doc — informed by 3-day failure and comprehensive audit

## Root Causes of Failure

1. **Hardcoded project path** in recovery.py — only works for hydrogenation jobs
2. **Upload source not stored in Job record** — recovery can't find original files
3. **60s recovery loop** — too slow for fast recovery
4. **No retry on instance creation** — one bad offer = permanent failure
5. **No validation** — files not verified, job start not verified

## Architecture Fix: Store Upload Source in Job

The fundamental fix: during job submission, store the `upload.source` path in the Job record so recovery can find the original files without hardcoded paths.

### Changes Required:

1. **cli/jobs.py**: Store `upload_json` in Job constructor (like retry_json fix)
2. **core/models.py**: Add `upload_json` field if not present (check first)
3. **core/recovery.py**: Read `upload_json` to find original files, remove hardcoded path
4. **daemon/monitor.py**: Reduce recovery loop to 15s, add retry on instance creation

### Implementation Priority:

| Fix | Impact | Effort | Priority |
|-----|--------|--------|----------|
| Store upload_json + remove hardcoded path | CRITICAL | 30 min | P0 |
| Reduce recovery loop to 15s | HIGH | 5 min | P0 |
| Retry instance creation (try 3 offers) | HIGH | 30 min | P1 |
| Validate file upload success | MEDIUM | 15 min | P1 |
| Verify job actually started after launch | MEDIUM | 15 min | P2 |
| Move NAMD params to job config | MEDIUM | 30 min | P2 |
