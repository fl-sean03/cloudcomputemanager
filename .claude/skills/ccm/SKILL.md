---
name: ccm
description: CloudComputeManager — GPU cloud management for Vast.ai. Use when submitting cloud jobs, managing instances, or working on the CCM platform.
argument-hint: "[topic or task]"
---

# CCM Agent Skill

**User request**: $ARGUMENTS

## Rules

1. **NEVER call `vastai` directly.** Use `ccm jobs submit` / `ccm jobs cancel`.
2. **ALWAYS set `project:`** in YAML (unique per campaign).
3. **ALWAYS include `progress:`** in YAML if you want dashboard progress/rate/ETA.
4. For SSH: `ccm exec <job_id> "cmd"` or `ccm ssh <job_id>`.

## Full Reference

Read **`docs/usage.md`** for: YAML schema, CLI commands, progress regex examples, SDK, resilience, multi-agent rules, dashboard column reference.

Read **`AGENTS.md`** for: development status, what's implemented.

If a topic is given, read the relevant source files and docs, then answer. If no topic, give a brief overview and ask what they need.
