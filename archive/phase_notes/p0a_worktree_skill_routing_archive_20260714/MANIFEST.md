# P0A worktree skill-routing archive (2026-07-14)

- source worktree: python_port_measurement_fix
- branch: fix/measurement-model-safety
- HEAD at archive time: 3265dbe96826e2fb843362c61913b1c6d7c47629
- content: 21 modified tracked .claude/skills files (158+/67-) as
  uncommitted_skill_routing.patch, plus the untracked worktree CLAUDE.md
  as untracked_CLAUDE.md.
- patch sha256: B5390F9ED5A5A05B61296F88B3F79E2E1A7F7567ED6F5140FB78CE7FD5EFCE92
- CLAUDE.md sha256: 5B242EBA2D3AE02193EE8151D385D9053329AB1F2B81814189075214BBD26182

Restore (if ever needed):
  git -C <worktree> apply uncommitted_skill_routing.patch
  copy untracked_CLAUDE.md -> <worktree>/CLAUDE.md

Note: the canonical routing changes live in python_port commit c3cb60e;
this archive preserves the worktree-local (owner-synced) variants that
were intentionally NOT committed with P0A (3265dbe).
