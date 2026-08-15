#!/usr/bin/env node

import { spawnSync } from "node:child_process";

const mode = process.argv[2];

if (!["--check", "--write"].includes(mode)) {
  console.error("Usage: node scripts/format-markdown.mjs --check|--write");
  process.exit(2);
}

// `--others --exclude-standard` adds untracked-but-not-ignored files to the
// tracked set. Without them a NEW Markdown file is invisible here until it is
// staged, so `npm run format` and `npm run format:check` both pass locally and
// CI then fails on the same file once the commit makes it tracked — which
// happened twice while writing the 2026-08 notes. Ignored files stay excluded.
const listedMarkdown = spawnSync(
  "git",
  ["ls-files", "--cached", "--others", "--exclude-standard", "--", "*.md", ":(exclude)data/**/*.md"],
  { encoding: "utf8" },
);

if (listedMarkdown.status !== 0) {
  if (listedMarkdown.stderr) {
    process.stderr.write(listedMarkdown.stderr);
  } else if (listedMarkdown.error) {
    console.error(listedMarkdown.error.message);
  }
  process.exit(listedMarkdown.status ?? 1);
}

// A path can appear in both lists (staged edits to a tracked file), and passing
// a duplicate to Prettier makes it format the file twice.
const files = [...new Set(listedMarkdown.stdout.split(/\r?\n/u).filter(Boolean))];

if (files.length === 0) {
  process.exit(0);
}

const prettier = process.platform === "win32" ? "prettier.cmd" : "prettier";
const result = spawnSync(prettier, [mode, ...files], {
  shell: process.platform === "win32",
  stdio: "inherit",
});

process.exit(result.status ?? 1);
