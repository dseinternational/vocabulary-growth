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

// cmd.exe caps a command line at 8191 characters, and `shell: true` -- which
// Windows needs to run the `.cmd` shim -- routes the whole argument list
// through it. The repository's Markdown grew past that: 177 files spell about
// 8,390 characters, and the failure is `The syntax of the command is
// incorrect.` from cmd itself, naming neither a length nor a file. Batching is
// the fix rather than raising anything, because the limit is not ours to raise
// and the next note would cross it again.
//
// Every batch runs even after one fails, so `--check` reports every offending
// file rather than only those in the first failing batch.
const BATCH_CHARS = 6000;

const batches = [[]];
let used = 0;
for (const file of files) {
  const cost = file.length + 3;
  if (used + cost > BATCH_CHARS && batches.at(-1).length > 0) {
    batches.push([]);
    used = 0;
  }
  batches.at(-1).push(file);
  used += cost;
}

let status = 0;
for (const batch of batches) {
  const result = spawnSync(prettier, [mode, ...batch], {
    shell: process.platform === "win32",
    stdio: "inherit",
  });
  status = status || (result.status ?? 1);
}

process.exit(status);
