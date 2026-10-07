#!/usr/bin/env node

import { spawnSync } from "node:child_process";

const mode = process.argv[2];

if (!["--check", "--write"].includes(mode)) {
  console.error("Usage: node scripts/format-markdown.mjs --check|--write");
  process.exit(2);
}

// Include untracked, non-ignored Markdown so new files are checked before staging.
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

// Keep Windows batches below cmd.exe's 8191-character command limit. The .cmd
// shim needs shell: true there. Run every batch so --check reports all failures.
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
