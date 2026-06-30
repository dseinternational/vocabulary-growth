#!/usr/bin/env node

import { spawnSync } from "node:child_process";

const mode = process.argv[2];

if (!["--check", "--write"].includes(mode)) {
  console.error("Usage: node scripts/format-markdown.mjs --check|--write");
  process.exit(2);
}

const trackedMarkdown = spawnSync("git", ["ls-files", "--", "*.md", ":(exclude)data/**/*.md"], {
  encoding: "utf8",
});

if (trackedMarkdown.status !== 0) {
  if (trackedMarkdown.stderr) {
    process.stderr.write(trackedMarkdown.stderr);
  } else if (trackedMarkdown.error) {
    console.error(trackedMarkdown.error.message);
  }
  process.exit(trackedMarkdown.status ?? 1);
}

const files = trackedMarkdown.stdout.split(/\r?\n/u).filter(Boolean);

if (files.length === 0) {
  process.exit(0);
}

const prettier = process.platform === "win32" ? "prettier.cmd" : "prettier";
const result = spawnSync(prettier, [mode, ...files], {
  shell: process.platform === "win32",
  stdio: "inherit",
});

process.exit(result.status ?? 1);
