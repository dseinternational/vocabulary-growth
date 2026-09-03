-- Copyright (c) 2026 Down Syndrome Education International and contributors
-- SPDX-License-Identifier: AGPL-3.0-or-later
--
-- Shapes the Markdown export for the content system. Applied to the `gfm`
-- format only (see _quarto.yml).
--
-- 1. Callouts become raw-HTML alert blocks. Left alone, Quarto exports a
--    callout as GitHub alert syntax (`> [!NOTE]`), which the content system's
--    Markdig pipeline does not parse (alert blocks are a separate Markdig
--    extension the pipeline does not enable), so a reader would see a
--    blockquote containing a literal "[!NOTE]". The site's own help pages write
--    these blocks by hand as `<div class="alert alert-warning">`; this emits
--    the same markup, with the callout's title as a bold first paragraph.
--
--    Quarto parses callouts into its own `Callout` AST node before any user
--    filter runs, so the handler below is for that node, not for a Div with a
--    `callout-*` class (a first version handled the Div and never fired). The
--    node's `content` is a single Block when the callout holds one block and a
--    list of blocks otherwise (checked on Quarto 1.10.18), so both are handled.
--
-- 2. HTML comments are dropped. The pages carry authoring notes as
--    `<!-- … -->` blocks, and Pandoc would otherwise pass them into the export,
--    where the content importer keeps them; they are for the source only.

local ALERT_CLASSES = {
  note = { "alert", "alert-info" },
  tip = { "alert", "alert-success" },
  important = { "alert", "alert-danger" },
  warning = { "alert", "alert-warning" },
  caution = { "alert", "alert-warning" },
}

local function content_blocks(content)
  local blocks = pandoc.List()
  if content == nil then
    return blocks
  end
  if pandoc.utils.type(content) == "Block" then
    blocks:insert(content)
    return blocks
  end
  for _, block in ipairs(content) do
    blocks:insert(block)
  end
  return blocks
end

function Callout(callout)
  local classes = ALERT_CLASSES[callout.type] or ALERT_CLASSES.note
  local blocks = pandoc.List()
  if callout.title ~= nil then
    local title = pandoc.utils.stringify(callout.title)
    if title ~= "" then
      blocks:insert(pandoc.Para({ pandoc.Strong({ pandoc.Str(title) }) }))
    end
  end
  blocks:extend(content_blocks(callout.content))
  return pandoc.Div(blocks, pandoc.Attr("", classes))
end

function RawBlock(el)
  if el.format == "html" and el.text:match("^%s*<!%-%-") then
    return {}
  end
  return nil
end
