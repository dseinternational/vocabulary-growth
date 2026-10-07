-- Copyright (c) 2026 Down Syndrome Education International and contributors
-- SPDX-License-Identifier: AGPL-3.0-or-later
--
-- Convert Quarto callouts to the content system's HTML alert markup.
-- Quarto exposes a Callout node; its content can be one block or a block list.
-- Drop source-only HTML comment blocks from the Markdown export.

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
