use std::io;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::path::Path;
use std::vec::IntoIter;

use pulldown_cmark::{Event, HeadingLevel, OffsetIter, Options, Parser, Tag};
use text_splitter::{Characters, ChunkConfig, TextSplitter};

use crate::adapter::{Adapter, Attributes, Chunk, Document, DocumentMetadata, SourceLocation};

pub(crate) struct MarkdownAdapter;

impl Adapter for MarkdownAdapter {
    fn open<'a>(
        &self,
        path: &Path,
        source: &'a str,
        chunk_size: NonZeroUsize,
    ) -> io::Result<Document<'a>> {
        Ok(Document {
            metadata: DocumentMetadata {
                source: path.to_path_buf(),
                attributes: Attributes::from_iter([("media_type".into(), "text/markdown".into())]),
            },
            chunks: Box::new(MarkdownChunks::new(source, chunk_size.get())),
        })
    }
}

/// Retains parsing state between calls to next(), borrowing the original text.
struct MarkdownChunks<'a> {
    events: OffsetIter<'a>,
    builder: ChunkBuilder<'a>,
    depth: usize,
    block_start: usize,
    metadata: bool,
    heading: Option<(HeadingLevel, String)>,
    finished: bool,
}

impl<'a> MarkdownChunks<'a> {
    /// Create an iterator over `source` with a positive character `limit` per chunk.
    fn new(source: &'a str, limit: usize) -> Self {
        assert!(limit > 0, "chunk size must be nonzero");
        let options = Options::ENABLE_TABLES
            | Options::ENABLE_TASKLISTS
            | Options::ENABLE_YAML_STYLE_METADATA_BLOCKS;
        Self {
            events: Parser::new_ext(source, options).into_offset_iter(),
            builder: ChunkBuilder::new(source, limit),
            depth: 0,
            block_start: 0,
            metadata: false,
            heading: None,
            finished: false,
        }
    }
}

impl Iterator for MarkdownChunks<'_> {
    type Item = Chunk;

    /// Advances this iterator and returns the next Markdown chunk, or `None` at exhaustion.
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        loop {
            // Finish the current block before advancing the parser or heading path.
            if let Some(range) = self.builder.split_ranges.next() {
                if let Some(chunk) = self.builder.emit(range) {
                    return Some(chunk);
                }
                continue;
            }

            let Some((event, range)) = self.events.next() else {
                self.finished = true;
                return self.builder.flush();
            };
            match event {
                Event::Start(tag) => {
                    let mut chunk = None;
                    if self.depth == 0 {
                        self.block_start = range.start;
                        self.metadata = matches!(tag, Tag::MetadataBlock(_));
                        if let Tag::Heading { level, .. } = tag {
                            // Emit the preceding section before changing its heading path.
                            chunk = self.builder.flush();
                            self.heading = Some((level, String::new()));
                        }
                    }
                    self.depth += 1;
                    if chunk.is_some() {
                        return chunk;
                    }
                }
                Event::End(_) => {
                    assert!(self.depth > 0, "End event without a matching Start");
                    self.depth -= 1;
                    if self.depth == 0 {
                        if let Some((level, text)) = self.heading.take() {
                            self.builder.headings.retain(|(parent, _)| *parent < level);
                            self.builder.headings.push((level, text));
                        } else if !self.metadata
                            && let Some(chunk) = self.builder.push(self.block_start..range.end)
                        {
                            return Some(chunk);
                        }
                    }
                }
                Event::Text(text) | Event::Code(text) => {
                    if let Some((_, title)) = &mut self.heading {
                        title.push_str(&text);
                    }
                }
                Event::SoftBreak | Event::HardBreak => {
                    if let Some((_, title)) = &mut self.heading {
                        title.push(' ');
                    }
                }
                Event::Rule | Event::Html(_) => {
                    if self.depth == 0
                        && let Some(chunk) = self.builder.push(range)
                    {
                        return Some(chunk);
                    }
                }
                // The enclosing block's source range already includes these tokens.
                Event::InlineHtml(_) | Event::TaskListMarker(_) => {}
                // The parser options leave math and footnote extensions disabled.
                Event::InlineMath(_) | Event::DisplayMath(_) | Event::FootnoteReference(_) => {}
            }
        }
    }
}

struct ChunkBuilder<'a> {
    source: &'a str,
    limit: usize,
    splitter: TextSplitter<Characters>,
    split_ranges: IntoIter<Range<usize>>,
    headings: Vec<(HeadingLevel, String)>,
    pending: Option<Range<usize>>,
    pending_chars: usize,
    // Locations advance with emitted chunks; no per-line offset array is needed.
    line_offset: usize,
    line_number: usize,
}

impl<'a> ChunkBuilder<'a> {
    /// Create a builder for spans in `source` with up to `limit` Unicode characters per chunk.
    fn new(source: &'a str, limit: usize) -> Self {
        Self {
            source,
            limit,
            splitter: TextSplitter::new(ChunkConfig::new(limit).with_trim(false)),
            split_ranges: Vec::new().into_iter(),
            headings: Vec::new(),
            pending: None,
            pending_chars: 0,
            line_offset: 0,
            line_number: 1,
        }
    }

    /// Add the next block's source `range` after draining any earlier split ranges.
    ///
    /// Returns the previous chunk if the block cannot fit alongside it. Buffers
    /// the new block or its split ranges for subsequent iteration.
    fn push(&mut self, range: Range<usize>) -> Option<Chunk> {
        if let Some(pending) = &self.pending {
            // Include the gap, but stop counting as soon as it cannot fit.
            let available = self.limit - self.pending_chars;
            let added_chars = self.source[pending.end..range.end]
                .chars()
                .take(available.saturating_add(1))
                .count();
            if added_chars <= available {
                self.pending = Some(pending.start..range.end);
                self.pending_chars += added_chars;
                return None;
            }
        }

        let previous = self.flush();
        let text = &self.source[range.clone()];
        let chars = text.chars().take(self.limit.saturating_add(1)).count();
        if chars <= self.limit {
            self.pending = Some(range);
            self.pending_chars = chars;
        } else {
            // Queue only source ranges. Keep the last piece pending so it can
            // still combine with the next block if both fit within the limit.
            let mut ranges: Vec<_> = self
                .splitter
                .chunk_indices(text)
                .map(|(offset, chunk)| {
                    let start = range.start + offset;
                    start..start + chunk.len()
                })
                .collect();
            self.pending = ranges.pop();
            self.pending_chars = self
                .pending
                .as_ref()
                .map_or(0, |span| self.source[span.clone()].chars().count());
            self.split_ranges = ranges.into_iter();
        }
        previous
    }

    /// Consume the pending span and return its chunk, if it contains non-whitespace text.
    fn flush(&mut self) -> Option<Chunk> {
        self.pending_chars = 0;
        let range = self.pending.take()?;
        self.emit(range)
    }

    /// Return a chunk for the source `range` and advance the line cursor.
    ///
    /// Returns `None` for an empty or whitespace-only span and leaves the cursor unchanged.
    fn emit(&mut self, range: Range<usize>) -> Option<Chunk> {
        let text = &self.source[range.clone()];
        if text.trim().is_empty() {
            return None;
        }

        // Scan each intervening byte at most once, including skipped frontmatter/gaps.
        let bytes = self.source.as_bytes();
        let line_start = self.line_number
            + (self.line_offset..range.start)
                .filter(|&offset| ends_line(bytes, offset))
                .count();
        let last = range.end - 1;
        let line_end = line_start
            + (range.start..last)
                .filter(|&offset| ends_line(bytes, offset))
                .count();
        self.line_offset = range.end;
        self.line_number = line_end + usize::from(ends_line(bytes, last));

        Some(Chunk {
            text: text.to_owned(),
            location: SourceLocation {
                byte_start: range.start,
                byte_end: range.end,
                line_start,
                line_end,
            },
            section_path: (!self.headings.is_empty()).then(|| {
                self.headings
                    .iter()
                    .map(|(_, title)| title.clone())
                    .collect()
            }),
            attributes: Attributes::new(),
        })
    }
}

/// Return whether the byte at `offset` in the complete Markdown `source` ends a line.
///
/// The last byte of LF, CRLF, or a standalone CR ends the line. Looking at the
/// complete source keeps a CRLF pair together even when a chunk ends at the CR.
fn ends_line(source: &[u8], offset: usize) -> bool {
    match source[offset] {
        b'\n' => true,
        b'\r' => source.get(offset + 1) != Some(&b'\n'),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk_markdown(source: &str, limit: usize) -> Vec<Chunk> {
        MarkdownChunks::new(source, limit).collect()
    }

    /// Assert that `chunks` reference exact spans of `source` within character `limit`.
    ///
    /// Verifies typed byte offsets and computes expected line numbers independently
    /// of the chunk builder's incremental cursor.
    fn assert_source_ranges(source: &str, chunks: &[Chunk], limit: usize) {
        for chunk in chunks {
            let start = chunk.location.byte_start;
            let end = chunk.location.byte_end;
            assert_eq!(chunk.text, source[start..end]);
            // Compute expected locations independently from the incremental cursor.
            let line_start = 1 + source.as_bytes()[..start]
                .iter()
                .filter(|b| **b == b'\n')
                .count();
            let line_end = 1 + source.as_bytes()[..end - 1]
                .iter()
                .filter(|b| **b == b'\n')
                .count();
            assert_eq!(chunk.location.line_start, line_start);
            assert_eq!(chunk.location.line_end, line_end);
            assert!(chunk.text.chars().count() <= limit);
            assert!(!chunk.text.trim().is_empty());
        }
    }

    /// Verify oversized blocks preserve their text and finish before the next heading.
    #[test]
    fn oversized_blocks_leave_their_remainder_for_the_next_call() {
        let block = format!("```\n{}\n```", "🙂text ".repeat(10_000));
        let source = format!("{block}\n\n# Later\n\nNext.\n");
        let chunks = chunk_markdown(&source, 20);
        let (last, code_chunks) = chunks.split_last().unwrap();
        assert_source_ranges(&source, &chunks, 20);
        assert!(code_chunks.iter().all(|chunk| chunk.section_path.is_none()));
        assert_eq!(
            code_chunks
                .iter()
                .map(|chunk| chunk.text.as_str())
                .collect::<String>(),
            block
        );
        assert_eq!(last.section_path, Some(vec!["Later".into()]));
        assert_eq!(last.text, "Next.\n");
    }

    /// Verify that an oversized paragraph splits at sentence boundaries within the limit.
    #[test]
    fn oversized_paragraph_prefers_sentence_boundaries() {
        let first = "Перше коротке речення. ";
        let second = "Друге речення довше.";
        let source = format!("{first}{second}");
        let chunks = chunk_markdown(&source, 30);
        assert_source_ranges(&source, &chunks, 30);
        let texts: Vec<_> = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
        assert_eq!(texts, [first, second]);
    }

    /// Verify section ancestry and typed line coordinates across headings and fenced code.
    #[test]
    fn heading_paths_follow_the_hierarchy_and_ignore_fenced_code() {
        let source = "Intro.\n\n# Root\n\n## **Child**\n\n```md\n# Not a heading\n```\n\n## Sibling\n\nText.\n";
        let chunks = chunk_markdown(source, 2000);
        let paths: Vec<_> = chunks
            .iter()
            .map(|chunk| chunk.section_path.clone())
            .collect();
        assert_eq!(
            paths,
            vec![
                None,
                Some(vec!["Root".into(), "Child".into()]),
                Some(vec!["Root".into(), "Sibling".into()]),
            ]
        );
        assert_eq!(chunks[1].text, "```md\n# Not a heading\n```");
        assert_eq!(chunks[1].location.line_start, 7);
        assert_eq!(chunks[2].text, "Text.\n");
        assert_eq!(chunks[2].location.line_start, 13);
        assert_source_ranges(source, &chunks, 2000);
    }

    #[test]
    fn unicode_splitting_obeys_the_limit_without_losing_text() {
        let source = "Український текст із 🙂 та словами без заголовків.";
        for limit in [1, 7, 20] {
            let chunks = chunk_markdown(source, limit);
            assert_source_ranges(source, &chunks, limit);
            let reconstructed: String = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
            // Whitespace-only chunks are intentionally omitted.
            assert_eq!(
                reconstructed.split_whitespace().collect::<String>(),
                source.split_whitespace().collect::<String>()
            );
        }
    }

    /// Verify omitted metadata does not shift the typed coordinates of merged body blocks.
    #[test]
    fn metadata_is_excluded_and_small_blocks_are_combined() {
        let source = "---\nlanguage: ua\n---\n\nFirst.\n\nSecond.\n";
        let chunks = chunk_markdown(source, 2000);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "First.\n\nSecond.\n");
        assert_eq!(chunks[0].location.line_start, 5);
        assert_eq!(chunks[0].location.line_end, 7);
        assert_source_ranges(source, &chunks, 2000);
    }

    /// Verify indexed coordinates for Markdown mixing LF, CRLF, and standalone CR.
    #[test]
    fn mixed_line_endings_keep_exact_source_coordinates() {
        let heading = "# Заголовок\r\n\r\n";
        let body = "Перший\rДругий\nТретій\r";
        let source = format!("{heading}{body}");
        let chunks = chunk_markdown(&source, 2000);
        assert_eq!(chunks.len(), 1);
        let chunk = &chunks[0];
        assert_eq!(chunk.text, body);
        assert_eq!(
            chunk.location,
            SourceLocation {
                byte_start: heading.len(),
                byte_end: source.len(),
                line_start: 3,
                line_end: 5,
            }
        );
    }

    /// Verify a CRLF split across chunks advances the cursor only at the LF.
    #[test]
    fn crlf_across_chunk_boundary_counts_as_one_line_break() {
        let mut builder = ChunkBuilder::new("A\r\nB", 2);
        let first = builder.emit(0..2).unwrap();
        assert_eq!(first.text, "A\r");
        assert_eq!((first.location.line_start, first.location.line_end), (1, 1));

        let second = builder.emit(2..4).unwrap();
        assert_eq!(second.text, "\nB");
        assert_eq!(
            (second.location.line_start, second.location.line_end),
            (1, 2)
        );
    }

    #[test]
    fn fitting_tables_lists_and_code_remain_whole() {
        let source = "| Name | Value |\n| --- | --- |\n| A | B |\n\n- first\n- second\n\n```rust\nlet x = 1;\n```\n";
        let chunks = chunk_markdown(source, 45);
        assert_source_ranges(source, &chunks, 45);
        for block in [
            "| Name | Value |\n| --- | --- |\n| A | B |\n",
            "- first\n- second\n",
            "```rust\nlet x = 1;\n```",
        ] {
            assert!(
                chunks.iter().any(|chunk| chunk.text.contains(block)),
                "Missing block {block:?} in {:?}",
                chunks.iter().map(|chunk| &chunk.text).collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    fn combining_unicode_blocks_counts_gaps_and_resets_after_flush() {
        let source = "я\n\n🙂\n\nю\n\nє";
        for (limit, expected) in [
            (4, vec!["я\n", "🙂\n", "ю\n\nє"]),
            (5, vec!["я\n\n🙂\n", "ю\n\nє"]),
        ] {
            let chunks = chunk_markdown(source, limit);
            let texts: Vec<_> = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
            assert_eq!(texts, expected);
            assert_source_ranges(source, &chunks, limit);
        }
    }

    #[test]
    fn oversized_block_remainder_can_combine_with_the_next_block() {
        let source = "abcdefghijk\n\nя";
        let chunks = chunk_markdown(source, 5);
        let texts: Vec<_> = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
        assert_eq!(texts, vec!["abcde", "fghij", "k\n\nя"]);
        assert_source_ranges(source, &chunks, 5);
    }

    #[test]
    fn empty_documents_have_no_chunks() {
        for source in [
            "",
            " \n\n",
            "---\nstatus: draft\n---\n",
            "# Title\n\n## Section\n",
        ] {
            assert!(chunk_markdown(source, 2000).is_empty());
        }
    }
}
