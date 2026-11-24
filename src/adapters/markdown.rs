use std::io;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::path::Path;

use pulldown_cmark::{Event, HeadingLevel, OffsetIter, Options, Parser, Tag};

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
    // An oversized block is consumed a piece at a time, without queuing its chunks.
    remaining: Option<Range<usize>>,
    finished: bool,
}

impl<'a> MarkdownChunks<'a> {
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
            remaining: None,
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
            if let Some(mut range) = self.remaining.take() {
                let chunk = self.builder.push(&mut range);
                if !range.is_empty() {
                    self.remaining = Some(range);
                }
                if chunk.is_some() {
                    return chunk;
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
                        } else if !self.metadata {
                            self.remaining = Some(self.block_start..range.end);
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
                    if self.depth == 0 {
                        self.remaining = Some(range);
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
    headings: Vec<(HeadingLevel, String)>,
    pending: Option<Range<usize>>,
    pending_chars: usize,
    // Locations advance with emitted chunks; no per-line offset array is needed.
    line_offset: usize,
    line_number: usize,
}

impl<'a> ChunkBuilder<'a> {
    fn new(source: &'a str, limit: usize) -> Self {
        Self {
            source,
            limit,
            headings: Vec::new(),
            pending: None,
            pending_chars: 0,
            line_offset: 0,
            line_number: 1,
        }
    }

    /// Consume some or all of range and return at most one completed chunk.
    fn push(&mut self, range: &mut Range<usize>) -> Option<Chunk> {
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
                range.start = range.end;
                return None;
            }
            // Leave range untouched so the next call can start the new chunk.
            if let Some(chunk) = self.flush() {
                return Some(chunk);
            }
        }

        // Split only one piece, retaining the remaining source range for next().
        if let Some((cut, _)) = self.source[range.clone()].char_indices().nth(self.limit) {
            let prefix = &self.source[range.start..range.start + cut];
            let boundary = prefix
                .rfind(char::is_whitespace)
                .map(|offset| offset + prefix[offset..].chars().next().unwrap().len_utf8())
                .filter(|offset| prefix[..*offset].chars().count() >= self.limit / 2)
                .unwrap_or(cut);
            let end = range.start + boundary;
            self.pending = Some(range.start..end);
            range.start = end;
            return self.flush();
        }

        self.pending_chars = self.source[range.clone()].chars().count();
        self.pending = Some(range.clone());
        range.start = range.end;
        None
    }

    /// Emit this builder's pending source span with typed byte and line coordinates.
    ///
    /// Returns a chunk and advances the line cursor, or `None` if there is no
    /// pending span or it contains only whitespace.
    fn flush(&mut self) -> Option<Chunk> {
        self.pending_chars = 0;
        let range = self.pending.take()?;
        let text = &self.source[range.clone()];
        if text.trim().is_empty() {
            return None;
        }

        // Scan each intervening byte at most once, including skipped frontmatter/gaps.
        let line_start = self.line_number
            + self.source[self.line_offset..range.start]
                .bytes()
                .filter(|byte| *byte == b'\n')
                .count();
        let next_line = line_start + text.bytes().filter(|byte| *byte == b'\n').count();
        let line_end = next_line - usize::from(text.ends_with('\n'));
        self.line_offset = range.end;
        self.line_number = next_line;

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

    /// Verify oversized blocks retain a remainder after each typed source span is emitted.
    #[test]
    fn oversized_blocks_leave_their_remainder_for_the_next_call() {
        let source = format!("```\n{}\n```\n\n# Later\n", "🙂text ".repeat(10_000));
        let mut chunks = MarkdownChunks::new(&source, 20);
        let first = chunks.next().unwrap();
        let remainder = chunks.remaining.as_ref().unwrap();
        assert_eq!(remainder.start, first.location.byte_end);
        assert!(remainder.len() > source.len() / 2);
        assert!(chunks.builder.pending.is_none());
        assert!(!chunks.finished);
        assert_source_ranges(&source, &[first, chunks.next().unwrap()], 20);
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
