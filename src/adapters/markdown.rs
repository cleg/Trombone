use std::io;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::path::Path;

use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag};

use crate::adapter::{Adapter, Attributes, Chunk, Document, DocumentMetadata};

pub(crate) struct MarkdownAdapter;

impl Adapter for MarkdownAdapter {
    fn open(&self, path: &Path, chunk_size: NonZeroUsize) -> io::Result<Document> {
        let source = std::fs::read_to_string(path)?;
        let chunks = chunk_markdown(&source, chunk_size.get());

        Ok(Document {
            metadata: DocumentMetadata {
                source: path.to_path_buf(),
                attributes: Attributes::from_iter([("media_type".into(), "text/markdown".into())]),
            },
            chunks: Box::new(chunks.into_iter().map(Ok)),
        })
    }
}

fn chunk_markdown(source: &str, limit: usize) -> Vec<Chunk> {
    let options = Options::ENABLE_TABLES
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_YAML_STYLE_METADATA_BLOCKS;
    let mut builder = ChunkBuilder::new(source, limit);
    let mut depth = 0;
    let mut block_start = 0;
    let mut metadata = false;
    let mut heading: Option<(HeadingLevel, String)> = None;

    for (event, range) in Parser::new_ext(source, options).into_offset_iter() {
        match event {
            Event::Start(tag) => {
                if depth == 0 {
                    block_start = range.start;
                    metadata = matches!(tag, Tag::MetadataBlock(_));
                    if let Tag::Heading { level, .. } = tag {
                        builder.flush();
                        heading = Some((level, String::new()));
                    }
                }
                depth += 1;
            }
            Event::End(_) => {
                assert!(depth > 0, "End event without a matching Start");

                depth -= 1;
                if depth == 0 {
                    if let Some((level, text)) = heading.take() {
                        builder.headings.retain(|(parent, _)| *parent < level);
                        builder.headings.push((level, text));
                    }
                    if !metadata {
                        builder.push(block_start..range.end);
                    }
                }
            }
            Event::Text(text) | Event::Code(text) => {
                if let Some((_, title)) = &mut heading {
                    title.push_str(&text);
                }
            }
            Event::SoftBreak | Event::HardBreak => {
                if let Some((_, title)) = &mut heading {
                    title.push(' ');
                }
            }
            Event::Rule | Event::Html(_) if depth == 0 => builder.push(range),
            _ => {}
        }
    }

    builder.flush();
    builder.chunks
}

struct ChunkBuilder<'a> {
    source: &'a str,
    limit: usize,
    headings: Vec<(HeadingLevel, String)>,
    pending: Option<Range<usize>>,
    pending_chars: usize,
    line_starts: Vec<usize>,
    chunks: Vec<Chunk>,
}

impl<'a> ChunkBuilder<'a> {
    fn new(source: &'a str, limit: usize) -> Self {
        Self {
            source,
            limit,
            headings: Vec::new(),
            pending: None,
            pending_chars: 0,
            line_starts: std::iter::once(0)
                .chain(source.match_indices('\n').map(|(offset, _)| offset + 1))
                .collect(),
            chunks: Vec::new(),
        }
    }

    fn push(&mut self, mut range: Range<usize>) {
        if let Some(pending) = &self.pending {
            // Include the gap between blocks, without recounting the accumulated text.
            let added_chars = self.source[pending.end..range.end].chars().count();
            if self.pending_chars + added_chars <= self.limit {
                self.pending = Some(pending.start..range.end);
                self.pending_chars += added_chars;
                return;
            }
            self.flush();
        }

        // Oversized blocks fall back to whitespace boundaries, then Unicode characters.
        while let Some((cut, _)) = self.source[range.clone()].char_indices().nth(self.limit) {
            let prefix = &self.source[range.start..range.start + cut];
            let boundary = prefix
                .rfind(char::is_whitespace)
                .map(|offset| offset + prefix[offset..].chars().next().unwrap().len_utf8())
                .filter(|offset| prefix[..*offset].chars().count() >= self.limit / 2)
                .unwrap_or(cut);
            let end = range.start + boundary;
            self.pending_chars = self.source[range.start..end].chars().count();
            self.pending = Some(range.start..end);
            self.flush();
            range.start = end;
        }

        if range.start < range.end {
            self.pending_chars = self.source[range.clone()].chars().count();
            self.pending = Some(range);
        }
    }

    fn flush(&mut self) {
        self.pending_chars = 0;
        let Some(range) = self.pending.take() else {
            return;
        };
        let text = &self.source[range.clone()];
        if text.trim().is_empty() {
            return;
        }

        let attributes = Attributes::from_iter([
            ("byte_start".into(), range.start.into()),
            ("byte_end".into(), range.end.into()),
            (
                "line_start".into(),
                self.line_starts
                    .partition_point(|start| *start <= range.start)
                    .into(),
            ),
            (
                "line_end".into(),
                self.line_starts
                    .partition_point(|start| *start < range.end)
                    .into(),
            ),
        ]);
        self.chunks.push(Chunk {
            text: text.to_owned(),
            section_path: (!self.headings.is_empty()).then(|| {
                self.headings
                    .iter()
                    .map(|(_, title)| title.clone())
                    .collect()
            }),
            attributes,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_source_ranges(source: &str, chunks: &[Chunk], limit: usize) {
        for chunk in chunks {
            let start = chunk.attributes["byte_start"].as_u64().unwrap() as usize;
            let end = chunk.attributes["byte_end"].as_u64().unwrap() as usize;
            assert_eq!(chunk.text, source[start..end]);
            assert!(chunk.text.chars().count() <= limit);
            assert!(!chunk.text.trim().is_empty());
        }
    }

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
                Some(vec!["Root".into()]),
                Some(vec!["Root".into(), "Child".into()]),
                Some(vec!["Root".into(), "Sibling".into()]),
            ]
        );
        assert!(chunks[2].text.contains("# Not a heading"));
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

    #[test]
    fn metadata_is_excluded_and_small_blocks_are_combined() {
        let source = "---\nlanguage: ua\n---\n\nFirst.\n\nSecond.\n";
        let chunks = chunk_markdown(source, 2000);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "First.\n\nSecond.\n");
        assert_eq!(chunks[0].attributes["line_start"], 5);
        assert_eq!(chunks[0].attributes["line_end"], 7);
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
    fn oversized_code_and_tables_still_respect_the_limit() {
        let source = "```\nдуже довгий рядок коду 🙂\n```\n\n| A | B |\n| --- | --- |\n| довге | значення |\n";
        let chunks = chunk_markdown(source, 10);
        assert_source_ranges(source, &chunks, 10);
        let reconstructed: String = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
        assert_eq!(
            reconstructed.split_whitespace().collect::<String>(),
            source.split_whitespace().collect::<String>(),
        );
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
        for source in ["", " \n\n", "---\nstatus: draft\n---\n"] {
            assert!(chunk_markdown(source, 2000).is_empty());
        }
    }
}
