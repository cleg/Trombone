use std::env;
use std::io::{self, IsTerminal, Write};
use std::path::Path;

use clap::builder::styling::{AnsiColor, Style};
use console::{Term, measure_text_width, truncate_str};

use crate::search::{SearchHit, SourceStatus};

pub(crate) trait SearchOutput {
    fn write_results(&mut self, hits: &[SearchHit]) -> io::Result<()>;
}

pub(crate) struct JsonOutput<W> {
    writer: W,
}

impl<W: Write> JsonOutput<W> {
    pub(crate) fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> SearchOutput for JsonOutput<W> {
    fn write_results(&mut self, hits: &[SearchHit]) -> io::Result<()> {
        serde_json::to_writer(&mut self.writer, hits)?;
        writeln!(self.writer)?;
        self.writer.flush()
    }
}

const SEPARATOR: &str = "────────────────────────────────────────────────";
const VISIBLE_SOURCES: usize = 5;

pub(crate) struct ConsoleOutput<W> {
    writer: W,
    color: bool,
    columns: Option<usize>,
}

impl<W: IsTerminal> ConsoleOutput<W> {
    /// Wrap stdout's `writer`, sizing interactive rows and honoring color preferences.
    ///
    /// Returns a renderer with a terminal width for TTY output and full-width
    /// source paths for redirected output.
    pub(crate) fn new(writer: W) -> Self {
        let is_terminal = writer.is_terminal();
        let color = is_terminal
            && !env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty())
            && env::var_os("TERM").as_deref() != Some("dumb".as_ref());
        let columns = is_terminal.then(|| {
            let (_, columns) = Term::stdout().size();
            usize::from(columns.max(1))
        });
        Self {
            writer,
            color,
            columns,
        }
    }
}

impl<W: Write> SearchOutput for ConsoleOutput<W> {
    /// Print each chunk in `hits` with up to five source rows and one verified excerpt.
    ///
    /// Returns success after flushing, or an error if writing or flushing fails.
    fn write_results(&mut self, hits: &[SearchHit]) -> io::Result<()> {
        let style = |style| if self.color { style } else { Style::new() };
        let lines = style(AnsiColor::Yellow.on_default());
        let distance = style(AnsiColor::Magenta.on_default());
        let separator = style(Style::new().dimmed());

        if hits.is_empty() {
            writeln!(self.writer, "No matching chunks.")?;
        }
        for (position, hit) in hits.iter().enumerate() {
            if position > 0 {
                writeln!(self.writer, "\n{separator}{SEPARATOR}{separator:#}\n")?;
            }
            writeln!(
                self.writer,
                "{distance}Cosine distance {:.4}{distance:#}",
                hit.distance,
            )?;
            for source in hit.sources.iter().take(VISIBLE_SOURCES) {
                let (color, detail, detail_style) = match source.status {
                    SourceStatus::Available => (
                        AnsiColor::Green,
                        format!(":{}-{}", source.line_start, source.line_end),
                        lines,
                    ),
                    SourceStatus::Missing => (AnsiColor::Red, " [missing]".into(), separator),
                    SourceStatus::ExcerptUnavailable => {
                        (AnsiColor::Yellow, " [no excerpt]".into(), separator)
                    }
                };
                let detail_width = self.columns.map_or(usize::MAX, |columns| columns - 1);
                let detail = truncate_str(
                    &detail,
                    detail_width,
                    if detail_width == 0 { "" } else { "…" },
                );
                let path_width = self
                    .columns
                    .map(|columns| columns - measure_text_width(&detail));
                let label = source_label(&source.path, path_width);
                let path_style = style(color.on_default().bold());
                writeln!(
                    self.writer,
                    "{path_style}{label}{path_style:#}{detail_style}{detail}{detail_style:#}",
                )?;
            }
            if hit.sources.len() > VISIBLE_SOURCES {
                let remaining = hit.sources.len() - VISIBLE_SOURCES;
                writeln!(
                    self.writer,
                    "{separator}(+{remaining} more files){separator:#}"
                )?;
            }
            if let Some(section_path) = &hit.section_path {
                writeln!(self.writer, "{}", section_path.join(" > "))?;
            }
            if let Some(text) = &hit.text {
                self.writer.write_all(text.as_bytes())?;
                if !text.ends_with('\n') {
                    writeln!(self.writer)?;
                }
            } else {
                writeln!(self.writer, "Matched indexed chunk. Excerpt unavailable.")?;
            }
        }
        self.writer.flush()
    }
}

/// Render `path` in one line, preserving its filename when the full path does
/// not fit in `width` terminal columns. Without a width, returns the full path.
fn source_label(path: &Path, width: Option<usize>) -> String {
    let is_line_control =
        |character: char| character.is_control() || matches!(character, '\u{2028}' | '\u{2029}');
    let full = path.to_string_lossy().replace(is_line_control, "�");
    let Some(width) = width.filter(|&width| measure_text_width(&full) > width) else {
        return full;
    };
    let Some(filename) = path.file_name() else {
        return truncate_str(&full, width, "…").into_owned();
    };
    let filename = filename.to_string_lossy().replace(is_line_control, "�");
    let short = format!("…{}{filename}", std::path::MAIN_SEPARATOR);
    if measure_text_width(&short) <= width {
        short
    } else {
        truncate_str(&filename, width, "…").into_owned()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::search::SearchSource;

    /// Verify terminal rows fit, show status colors and ranges, and cap the file list.
    ///
    /// Returns success after checking the rendered rows and single excerpt, or
    /// propagates an output or UTF-8 decoding failure.
    #[test]
    fn terminal_search_caps_and_fits_source_rows() -> crate::error::Result<()> {
        let columns = 32;
        let mut output = ConsoleOutput {
            writer: Vec::new(),
            color: true,
            columns: Some(columns),
        };
        let files = [
            ("перший.md", SourceStatus::Available),
            ("missing.md", SourceStatus::Missing),
            ("changed.md", SourceStatus::ExcerptUnavailable),
            ("fourth.md", SourceStatus::Available),
            ("fifth.md", SourceStatus::Available),
            ("hidden.md", SourceStatus::Available),
        ];
        output.write_results(&[SearchHit {
            sources: files
                .into_iter()
                .map(|(name, status)| SearchSource {
                    path: PathBuf::from("some-very-long-parent-directory-with-文").join(name),
                    line_start: 12,
                    line_end: 15,
                    status,
                })
                .collect(),
            line_start: 12,
            line_end: 15,
            section_path: None,
            distance: 0.25,
            text: Some("Chunk text.".into()),
        }])?;

        let rendered = String::from_utf8(output.writer)?;
        let rows: Vec<_> = rendered.lines().collect();
        assert_eq!(rows.len(), 8);
        assert!(
            rows[1..=5]
                .iter()
                .all(|row| measure_text_width(row) <= columns)
        );
        for (row, color, detail) in [
            (rows[1], AnsiColor::Green, "перший.md:12-15"),
            (rows[2], AnsiColor::Red, "missing.md [missing]"),
            (rows[3], AnsiColor::Yellow, "changed.md [no excerpt]"),
        ] {
            assert!(row.starts_with(&format!("{}", color.on_default().bold())));
            assert!(console::strip_ansi_codes(row).ends_with(detail));
        }
        assert_eq!(console::strip_ansi_codes(rows[6]), "(+1 more files)");
        assert_eq!(rows[7], "Chunk text.");
        assert!(!rendered.contains("hidden.md"));
        Ok(())
    }
}
