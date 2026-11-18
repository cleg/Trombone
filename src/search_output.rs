use std::env;
use std::io::{self, IsTerminal, Write};

use clap::builder::styling::{AnsiColor, Style};

use crate::search::SearchHit;

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

pub(crate) struct ConsoleOutput<W> {
    writer: W,
    color: bool,
}

impl<W: IsTerminal> ConsoleOutput<W> {
    pub(crate) fn new(writer: W) -> Self {
        let color = writer.is_terminal()
            && !env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty())
            && env::var_os("TERM").as_deref() != Some("dumb".as_ref());
        Self { writer, color }
    }
}

impl<W: Write> SearchOutput for ConsoleOutput<W> {
    fn write_results(&mut self, hits: &[SearchHit]) -> io::Result<()> {
        let style = |style| if self.color { style } else { Style::new() };
        let path = style(AnsiColor::Cyan.on_default().bold());
        let lines = style(AnsiColor::Yellow.on_default());
        let distance = style(AnsiColor::Magenta.on_default());
        let separator = style(Style::new().dimmed());

        if hits.is_empty() {
            writeln!(self.writer, "No matching excerpts.")?;
        }
        for (position, hit) in hits.iter().enumerate() {
            if position > 0 {
                writeln!(self.writer, "\n{separator}{SEPARATOR}{separator:#}\n")?;
            }
            writeln!(
                self.writer,
                "{path}{}{path:#}:{lines}{}-{}{lines:#} {distance}(cosine distance {:.4}){distance:#}",
                hit.path.display(),
                hit.line_start,
                hit.line_end,
                hit.distance,
            )?;
            if let Some(section_path) = &hit.section_path {
                writeln!(self.writer, "{}", section_path.join(" > "))?;
            }
            self.writer.write_all(hit.text.as_bytes())?;
            if !hit.text.ends_with('\n') {
                writeln!(self.writer)?;
            }
        }
        self.writer.flush()
    }
}
