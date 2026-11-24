use std::fs;
use std::io;
use std::process::Command;

/// Verify indexing rejects an absent or empty root list before opening storage or Ollama.
///
/// Returns success for all invalid lists and a successful compact without roots,
/// or an error setting up the fixture or launching the CLI.
#[test]
fn indexing_requires_roots_but_compaction_does_not() -> io::Result<()> {
    let directory = tempfile::tempdir()?;
    let config = directory.path().join("trombone.toml");
    let storage = "[database]\npath = '.'\n[ollama]\nmodel = 'unused'\nurl = 'invalid-url'\n";
    for roots in ["", "roots = []\n", "roots = ['notes', '']\n"] {
        fs::write(&config, format!("{roots}{storage}"))?;
        let result = Command::new(env!("CARGO_BIN_EXE_tb"))
            .arg("--config")
            .arg(&config)
            .args(["index", "--chunk-size", "1000", "--no-compact"])
            .output()?;
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(!result.status.success());
        assert!(error.contains("Failed to load config"), "{error}");
        assert!(error.contains("roots must contain"), "{error}");
        assert!(!directory.path().join("trombone.sqlite").exists());
    }

    let result = Command::new(env!("CARGO_BIN_EXE_tb"))
        .args(["index", "notes"])
        .output()?;
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("unexpected argument 'notes'"));

    fs::write(&config, storage)?;
    let result = Command::new(env!("CARGO_BIN_EXE_tb"))
        .arg("--config")
        .arg(&config)
        .arg("compact")
        .output()?;
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(())
}
