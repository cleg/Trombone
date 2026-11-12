use std::io;

use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

pub(crate) fn round_trip(embedding: &[f32]) -> Result<(u64, f32), Box<dyn std::error::Error>> {
    const KEY: u64 = 42;

    let directory = tempfile::tempdir()?;
    let path = directory.path().join("probe.usearch");
    let path = path
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Index path must be UTF-8"))?;
    let options = IndexOptions {
        dimensions: embedding.len(),
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        multi: false,
        ..IndexOptions::default()
    };

    {
        let index = Index::new(&options)?;
        index.reserve(1)?;
        index.add(KEY, embedding)?;
        index.save(path)?;
    }

    let index = Index::new(&options)?;
    index.view(path)?;
    let matches = index.search(embedding, 1)?;
    match (matches.keys.as_slice(), matches.distances.as_slice()) {
        ([KEY], [distance]) => Ok((KEY, *distance)),
        _ => Err(io::Error::other("USearch did not return the inserted key").into()),
    }
}
