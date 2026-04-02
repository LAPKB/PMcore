use std::fs;
use std::path::Path;

use pharmsol::Equation;
use serde::{Deserialize, Serialize};

use crate::estimation::nonparametric::NonparametricWorkspace;
use crate::output::shared::shared_output_file_names;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactIndex {
    pub files: Vec<String>,
    pub expected_files: Vec<String>,
    pub missing_files: Vec<String>,
    pub shared_expected_files: Vec<String>,
    pub method_specific_expected_files: Vec<String>,
}

pub(crate) fn nonparametric_artifacts<E: Equation>(
    result: &NonparametricWorkspace<E>,
) -> ArtifactIndex {
    artifact_index(
        result.output_folder(),
        result.should_write_outputs(),
        crate::output::nonparametric::output_file_names(result),
    )
}

fn artifact_index(
    folder: &str,
    should_write_outputs: bool,
    mut expected_files: Vec<String>,
) -> ArtifactIndex {
    if !should_write_outputs {
        return ArtifactIndex::default();
    }

    expected_files.sort();
    expected_files.dedup();

    let shared_output_files = shared_output_file_names();
    let shared_expected_files = expected_files
        .iter()
        .filter(|file| shared_output_files.contains(*file))
        .cloned()
        .collect::<Vec<_>>();
    let method_specific_expected_files = expected_files
        .iter()
        .filter(|file| !shared_output_files.contains(*file))
        .cloned()
        .collect::<Vec<_>>();

    let path = Path::new(folder);
    if !path.exists() {
        return ArtifactIndex {
            files: Vec::new(),
            missing_files: expected_files.clone(),
            expected_files,
            shared_expected_files,
            method_specific_expected_files,
        };
    }

    let mut files = expected_files
        .iter()
        .filter(|file| {
            fs::metadata(path.join(file))
                .map(|meta| meta.is_file())
                .unwrap_or(false)
        })
        .cloned()
        .collect::<Vec<_>>();
    files.sort();

    let missing_files = expected_files
        .iter()
        .filter(|file| !files.contains(file))
        .cloned()
        .collect();

    ArtifactIndex {
        files,
        expected_files,
        missing_files,
        shared_expected_files,
        method_specific_expected_files,
    }
}
