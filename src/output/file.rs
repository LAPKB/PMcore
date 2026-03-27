use anyhow::{Context, Result};
use std::fs::{create_dir_all, File, OpenOptions};
use std::path::{Path, PathBuf};

/// Contains all the necessary information of an output file.
#[derive(Debug)]
pub struct OutputFile {
    file: File,
    relative_path: PathBuf,
}

impl OutputFile {
    pub fn new(folder: &str, file_name: &str) -> Result<Self> {
        let relative_path = Path::new(folder).join(file_name);

        if let Some(parent) = relative_path.parent() {
            create_dir_all(parent)
                .with_context(|| format!("Failed to create directories for {:?}", parent))?;
        }

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&relative_path)
            .with_context(|| format!("Failed to open file: {:?}", relative_path))?;

        Ok(Self {
            file,
            relative_path,
        })
    }

    pub fn file(&self) -> &File {
        &self.file
    }

    pub fn file_owned(self) -> File {
        self.file
    }

    pub fn relative_path(&self) -> &Path {
        &self.relative_path
    }
}