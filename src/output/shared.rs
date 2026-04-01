use anyhow::Result;
use csv::WriterBuilder;
use serde::Serialize;

use crate::algorithms::Algorithm;
use crate::api::{OutputPlan, RuntimeOptions};
use crate::output::OutputFile;
use crate::results::{DiagnosticsBundle, FitSummary};

pub(crate) fn shared_output_file_names() -> Vec<String> {
    vec![
        "settings.json",
        "summary.json",
        "summary.csv",
        "diagnostics.json",
        "predictions.csv",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RunConfiguration {
    pub algorithm: Algorithm,
    pub output: OutputPlan,
    pub runtime: RuntimeOptions,
    pub parameter_names: Vec<String>,
}

impl RunConfiguration {
    pub(crate) fn new(
        algorithm: Algorithm,
        output: &OutputPlan,
        runtime: &RuntimeOptions,
        parameter_names: Vec<String>,
    ) -> Self {
        Self {
            algorithm,
            output: output.clone(),
            runtime: runtime.clone(),
            parameter_names,
        }
    }

    pub(crate) fn output_path(&self) -> &str {
        self.output.path.as_deref().unwrap_or("outputs/")
    }

    pub(crate) fn should_write_outputs(&self) -> bool {
        self.output.write
    }
}

pub(crate) fn write_settings(folder: &str, configuration: &RunConfiguration) -> Result<()> {
    let outputfile = OutputFile::new(folder, "settings.json")?;
    let mut file = outputfile.file_owned();
    let serialized = serde_json::to_string_pretty(configuration)?;
    std::io::Write::write_all(&mut file, serialized.as_bytes())?;
    Ok(())
}

pub fn write_summary(folder: &str, summary: &FitSummary) -> Result<()> {
    let outputfile = OutputFile::new(folder, "summary.json")?;
    let mut file = outputfile.file_owned();
    let serialized = serde_json::to_string_pretty(summary)?;
    std::io::Write::write_all(&mut file, serialized.as_bytes())?;

    let outputfile = OutputFile::new(folder, "summary.csv")?;
    let mut writer = WriterBuilder::new()
        .has_headers(true)
        .from_writer(outputfile.file_owned());
    writer.write_record(["metric", "value"])?;
    writer.write_record([
        "objective_function",
        &summary.objective_function.to_string(),
    ])?;
    writer.write_record(["converged", &summary.converged.to_string()])?;
    writer.write_record(["iterations", &summary.iterations.to_string()])?;
    writer.write_record(["subject_count", &summary.subject_count.to_string()])?;
    writer.write_record(["observation_count", &summary.observation_count.to_string()])?;
    writer.write_record(["parameter_count", &summary.parameter_count.to_string()])?;
    writer.write_record(["algorithm", &summary.algorithm])?;
    writer.flush()?;

    Ok(())
}

pub fn write_diagnostics(folder: &str, diagnostics: &DiagnosticsBundle) -> Result<()> {
    let outputfile = OutputFile::new(folder, "diagnostics.json")?;
    let mut file = outputfile.file_owned();
    let serialized = serde_json::to_string_pretty(diagnostics)?;
    std::io::Write::write_all(&mut file, serialized.as_bytes())?;
    Ok(())
}

pub fn write_csv_rows<T: Serialize>(
    folder: &str,
    file_name: &str,
    rows: impl IntoIterator<Item = T>,
) -> Result<()> {
    let outputfile = OutputFile::new(folder, file_name)?;
    let mut writer = WriterBuilder::new()
        .has_headers(true)
        .from_writer(outputfile.file_owned());

    for row in rows {
        writer.serialize(row)?;
    }

    writer.flush()?;
    Ok(())
}
