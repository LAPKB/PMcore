use crate::prelude::settings::run::Data;
use tracing::Level;
use tracing_subscriber::fmt;

pub fn setup_log(settings: &Data) {
    if let Some(log_path) = &settings.parsed.paths.log_out {
        if std::fs::remove_file(log_path).is_ok() {};

        let subscriber = fmt::Subscriber::builder()
            .with_max_level(Level::INFO)
            .with_writer(std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(log_path)
                .unwrap())
            .with_ansi(false)
            .finish();

        tracing::subscriber::set_global_default(subscriber)
            .expect("Setting default subscriber failed");
    }
}
