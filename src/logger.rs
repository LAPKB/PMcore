use crate::prelude::settings::run::Data;
use log::LevelFilter;
use log4rs::append::file::FileAppender;
use log4rs::config::{Appender, Config, Root};
use log4rs::encode::pattern::PatternEncoder;
use std::fs;

pub fn setup_log(settings: &Data) {
    if let Some(log_path) = &settings.parsed.paths.log_out {
        if fs::remove_file(log_path).is_ok() {};
        let logfile = FileAppender::builder()
            .encoder(Box::new(PatternEncoder::new("{l}: {m}\n")))
            .build(log_path)
            .unwrap();

        let config = Config::builder()
            .appender(Appender::builder().build("logfile", Box::new(logfile)))
            .build(Root::builder().appender("logfile").build(LevelFilter::Info))
            .unwrap();

        log4rs::init_config(config).unwrap();
    };
}
