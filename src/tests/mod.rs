#[cfg(test)]
use super::base::*;

#[test]
fn basic_sobol(){
    assert_eq!(lds::sobol(5, vec![(0.,1.),(0.,1.),(0.,1.)], 347), ndarray::array![
        [0.10731888, 0.14647412, 0.58510387],
        [0.9840305, 0.76333654, 0.19097507],
        [0.3847711, 0.73466134, 0.2616291],
        [0.70233, 0.41038263, 0.9158684],
        [0.60167587, 0.61712956, 0.62639713]
    ])
}

#[test]
fn scaled_sobol(){
    assert_eq!(lds::sobol(5, vec![(0.,1.),(0.,2.),(-1.,1.)], 347), ndarray::array![
        [0.10731888, 0.29294825, 0.17020774],
        [0.9840305, 1.5266731, -0.61804986],
        [0.3847711, 1.4693227, -0.4767418],
        [0.70233, 0.82076526, 0.8317368],
        [0.60167587, 1.2342591, 0.25279427]
    ])
}

#[test]
fn read_mandatory_settings(){
    let settings = settings::read("config.toml".to_string());
    assert_eq!(settings.paths.data, "data.csv");
    assert_eq!(settings.config.cycles, 1024);
    assert_eq!(settings.config.engine, "NPAG");
}

#[test]
fn read_test_datafile(){
    let scenarios = datafile::parse("test.csv".to_string());
    if let Ok(scenarios) = scenarios {
        assert_eq!(scenarios.len(), 20);
        assert_eq!(scenarios.last().unwrap().id, "20");
        assert_eq!(scenarios.last().unwrap().time, 
            [0.0,24.0,48.0,72.0,96.0,120.0,120.0,
            120.77,121.75,125.67,128.67,143.67,]);
        
    }
}