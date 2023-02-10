#[cfg(test)]
use super::base::*;

#[test]
fn basic_sobol(){
    assert_eq!(lds::sobol(5, vec![(0.,1.),(0.,1.),(0.,1.)], 347), ndarray::array![
        [0.10731887817382813, 0.14647412300109863, 0.5851038694381714],
        [0.9840304851531982, 0.7633365392684937, 0.19097506999969482],
        [0.38477110862731934, 0.734661340713501, 0.2616291046142578],
        [0.7023299932479858, 0.41038262844085693, 0.9158684015274048],
        [0.6016758680343628, 0.6171295642852783, 0.6263971328735352]
    ])
}

#[test]
fn scaled_sobol(){
    assert_eq!(lds::sobol(5, vec![(0.,1.),(0.,2.),(-1.,1.)], 347), ndarray::array![
        [0.10731887817382813, 0.29294824600219727, 0.17020773887634277],
        [0.9840304851531982, 1.5266730785369873, -0.6180498600006104],
        [0.38477110862731934, 1.469322681427002, -0.4767417907714844],
        [0.7023299932479858, 0.8207652568817139, 0.8317368030548096],
        [0.6016758680343628, 1.2342591285705566, 0.2527942657470703]
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