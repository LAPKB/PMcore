use eyre::Result;

fn main() -> Result<()> {
    let scenarios = np_core::base::datafile::parse(&"examples/two_eq_lag.csv".to_string()).unwrap();
    let scenario = scenarios.last().unwrap();
    dbg!(&scenario);
    // dbg!(simple_sim(
    //     &Engine::new(Sim {}),
    //     scenario,
    //     &Array1::from(vec![
    //         0.7012468470182522,
    //         0.046457990962687504,
    //         82.4722461587669,
    //         1.4065258528674902
    //     ])
    // ));
    // dbg!(simple_sim(
    //     &Engine::new(Sim {}),
    //     scenario,
    //     &Array1::from(vec![
    //         0.45653373718261686,
    //         0.046457990962687504,
    //         82.4722461587669,
    //         1.4065258528674902
    //     ])
    // ));
    // dbg!(simple_sim(
    //     &Engine::new(Sim {}),
    //     scenario,
    //     &Array1::from(vec![
    //         0.7012468470182522,
    //         0.053580406975746155,
    //         82.4722461587669,
    //         1.4065258528674902
    //     ])
    // ));
    // dbg!(simple_sim(
    //     &Engine::new(Sim {}),
    //     scenario,
    //     &Array1::from(vec![
    //         0.7012468470182522,
    //         0.046457990962687504,
    //         54.13560247421265,
    //         1.4065258528674902
    //     ])
    // ));
    // dbg!(simple_sim(
    //     &Engine::new(Sim {}),
    //     scenario,
    //     &Array1::from(vec![
    //         0.7012468470182522,
    //         0.046457990962687504,
    //         82.4722461587669,
    //         1.799925994873047
    //     ])
    // ));
    // dbg!(simple_sim(
    //     &Engine::new(Sim {}),
    //     scenario,
    //     &Array1::from(vec![
    //         0.7012468470182522,
    //         0.046457990962687504,
    //         82.4722461587669,
    //         0.
    //     ])
    // ));
    // dbg!(simple_sim(
    //     &Engine::new(Sim {}),
    //     scenario,
    //     &Array1::from(vec![
    //         0.7012468470182522,
    //         0.046457990962687504,
    //         82.4722461587669,
    //         4.
    //     ])
    // ));
    // dbg!(simple_sim(
    //     &Engine::new(Sim {}),
    //     scenario,
    //     &Array1::from(vec![
    //         0.45653373718261686,
    //         0.053580406975746155,
    //         54.13560247421265,
    //         1.799925994873047
    //     ])
    // ));

    Ok(())
}
