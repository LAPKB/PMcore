use std::collections::HashMap;
use std::error::Error;

type Record = HashMap<String, String>;

//This structure represents a single row in the CSV file
#[derive(Debug)]
struct Event{
    id: String,
    evid: isize,
    time: f64,
    dur: Option<f64>,
    dose: Option<f64>,
    // addl: Option<isize>,
    // ii: Option<isize>,
    input: Option<usize>,
    out: Option<f64>,
    // outeq: Option<isize>,
    // c0: Option<f32>,
    // c1: Option<f32>,
    // c2: Option<f32>,
    // c3: Option<f32>,
    // cov: HashMap<String, f32>
}
pub fn parse(path: String) -> Result<Vec<Scenario>, Box<dyn Error>> {

    let mut rdr = csv::ReaderBuilder::new()
        // .delimiter(b',')
        // .escape(Some(b'\\'))
        .comment(Some(b'#'))
        .from_path(path).unwrap();
    let mut events: Vec<Event> = vec![];

    for result in rdr.deserialize() {
        let mut record: Record = result?;
        
        events.push(Event{
            id: record.remove("ID").unwrap(),
            evid: record.remove("EVID").unwrap().parse::<isize>().unwrap(),
            time: record.remove("TIME").unwrap().parse::<f64>().unwrap(),
            dur: record.remove("DUR").unwrap().parse::<f64>().ok(),
            dose: record.remove("DOSE").unwrap().parse::<f64>().ok(),
            // addl: record.remove("ADDL").unwrap().parse::<isize>().ok(),
            // ii: record.remove("II").unwrap().parse::<isize>().ok(),
            input: record.remove("INPUT").unwrap().parse::<usize>().ok(),
            out: record.remove("OUT").unwrap().parse::<f64>().ok(),
            // outeq: record.remove("OUTEQ").unwrap().parse::<isize>().ok(),
            // c0: record.remove("C0").unwrap().parse::<f32>().ok(),
            // c1: record.remove("C1").unwrap().parse::<f32>().ok(),
            // c2: record.remove("C2").unwrap().parse::<f32>().ok(),
            // c3: record.remove("C3").unwrap().parse::<f32>().ok(),
            // cov: record.into_iter().map(|(key,value)|return (key, value.parse::<f32>().unwrap())).collect()
        });

    }


    let mut scenarios: Vec<Scenario> = vec![];
    
    let ev_iter = events.group_by_mut(|a,b| a.id == b.id);
    
    for group in ev_iter{
        scenarios.push(parse_events_to_scenario(group));
    }

    // dbg!(&scenarios);


    Ok(scenarios)
}

#[derive(Debug)]
pub struct Dose {
    pub time: f64,
    pub dose: f64,
    pub compartment: usize
}

#[derive(Debug)]
pub struct Infusion {
    pub time: f64,
    pub dur: f64,
    pub amount: f64,
    pub compartment: usize
}
//This structure represents a full set of dosing events for a single ID
//TODO: I should transform the ADDL and II elements into the right dose events
#[derive(Debug)]
pub struct Scenario{
    pub id: String, //id of the Scenario
    pub time: Vec<f64>, //ALL times
    pub infusions: Vec<Infusion>,
    // pub time_infusion: Vec<f64>,
    pub doses: Vec<Dose>,
    pub time_obs: Vec<f64>, //obs times
    // pub infusion: Vec<(f64,f64,usize)>,// dose, dur, comp
    pub obs: Vec<f64>, // obs @ time_obs
}

// Current Limitations:

// This version does not handle
// *  EVID!= 1 or 2
// *  ADDL & II
// *  OUTEQ
// *  INPUT
// *  DUR
// *  C0, C1, C2, C3

//TODO: time needs to be expanded with the times relevant to ADDL and II
//TODO: Also dose must be expanded because of the same reason

// dose: , //This should be a map (or function) with dose values for every time
// cov: , //this should be a matrix (or function ), with values for each cov and time
// rateiv: ,//this one should be a function, tht returns the rate calculation for RATEIV
            //based on the DOSE and DUR values

fn parse_events_to_scenario(events: &[Event]) -> Scenario{
    let mut time: Vec<f64> = vec![];
    // let mut time_infusion: Vec<f64> = vec![];  
    let mut doses: Vec<Dose> = vec![];
    let mut infusions: Vec<Infusion> = vec![];
    let mut time_obs: Vec<f64> = vec![];
    // let mut infusion: Vec<(f64,f64, usize)> = vec![];
    let mut obs: Vec<f64> = vec![];
    for event in events {
        time.push(event.time);
        
        if event.evid == 1 { //dose event
            if event.dur.unwrap_or(0.0) > 0.0 {
                infusions.push(
                    Infusion { 
                        time: event.time,
                        dur: event.dur.unwrap().into(),
                        amount: event.dose.unwrap().into(),
                        compartment: event.input.unwrap() - 1
                    });
            } else {
                doses.push(
                    Dose { 
                        time: event.time, 
                        dose: event.dose.unwrap(), 
                        compartment: event.input.unwrap() - 1
                    });
            }
            
        } else if event.evid == 0 { //obs event
            obs.push(event.out.unwrap());
            time_obs.push(event.time);
        }
        
    }

    Scenario { 
        id: events[0].id.clone(),
        time,
        doses,
        infusions,
        time_obs,
        obs
     }
}