pub mod algorithms {
    pub mod npag;
}
pub mod routines {
    pub mod datafile;
    pub mod initialization {
        pub mod sobol;
    }
    pub mod optimization {
        pub mod expansion;
        pub mod optim;
    }
    pub mod output;

    pub mod settings {
        pub mod run;
        pub mod simulator;
    }
    pub mod evaluation {
        pub mod ipm;
        pub mod prob;
        pub mod qr;
        pub mod sigma;
    }
    pub mod simulation {
        pub mod predict;
        pub mod simulator;
    }
    pub mod temp;
}
pub mod tui;

pub mod prelude {
    pub use crate::prelude::evaluation::{prob, sigma, *};
    pub use crate::routines::initialization::*;
    pub use crate::routines::optimization::*;
    pub use crate::routines::simulation::*;
    pub use crate::routines::*;
    pub use crate::tui::ui::*;
}

//Tests
mod tests;
