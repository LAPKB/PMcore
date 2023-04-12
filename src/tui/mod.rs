pub mod actions;
pub mod inputs;
pub mod state;
pub mod ui;

use log::{debug, warn};

use self::actions::{Action, Actions};
use self::inputs::key::Key;
use self::state::AppState;
use std::fs::File;

#[derive(Debug, PartialEq, Eq)]
pub enum AppReturn {
    Exit,
    Continue,
}

/// The main application, containing the state
pub struct App {
    /// Contextual actions
    actions: Actions,
    /// State
    state: AppState,
}

impl App {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let actions = vec![Action::Quit, Action::Stop].into();
        let state = AppState::new();

        Self { actions, state }
    }

    /// Handle a user action
    pub fn do_action(&mut self, key: Key) -> AppReturn {
        if let Some(action) = self.actions.find(key) {
            debug!("Run action [{:?}]", action);
            match action {
                Action::Quit => AppReturn::Exit,
                Action::Stop => {
                    // Write the "stop.txt" file
                    log::info!("Stop signal received - writing stopfile");
                    let filename = "stop";
                    File::create(filename).unwrap();
                    AppReturn::Continue
                }
            }
        } else {
            warn!("No action associated to {}", key);
            AppReturn::Continue
        }
    }

    pub fn actions(&self) -> &Actions {
        &self.actions
    }
    pub fn state(&self) -> &AppState {
        &self.state
    }
}
