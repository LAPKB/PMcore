pub mod actions;
pub mod components;
pub mod inputs;
pub mod state;
pub mod ui;

use crate::prelude::output::NPCycle;

use self::actions::{Action, Actions};
use self::inputs::key::Key;
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
    state: NPCycle,
    /// Index for tab
    tab_index: usize,
    /// Tab titles
    tab_titles: Vec<&'static str>,
}

impl App {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let actions = vec![Action::Quit, Action::Stop, Action::Next].into();
        let state = NPCycle::new();
        let tab_index = 0;
        let tab_titles = vec!["Logs", "Plot", "Parameters"];

        Self {
            actions,
            state,
            tab_index,
            tab_titles,
        }
    }

    /// Handle a user action
    pub fn do_action(&mut self, key: Key) -> AppReturn {
        if let Some(action) = self.actions.find(key) {
            tracing::debug!("Run action [{:?}]", action);
            match action {
                Action::Quit => AppReturn::Exit,
                Action::Stop => {
                    // Write the "stop.txt" file
                    tracing::info!("Stop signal received, program will stop after current cycle");
                    let stopfile = "stop";
                    File::create(stopfile).unwrap();
                    AppReturn::Continue
                }
                Action::Next => {
                    self.tab_index += 1;
                    if self.tab_index >= self.tab_titles.len() {
                        self.tab_index = 0;
                    }
                    AppReturn::Continue
                }
            }
        } else {
            tracing::trace!(
                "The {} key was registered, but it has no associated action",
                key
            );
            AppReturn::Continue
        }
    }

    pub fn actions(&self) -> &Actions {
        &self.actions
    }
    pub fn state(&self) -> &NPCycle {
        &self.state
    }
}
