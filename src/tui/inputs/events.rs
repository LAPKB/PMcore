use std::thread;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use std::time::Duration;

use super::key::Key;
use super::InputEvent;

/// A small event handler that wrap crossterm input and tick event. Each event
/// type is handled in its own thread and returned to a common `Receiver`
pub struct Events {
    rx: UnboundedReceiver<InputEvent>,
    // Need to be kept around to prevent disposing the sender side.
    _tx: UnboundedSender<InputEvent>,
}

impl Events {
    /// Constructs an new instance of `Events` with the default config.
    pub fn new(tick_rate: Duration) -> Events {
        let (tx, rx) = unbounded_channel();

        let event_tx = tx.clone();
        thread::spawn(move || {
            loop {
                // poll for tick rate duration, if no event, sent tick event.
                if crossterm::event::poll(tick_rate).unwrap() {
                    if let crossterm::event::Event::Key(key) = crossterm::event::read().unwrap() {
                        let key = Key::from(key);
                        event_tx.send(InputEvent::Input(key)).unwrap();
                    }
                }
            }
        });

        Events { rx, _tx: tx }
    }

    pub fn next(&mut self) -> Option<InputEvent> {
        match self.rx.try_recv(){
            Ok(event) => Some(event),
            Err(_) => None
        }
    }
}