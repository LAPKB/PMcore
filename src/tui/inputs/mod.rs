use self::key::Key;

pub mod events;
pub mod key;

#[derive(Debug)]
pub enum InputEvent {
    Input(Key),
}
