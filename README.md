# NPcore
Rust Library with the building blocks needed to create new Non-Parametric algorithms and it's integration with Pmetrics

## Actual functionality

* Runs models using ODEs
* Basic NPAG implementation

## Examples

There are two examples implemented in this repository. Both of them using NPAG

run the following commands to run them:
'''
cargo run --example two_eq_lag --release
cargo run --example bimodal_ke --release
'''

Look at the corresponding examples/*.toml file to change the configuration of each run.


## NOTES

At the moment this library requires the nightly build of the rust compiler, make sure 
nightly is enabled by typing.

'''
rustup install nightly
rustup default nightly
'''
