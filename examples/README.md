# Examples
## Conway's Game of Life
- Conway's Game of Life is a well-known cellular automaton in which each generation of a population "evolves" from the previous one according to a set of predefined rules. This particular implementation is inspired in the very famous APL "one-liner" [implementation](https://aplwiki.com/wiki/Conway%27s_Game_of_Life).
```
cargo run --release --example conway_gol
```
<img src="https://github.com/JErnestoMtz/rapl/blob/main/graphics/gol.gif" width="300">

## Ising Model
- The [Ising model](https://en.wikipedia.org/wiki/Ising_model) is a mathematical model of interacting particles with binary (spin up or down) degrees of freedom, for example magnetic moments in a ferromagnetic material. This is a simple implementation of this model that produces an animation of the spin lattice in the terminal:
```
cargo run --release --example ising_model
```
<img src="https://github.com/JErnestoMtz/rapl/blob/main/graphics/Ising.gif" width="300">


