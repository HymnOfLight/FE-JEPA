### Checks of the elements

| Element | Rigid-body residual | Patch test, energy error | Cantilever tip deflection / Timoshenko: 10x1x1, 20x2x1, 40x4x2, 80x8x4 |
|---|---|---|---|
| H8I | 4e-16 | 0e+00 | 0.987, 0.988, 0.993, 0.994 |
| H8 | 3e-16 | 7e-16 | 0.658, 0.870, 0.959, 0.986 |
| T4 | 2e-16 | 2e-16 | 0.211, 0.514, 0.795, 0.933 |
| H20 | 1e-15 | 4e-16 | 0.983, 0.992, 0.994, 0.995 |
| H20R | 1e-15 | 2e-16 | 0.986, 0.993, 0.995, 0.995 |
| H27 | 8e-16 | 3e-16 | 0.987, 0.993, 0.995, 0.995 |
| T10 | 6e-16 | 7e-16 | 0.978, 0.991, 0.994, 0.995 |

### FS1: initial stiffness S (kNm/mrad); tests 18.19 and 16.84

Refinement (h in mm; layers through the end plate, each beam flange, half the web; the last column adds plate layers only):

| Element | h 20, 1/1/1 | h 10, 1/1/1 | h 5, 2/2/1 | h 2.5, 4/4/2 | h 2.5, 8/4/2 | Unknowns at the finest |
|---|---|---|---|---|---|---|
| H8I |  | 37.86 | 37.35 | 37.39 | 37.25 | 179,347 |
| H8 |  | 43.78 | 39.30 | 38.03 | 37.73 | 179,347 |
| T4 |  | 59.97 | 47.63 | 41.22 | 39.63 | 179,347 |
| H20 | 37.45 | 37.54 | 37.65 |  |  | 80,773 |
| H20R | 36.22 | 37.11 | 37.48 |  |  | 80,773 |
| H27 | 38.64 | 37.34 | 37.54 |  |  | 145,640 |
| T10 | 40.00 | 38.18 | 37.91 |  |  | 145,640 |

Layers through the end plate at h = 5 mm (2 per flange, 1 per half web):

| Element | 1 layer | 2 layers | 4 layers | 8 layers |
|---|---|---|---|---|
| H8I | 37.03 | 37.35 | 37.06 | 36.91 |
| H8 | 40.97 | 39.30 | 38.53 | 38.28 |
| T4 | 56.62 | 47.63 | 43.57 | 42.28 |
| H20 | 38.18 | 37.65 | 37.46 |  |
| H20R |  | 37.48 |  |  |
| H27 |  | 37.54 |  |  |
| T10 |  | 37.91 |  |  |

Finest mesh of each element and its ratio to the two tests:

| Element | Mesh | Unknowns | S | S / test a | S / test b | PDAS iterations | Largest optimality residual |
|---|---|---|---|---|---|---|---|
| H8I | h 2.5, 8/4/2 | 179,347 | 37.25 | 2.05 | 2.21 | 24 | 2e-13 |
| H8 | h 2.5, 8/4/2 | 179,347 | 37.73 | 2.07 | 2.24 | 25 | 1e-13 |
| T4 | h 2.5, 8/4/2 | 179,347 | 39.63 | 2.18 | 2.35 | 28 | 7e-14 |
| H20 | h 5, 4/2/1 | 113,541 | 37.46 | 2.06 | 2.22 | 27 | 2e-14 |
| H20R | h 5, 2/2/1 | 80,773 | 37.48 | 2.06 | 2.23 | 27 | 1e-13 |
| H27 | h 5, 2/2/1 | 145,640 | 37.54 | 2.06 | 2.23 | 27 | 7e-14 |
| T10 | h 5, 2/2/1 | 145,640 | 37.91 | 2.08 | 2.25 | 31 | 2e-14 |

### FS3: initial stiffness S (kNm/mrad); tests 23.23 and 21.56

Refinement (h in mm; layers through the end plate, each beam flange, half the web; the last column adds plate layers only):

| Element | h 20, 1/1/1 | h 10, 1/1/1 | h 5, 2/2/1 | h 2.5, 4/4/2 | h 2.5, 8/4/2 | Unknowns at the finest |
|---|---|---|---|---|---|---|
| H8I |  | 65.80 | 65.58 | 65.08 | 64.90 | 184,130 |
| H8 |  | 69.41 | 66.80 | 65.55 | 65.20 | 184,130 |
| T4 |  | 83.21 | 73.51 | 68.12 | 66.56 | 184,130 |
| H20 | 65.65 | 65.75 | 65.21 |  |  | 83,238 |
| H20R | 64.93 | 65.25 | 65.03 |  |  | 83,238 |
| H27 | 66.07 | 65.72 | 65.11 |  |  | 150,102 |
| T10 | 67.00 | 66.33 | 65.43 |  |  | 150,102 |

Layers through the end plate at h = 5 mm (2 per flange, 1 per half web):

| Element | 1 layer | 2 layers | 4 layers | 8 layers |
|---|---|---|---|---|
| H8I | 64.94 | 65.58 | 65.21 | 65.01 |
| H8 | 68.06 | 66.80 | 65.95 | 65.62 |
| T4 | 80.98 | 73.51 | 69.50 | 68.06 |
| H20 | 65.89 | 65.21 | 64.98 |  |
| H20R |  | 65.03 |  |  |
| H27 |  | 65.11 |  |  |
| T10 |  | 65.43 |  |  |

Finest mesh of each element and its ratio to the two tests:

| Element | Mesh | Unknowns | S | S / test a | S / test b | PDAS iterations | Largest optimality residual |
|---|---|---|---|---|---|---|---|
| H8I | h 2.5, 8/4/2 | 184,130 | 64.90 | 2.79 | 3.01 | 21 | 1e-13 |
| H8 | h 2.5, 8/4/2 | 184,130 | 65.20 | 2.81 | 3.02 | 21 | 1e-13 |
| T4 | h 2.5, 8/4/2 | 184,130 | 66.56 | 2.87 | 3.09 | 23 | 3e-13 |
| H20 | h 5, 4/2/1 | 118,412 | 64.98 | 2.80 | 3.01 | 27 | 9e-14 |
| H20R | h 5, 2/2/1 | 83,238 | 65.03 | 2.80 | 3.02 | 31 | 4e-14 |
| H27 | h 5, 2/2/1 | 150,102 | 65.11 | 2.80 | 3.02 | 24 | 5e-14 |
| T10 | h 5, 2/2/1 | 150,102 | 65.43 | 2.82 | 3.03 | 29 | 4e-14 |

