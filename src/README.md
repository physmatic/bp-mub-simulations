Run `python -m src.generate_mubs` to create and plot the MUB circuits.
* These circuits are of `QUBIT_NUM` size, defined in `consts.py`.
* The program produces `2**QUBIT_NUM + 1` circuits, where each circuit produces an orthonormal basis (each basis vector corresponds to a different standard input).
* The produced bases of the circuits are mutually unbiased (This validation is part of the script)

Run `python -m src.plot_runtime` to see the runtime of calculating a random circuit as a function of `QUBIT_NUM`.
This should display a runtime graph of `O(n**3)`, like the efficient implementation described in the paper.
