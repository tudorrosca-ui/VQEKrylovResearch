import pennylane as qml


molecule_name = "H2"
bond_length = 0.742  # LiH 1.57
# get molecule data using pennylane
DATASET = qml.data.load("qchem", molname=molecule_name, bondlength=bond_length, basis="STO-3G")[0]
COORDS = DATASET.molecule.coordinates
SYMBOLS = DATASET.molecule.symbols
CHARGE = DATASET.molecule.charge

# get pennylane data into readable form for pySCFDriver
atom_info = ""
for i in range(len(SYMBOLS)):
    atom_info += SYMBOLS[i] + " " + str(COORDS[i][0]) + " " + str(COORDS[i][1]) + " " + str(COORDS[i][2]) + "; "

atom_info = atom_info[0:-2]

print(atom_info)
