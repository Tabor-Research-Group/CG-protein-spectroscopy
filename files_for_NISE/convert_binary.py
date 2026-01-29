import struct
import numpy as np

# Function to convert text file to binary file
def convert_to_binary(input_filename, output_filename):
    with open(input_filename, 'r') as input_file, open(output_filename, 'wb') as output_file:
        for line in input_file:
            parts = line.split()
            frame_number = int(parts[0])
            hamiltonian_entries = np.array(parts[1:], dtype=np.float32)

            # Write frame number as a 32-bit float
            output_file.write(struct.pack('f', frame_number))

            # Write Hamiltonian entries as 32-bit floats
            output_file.write(hamiltonian_entries.tobytes())

# Example usage
convert_to_binary('hamiltonians_groundtruth.dat', 'hamiltonians_groundtruth.bin')
convert_to_binary('hamiltonians_predicted.dat', 'hamiltonians_predicted.bin')
convert_to_binary('dipoles_groundtruth.dat', 'dipoles_groundtruth.bin')
convert_to_binary('dipoles_predicted.dat', 'dipoles_predicted.bin')
