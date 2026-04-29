import struct
import argparse
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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+')
    filenames = parser.parse_args().filenames
    
    for f in filenames:
        output = f[:-4] + '.bin'
        convert_to_binary(f, output)

if __name__ == "__main__":
    main()
