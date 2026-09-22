# standard lib imports
import argparse
import pickle

# 3rd party lib imports
import MDAnalysis as mda

# my lib imports
from train.physics import calculate_torii_dipole_batch

def extract_from_universe(u):
    osc_types = list(set(u.residues.resnames))
    if 'ASN' in osc_types:
        osc_types.append('ASN-SC')
        asn_res = u.select_atoms('resname ASN').residues
        CG = u.select_atoms('resname ASN and name CG')
        OD1 = u.select_atoms('resname ASN and name OD1')
        ND2 = u.select_atoms('resname ASN and name ND2')
    else:
        asn_res = None
    if 'GLN' in osc_types:
        osc_types.append('GLN-SC')
        gln_res = u.select_atoms('resname GLN').residues
        CD = u.select_atoms('resname GLN and name CD')
        OE1 = u.select_atoms('resname GLN and name OE1')
        NE2 = u.select_atoms('resname GLN and name NE2')

    else:
        gln_res = None
    
    carbons = u.select_atoms('name C')[:-1]
    oxygens = u.select_atoms('name O')[:-1]
    nitrogens = u.select_atoms('name N')[1:]
    alpha_carbons = u.select_atoms('name CA')
    
    phi_sel = []
    psi_sel = []
    for res in u.residues:
        phi_sel.append(res.phi_selection())
        psi_sel.append(res.psi_selection())
    
    data = {key: [] for key in osc_types}
    n = len(u.residues)
    for ts in u.trajectory[:]:
        C_prev_positions = carbons.positions
        O_prev_positions = oxygens.positions
        N_curr_positions = nitrogens.positions
        CA_positions = alpha_carbons.positions
        
        dipoles = calculate_torii_dipole_batch(C_prev_positions, O_prev_positions, N_curr_positions)
        for i, res in enumerate(u.residues):
            if res.resid < n:
                osc = {
                        'type':'backbone',
                        'residue_key':(res.resid, res.resname),
                        'frame':ts.frame,
                        'oscillator_index':i,
                        'predicted_atoms':{'C_prev':C_prev_positions[i], 'O_prev':O_prev_positions[i], 'N_curr':N_curr_positions[i], 'CA_prev':CA_positions[i], 'CA_curr':CA_positions[i+1]},
                        'predicted_dipole': dipoles[i],
                       }
                if i == 0:
                    psi_N = None
                    phi_N = None
                    psi_C = psi_sel[i].dihedral.value()
                    phi_C = phi_sel[i+1].dihedral.value()
                elif i == n - 1:
                    psi_N = psi_sel[i-1].dihedral.value()
                    phi_N = phi_sel[i].dihedral.value()
                    psi_C = None
                    phi_C = None
                else:
                    psi_N = psi_sel[i-1].dihedral.value()
                    phi_N = phi_sel[i].dihedral.value()
                    psi_C = psi_sel[i].dihedral.value()
                    phi_C = phi_sel[i+1].dihedral.value() if phi_sel[i+1] else None
                osc['predicted_rama_nnfs'] = {'psi_N':psi_N, 'phi_N':phi_N, 'psi_C':psi_C, 'phi_C': phi_C}
                data[res.resname].append(osc)
                
        if asn_res is not None:
            CG_pos = CG.positions
            OD1_pos = OD1.positions
            ND2_pos = ND2.positions
            dipoles = calculate_torii_dipole_batch(CG_pos, OD1_pos, ND2_pos)
            for j, res in enumerate(asn_res):
                i += 1
                osc = {
                        'type':'sidechain',
                        'residue_key':(res.resid, res.resname),
                        'frame':ts.frame,
                        'oscillator_index':i,
                        'predicted_atoms':{'CG':CG_pos[j], 'OD1':OD1_pos[j], 'ND2':ND2_pos[j]},
                        'predicted_dipole': dipoles[j]
                       }
                data['ASN-SC'].append(osc)
    
        if gln_res is not None:
            CD_pos = CD.positions
            OE1_pos = OE1.positions
            NE2_pos = NE2.positions
            dipoles = calculate_torii_dipole_batch(CD_pos, OE1_pos, NE2_pos)
            for j, res in enumerate(gln_res):
                i += 1
                osc = {
                        'type':'sidechain',
                        'residue_key':(res.resid, res.resname),
                        'frame':ts.frame,
                        'oscillator_index':i,
                        'predicted_atoms':{'CD':CD_pos[j], 'OE1':OE1_pos[j], 'NE2':NE2_pos[j]},
                        'predicted_dipole': dipoles[j],
                       }

                data['GLN-SC'].append(osc)
    
    return data


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-f', '--input', required=True, nargs='+')
    parser.add_argument('-o', '--output', default='extracted_inputs.pkl')
    
    args = parser.parse_args()
    
    u = mda.Universe(*args.input)
    
    data = extract_from_universe(u)
            
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    
    
    
if __name__ == '__main__':
    main()
