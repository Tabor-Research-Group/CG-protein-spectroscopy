# CG Protein Spectroscopy 
This repository contains code for predicting amide-I region IR spectra from MARTINI CG MD trajectories. 

The workflow consists of two stages a minimal, spectroscopically-motivated backmapping DDPM and a transformer-based site frequency perturbation model. Other backmapping methods may be used in place of the DDPM presented here. 

## Dependencies
The code has the following dependencies:
- Python=3.11
- scikit-learn
- MDAnalysis
- torch 
- torchvision
- torch_geometric

Final IR spectra are obtained using the [NISE_2017](https://github.com/GHlacour/NISE_2017) software package. 

## Example Inference Workflow

See `/backmapping` and `/freq_model` for additional details of the two stages. Contents of the files passed to the `--config` flags should be updated for your systems. 

```bash
# Backmapping Model Inference
python /backmapping/inference_cg_only.py --config /backmapping/config_inference_traj.yaml --checkpoint /backmapping/best.pt
 
# Site Frequency Model Inference
python /freq_model/inference_spectra_cg_only.py --config /freq_model/config_inference.json

# Convert model outputs to NISE binary inputs
python /files_for_NISE/convert_binary.py {target_files}

# See NISE_2017 for usage
{path_to_nise}/NISE_2017/bin/NISE nise_input 
```

Other backmapping models can be used in place of the model provided here. The script `/freq_model/extract_generic.py` can prepare backmapped trajectories for use with the site frequency model.

```
python /freq_model/extract_generic.py -f backmapped_trajectory.pdb 
```

