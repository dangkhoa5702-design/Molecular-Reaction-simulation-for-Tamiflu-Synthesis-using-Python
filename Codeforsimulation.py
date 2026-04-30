"""
Interactive 3D Reaction Mechanism Animation for Oseltamivir Synthesis
Features: Morphing molecules, reaction arrows, timeline slider, 3D grid, labels
JupyterLab/py3Dmol | April 28, 2026
"""
import time, numpy as np
from IPython.display import HTML, display
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
from ipywidgets import interact, FloatSlider, Play# Complete oseltamivir synthesis pathway (11 steps)
pathway_smiles = {
    0: "O[C@@H]1C[C@H](O)[C@@H](O)C(C(O)=O)=C1",  # Shikimic acid
    1: "CCOC(=O)C1=C[C@@H](O)[C@H](O)[C@H](O)C1",  # Ethyl ester
    2: "CCOC(=O)C1=C[C@@H](O)[C@H]2OC(CC)(CC)O[C@H]2C1",  # Acetal
    3: "CCOC(=O)C1=C[C@@H](OS(=O)(=O)C)[C@H]2OC(CC)(CC)O[C@H]2C1",  # Mesylate
    4: "CCOC(=O)C1=CC2O[C@@H]2C[C@H]1OC(CC)CC",  # Epoxide
    5: "CCOC(=O)[C@H]1C[C@@H]2O[C@@H]2C=C1N=[N+]=[N-]",  # Azide
    6: "CCOC(=O)[C@H]1C[C@@H]2O[C@@H]2C=C1NC(=O)C",  # Acetamide
    7: "CCOC(=O)[C@@H]1[C@H](NC(=O)C)C[C@H]2COC(CC=C)C[C@@H]12",  # Allyl ether
    8: "NC[C@@H]1O[C@H](CO)C[C@H](NC(=O)C)[C@H]1C(O)=O",  # Amino alcohol
    9: "CCOC(=O)[C@@H]1[C@H](NC(=O)C)C[C@H](CO[C@H](CO)CO)O1",  # Triol ester
    10: "CCOC(=O)[C@@H]1[C@H](NC(=O)C)C[C@@H](N)C[C@H]1NC(C)=O"  # Oseltamivir
}
step_names = {
    0: "(-)-Shikimic Acid", 1: "Ethyl Ester", 2: "Pentylidene Acetal", 
    3: "Mesylate", 4: "Epoxide", 5: "Azide Opening", 6: "Acetamide",
    7: "Allyl Ether", 8: "Deprotected", 9: "Triol Ester", 10: "Oseltamivir"
}
reagents = {
    1: "EtOH/H⁺", 2: "3-Pentanone/TsOH", 3: "MsCl/Et₃N", 4: "KOtBu",
    5: "NaN₃/NH₄Cl", 6: "Ac₂O", 7: "AllylOH/NaH", 8: "Pd/C/H₂",
    9: "EtOH/HCl", 10: "H₃PO₄"
}
def generate_3d_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=5, params=AllChem.ETKDG())
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    return mol# Precompute all 3D molecules
print("Generating 3D structures...")
mols_3d = {i: generate_3d_mol(smiles) for i, smiles in pathway_smiles.items()}
# 1. INITIALIZE ALL GLOBAL VARIABLES
reagent_smiles = {
    1: "CCO",        # EtOH
    2: "CCC(=O)CC",  # 3-Pentanone
    3: "CS(=O)(=O)Cl", # MsCl
    4: "CC(C)(C)[O-]",# tBuO-
    5: "[N-][N+]=N",  # N3-
    6: "CC(=O)OC(=O)C",# Ac2O
    7: "C=CCO",       # Allyl alcohol
    8: "[H][H]"       # H2
}
reagents = {
    1: "EtOH/H⁺", 2: "3-Pentanone/TsOH", 3: "MsCl/Et₃N", 4: "KOtBu",
    5: "NaN₃/NH₄Cl", 6: "Ac₂O", 7: "AllylOH/NaH", 8: "Pd/C/H₂",
    9: "EtOH/HCl", 10: "H₃PO₄"
}
step_names = {
    0: "(-)-Shikimic Acid", 1: "Ethyl Ester", 2: "Pentylidene Acetal", 
    3: "Mesylate", 4: "Epoxide", 5: "Azide Opening", 6: "Acetamide",
    7: "Allyl Ether", 8: "Deprotected", 9: "Triol Ester", 10: "Oseltamivir"
}
# 2. DEFINITION OF THE ANIMATION FUNCTION
def create_reaction_scene(progress=0.0):
    step = min(int(progress * 10), 9)
    alpha = progress * 10 - step
    # 1. Morphing Logic
    mol_start, mol_end = mols_3d[step], mols_3d[step+1]
    mol_morph = Chem.Mol(mol_end)
    mol_morph.RemoveAllConformers()
    conf = Chem.Conformer(mol_morph.GetNumAtoms())
    match = mol_end.GetSubstructMatch(mol_start)
    c1, c2 = mol_start.GetConformer(), mol_end.GetConformer()
    for i in range(mol_morph.GetNumAtoms()):
        s_idx = match.index(i) if i in match else -1
        p2 = c2.GetAtomPosition(i)
        p1 = c1.GetAtomPosition(s_idx) if s_idx != -1 else p2
        eased = alpha**2 * (3 - 2*alpha)
        p_m = (1 - eased) * np.array([p1.x, p1.y, p1.z]) + eased * np.array([p2.x, p2.y, p2.z])
        conf.SetAtomPosition(i, p_m)
    mol_morph.AddConformer(conf)
    # 2. Rendering Setup
    view = py3Dmol.view(width=900, height=600)
    view.addModel(Chem.MolToMolBlock(mol_morph), 'sdf') # Model 0: Molecule
    # 3. Kinetic Reagent
    if step + 1 in reagent_smiles:
        reg_mol = Chem.AddHs(Chem.MolFromSmiles(reagent_smiles[step + 1]))
        AllChem.EmbedMolecule(reg_mol)
        view.addModel(Chem.MolToMolBlock(reg_mol), 'sdf') # Model 1: Reagent
        view.setStyle({'model': 1}, {'sphere': {'color': 'red', 'scale': 0.4}})
        # Label all atoms in the reagent
        for atom in reg_mol.GetAtoms():
            pos = reg_mol.GetConformer().GetAtomPosition(atom.GetIdx())
            view.addLabel(f"{atom.GetSymbol()}", {'position': {'x': pos.x, 'y': pos.y, 'z': pos.z}, 'fontSize': 10, 'fontColor': 'red'})
    # 4. Label all atoms in the primary molecule
    for atom in mol_morph.GetAtoms():
        pos = mol_morph.GetConformer().GetAtomPosition(atom.GetIdx())
        view.addLabel(f"{atom.GetSymbol()}{atom.GetIdx()}", {
            'position': {'x': pos.x, 'y': pos.y, 'z': pos.z},
            'fontSize': 10, 'fontColor': 'blue', 'showBackground': False
        })
    # 5. Persistent Grid and Info
    view.addBox({'center': {'x': 0, 'y': 0, 'z': 0}, 'dimensions': {'w': 12, 'h': 12, 'd': 12}, 
                 'color': 'gray', 'opacity': 0.05, 'wireframe': True})
    view.setStyle({'model': 0}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.1}, 'sphere': {'scale': 0.2}})
    view.addLabel(f"Phase: {step_names[step+1]}", {'position': {'x': 0, 'y': 6, 'z': 0}, 'useScreen': True, 'fontSize': 18})
    
    view.zoomTo()
    view.setBackgroundColor('white')
    return view
    # 3. LAUNCH INTERACTIVE INTERFACE
from ipywidgets import interact, FloatSlider, Play
interact(create_reaction_scene, progress=FloatSlider(min=0, max=1, step=0.01, value=0))
print("🎬 INTERACTIVE 3D OSALTAMIVIR SYNTHESIS ANIMATION")
print("Use slider or play button to watch molecules morph!")
interact(create_reaction_scene, 
         progress=FloatSlider(min=0, max=1, step=0.01, value=0, 
                             description='Reaction Progress', 
                             style={'description_width': 'initial'}),
         continuous_update=False)# Auto-play button
play_widget = Play(value=0, min=0, max=1, step=0.02, interval=200, 
                   description="Play", disabled=False)
interact(lambda progress: create_reaction_scene(progress), progress=play_widget)