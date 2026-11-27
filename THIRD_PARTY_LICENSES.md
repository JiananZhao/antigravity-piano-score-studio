# Third-Party Licenses & Attribution

This plugin repository contains only MIT-licensed code, documentation, and workflow rules. External audio libraries, models, and tools installed locally via `install.ps1` / `install.sh` / `setup_assets.py` carry their own licenses:

1. **Salamander Grand Piano V3** (~714 MB FLAC)
   - Author: Alexander Holm; SFZ remapping (kinwie / SFZ Instruments)
   - License: **Creative Commons Attribution 3.0 Unported (CC-BY 3.0)**
   - Source: https://archive.org/details/SalamanderGrandPianoV3 / https://github.com/sfzinstruments/SalamanderGrandPianoV3
   - Requirement: Retain attribution to Alexander Holm when distributing rendered samples or redistributing the SFZ library.

2. **sfizz** (~11 MB)
   - License: **BSD-2-Clause**
   - Source: https://github.com/sfztools/sfizz

3. **MuScriptor** (Code & Weights)
   - Code License: **MIT** (https://github.com/muscriptor/muscriptor)
   - Model Weights (`MuScriptor/muscriptor-medium`): **CC BY-NC 4.0 (Non-Commercial)**. Each user must accept the license on HuggingFace and download weights with their own token; do NOT bundle weights into public repositories.

4. **MuseScore Studio 4**
   - License: **GPLv3** (https://github.com/musescore/MuseScore)
   - Installed separately or invoked locally as an external CLI binary.
