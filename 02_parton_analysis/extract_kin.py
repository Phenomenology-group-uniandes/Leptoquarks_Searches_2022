import os
def add_parent_lib_path(name="Pheno_BSM"):
    import sys
    sys.path.append(
        os.path.join(
            sys.path[0].split(name)[0],
            name
        )
    )
add_parent_lib_path()

from delphes_reader.lhereader import LHE_Loader 
from delphes_reader.lhereader import readLHEF 
from delphes_reader.lhereader import get_kinematics_row
from delphes_reader.lhereader import get_event_by_child
from delphes_reader.root_analysis import make_histograms