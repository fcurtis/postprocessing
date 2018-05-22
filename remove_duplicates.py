# Generic python modules
from os import listdir, mkdir
from os.path import isfile, isdir, join
import sys
import shutil
import time

# Specific python modules
from mpi4py import MPI 
from pymatgen.analysis.structure_matcher import (StructureMatcher,
                                                ElementComparator,
                                                SpeciesComparator,
                                                FrameworkComparator)
# Gator modules
sys.path.append("../gator/src/")                                               
from structures.structure import Structure
from core.file_handler import write_data 



def main():
    '''
    Peforms a duplicate check on a folder of jsons
    using pymatgen and MPI4py 
    '''

    # User defined settings and paths
    in_folder = "test_pool"          #folder of jsons to be checked                                           
    energy_name = "energy" #stored energy name 
    L_tol =.2       # lattice length tolerance pymatgen                                                       
    S_tol = .4      # site tolerance pymatgen                                                
    Angle_tol = 3   # lattice angle tolerance pymatgen

    # MPI communicator
    comm = MPI.COMM_WORLD

    # Remove Duplicates
    comp = ComparisonMPI4py(comm, 
                            in_folder, 
                            energy_name,
                            L_tol,
                            S_tol,
                            Angle_tol)
    comp.compare_all_structures()
    return


class ComparisonMPI4py():
    '''
    Compares jsons in in_folder using pymatgen's StructureMatcher

    If more than one duplicate of the same structure exists, 
    the one with the lowest stored energy_name is saved

    Saves a folder of all the non-duplicates as "<in_folder>_no_dups"
    '''
    def __init__(self, comm, in_folder, energy_name, L_tol, S_tol, Angle_tol):
        self.in_jsons = listdir(in_folder)
        self.in_IDs = [i.strip(".json") for i in self.in_jsons]
        self.in_folder = in_folder
        self.energy_name = energy_name
        self.comm = comm
        self.L_tol = L_tol
        self.S_tol = S_tol
        self.Angle_tol = Angle_tol

    def compare_all_structures(self):
        '''
        Checks an entire folder of json structures for any 
        duplicates

        Saves new folder with all non duplicates
        '''
        # Store beginning time
        start_time = time.time()

        # Output number of parallel processes                                  
        processes = self.comm.Get_size()                                       
        if self.comm.Get_rank() == 0:                                          
            print("--- Duplicate check with %i parallel processes ---\n" % processes) 

        # Keep track of structures already checked
        # Stores non_duplicate_Ids of non duplicate structures
        non_duplicate_IDs = []
        checked = []
        comp_list  = self.in_IDs
        for ID in comp_list:
            if ID not in checked:
                if self.comm.Get_rank() == 0:                                          
                    print("-- %i/%i structures checked" % 
                             (len(checked), len(comp_list)))
                dup_IDs = self.check_if_duplicate(ID, comp_list)
                for dup in dup_IDs:
                    checked.append(dup)
                lowest_energy_ID = self.get_lowest_energy(dup_IDs)
                non_duplicate_IDs.append(lowest_energy_ID)
        self.comm.Barrier()

        # Save non duplicate jsons in new folder
        if self.comm.Get_rank() ==0:
            out_folder = self.in_folder+ "_no_dups"
            if not isdir(out_folder):
                mkdir(out_folder)
            for ID in non_duplicate_IDs:
                shutil.copy(join(self.in_folder,ID + ".json"), join(out_folder, ID+".json")) 
            print ("-- Time for duplicate check %s seconds" % (time.time()-start_time))

    def get_lowest_energy(self, dup_IDs):
        '''
        Given the structure IDs of duplicate structures,
        returns the structure ID of the structure with the
        lowest stored energy
        '''

        if self.comm.Get_rank() == 0:
            print ("-- %s are duplicates" % (dup_IDs))
            energies = []
            for ID in dup_IDs:
                struct = self.return_structure(ID)
                energy = struct.get_property(self.energy_name)
                if energy == None:
                    message = "Energy name %s not found for %s" %(self.energy_name, ID)
                    raise ValueError(message)
                energies.append([ID, energy])

            # Sort duplicate structures by energy
            sorted_en = sorted(energies, key=lambda x: x[1])

            # Return ID of the lowest-energy duplicate
            lowest_energy_ID, lowest_energy = sorted_en[0]
            if self.comm.Get_rank() == 0:
                print ("-- %s has the lowest energy of %.3f eV\n" %(lowest_energy_ID,
                                                               lowest_energy))      
            return sorted_en[0][0]

    def return_structure(self, ID):
        '''
        Given the ID of a structure, returns a GAtor Structure object
        containing its properties 
        '''
        struct_path = join(self.in_folder, ID+ str(".json"))
        struct = Structure()                       
        struct.build_geo_from_json_file(struct_path)
        return struct 

    def check_if_duplicate(self, struct_ID, comp_list_IDs):
        '''
        Checks if input Structure is a duplicate of any others 
        in the comp_list
        '''       
        # Return GAtor Structure object from ID
        struct = self.return_structure(struct_ID)
  
        # Get list of Structures to compare
        comp_list = []
        for i in range(len(comp_list_IDs)):
            comp_list.append([comp_list_IDs[i], 
                             self.return_structure(comp_list_IDs[i])])

        # Perform duplicate check
        # Checks for possibility of more than one duplicate per 
        # compared structure                                                                               
        i = 0                                                                  
        dups_list = []
        dups_list.append(struct_ID)
        while i < len(comp_list):                          
            compare = comp_list[i:self.comm.Get_size()+i]      
            if len(compare) < self.comm.Get_size():                            
                while len(compare) < self.comm.Get_size():                     
                    compare.append(comp_list[0])                               
            ID, structc = self.comm.scatter(compare, root=0)                   
            fit = self.compute_pymatgen_fit(struct, structc)                   
            fit_list = self.comm.gather(fit, root=0)                           
            if self.comm.Get_rank() == 0:                                      
                for fit in fit_list:                                           
                    if fit is False:                                           
                        continue                                               
                    else:            
                        ID = fit
                        if ID != struct_ID:
                            dups_list.append(ID)                                                                                            
            self.comm.Barrier()                                                                     
            i += self.comm.Get_size()                                                                                                       
        # Return  IDs of duplicate structure(s) in dups_list
        self.comm.Barrier()
        dups_list = self.comm.bcast(dups_list, root=0)               
        return dups_list


    def compute_pymatgen_fit(self, s1, s2):      
        '''
        Compares two GAtor Structure objects for
        similiarity using pymatgen's StructureMatcher
        '''                                                              
        sm =  (StructureMatcher(ltol=self.L_tol,                                    
                    stol=self.S_tol,                                                
                    angle_tol=self.Angle_tol,                                       
                    primitive_cell=True,                                       
                    scale=False,                                               
                    attempt_supercell=False,                                   
                    comparator=SpeciesComparator()))                           
                                                                               
        sp1 = s1.get_pymatgen_structure()                                      
        sp2 = s2.get_pymatgen_structure()                                      
        fit = sm.fit(sp1, sp2)                                                 

        # If the structure is a duplicate 
        # return the ID of the duplicate                                                                        
        if fit:                                                                
            return s2.struct_id  
        # Else return False                                              
        return fit 

if __name__ == "__main__":
    main()