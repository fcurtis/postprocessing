# Python default modules
from os import listdir, mkdir
from os.path import isfile, isdir, join
import sys

# Pymatgen modules
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer as SGA               
from pymatgen.io.cif import CifWriter                                          

# Gator modules
sys.path.append("../gator/src")
from structures.structure import Structure
from core.file_handler import write_data 


def main():
    '''
    Converts an input folder of structures saved as jsons
    and outputs a folder of cif files.

    Symmetry is computed using Pymatgen.

    The energy name is input because the cifs are sorted by their energy rank
    The ranking is saved in the cif filename for easy visualization with 
    e.g. mercury. The spacegroup is also saved in the cif name.
    '''
    in_folder = "test_pool"                                                        
    out_folder = "test_pool_cifs" 
    energy_name = "energy"
    symmetry_tolerance = 0.85 # pymatgen symmetry tolerance detection 
    jsontocif = JsonToCif(in_folder, 
                          out_folder, 
                          energy_name, 
                          symmetry_tolerance)
    jsontocif.make_cifs()

class JsonToCif():
    '''
    Converts jsons to cifs using Pymatgen
    '''
    def __init__(self, in_folder, out_folder, energy_name, symmetry_tolerance):
        self.in_jsons = [join(in_folder,d) for d in listdir(in_folder)]
        self.out_folder = out_folder
        self.energy_name = energy_name
        self.symprec = symmetry_tolerance

    def make_cifs(self):
        properties = self.return_sorted_properties()
        self.write_symmetry_cifs(properties) 

    def return_sorted_properties(self):
        '''
        Returns a sorted array containing a GAtor Structure(), its energy,
        and its ID as given by GAtor/Genarris
        '''
        properties = []                                                            
        for j in self.in_jsons:                                                            
            struct = Structure() # Gator Structure object                                            
            struct.build_geo_from_json_file(j)                                     
            energy = struct.get_property(self.energy_name)                                                              
            ID = struct.struct_id                                                  
            properties.append([struct, energy, ID])                            
        properties.sort(key=lambda x: x[1])  
        return properties

    def write_symmetry_cifs(self, properties):
        '''
        Computes symmetry of structure using Pymatgen
        Then writes cit to out_folder
        '''
        count =1
        for struct, en, ID in properties:
            # identify space group
            structp = struct.get_pymatgen_structure()
            sg = SGA(structp, symprec= self.symprec).get_space_group_number() 

            # Pymatgen CIFwriter
            cw = CifWriter(structp,symprec=self.symprec)

            # Name output cif file
            if count < 10:
                count = "0" + str(count) # makes, e.g. 1 be 01 
            if not isdir(self.out_folder): mkdir(self.out_folder)
            name = join(self.out_folder,str(count)+"_"+str(ID)+"_"+str(sg)+".cif") 
            cw.write_file(name)

            # Increment count
            count = int(count)
            count+=1
   
            print ("Done with struct %s" % (ID))



if __name__ == "__main__":
    main()