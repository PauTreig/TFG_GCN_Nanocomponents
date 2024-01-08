#!/usr/bin/python
import sys
import math
class Bond:
    def __init__(self):
        self.atom1 = None;
        self.atom2 = None;
        self.low = 0.0;
        self.high = 0.0;
        self.order = None;
        self.length = 0.0;
    def Read(self,line):
        parts = line.split();
        if (len(parts) != 5):
            return False;
        self.atom1 = parts[0];
        self.atom2 = parts[1];
        self.low = float(parts[2]);
        self.high = float(parts[3]);
        self.order = parts[4];
        return True
    def Satisfy(self,atom1,atom2,dist):
        if ( (atom1 == self.atom1 and atom2 == self.atom2) or 
             (atom1 == self.atom2 and atom2 == self.atom1) ):
            if float(dist) > self.low and float(dist) < self.high:
                return True;
        return False;

    def Show(self):
        print(str(self.atom1) + " " + str(self.atom2) + " " + str(self.low)+ " " +
              str(self.high) + " " + str(self.order)+" actual length = "+str(self.length));
        

class Atom:
    def __init__(self):
        self.name = None;
        self.x = self.y = self.z = 0.0
    def Show(self):
        print(str(self.name) + " " +
              str(self.x) + " " +
              str(self.y) + " " +
              str(self.z) )

class Molecule:
    def __init__(self):
        self.atoms = [];
        self.bonds = [];
    def FindBonds(self,rules):
        for i in range(0,len(self.atoms)):
            for j in range(i+1,len(self.atoms)):
                a1 = self.atoms[i];
                a2 = self.atoms[j];
                length = math.sqrt (  (a1.x-a2.x)**2 +
                                 (a1.y-a2.y)**2 +
                                 (a1.z-a2.z)**2)
                order = rules.CheckRules(a1.name,a2.name,length);
                if ( order != None ):
                    bond = Bond();
                    bond.atom1 = int(i);
                    bond.atom2 = int(j);
                    bond.length = length;
                    bond.order = order;
                    self.bonds.append(bond)

    def ParseXYZFileLines(self,lines):
        for i in range(2,len(lines)):
            pars = lines[i].split();
            if (len(pars) != 4):
                return False
            a = Atom();
            a.name = pars[0]
            try:
                a.x = float(pars[1])
                a.y = float(pars[2])
                a.z = float(pars[3])
            except ValueError:
                return False

            self.atoms.append(a)

        return True;


    def Show(self):
        print("  "+str(len(self.atoms)));
        print("xyz2mo2.py");
        for i in self.atoms:
            i.Show();
        for i in self.bonds:
            i.Show();


    def ShowAsMol2(self):
        print("@<TRIPOS>MOLECULE")
        print("mol")
        print(str(len(self.atoms))+" "+str(len(self.bonds)));
        print("SMALL")
        print("NO_CHARGES")
        print("")
        print("")
        print("@<TRIPOS>ATOM")
        for i in range(0,len(self.atoms)):
            print(str(i+1) + "  " + str(self.atoms[i].name) + " " +
                  str(self.atoms[i].x) + " " + 
                  str(self.atoms[i].y) + " " +
                  str(self.atoms[i].z) + " " +
                  str(self.atoms[i].name) )
        print("@<TRIPOS>BOND")
        for i in range(0,len(self.bonds)):
            b = self.bonds[i]
            print(str(i+1)+" "+str(b.atom1+1)+" "+str(b.atom2+1)+" "+str(b.order))


class XYZFile:
    def __init__(self):
        self.name = None;
        self.molecules = [];

    def ParseFile(self,file_name):
        file = None;
        try:
            file = open(file_name,"r")
        except IOError:
            sys.stderr.write("Can't open xyz file <"+str(file_name)+ "> \n")
            return False

        content = file.readlines()
        
        i = 0;
        while i < len(content):
            pars = content[i].strip().split()
            if(len(pars)==0):
                i = i+1
                continue;
            
            atom_count = None;
            try:
                atom_count = int(pars[0])
            except ValueError:
                sys.stderr.write("Error occured while reading line " +
                                 str(i+1) + " from file <" + str(file_name) + ">, an atom count is expected.\n")
                return False
                
            end_line_no = i+2+atom_count;

            if(atom_count == 0 or end_line_no > len(content)):
                sys.stderr.write("Error occured while reading the molecule starting from line " +
                                 str(i+1) + " from file <" + str(file_name) + ">, check the atom count.\n")
                return False

            
            m = Molecule();
            result = m.ParseXYZFileLines(content[i:end_line_no])
            if (result):
                self.molecules.append(m)
            else:
                sys.stderr.write("Error occured while reading the molecule starting from line " +
                                 str(i+1) + " from file <" + str(file_name) + ">\n")
            i = end_line_no

        if(len(self.molecules) == 0):
            sys.stderr.write("Error occured while reading the xyz file <" + str(file_name) +">, no"
                             "molecular structure was read in.")
            return False


        file.close();
        return True;


class BondRules:
    def __init__(self):
        self.rules = []

    def ParseFile(self,file_name):
        file = None;
        try:
            file = open(file_name,"r")
        except IOError:
            sys.stderr.write("Can't open bond rule file <"+ str(file_name)+">/n")
            return False

        lines = file.readlines();

        for i in range(0,len(lines)):
            l = lines[i].strip();
            if (len(l)==0):
                continue;
            rule = Bond();
            result = rule.Read(l);
            if (not result):
                sys.stderr.write("Error in parsing <"+str(file_name+"> at line "+str(i+1)+" :\n"))
                sys.stderr.write(lines[i]+"\n")
                sys.stderr.write("Format error.\n")
                return False;
            self.rules.append(rule);
        file.close();
        return True;

    def Show(self):
        for i in self.rules:
            i.Show();
    def CheckRules(self,a1,a2,dist):
        for i in reversed(self.rules):
            if ( i.Satisfy(a1, a2, dist)):
                return i.order
        return None;

    


def main(argc,argv):
    if argc < 2 or argc > 4:
        sys.stderr.write(
"""
            Usage : xyz2mol2.py xyzfile [bond_rule_file]
            The [bond_rule_file] is a txt files which defines bond rules, they can include multiple lines, each line is like below:
            AtomType1 AtomType2 MinBondLength MaxBondLength BondType
            For example in the following lines
            C O 1.2 1.6 1
            C O 1.1 1.2 ar
            The above two lines ask the program to judge any two C, O atoms that are within 1.2 Angstrom(A) but at least 1.1 A
            apart form a aromatic bond, any two C, O atoms that are at least 1.2 A apart but within 1.6 A from each other form
            a single bond. Rules are checked in reverse order, which means that a rule appears later in the file overrides rules
            that appeared eariler, if it controdicts with the earlier rules.
            Other specifications include:
            1. In each rule, the order of names of two atoms doesn't matter, meaning "C O 1.2 1.6 1" and "O C 1.2 1.6 1" are
               exactly the same.
            2. The 3rd parameter of this program, [bond_rule_file], is optional. There is a default bond rule file, named
               "xyz2mol2.default.bond.rules", comes together with the xyz2mol2.py program and read by the program each time
               it runs, to recognize some commonly encountered bonds in organic molecules. However a user defined [bond_rule_file]
               can be supplied if some bonds are not recognized or special rules apply. Remember that rules in [bond_rule_file]
               will override the default rules, if there are controdictions.
            3. The 5th segment of each line in bond_rule_file defines the bond type, it can be any string that conforms with the
               TRIPOS mol2 file format specification, such as 1, 2, 3, ar, am, etc. More information can be obtained from
               www.tripos.com.
""")
        exit()


    rules = BondRules()
    result = rules.ParseFile("./xyz2mol2.default.bond.rules")
    if ( not result):
        exit()
        
    if (argc == 3):
        result = rules.ParseFile(argv[2]);
        if ( not result ):
            exit()

#    rules.Show()
            
    xyzfile = XYZFile()
    xyzfile.ParseFile(argv[1])

    for i in xyzfile.molecules:
        i.FindBonds(rules)

    for i in xyzfile.molecules:
        i.ShowAsMol2()

        

main(len(sys.argv),sys.argv)