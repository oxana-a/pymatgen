# coding: utf-8

__author__ = "mhsiron"
__version__ = "0.1"
__maintainer__ = "Martin Siron"
__email__ = "mhsiron@lbl.gov"
__status__ = "Beta"
__date__ = "11/02/19"

import os
import subprocess
import shutil
import numpy as np
from pymatgen.core import Element

from monty.dev import requires
from monty.os.path import which
from monty.io import zopen
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar

DDEC6EXE = which("Chargemol") or which("Chargemol.exe")


class DDEC6Analysis:
    """
    DDEC6 Analysis
    """
    @requires(which("Chargemol") or which("Chargemol.exe"),
              "DDEC6Analysis requires the executable Chargemol to be in the "
              "system path. Please download the library at "
              "http://sourceforge.com/ddec6")
    def __init__(self, chgcar_filename="CHGCAR.gz",
                 potcar_filename="POTCAR.gz",
                 aeccar_filenames=None, run=True, ad_dir=None,
                 custom_command=None, gzipped=True):
        """
        Initializes the DDEC6 Analysis. Either runs DDEC6 executable,
         or simply analyzes the files created by DDEC6
        :param chgcar_filename: (str) path to CHGCAR file
        :param potcar_filename: (str) path to CHGCAR file
        :param aeccar_filenames: (list) paths to AECCARs in order
            (AECCAR0, AECCAR1, AECCAR2)
        :param run: (bool) whether or not to run DDEC6
        :param ad_dir: custom directory for atomic_densities. Otherwise
            gets value from "DDEC6_ATOMIC_DENSITIES_DIR" environment variable
        :param custom_command: (str) custom command to run with DDEC6
            executable
        :param gzipped: whether or not the files are gzipped,
            they will be unzipped if so
        """
        if aeccar_filenames is None:
            aeccar_filenames = ["AECCAR0.gz", "AECCAR1.gz", "AECCAR2.gz"]

        # Set properties
        self.species_count = 0
        self.atomic_charges = []
        self.species = []
        self.coords = []
        self.bond_orders = {}
        self.potcar = None

        self.chgcar = Chgcar.from_file(chgcar_filename)
        self.potcar = Potcar.from_file(potcar_filename)

        # Set paths
        self._chgcarpath = os.path.abspath(chgcar_filename)
        self._potcarpath = os.path.abspath(potcar_filename)
        self._aeccarpaths = [os.path.abspath(aeccar) for aeccar in
                             aeccar_filenames]

        if run:
            self._execute_ddec6(ad_dir, custom_command,
                                gzipped)
        else:
            self._from_data_dir()

    def _execute_ddec6(self, ad_dir, custom_command, gzipped):
        """
        Internal command to execute DDEC6, data analysis runs after
        :param ad_dir: custom directory for atomic_densities. Otherwise
            gets value from "DDEC6_ATOMIC_DENSITIES_DIR" environment variable
        :param custom_command: (str) custom command to run with DDEC6
            executable
        :param gzipped: whether or not the files are gzipped,
            they will be unzipped if so
        """

        # Unzip if needed:
        if gzipped:
            with zopen(self._chgcarpath, 'rt') as f_in:
                with open("CHGCAR", "wt") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            with zopen(self._potcarpath, 'rt') as f_in:
                with open("POTCAR", "wt") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            for n, aeccarpath in enumerate(self._aeccarpaths):
                print(aeccarpath)
                with zopen(aeccarpath, 'rt') as f_in:
                    with open("AECCAR" + str(n), "wt") as f_out:
                        shutil.copyfileobj(f_in, f_out)

        # write job_script file:
        self._write_jobscript_for_ddec6(ad_dir=ad_dir, net_charge=0.0,
                                        periodicity=[True, True, True])

        # command arguments for DDEC6
        args = [DDEC6EXE]
        if custom_command is not None:
            args = [DDEC6EXE, custom_command]
        rs = subprocess.Popen(args,
                              stdout=subprocess.PIPE,
                              stdin=subprocess.PIPE, close_fds=True)
        stdout, stderr = rs.communicate()
        if rs.returncode != 0:
            raise RuntimeError("DDEC6 exited with return code %d. "
                               "Please check your DDEC6 installation."
                               % rs.returncode)

        self._from_data_dir()

    def _from_data_dir(self):
        """
        Internal command to load XYZ files and process post DDEC6 executable
        """
        # load files created
        # Atomic Charges
        # atomic_charges_lines = []
        with open("DDEC6_even_tempered_net_atomic_charges.xyz",
                  "r") as f:
            atomic_charges_lines = [line.strip() for line in f]
        # Atomic Charges
        # bond_orders_lines = []
        with open("DDEC6_even_tempered_bond_orders.xyz",
                  "r") as f:
            bond_orders_lines = [line.strip() for line in f]
        # Overlap Populations
        # overlap_populations_lines = []
        with open("overlap_populations.xyz",
                  "r") as f:
            overlap_populations_lines = [line.strip() for line in f]

        self.raw_data = {
            "atomic_charges": atomic_charges_lines,
            "bond_orders": bond_orders_lines,
            "overlap_populations": overlap_populations_lines,
        }

        self._get_charge_info()
        self._get_bond_order_info()

    def get_charge_transfer(self, index=None, element=None, ):
        """
        Get charge for a select index or average of charge for a Element
        :param index: (int) specie index
        :param element: (Element) Pymatgen element
        :return: atomic charge, or avg of atomic charge for an element
        """

        if index is not None:
            print(index)
            print(self.atomic_charges[index])
            return self.atomic_charges[index]
        elif element is not None:
            charges = []
            for c_element, c_charges in zip(self.species, self.atomic_charges):
                if c_element == element:
                    charges.append(c_charges)
            return np.average(charges)
        else:
            return self.atomic_charges

    def get_charge(self, atom_index):
        """
           #     Calculates difference between the valence charge of an atomic specie
           #     and its DDEC6 calculated charge
           #     :param index: (int) specie index
           #     :return: charge transfer
        """
        potcar_indices = []
        for i, v in enumerate(self.chgcar.poscar.natoms):
            potcar_indices += [i] * v
        nelect = self.potcar[potcar_indices[atom_index]].nelectrons
        print(nelect)
        return nelect+self.get_charge_transfer(index=atom_index)

    def get_bond_order(self, index_from, index_to):
        """
        Returns bond order index of species connected to certain specie
        :param index_from: (int) specie originating
        :param index_to: (int) bonded to this specie
        :return: bond order index
        """
        if not self.bond_orders[index_from].get("all_bonds", False):
            print(
                "DDEC6 did not find specie {} to be connected to any specie".format(
                    index_from))
            return None
        elif not self.bond_orders[index_from].get("all_bonds", {}).get(
                index_to, False):
            print(
                "DDEC6 did not find specie {} to be connected to specie {}".format(
                    index_from, index_to))
            return None
        else:
            return self.bond_orders[index_from].get("all_bonds", {}).get(
                index_to, {}).get("bond_order", None)

    def _get_info_from_xyz(self, raw_data_key, info_array):
        """
        Internal command to process XYZ files
        :param raw_data_key: key in raw_data collection
        :param info_array: key to analyze in XYZ header
        :return:
        """
        species_count = self.species_count

        # in all files
        all_info = {"coords": np.zeros([species_count, 3], dtype="float64"),
                    "species": []}

        # all_info["coords"] = np.zeros([species_count, 3], dtype="float64")
        # all_info["species"] = []

        for element in info_array:
            all_info[element] = np.zeros(species_count, dtype="float64")

        for line_number, line_content in enumerate(all_info["coords"]):
            line = self.raw_data[raw_data_key][line_number + 2].split()
            all_info["species"].append(Element(line[0]))
            all_info["coords"][line_number][:] = line[1:4]
            for num, element in enumerate(info_array):
                all_info[element][line_number] = line[4 + num]

        return all_info

    def _write_jobscript_for_ddec6(self, ad_dir=None, net_charge=0.0,
                                   periodicity=None):
        """
        Writes job_script.txt for DDEC6 execution
        :param ad_dir: (str) atomic densities reference directory
        :param net_charge: (float) net charge of structure, 0.0 is default
        :param periodicity: (list of booleans) periodicity among a,b, and c
        """
        if periodicity is None:
            periodicity = [True, True, True]
        self.net_charge = net_charge
        self.periodicity = periodicity

        # Net Charge
        lines = ["<net charge>", net_charge, "</net charge>", ""]

        # lines.append("<net charge>")
        # lines.append(net_charge)
        # lines.append("</net charge>")
        # lines.append("")

        # Periodicity
        per_a = ".true." if periodicity[0] else ".false."
        per_b = ".true." if periodicity[1] else ".false."
        per_c = ".true." if periodicity[2] else ".false."
        lines.append("<periodicity along A, B, and C vectors>")
        lines.append(per_a)
        lines.append(per_b)
        lines.append(per_c)
        lines.append("</periodicity along A, B, and C vectors>")
        lines.append("")

        # atomic_densities dir
        ad_dir = ad_dir or os.environ.get("DDEC6_ATOMIC_DENSITIES_DIR", ".")
        lines.append("<atomic densities directory complete path>")
        lines.append(ad_dir)
        lines.append("</atomic densities directory complete path>")
        lines.append("")
        lines.append("<charge type>")
        lines.append("DDEC6")
        lines.append("</charge type>")

        with open('job_control.txt', 'w') as fh:
            for line in lines:
                fh.write('%s\n' % line)

    def _get_charge_info(self):
        """
        Internal command to process atomic charges, species, and coordinates
        """
        self.species_count = int(self.raw_data["atomic_charges"][0])
        self.atomic_charges = []
        self.species = []
        self.coords = []
        for line in self.raw_data["atomic_charges"][2:2 + self.species_count]:
            self.atomic_charges.append(float(line.split()[-1]))
            self.species.append(Element(line.split()[0]))
            self.coords.append(line.split()[1:-2])

    def _get_bond_order_info(self):
        """
        Internal command to process bond order information
        """
        # Get meta data
        # bo_xyz = self._get_info_from_xyz("bond_orders", ["bond_orders"])

        # Get where relevant info for each atom starts
        bond_order_info = {}
        for line_number, line_content in enumerate(
                self.raw_data["bond_orders"]):
            if "Printing" in line_content:
                species_index = line_content.split()[5]
                bond_order_info[int(species_index) - 1] = {
                    "start": line_number}

        # combine all relevant info
        for atom, content in bond_order_info.items():
            try:
                for bo_line in self.raw_data["bond_orders"][
                               bond_order_info[atom]["start"] + 2:
                               bond_order_info[atom + 1]["start"] - 4]:

                    # Find total bond order
                    total_bo = float(bo_line.split()[-1])

                    # Find current info
                    c_bonded_to = int(bo_line.split()[12]) - 1
                    c_bonded_to_element = Element(bo_line.split()[14])
                    c_bonded_to_bo = float(bo_line.split()[20])
                    c_direction = (
                        int(bo_line.split()[4][:-1]),
                        int(bo_line.split()[5][:-1]),
                        int(bo_line.split()[6][:-1]))

                    c_bo_by_bond = {c_bonded_to: {
                        "element": c_bonded_to_element,
                        "bond_order": c_bonded_to_bo,
                        "direction": c_direction}
                    }
                    bo_by_bond = c_bo_by_bond
                    if bond_order_info[atom].get("all_bonds"):
                        bo_by_bond = bond_order_info[atom].get("all_bonds")
                    bo_by_bond.update(c_bo_by_bond)

                    # update bondings, total_bo
                    bond_order_info[atom].update({
                        "all_bonds": bo_by_bond,
                        "total_bo": total_bo
                    })

            except:
                for bo_line in self.raw_data["bond_orders"][
                               bond_order_info[atom]["start"] + 2:-3]:
                    # Find total bond order
                    # Find total bond order
                    total_bo = float(bo_line.split()[-1])

                    # Find current info
                    c_bonded_to = int(bo_line.split()[12]) - 1
                    c_bonded_to_element = Element(bo_line.split()[14])
                    c_bonded_to_bo = float(bo_line.split()[20])
                    c_direction = (
                        int(bo_line.split()[4][:-1]),
                        int(bo_line.split()[5][:-1]),
                        int(bo_line.split()[6][:-1]))

                    c_bo_by_bond = {c_bonded_to: {
                        "element": c_bonded_to_element,
                        "bond_order": c_bonded_to_bo,
                        "direction": c_direction}
                    }
                    bo_by_bond = c_bo_by_bond
                    if bond_order_info[atom].get("all_bonds"):
                        bo_by_bond = bond_order_info[atom].get("all_bonds")
                    bo_by_bond.update(c_bo_by_bond)

                    # update bondings, total_bo
                    bond_order_info[atom].update({
                        "all_bonds": bo_by_bond,
                        "total_bo": total_bo
                    })
        self.bond_orders = bond_order_info

    def update_structure(self):
        """
        Takes CHGCAR's structure object and updates it with atomic charges,
        and bond orders
        :return: updated structure
        """
        structure = self.chgcar.structure
        structure.add_site_property("atomic_charges_ddec6",
                                    self.atomic_charges)
        structure.add_site_property("bond_orders_ddec6", self.bond_orders)
        return structure
