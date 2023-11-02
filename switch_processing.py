import numpy as np
from nupack import *

import draw_rna
from draw_rna.ipynb_draw import draw_struct


"""
    Sum along the diagonal of the base pair probability matrix
    The base pair probability matrix is a two dimensional square matrix with each element (i,j) in the square
    matrix representing the probability that the base at position i is bound to the base at position j
    The elements on the diagonal represents the probability that the base at that position is unbound
    Summing along the diagonal and then dividing by the length of the sequence returns the average probability
    that a base is single stranded: each base is unbound

    This is a helper function for the evaluate sequence function
"""
def find_single_strandedness(base_pair_probability_matrix):
    assert base_pair_probability_matrix.ndim == 2, "matrix must be two dimensional"
    assert base_pair_probability_matrix.shape[0] == base_pair_probability_matrix.shape[1], "matrix must be square"

    sum_of_diagonal = np.sum(np.diag(base_pair_probability_matrix))
    length_of_sequence = base_pair_probability_matrix.shape[0]
    return sum_of_diagonal/length_of_sequence


# ==========depricated========== #
# def compare_complex_concentrations(seq_complex, tube_result):
#     #==========Check that none of the multi strand complexes have higher concentration than single strand==========#
#     seq_complex_concentration = tube_result["t1"].complex_concentrations[seq_complex]
#     print(f"sequence complex concaentration: {seq_complex_concentration}")
#     for complex_obj, concentration in tube_result["t1"].complex_concentrations.items():
#         print(f"testing {complex_obj.name} with concentratin {concentration}")
#         if complex_obj.name != "(seq)" and concentration > seq_complex_concentration:
#             print(f"warning: complex {complex_obj.name} has higher concentration that seq conpmex")
#             print(f"{complex_obj.name} has concentration: {concentration}")
#             print(f"(seq) has concentration: {seq_complex_concentration}")


"""
    Seq_Complex: This is the nupack Complex object representing the sequence being evaluated not bound to any
                    sequence other than itself
    Tube Result: this is the nupack Tube_Result object returned by the tube analysis function called on the
                    test tube object containing just one sequence
    This is a helper function for the evaluate sequence function. It returns the ratio of sequences in a hypothetical
    test tube that are not bound to any other sequence. For example, 70% of the seq could be free and the other 30%
    could be bound to itself in a complex: (seq+seq) (I apologize if my explanation is lackluster, my biology is not great)
"""
def find_ratio_unbound_strands(seq_complex, tube_result):
    seq_complex_concentration = tube_result["t1"].complex_concentrations[seq_complex]
    total_concentration = 0
    for complex_obj, concentration in tube_result["t1"].complex_concentrations.items():
        total_concentration += concentration
    return seq_complex_concentration/total_concentration


"""
    sequence: this is the string of bases representing the sequence to be evaluated
    desired_structure: this is the dot-paren representation of the desired secondary structure for the sequence being
                        evaluated
    unbound_region: this is a tuple of integers stating one region that is meant to be unbound in the sequence that is
                        being evaluated. The boundaries are inclusive and they describe the zero-indexed position of
                        the first and last base in the unbound region
                        ex: "AAAAAACGCGCGCGC" if the first six bases are meant to be unbound the input would be: (0, 5)

    This function returns a dictionary of metrics about each sequence. It uses the nupack analysis package to simulate
    a test tube with the sequence in it as well as the nupack utilities package
"""
def evaluate_sequence(sequence: str, desired_structure: str, unbound_region: tuple[int,int]):
    my_model = Model(material="RNA")
    strand_seq = Strand(sequence=sequence, name="seq")
    seq_complex = Complex([strand_seq], name="(seq)")
    t1 = Tube(
        strands={strand_seq: 1e-6}, 
        complexes=SetSpec(max_size=3, include=[seq_complex]), 
        name='t1')
    tube_results = tube_analysis(tubes=[t1], model=my_model, compute=['pfunc', 'pairs', 'mfe'])

    #=====Find Ratio of unbound strands=====#
    ratio_unbound_strands = find_ratio_unbound_strands(seq_complex=seq_complex, tube_result=tube_results)

    #=====Find Single-Strandedness of unbound Region=====#
    seq_base_pair_probability_matrix = tube_results[seq_complex].pairs.to_array()
    toehold_region_sub_matrix = \
        seq_base_pair_probability_matrix[unbound_region[0]:unbound_region[1] + 1, unbound_region[0]:unbound_region[1] + 1]
    l_toehold_region = find_single_strandedness(toehold_region_sub_matrix)

    #=====Find Normalized Ensemble Defect of Toehold Sequence=====#
    normalized_ensemble_defect = defect(strands=sequence, structure=desired_structure, model=my_model)

    return {
        "ratio_unbound_strands": ratio_unbound_strands,
        "l_toehold_region": l_toehold_region,
        "sequence_normalized_ensemble_defect": normalized_ensemble_defect
    }


"""
    target: this input is a string representing the target strand sequence of bases
    switch: this input is a string representing the toehold switch sequence of bases
    desired_secondary_structure: this input is a string representing the secondary structure of the (target+switch) complex
                                    out of habit I have put the target before the switch in the complex

    Lenght(secondary structure string) = 1 + len(target) + len(switch) because the secondary structure string has a "+"
    character connecting the two strands together as well as the information for each base in the target and switch strands

    This function returns a dictionary holding the normalized complex ensemble defect for the target switch complex
    This is calculated using the base pair probability matrix under the hood
"""
def evaluate_target_switch_complex(target: str, switch: str, desired_secondary_structure: str):
    assert len(target) + len(switch) + 1 == len(desired_secondary_structure), \
    "lenght of target strand and switch strand doesn't match up with the length of the desired secondary structure"
    normalized_complex_ensemble_defect = defect(strands=[target, switch], structure=desired_secondary_structure, model=Model(material="RNA"))
    return {
        "normalized_complex_ensemble_defect": normalized_complex_ensemble_defect
    }


import matplotlib.pyplot as plt
from draw_rna import draw
def draw_sequence(sequence: str, secondary_structure: str, colors: str, file_name: str):
    draw.draw_rna(sequence=sequence, secstruct=secondary_structure, color_list=colors)
    plt.savefig(file_name)
    plt.close()

def get_mfe_secondary_structure(sequence):
    if isinstance(sequence, str):
        temp = [sequence]
    else:
        temp = sequence
    return str(mfe(strands=temp, model=Model(material="RNA"))[0].structure)