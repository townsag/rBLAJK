import time

start = time.time()

import numpy as np
from nupack import *
import pandas as pd

cox_src = "TTAAAACAGCCTGTGGGTTGTACCCACCCACAGGGCCCACTGGGCGCTAGCACTCTGATTCTACGGAATCCTTGTGCGCCTGTTTTATGTCCCTTCCCCCAATCAGTAACTTAGAAGCATTGCACCTCTTTCGACCGTTAGCAGGCGTGGCGCACCAGCCATGTCTTGGTCAAGCACTTCTGTTTCCCCGGACCGAGTATCAATAGACTGCTCACGCGGTTGAGGGAGAAAACGTCCGTTACCCGGCTAACTACTTCGAGAAGCCTAGTAGCACCATGAAAGTTGCAGAGTGTTTCGCTCAGCACTTCCCCCGTGTAGATCAGGTCGATGAGTCACTGCGATCCCCACGGGCGACCGTGGCAGTGGCTGCGTTGGCGGCCTGCCTGTGGGGTAACCCACAGGACGCTCTAATATGGACATGGTGCAAAGAGTCTATTGAGCTAGTTAGTAGTCCTCCGGCCCCTGAATGCGGCTAATCCTAACTGCGGAGCACATACCCTCGACCCAGGGGGCAGTGTGTCGTAACGGGCAACTCTGCAGCGGAACCGACTACTTTGGGTGTCCGTGTTTCCTTTTATTCTTATACTGGCTGCTTATGGTGACAATTGAAAGATTGTTACCATATAGCTATTGGATTGGCCATCCGGTGTGCAACAGAGCTATTATTTACCTATTTGTTGGGTATATACCACTCACATCCAGAAAAACCCTCGACACACTAGTATACATTCTTTACTTGAATTCTAGAAAATGGGGTCACAAGTCTCAACCCAACGATCGGGTTCCCACGAAAATTCGAACTCAGCATCAGAAGGA".replace("T","U")
binding_site_length = 36
my_model = Model(material='rna')

def compute_single_strandedness(bpps: np.array):
    assert bpps.ndim == 2, "the base pair probability array is not two dimensional"
    assert bpps.shape[0] == bpps.shape[1], "the base pair probabilities array is not square"
    return np.trace(bpps)/bpps.shape[0]

def design_series_A_sensor(target_sequence: str, binding_site_length, my_model, trials=3):
    assert binding_site_length == 36, "binding site length must be 36 for this version of series A sensors"
    assert my_model.material == 'RNA', "this design series is for RNA"
    LacZ_head = "ATGACCATGATTACGGATTCA"

    leader = Domain("GGG", name="leader")
    a = Domain(target_sequence[0:3], name="a")
    b = Domain(target_sequence[3:6], name="b")
    c = Domain(target_sequence[6:36], name="c")
    x = Domain("N3", name="x")
    g = Domain("N3", name="g")
    e = Domain("SN4S", name="e")
    f = Domain("N3", name="f")
    rbs = Domain("AGAGGAGA", name="rbs")
    start = Domain("AUG", name="start")
    y = Domain("N3", name="y")

    lacZ = Domain(LacZ_head, name="lacz")

    switch = TargetStrand([leader, ~c, ~b, ~a, ~x, g, ~e, f, rbs, e, start, x, a, b, y, ~a, ~x, lacZ], name="switch")
    rna = TargetStrand([a, b, c], name="rna")

    switch_complex = TargetComplex([switch], '.33(9.3(6.11)6.3)9.30', name='switch_complex')
    switch_rna_complex = TargetComplex([rna, switch], "(36+.3)36.6(6.11)6.3(6.6)6.21")

    t1 = TargetTube(on_targets={switch_rna_complex: 1e-7, switch_complex: 1e-8}, name='t1', off_targets=SetSpec(max_size=3))
    my_tubes = [t1]

    options = DesignOptions(f_stop=0.02)

    my_design = tube_design(tubes=my_tubes, hard_constraints=[],soft_constraints=[], defect_weights=None,
                            options=options,model=my_model)

    my_results = my_design.run(trials=trials)
    # either sort by weighted ensemble defect or rna+switch complex defect
    my_sorted_results = sorted(my_results, key=lambda x: x.defects.weighted_ensemble_defect)
    best_result = my_sorted_results[0]
    return t1, best_result

# pairs is a function from the nupack library, it returns a matrix of shape (len_source, len_source)
probability_matrix_array = pairs(strands=[cox_src], model=my_model).to_array()

L_mrna_list = []
for i in range(0, len(cox_src) - binding_site_length + 1):
    # calculate the "single-strandedness for each 30 nucleotide window
    # ex: strand of 31 nucleotides has 31 - 30 + 1 = 2 windows
    # each entry in L_array is the single-strandedness score of the
    # 30nt long section after and including the base at that index i
    temp_sum = 0
    for ii in range(i, i + binding_site_length):
        # each entry in the base pair probability matrix[i, j] represents the porbability that the base at position
        # i is paired to the base at position j. If i == j this represent the probability that the base is unpaired
        # we sum the probabilities that each base is unpaired
        temp_sum += probability_matrix_array[ii,ii]
    # compute the average probability that a base in the 30 nucleotide window will be unpaired
    L_mrna_list.append(temp_sum/binding_site_length)
# creates an array of shape (src_len - BSL + 1, 1) where len is the length of the 
L_mrna_array = np.array(L_mrna_list)


columns = ['l_mrna','l_toehold','d_sensor_design','normalized_defect_switch','normalized_defect_mrna_switch','switch_mfe','mrna_switch_complex_mfe','binding_site_sequence','switch_mfe_sequence','switch_2ndary_structure','mrna_switch_complex_2ndary_structure']
series_A_df = pd.DataFrame(columns=columns)
series_A_df.index.name = "binding_site_start_index"


for potential_binding_site_index, l_mrna in enumerate(L_mrna_array[:2]):
    temp_pbs_sequence = cox_src[potential_binding_site_index:potential_binding_site_index+binding_site_length]
    print("designing for: " + temp_pbs_sequence)
    print("time elapsed: ", time.time() - start)
    assert len(temp_pbs_sequence) == binding_site_length, "length of temp pbs sequence is not the same as binding site length"

    # design a toehold for this potential binding site
    temp_target_tube, temp_best_result = design_series_A_sensor(temp_pbs_sequence, binding_site_length, my_model,trials=3)
    # Tube object based on TargetTube with designed sequences, has correct concentrations and any changes to secondary struture
    t1_designed = temp_best_result.to_analysis(temp_target_tube)
    # Calculate the MFE structure for each on-target complex in the design ensemble
    tube_results = complex_analysis(t1_designed, compute=['mfe','pairs'], model=my_model)

    print("time elapsed: ", time.time() - start)
    # switch bpps matrix
    temp_switch_bpps = tube_results["switch_complex"].pairs.to_array()
    toehold_region_length = 33
    l_toehold = compute_single_strandedness(temp_switch_bpps[:toehold_region_length, :toehold_region_length])

    # ensemble defect for design
    # todo: find out what exactly is meant by normalized ensemble defect
    d_sensor = temp_best_result.defects.ensemble_defect
    nd_switch = temp_best_result.defects.complexes.loc[temp_best_result.defects.complexes['complex_name'] == 'switch_complex', "normalized"].values[0]
    nd_rna_switch = temp_best_result.defects.complexes.loc[temp_best_result.defects.complexes['complex_name'] == '(rna+switch)', "normalized"].values[0]

    print("should be the same:")
    print(temp_pbs_sequence)
    print(temp_best_result.to_analysis["rna"])
    assert temp_pbs_sequence == str(temp_best_result.to_analysis["rna"]), "potential binding site is not the same in the designed sequence"

    switch_sequence = temp_best_result.to_analysis["switch"]
    switch_secondary_structure = str(tube_results["switch_complex"].mfe[0].structure)
    mrna_switch_complex_secondary_structure = str(tube_results["(rna+switch)"].mfe[0].structure)

    switch_mfe = tube_results["switch_complex"].free_energy
    rna_switch_complex_mfe = tube_results["(rna+switch)"].free_energy

    temp_row = {
        'l_mrna': l_mrna,
        'l_toehold': l_toehold,
        'd_sensor_design': d_sensor,
	    'normalized_defect_switch': nd_switch,
        'normalized_defect_mrna_switch': nd_rna_switch,
        'switch_mfe': switch_mfe,
        'mrna_switch_complex_mfe': rna_switch_complex_mfe,
        'binding_site_sequence': temp_pbs_sequence,
        'switch_mfe_sequence': switch_sequence,
        'switch_2ndary_structure': switch_secondary_structure,
        'mrna_switch_complex_2ndary_structure': mrna_switch_complex_secondary_structure
    }

    series_A_df.loc[potential_binding_site_index] = temp_row

end = time.time()
print("elapsed seconds: ", end-start)

series_A_df.to_csv("test_series_a.csv")