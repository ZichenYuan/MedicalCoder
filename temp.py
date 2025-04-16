# from codify import Codify
# from utils import get_codes

# dia = ["Pneumonia, organism unspecified", "Congestive heart failure, unspecified", "Mechanical ventilation, unspecified", "Heart failure, unspecified", "Pleural effusion, unspecified", "Ectopic beats", "Edema, unspecified", "Fever, unspecified", "Respiratory distress, unspecified"]
# evi = ["Radiological findings of right upper lobe pneumonia and new left lung opacities", "History of severely depressed left ventricular systolic function and regional wall motion abnormalities", "Presence of bilateral pleural effusions", "Use of intubation and mechanical ventilation", "Use of central venous catheter and various medications", "Ongoing investigations for cause of fever spikes", "Complex medical condition requiring ongoing monitoring and management", "Clinical notes indicating discordant breathing and hemodynamic monitoring", "Patient's history of MRSA pneumonia"]

# # query = ['pneumonia, specifically MRSA pneumonia, with radiographs revealing consolidations in the right upper lobe and left lower lobe, and worsening opacities in the left lingula and left lower lobe.',
# # 'moderate left and trace right pleural effusions, indicating other diseases of the pleura.',
# # 'A Swan-Ganz catheter has been placed in the right pulmonary artery, indicating a disease of pulmonary circulation.',
# # 'severely depressed left ventricular systolic function, resting regional wall motion abnormalities, mildly dilated aortic root and ascending aorta, mildly thickened aortic valve leaflets, trace aortic regurgitation, and mild to moderate mitral regurgitation, indicating diseases of the heart.',
# # ' a softly distended abdomen and has not had a bowel movement, indicating other diseases of the digestive system.',
# # 'edema from the thighs down to the ankles and is weeping serous fluid from old IV/arterial line sites, indicating other disorders of the skin and subcutaneous tissue.',
# # 'febrile, producing thick tan sputum, and requires frequent suctioning, indicating other general symptoms.',
# # 'Tendotracheal tube, right jugular central venous line, left subclavian central venous catheter, and is allergic to heparin, indicating complications of surgical and medical care.',
# # 'sedated with fentanyl and ativan, indicating delirium, dementia, and amnestic and other cognitive disorders.']
# query = ["Pneumonia due to Methicillin Resistant Staphylococcus Aureus", "Congestive heart failure, unspecified", "Fever, unspecified", "Gastrointestinal hemorrhage, unspecified", "Complications of internal prosthetic device, implant, and graft due to catheter", "Cellulitis and abscess of unspecified sites", "Altered mental status, unspecified", "Acute renal failure, unspecified", "Allergy, unspecified"]
# codify = Codify()
# for i in range(len(query)):
#     # result1 = codify.get_ranked_top_k_icd_codes(3, dia[i])
#     # result2 = codify.get_ranked_top_k_icd_codes(3, evi[i])
#     result3 = codify.get_ranked_top_k_icd_codes(3, query[i])
#     # code = get_codes(result1)
#     # code2 = get_codes(result2)
#     # print(f'code:{code}')
#     # print(f'code2:{code2}')
#     code3 = get_codes(result3)
#     print(f'code3:{code3}')

import pickle

csv_file_path = "mimic3_full.csv"
# Try to load existing samples from file
sample_file = "random_samples.pkl"

with open(sample_file, 'rb') as f:
    descriptions, codes_list, document_metadatas, ids = pickle.load(f)
    print("Loaded existing random samples")

clinical_notes = descriptions[4]

print(clinical_notes)
print(codes_list[4])