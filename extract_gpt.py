from codify import Codify
from config import groq_client, openai_client
import os
import openai
from langchain_openai import ChatOpenAI
from agent import Agent
from pydantic import BaseModel
from typing import List

# Set your OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Example clinical notes
clinical_notes = """
assessment as noted

neuro: A+O X 3, SOME EXPRESSIVE APHASIA, AMBULATES WELL, MAE=, STRONG, PERL, SPEACH CLEAR. C/O L  EYE/HEADACHE-GETTING PERCOCET WITH RELIEF, NO SEIZURE ACTIVITY NOTED, ON KEPRA AS ORDERED

CV:STABLE

RES:LS CLEAR, NO DISTRESS

GI:ON CL LIQ DIET TOL WELL.
GU: BRISK O/U

LABS: K-3.9, CON'T ON 125/H  40K/NS

PLAN: TO CT TODAY AND TRANSFER IF UNCHANGED NURSING NOTE:
ADMISSION NOTE:

28 year old male admitted on [**4-1**] after c/o of severe H/A while watching TV.  Cat scan head showed left temporal bleed.  Initially admitted to 5 [**Hospital Ward Name **] for observation then to Sicu.  MRI head and transfer to T-sicu for further monitoring.

PMH:  asthma (cold weather induced)
      depression ( 14 years age)

ALLERGIES: Sulfa ? reaction

PSH:  none

REVIEW OF SYSTEMS:

CV:  Hemodynamic status stable, hr= 55-60, SB to nsr, bp=108-122/70, + dp bil.

RESP:  Maintained on RA, sat=98%, resp. rate= 16, BS clear

GI:  NPO, started on clear liqs. 30cc/hr, npo after MN for angio. in am.

GU:  Voids into urinal

NEURO:  Upon admission c/o of h/a, left side head, scale 7, tylenol given with effect, h/a at 2, PERL, 6mm -7mm bil. A+O x 3, initially c/o of "visual shadows", denies n/v.  Muscle st. +5/+5 bil. upper and lower ext., - pronator drift.  DIlantin as per order.

SKIN:  Intact

ENDO:  Intact

ID:  Afebrile

SOCIAL:  Mother, grandmother and friends in to visit.  Pt. informed of need for angio. in am.  Family members informed of status by Neuro-[**Doctor First Name **].

PLAN: Continue to monitor neuro status, clear liqs. as per order, NPO after MN, monitor dilantin levels, monitor lytes, monitor BS.  Provide supprt to family. NSG NOTE
ROS:

    CV:ART LINE DC'D.HEMODYNAMICALLY STABLE.PULSE 50-55,NSBRADY,NO ECTOPI.

    RESP:LUNGS CLEAR,ON RA WITH SAT 98-100.

    NEURO:ALERT,ORIENTATED X 3.SOMETIMES NEEDS SIMPLE EXPLANATIONS OF REASON FOR MEDS,PROCEDURES.OCCASIONALLY NEEDS CUES/CHOICES WHEN ASKED WHERE ARE YOU,WILL CHOOSE HOSPITAL IF GIVEN THAT CHOICE WITH OTHER CHOICES.COOPERATIVE,STEADY ON FEET,FOLLOWS COMMANDS,USES URINAL.
CO HEADACHE L EYE AREA.GIVEN TYLENOL WITH MOD EFFECT.NOW HAS ORDER FOR
  NARCOTIC IF NEEDED

 GU :VOIDING IN SUFFICENT AMTS.ON IVF,NS WITH 40 KCL AT 125 HR.

    GI:TAKING CLEAR LIQUIDS WITH OUT NAUSEA.

    LYTES:AM K 3.2.RECEIVED 40 JCL PO AND IVF CHANGED TO NS +40 KCL.

     SOCIAL:DR [**First Name (STitle) **] SPOKE TO PTS FATHER ON PHONE AND MOTHER/PATIENT IN RM.

     PLAN:? TRANSFER TO FLOOR IF DR [**Last Name (STitle) 621**] OK.DOESN'T NEED STEPDOWN PER T-SICU ATTENDING.CALLED OUT TO FLOOR BUT NO BED YET.NS HO TRYING TO
GET SPEAK WITH DR [**Last Name (STitle) 621**] FOR OK. ROS:

Neuro: Alert oriented x's 3 w/ques. MAE x's 4 strong and = to command. PEARLLA. No seizure activity noted. On Kappra. Occasional use of inappropriate words ie: will state "This is the first time I have talked in four years"  When he really wanted to say days instead of years.

CV: RSR->SB w/o ectopy. VSS. Peripheral pulses palpable w/ease. Has right radial ABP line.

Resp: On room air, no resp distress = rise and fall of chest. Lung sounds clear.

GI: Taking clear liquids. Occasional episodes of frequent belching. Abd soft w/active bowel sounds.

GU: Voids per urinal in QS.

Labs: K 3.2 this AM. Awaiting repletion orders.

Social: Mother and grandmother in to visit at beginning of shift. Father calls and [**Name2 (NI) 4765**] via phone.

Plan: Monitor, comfort, mobilize. Nursing NOte:
REVIEW OF SYSTEMS:

CV:  Hemodynamic status stable, hr=57 SB to 81 nsr, no ectopy, + dp bil, ext. warm pink.  Right art. line placed.

RESP: Maintained on RA, sat= 99-100%, resp. rate=15, BS clear

GI:  c/o of nausea this am with belching, no further c/o throughout the day, tol. sips clear.  Abd. soft, hypoactive BS

GU:  Voids, 100-200/hr

SKIN: Intact, DSD right groin, no hematoma

NEURO:  Angiogram done, tol. procedure well, c/o of h/a prior to angio, rated as 7, tylenol given with minimal relief of h/a, also c/o nausea.  Continued with expressive aphasia this am.  Post angio describes h/a at top of head, rates as [**2-3**], right pupil 6mm, left 5mm, reactive to light.  Upper ext. +5/+5, lower ext. +5/+5.  Slight improvement of expressive aphasia.  Oriented to person, time, disoriented to place.  Angio site, right groin, DSD, no hematoma. No sz. activity noted.

ID:  Low grade temp at 100.1

ENDO:  No sl. scale

SOCIAL:  Mother and grandmother in to visit, spoke with Neuro. post angiogram.  Brother and friend in to visit.

PLAN:  Continue to monitor v/s, neuro. assessment, assess right groin site, monitor dilantin  levels, schedure meeting with Social worker in am, monitor lytes.  Provide support to family. nursing care note
Took over care for pt in MRI at 1015 am. Pt AA&Ox3 MAE FC. Speech clear. Pt monitored during MRI/A/V. VSS.  Pt to be transferred per nursing supervisor from SICU to T/SICU. Report given. Pt transported with RN to T/SICU at 1130am. SIGNIFICANT EVENTS:

CT SCAN FOR NERUO CHANGES
SEIZURE LIKE ACTIVITY


ROS:

NERUO: INITIALLY PRESENTED TO BE ALERT ORIENTED AND FOLLOWING COMMANDS. DILANTIN IV STARTED. BECOMES CHILLED AND HAVING DIFFICULTY SPEAKING, CONFUSED, USING INAPPROPRIATE WORDS, AND WORD SALAD. YET FOLLOWS COMMANDS. DILANTIN STOPPED, CHILLS GO AWAY AND BECOMES SLIGHTLY MORE APPROPRIATE. APPROX 30 CC OF DILANTIN REMAIN, INFUSION RESUMED AND CHILLS RETURN AT THE END OF THE INFUSION AND HAS SLIGHT RETURN OF SPEAKING ISSUES AS NOTED ABOVE BUT NOT TO THE EXTENT OF ORIGIONAL OCCURANCE. TAKEN FOR STAT CT => NO CHANGE IN BLEED. SLEEPS FOR SEVERAL HOURS AFTER RETURNING FROM CT THEN AWAKE AND HAVING SIMMILAR DIFFICULTIES W/SPEACH AS HE DID EARLIER ALONG W/COGGWHEEL LIKE MOVEMENTS OF RIGHT ARM,LIFTING ARM INTO AIR, HO AWARE AND ASSESSES PATIENT. NURSE RETURNS TO ROOM TO FIND PATIENT LYING ON RIGHT SIDE W/EYES DEVIATED TO THE RIGHT AND HAVING REPETIVE MOVEMENT OF EYEBROWS, AND IN FACE. UNABLE TO GET PATIENT TO STOP W/VERBAL QUES. HO AT SIDE AND ASSESSES PATIENT. PATIENT BECOMES SLIGHTLY RIDGED AND BEGINS KICKING LIKE MOVEMENTS OF LOWER EXTREMITIES, LEFT ARM NOT INVOLVED. THIS LASTS FOR SEVERAL MINUTES THEN STOPS. THEN NOTED CHEWING LIKE MOVEMENTS OF MOUTH. APHASIC AFTER THIS. ATIVAN .5 MG GIVEN PRIOR TO THIS AS PATIENT WAS RESTLESS AND HAD PULLED OUT A-LINE. 0.5 MG OF ATIVEN GIVEN AFTER THIS EVENT AS WELL. THEN SLEEPS AND AWAKEND FOR HOURLY NEURO CHECKS REMAINING APHASIC UNTIL ~ 0500 THEN BEGINNING TO SPEAK AGAIN BUT USING INAPROPRIATE WORDS I.E. WHEN ASKED WHAT IS NAME IS HE STATES "TWENTY-EIGHT THOUSAND NIMBLE BEE'S". HO AWARE.

CV: RSR W/O ECTOPY. VSS. P BOOTS ON FOR DVT PROPHYLAXIS.

RESP: LUNG SOUNDS CLEAR, NO RESP DISTRESS. O2 APPLIED AT TIME OF SEIZURE LIKE ACTIVITY. SATS 95% OR >

GI: ABD SOFT W/ACTIVE BS THOUGHOUT. NOTED TO BE BELCHING FREQUENTLY. PO PROTONIX PROPHYLACTILY. NPO AFTER MN FOR ANGIO THIS AM.

GU: VOIDS PER URINAL X'S 1 THIS SHIFT.

LABS: WNL

SOCIAL: FAMILY VISITED AT BEGINNING OF SHIFT. NO FURTHER CONTACT W/FAMILY.

PLAN: ANGIOGRAM THIS AM. CONTINUE TO MONITOR. Admission Date:  [**2146-4-1**]              Discharge Date:   [**2146-4-8**]

Date of Birth:  [**2118-1-25**]             Sex:   M

Service: NEUROSURGERY

Allergies:
Patient recorded as having No Known Allergies to Drugs

Attending:[**First Name3 (LF) 9223**]
Chief Complaint:
Headache

Major Surgical or Invasive Procedure:
Angiogram


History of Present Illness:
28yo male presents with sudden onset headache, was also confused
and unable to give coherent history. Did call grandmother
complaining of sudden headache, found sometime later by police
and brought to ED.

Past Medical History:
none

Social History:
+alcohol
+drugs (?types)

Family History:
aunt and grandmother died of cerebral aneurysms

Physical Exam:
97.1 80 120/86 18 98%
no apparent distress, no trauma, neck supple, lungs clear, heart
reg rate and rhythm, abdomen soft, extrem warm
Neuro: AAO to location, speech nonsensical at times, difficulty
completing commands, pupils 7mm and reactive, motor appears
intact, sensation intact to light touch, finger to nose done
well, DTR 2+ UE 1+ LE toes downgoing

Pertinent Results:
[**2146-4-1**] 04:15PM BLOOD WBC-14.6* RBC-5.21 Hgb-15.7 Hct-43.4
MCV-83 MCH-30.2 MCHC-36.2* RDW-12.6 Plt Ct-328
[**2146-4-1**] 04:15PM BLOOD Neuts-84.9* Lymphs-9.9* Monos-4.6 Eos-0.2
Baso-0.4
[**2146-4-1**] 04:15PM BLOOD Plt Ct-328
[**2146-4-1**] 04:15PM BLOOD PT-12.9 PTT-29.3 INR(PT)-1.0
[**2146-4-1**] 04:15PM BLOOD Glucose-120* UreaN-15 Creat-1.0 Na-141
K-3.4 Cl-103 HCO3-21* AnGap-20
[**2146-4-1**] 04:15PM BLOOD Calcium-9.9 Phos-1.0* Mg-1.9
[**2146-4-3**] 03:11AM BLOOD Phenyto-8.5*

Head CT: left temporal lobe bleed

Brief Hospital Course:
Admitted to ICU, loaded with Dilantin, tight blood pressure
control, close neurological moitoring q1hour.BRAIN
MRI:IMPRESSION: Left temporal hemorrhage with mild surrounding
edema. No definite intrinsic parenchymal abnormal enhancement
seen. A developmental venous anomaly is seen in this region.
Although this could be an incidental association, developmental
venous anomaly is often seen in association with cavernous
malformations. Subtle area of signal or chronic blood products
adjacent to the occipital [**Doctor Last Name 534**] of the left lateral ventricle
also indicates a previous area of hemorrhage, which could be
related to a cavernous malformation.

MRA OF THE HEAD:IMPRESSION: Normal MRA of the head.

MRV OF THE HEAD:IMPRESSION: Normal MRV of the head.
On [**2146-4-3**] he was found to have some increase in
expressive/receptive aphasia.  He had reaction to dilantin and
was switched to Keppra and weaned off Dilantin. He underwent
angiogram [**2146-4-3**] IMPRESSION: Selective diagnostic cerebral
arteriography of the bilateral carotid and vertebral arteries
was performed without evidence of aneurysm or vascular
malformation. There is a suggestion of a small venous angioma in
the left temporal region identified on delayed imaging during
selective injection of the left internal carotid artery.

His aphasia gradually improved. He continued to do well and was
discharged to home [**2146-4-8**].


Medications on Admission:
none

Discharge Medications:
1. Levetiracetam 500 mg Tablet Sig: One (1) Tablet PO BID (2
times a day).
Disp:*60 Tablet(s)* Refills:*2*
2. Oxycodone-Acetaminophen 5-325 mg Tablet Sig: 1-2 Tablets PO
Q4-6H (every 4 to 6 hours) as needed.
Disp:*40 Tablet(s)* Refills:*0*


Discharge Disposition:
Home

Discharge Diagnosis:
Cerebral hemorrhage


Discharge Condition:
neurologically stable

Discharge Instructions:
Call for any problems.

Followup Instructions:
Follow up with Dr. [**Last Name (STitle) 1132**] in 3 weeks with MRI with and without
contrast of head. Call [**Telephone/Fax (1) 3231**] for appt.


                             [**Name6 (MD) **] [**Last Name (NamePattern4) **] MD, [**MD Number(3) 9225**]

Completed by:[**2146-4-8**] [**2146-4-5**] 8:55 AM
 CT HEAD W/O CONTRAST                                            Clip # [**Clip Number (Radiology) 44528**]
 Reason: any changes
 Admitting Diagnosis: INTRACRANIAL BLEED
 ______________________________________________________________________________
 [**Hospital 2**] MEDICAL CONDITION:
  28 year old man with bleed
 REASON FOR THIS EXAMINATION:
  any changes
 No contraindications for IV contrast
 ______________________________________________________________________________
                                 FINAL REPORT
 INDICATION:  Intraparenchymal hemorrhage.

 TECHNIQUE:  Axial noncontrast CT imaging of the brain.  Comparison is made to
 a prior study from asthma [**2146-4-2**].

 FINDINGS:  There is an unchanged focus of intraparenchymal hemorrhage within
 the left temporal lobe.  New areas of hemorrhage are identified.  The
 ventricles and sulci are normal in size and symmetrical.  There is no evidence
 of subfalcine or uncal herniation.  There is no shift of normally midline
 structures.  The CT evidence of acute major vascular territorial infarction.

 Bone windows show clear paranasal sinuses with no evidence of fracture.

 IMPRESSION:  Unchanged focus of intraparenchymal hemorrhage within the left
 temporal lobe. [**2146-4-1**] 7:20 PM
 CT HEAD W/O CONTRAST                                            Clip # [**Clip Number (Radiology) 44522**]
 Reason: rule out aneurysm/bleed
 ______________________________________________________________________________
 [**Hospital 2**] MEDICAL CONDITION:
  28 year old man with acute onset of severe headache,confusion.  family history
  of cerebral aneurysm.
 REASON FOR THIS EXAMINATION:
  rule out aneurysm/bleed
 No contraindications for IV contrast
 ______________________________________________________________________________
 WET READ: MMBn FRI [**2146-4-1**] 10:22 PM
  3.1 x 2.5 cm intraparenchymal hemorrhage in lrft temporoparietal area,
 ______________________________________________________________________________
                                 FINAL REPORT *ABNORMAL!
 INDICATION:  28-year-old man with acute onset of severe headache and
 confusion.  Family history of cerebral aneurysm.  Evaluate.

 There is a 2.5 x 3.1-cm intraparenchymal hemorrhage in the left
 temporal and frontal lobes, at the level of the fourth ventricle.  This is
 causing mild mass effect, but there is no shift of normally midline
 structures.  No additional high density foci are identified.  The surrounding
 soft tissue and osseous structures are unremarkable.

 IMPRESSION:  3.1 x 2.5-cm intraparenchymal hemorrhage in left parietotemporal
 [**Doctor Last Name 218**], at the level of the fourth ventricle.  Mild mass effect, but no shift of
 normally midline structures.

 These findings were called to Dr. [**Known firstname 14288**] [**Last Name (NamePattern1) 44523**] at the time of observation. [**2146-4-3**] 1:09 PM
 CAROT/CEREB [**Hospital1 80**]                                                  Clip # [**Clip Number (Radiology) 44527**]
 Reason: bleed
 Admitting Diagnosis: INTRACRANIAL BLEED
  Contrast: OPTIRAY Amt: 290
 ********************************* CPT Codes ********************************
 * [**Numeric Identifier 82**] SEL CATH 3RD ORDER [**Last Name (un) 83**]         [**Numeric Identifier 84**] SEL CATH 2ND ORDER               *
 * -59 DISTINCT PROCEDURAL SERVICE       [**Numeric Identifier 84**] SEL CATH 2ND ORDER               *
 * -59 DISTINCT PROCEDURAL SERVICE       [**Numeric Identifier 85**] ADD'L 2ND/3RD ORDER              *
 * [**Numeric Identifier 85**] ADD'L 2ND/3RD ORDER             [**Numeric Identifier 86**] CAROTID/CEREBRAL BILAT           *
 * [**Numeric Identifier 3005**] EXT CAROTID UNILAT              [**Numeric Identifier 88**] VERT/CAROTID A-GRAM              *
 * [**Numeric Identifier 88**] VERT/CAROTID A-GRAM             -59 DISTINCT PROCEDURAL SERVICE        *
 * C1769 GUID WIRES INFU/PERF            C1894 INT/SHTH NOT/GUID EP NON-LASER   *
 * NON-IONIC 200 CC SUPPLY                                                      *
 ****************************************************************************
 ______________________________________________________________________________
 [**Hospital 2**] MEDICAL CONDITION:
  28 year old man with
 REASON FOR THIS EXAMINATION:
  bleed
 ______________________________________________________________________________
                                 FINAL REPORT
 HISTORY:  28-year-old male with recent intracerebral hemorrhage.  Evaluate for
 aneurysm or vascular malformation.

 PROCEDURE/FINDINGS:  The procedure was performed by Dr. [**First Name4 (NamePattern1) 1439**] [**Last Name (NamePattern1) 1440**] and Dr.
 [**First Name8 (NamePattern2) 4903**] [**Last Name (NamePattern1) 91**].  Dr. [**Last Name (STitle) 91**], the staff radiologist, was present and supervising
 throughout.  Informed consent was obtained from the patient after explaining
 the risks, indications, and alternative management.  Risks explained included
 stroke, loss of vision and speech (temporary or permanent) with possible
 treatment with stent and coils if needed.

 The patient was brought to the interventional neuroradiology theater and
 placed on the biplane table in the supine position.  Both groins were prepped
 and draped in the standard sterile fashion.  Access to the right common
 femoral artery was obtained in the retrograde fashion using a 19 gauge single
 wall puncture needle, under local anesthesia using 1% Lidocaine mixed with
 sodium bicarbonate and with aseptic precautions.  Through the access needle, a
 0.035 [**Last Name (un) 94**] wire was introduced and, after a  small incision was made using
 a 11 blade scalpel, the access needle was removed.  Over the [**Last Name (un) 94**] wire, a 4
 French angiographic sheath was placed.  After the inner dilator was removed,
 the sheath was assembled to a continuous saline infusion (mixed with Heparin
 500 units in 500 cc of saline).  Through this sheath, a 4 French Berenstein
 catheter was introduced and connected to a continuous saline infusion (mixed
 with Heparin 1000 units in 1000 ML of saline).

 The following vessels were selectively catheterized using a 0.035 angled
 Glidewire in combination with the Berenstein catheter and selective
 arteriograms were obtained in the AP and lateral projections:

 right internal carotid artery,
 right vertebral artery,
 left internal carotid artery,
                                                             (Over)

 [**2146-4-3**] 1:09 PM
 CAROT/CEREB [**Hospital1 80**]                                                  Clip # [**Clip Number (Radiology) 44527**]
 Reason: bleed
 Admitting Diagnosis: INTRACRANIAL BLEED
  Contrast: OPTIRAY Amt: 290
 ______________________________________________________________________________
                                 FINAL REPORT
 (Cont)
 left external carotid artery,
 and left vertebral artery.

 After the films were reviewed, including the 3-D reconstructions, no evidence
 of aneurysm or AV malformation was identified within the aforementioned
 vessels.  There is a suggestion of a small venous angioma in the left temporal
 region which was identified on delayed imaging during selective injection of
 the left internal carotid artery.  Following arteriography, the 4 French
 Berenstein catheter was removed followed by the 4 French angiographic sheath.
 Manual pressure was held at the arterial puncture site until hemostasis was
 achieved.  A dry sterile dressing was applied.  The patient was transferred to
 the surgical intensive care unit following the procedure in stable condition.

 COMPLICATIONS:  None.

 MEDICATIONS:  1% Lidocaine.

 CONTRAST:  290 cc of 50% Optiray 240.

 IMPRESSION:  Selective diagnostic cerebral arteriography of the bilateral
 carotid and vertebral arteries was performed without evidence of aneurysm or
 vascular malformation.  There is a suggestion of a small venous angioma in the
 left temporal region identified on delayed imaging during selective injection
 of the left internal carotid artery.

 These findings were discussed with the referring clinical service at the time
 of the procedure. [**2146-4-2**] 9:42 AM
 MR HEAD W & W/O CONTRAST; MRA BRAIN W/O CONTRAST                Clip # [**Clip Number (Radiology) 44525**]
 MR CONTRAST GADOLIN
 Reason: Please do mri with gado/mra/mrv
 Admitting Diagnosis: INTRACRANIAL BLEED
  Contrast: MAGNEVIST Amt: 15
 ______________________________________________________________________________
 [**Hospital 2**] MEDICAL CONDITION:
  28 year old man with acute confusional state, now with left temporal bleed.
 REASON FOR THIS EXAMINATION:
  Please do mri with and without gado/, also mra/mrv - we are looking for a
  bleeding cause (aneurysm, avm, cav mal)
 ______________________________________________________________________________
                                 FINAL REPORT
 EXAM:  MRI of the brain.

 CLINICAL INFORMATION:  The patient with left-sided temporal hemorrhage, for
 further evaluation.

 TECHNIQUE:  T1 sagittal and axial, and FLAIR T2 susceptibility, and diffusion
 axial images of the brain were obtained before gadolinium.  T1 axial and
 coronal images of the brain were obtained following gadolinium.  3D time-of-
 flight MRA of the circle of [**Location (un) **] was acquired.  Additionally, 2D time-of-
 flight MRV of the head was obtained.

 BRAIN MRI:

 Again, blood products are identified in the left temporal lobe in the temporal
 polar region extending to the subinsular region on the left side.  The blood
 products are slightly hyperintense on T1-weighted images with low signal on
 susceptibility-weighted images.  The findings indicate hyperacute/acute
 hematoma in the left temporal region.  Following gadolinium administration, no
 distinct intrinsic enhancement is seen within the region of hematoma. However,
 a linear venous structure is seen in the adjacent brain and also in the left
 medial temporal lobes, which is suspicious for a developmental venous anomaly
 as suggested by the CTA examination of [**2146-4-1**].  Additionally, on
 susceptibility-weighted images, an area of low signal is seen adjacent to the
 occipital [**Doctor Last Name 218**] of the left lateral ventricle indicating chronic blood
 products.  No definite enhancement is seen in this region.  There is no
 midline shift or hydrocephalus seen.

 IMPRESSION:  Left temporal hemorrhage with mild surrounding edema.  No
 definite intrinsic parenchymal abnormal enhancement seen.  A developmental
 venous anomaly is seen in this region.  Although this could be an incidental
 association, developmental venous anomaly is often seen in association with
 cavernous malformations.  Subtle area of signal or chronic blood products
 adjacent to the occipital [**Doctor Last Name 218**] of the left lateral ventricle also indicates a
 previous area of hemorrhage, which could be related to a cavernous
 malformation.

 MRA OF THE HEAD:

 The 3D time-of-flight MRA of the circle of [**Location (un) **] demonstrates normal flow
                                                             (Over)

 [**2146-4-2**] 9:42 AM
 MR HEAD W & W/O CONTRAST; MRA BRAIN W/O CONTRAST                Clip # [**Clip Number (Radiology) 44525**]
 MR CONTRAST GADOLIN
 Reason: Please do mri with gado/mra/mrv
 Admitting Diagnosis: INTRACRANIAL BLEED
  Contrast: MAGNEVIST Amt: 15
 ______________________________________________________________________________
                                 FINAL REPORT
 (Cont)
 signal within the arteries of anterior and posterior circulation.

 IMPRESSION:  Normal MRA of the head.

 MRV OF THE HEAD:

 The MRV of the head demonstrates normal flow signal within the superior
 sagittal and deep venous system.

 IMPRESSION:  Normal MRV of the head. [**2146-4-2**] 8:36 PM
 CT HEAD W/O CONTRAST                                            Clip # [**Clip Number (Radiology) 44526**]
 Reason: worsening aphasia
 Admitting Diagnosis: INTRACRANIAL BLEED
 ______________________________________________________________________________
 [**Hospital 2**] MEDICAL CONDITION:
  28 year old man with
 REASON FOR THIS EXAMINATION:
  worsening aphasia
 No contraindications for IV contrast
 ______________________________________________________________________________
                                 FINAL REPORT
 HISTORY:  Intraparenchymal hemorrhage, now with worsening aphasia.

 COMPARISON:  [**2146-4-1**].

 TECHNIQUE:  Noncontrast head CT.

 FINDINGS:  The left frontotemporal hemorrhage appears unchanged in extent.
 There is no evidence of additional new hemorrhage since the previous study.
 Acute blood products are also again noted in the occipital [**Doctor Last Name 218**] of the left
 lateral ventricle.  There is stable mild edema surrounding the
 intraparenchymal hemorrhage.  There is stable obliteration of the left
 temporal [**Doctor Last Name 218**] and mild compression of the left frontal [**Doctor Last Name 218**]. The third
 ventricle remains slitlike.  Overall, the extent of mass effect is unchanged.
 There is no CT evidence of an acute major territorial ischemic infarction. The
 visualized osseous structures appear unremarkable. Visualized paranasal
 sinuses and mastoid air cells are normally aerated.

 IMPRESSION:  Stable left frontotemporal intraparenchymal hemorrhage with mild
 mass effect.
                                                                       DFDkq [**2146-4-1**] 9:45 PM
 CTA HEAD W&W/O C & RECONS; CT 150CC NONIONIC CONTRAST           Clip # [**Clip Number (Radiology) 44524**]
 Reason: any evidence of aneurysms?
  Contrast: OPTIRAY Amt: 122
 ______________________________________________________________________________
 [**Hospital 2**] MEDICAL CONDITION:
  28 year old man with left temporal lobe bleed; multiple family members with
  cerebral aneurysm (sudden death)
 REASON FOR THIS EXAMINATION:
  any evidence of aneurysms?
 No contraindications for IV contrast
 ______________________________________________________________________________
                                 FINAL REPORT  (REVISED)
 INDICATION:  Left temporal intraparenchymal hemorrhage.  Strong family history
 of cerebral aneurysm.  Evaluate for aneurysm.

 TECHNIQUE:  Multidetector helical axial imaging of the head was performed
 following the administration of 150 cc of Optiray contrast.  Initial coronal
 and sagittal reformatting was performed, followed by 3D work station images.

 FINDINGS:  The large intraparenchymal hemorrhage within the left temporal lobe
 is again seen, relatively unchanged from 2 hours previous.  Examination of the
 intracranial arteries demonstrates normal patency of the major tributaries of
 the circle of [**Location (un) **], including the anterior, middle and posterior cerebral
 arteries.  There is tortuousity of what may be an early branching right
 anterior temporal artery division of the right middle cerebral artery, vs. a
 contiguous vein.  A small aneurysm may be difficult to detect in this locale
 without the use of conventional catheter angiography.  However, this
 questionable abnormality is on the side opposite of the hemorrhage.

 Posterior to the left temporal hemorrhage, there is a curvilinear area, about
 1.3cm in length, which may represent either the major venous tributary of a
 developmental venous anomaly, a component of an arteriovenous malformation, or
 simply a displaced vein.  No definite pathological vascularity can be
 discerned within the hemorrhage itself.

 IMPRESSION:

 1)  Acute left temporal intraparenchymal hemorrhage.

 2)  Curvilinear region of contrast enhancement posterior to the left temporal
 intraparenchymal hemorrhage.  The finding may represent the major venous
 tributary of a developmental venous anomaly, a component of an arteriovenous
 malformation, or a displaced vein. No definite aneurysm is identified on the
 side of the hemorrhage, but there is a questionable tiny aneurysm v. a
 tortuous right middle cerebral artery branch (see above report).

 The findings were immediately discussed with Dr. [**Last Name (STitle) 490**], the emergency room
 attending physician, [**Name10 (NameIs) **] the neurosurgical resident on-call at pager [**Numeric Identifier 4949**].
 Conventional angiography was recommended.  Dr. [**Last Name (STitle) 831**] was also consulted re:
 the findings of this case by me (JK).

                                                             (Over)

 [**2146-4-1**] 9:45 PM
 CTA HEAD W&W/O C & RECONS; CT 150CC NONIONIC CONTRAST           Clip # [**Clip Number (Radiology) 44524**]
 Reason: any evidence of aneurysms?
  Contrast: OPTIRAY Amt: 122
 ______________________________________________________________________________
                                 FINAL REPORT  (REVISED)
"""

class ExtractModel(BaseModel):
    diagnosis: List[str]
    evidence: List[str]

def extract_k_key_points(query: str, k:int):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                      temperature=0,
                        max_tokens=300,
                        timeout=None,
                        max_retries=2)
    
    system_prompt = f"""
    You are a medical expert that can extract key points from a clinical query.
    You will be given clinical notes written by different people for the same patient.
    You must extract the key diagonsis from the notes based on ICD-9 paradigm and give your supporting evidence.
    There might be typos and format errors in clinical notes.
    The response should be a list of diagnosis and a list of evidence. Length of both lists should be equal to {k}.
    Example:
    ["diagnosis1", "diagnosis2", "diagnosis3", "diagnosis4", "diagnosis5"...]
    ["evidence1", "evidence2", "evidence3", "evidence4", "evidence5"...]
    Do not return any other information.

    query: {query}
    """
    # context = f"""
    # query: {query}
    # """
    
    # Make the LangChain API call
    response = llm.predict(system_prompt)
    decisions = response.strip('/n').split('\n')
    diagonosis = decisions[0].strip('[').strip(']').split(',')
    evidence = decisions[1].strip('[').strip(']').split(',')
    return diagonosis,evidence


# extracted = extract_k_key_points(clinical_notes, 5)
# print(extracted)
# descriptions = ['"Left temporal intraparenchymal hemorrhage"', ' "Expressive aphasia"', ' "Developmental venous anomaly"', ' "Cavernous malformation"', ' "Aneurysm"']
# evidence = ['"CT scan head showed left temporal bleed"', ' "MRI head showed left temporal hemorrhage with mild surrounding edema"', ' "Angiogram showed suggestion of a small venous angioma in the left temporal region"', ' "MRI brain showed acute left temporal intraparenchymal hemorrhage with mild mass effect"', ' "CTA head showed acute left temporal intraparenchymal hemorrhage and curvilinear region of contrast enhancement posterior to the hemorrhage"']
 
# codify = Codify()
# for i in range(len(evidence)):
#     result = codify.get_ranked_top_k_icd_codes_with_evidence(3, descriptions[i],evidence[i])
#     print(f'rag result:{result}')

