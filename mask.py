"""
whole script for MVPA analysis 12-09-2024
"""

import pandas as pd
import os
from nilearn.image import index_img
from nilearn.decoding import Decoder
from nilearn.image import math_img, binarize_img, concat_imgs, load_img, threshold_img

import nibabel as nib

#set path
participant = [f"C{str(i).zfill(3)}" for i in range(3,119)]
for participant in participant:
    self_other_path=os.path.join ("/host/corin/tank/self_ref_trait_words/preprocess/", participant) #memory data path
    #test if self_other_path exists, if not, skip the participant
    if not os.path.exists(self_other_path):
        continue

    mask=r"/host/cicero/local_raid/data/jhu/yh/Code/Mask/SREfromNeurosynth/raw/self_referential_association-test_z_FDR_0.01.nii.gz"#original mask path
    behavior=r"/host/corin/tank/self_ref_trait_words/stimuli/Children_Memory/Presentation_Files"#behavioral data path/label
    
    output_path = os.path.join("/host/corin/tank/Qin/mvpa_output", participant) #output path
    #create output folder if not exist
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    #load & concat training set
    total_condition = pd.DataFrame()
    total_label = pd.DataFrame()
    total_img = []

    for i in range(4):
        self_other_file ='w'+ participant + f'_mem_run{i+1}_me.nii'
        self_other_data = os.path.join(self_other_path, self_other_file)
        self_other_data = load_img(self_other_data)
        file_label = os.path.join(behavior, f'run{i+1}.txt')

        df = pd.read_csv(file_label, sep='\t', header=None)
        df["run"] = i+1
        df = df.iloc[:, :2] 
        conditions = df.iloc[:,1]
        condition_mask = conditions.isin([1,2]) 
        condition_df = df[condition_mask]
        time_points = condition_df.iloc[:,0]
        extracted_img = index_img(self_other_data, time_points)
        condition_extracted_img = condition_df.iloc[:,1]
        condition_extracted_img = pd.DataFrame(condition_extracted_img,index=None)
        total_condition= pd.concat([total_condition,condition_extracted_img])
        total_img.append(extracted_img)

    total_img = concat_imgs(total_img)
    total_label["run"]=pd.Series([1]*40+[2]*40+[3]*40+[4]*40) 
    total_label["condition"]=total_condition.values

    #define training set & label
    X_IMG = total_img
    y = total_label["condition"]

    #train the mask
    self_other_mask_bi=binarize_img(mask)  #binarize the mask

    decoder = Decoder(
            estimator="svc_l2",
            mask=self_other_mask_bi,
            standardize="zscore_sample",
            screening_percentile=5,
            scoring="accuracy",
        )

    decoder.fit(X_IMG , y)     #avova svm

    # split the weight image into positive and negative parts; weight_img1=hit, weight_img4=cr, weight_img5=rest
    weight_img1 = decoder.coef_img_[1]
    positive_img1= threshold_img(weight_img1, threshold=0, two_sided=False)
    negitive_img1 = math_img("img1 - img2", img1=weight_img1, img2=positive_img1)
    weight_img2 = decoder.coef_img_[2]
    positive_img2= threshold_img(weight_img2, threshold=0, two_sided=False)
    negitive_img2 = math_img("img1 - img2", img1=weight_img2, img2=positive_img2)
    #combine the positive and negative parts of the weight images (sum of absolute value of positive and negative parts for each feature)
    positive_whole=math_img("img1+img2-img3-img4",
                            img1=positive_img1,img2=positive_img2,img3=negitive_img1,img4=negitive_img2)
    #sort the features based on the absolute value of the weight sum
    positive_whole_data = positive_whole.get_fdata()
    sorted_data = np.sort(positive_whole_data.flatten())[::-1]
    #extract the top 500 features
    threshold = sorted_data[500]
    positive_whole_data[positive_whole_data < threshold] = 0
    positive_whole_data[positive_whole_data >= threshold] = 1
    positive_whole_nii = nib.Nifti1Image(positive_whole_data, positive_whole.affine, positive_whole.header)
   
    weight_img1_path = os.path.join(output_path, 'weight_img1.nii.gz')
    weight_img1.to_filename(weight_img1_path)
    weight_img2_path = os.path.join(output_path, 'weight_img2.nii.gz')
    weight_img2.to_filename(weight_img2_path)
    whole_mask_path = os.path.join(output_path, 'whole_mask.nii.gz')
    positive_whole_nii.to_filename(whole_mask_path)


