import pandas as pd

input_file = "challenge-2024-training_metadata_2025-03-16.csv"
output_file = "filtered_data.csv"

# Define the filters "key: value" pairs where key is the column name and value is the filtering criteria.
### KNOWN FILTERS ###
# age_approx : 
#   positive INT increments of 5
# anatom_site_general :
#  'lower extremity' 'head/neck' 'posterior torso' 'anterior torso' 'upper extremity',
# benign_malignant :
#  'benign' 'malignant' 'indeterminate',
# clin_size_long_diam_mm : 
#   positive FLOAT,
# diagnosis : 'lichenoid keratosis' 'nevus' 'squamous cell carcinoma' 'basal cell carcinoma'
#   'actinic keratosis' 'seborrheic keratosis' 'atypical melanocytic proliferation' 'melanoma'
#   'pigmented benign keratosis' 'solar lentigo' 'dermatofibroma' 'AIMP'
#   'other' 'angiofibroma or fibrous papule' 'scar' 'lentigo NOS'
#   'melanoma metastasis' 'verruca' 'angioma' 'adnexal tumor' 'acrochordon,
# diagnosis_1 : 'Benign' 'Malignant' 'Indeterminate',
# diagnosis_2 : 'Benign epidermal proliferations' 'Benign melanocytic proliferations'
#   'Malignant epidermal proliferations'
#   'Malignant adnexal epithelial proliferations - Follicular'
#   'Indeterminate epidermal proliferations'
#   'Indeterminate melanocytic proliferations'
#   'Malignant melanocytic proliferations (Melanoma)'
#   'Benign soft tissue proliferations - Fibro-histiocytic' 'Cysts'
#   'Flat melanotic pigmentations - not melanocytic nevus'
#   'Inflammatory or infectious diseases'
#   'Benign soft tissue proliferations - Vascular'
#   'Benign adnexal epithelial proliferations - Apocrine or Eccrine'
#   'Benign adnexal epithelial proliferations - Follicular',
# diagnosis_3 :
#   'Lichen planus like keratosis' 'Nevus' 'Squamous cell carcinoma in situ'
#   'Basal cell carcinoma' 'Squamous cell carcinoma, Invasive'
#   'Solar or actinic keratosis' 'Seborrheic keratosis'
#   'Atypical melanocytic neoplasm' 'Melanoma in situ' 'Melanoma Invasive'
#   'Melanoma, NOS' 'Pigmented benign keratosis' 'Solar lentigo'
#   'Dermatofibroma' 'Atypical intraepithelial melanocytic proliferation'
#   'Trichilemmal or isthmic-catagen or pilar cyst' 'Angiofibroma' 'Scar'
#   'Lentigo NOS' 'Melanoma metastasis' 'Verruca'
#   'Squamous cell carcinoma, NOS' 'Hemangioma' 'Hidradenoma'
#   'Fibroepithelial polyp',
# diagnosis_4 : ...
# diagnosis_5 :
#   'Blue nevus, Cellular',
# diagnosis_confirm_type :
#   'single contributor clinical assessment' 'histopathology',
# image_type :
#   'TBP tile: close-up'
# sex : 
#   'male' 'female'

def unique_values_summary(df, column):
    """Get the number of unique values and list them for a given column."""
    unique_values = df[column].dropna().unique()
    print(f"Number of unique values in '{column}': {len(unique_values)}")
    print(f"Unique values: {unique_values}")
    return unique_values

def load_data(file_path):
    return pd.read_csv(file_path, low_memory=False)

df = load_data(input_file)
unique_values_summary(df, "image_type")