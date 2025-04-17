import pandas as pd

input_file = "challenge-2018-task-3-training_metadata_2025-03-16.csv"
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

# Combine the filters, if you want to filter the same key, write the key and value SEPARATELY on NEW key:value pair.
filters ={
        "diagnosis_1" : "Malignant"
        ,"benign_malignant": "malignant"
        ,"sex" : "female"
        }

def load_data_to_pd_dataframe(file_path):
    return pd.read_csv(file_path, low_memory=False)

def filter_data(df, filters):
    """Filter the DataFrame based on specified conditions.
    Args:
        df (pd.DataFrame): The original dataset.
        filters (dict): A dictionary where keys are column names and values are filtering criteria.
    """
    query_str = " & ".join([f"{col} == '{val}'" if isinstance(val, str) else f"{col} == {val}" for col, val in filters.items()])
    return df.query(query_str)

def records_count_with_applied_filters(df, filters):
    """Compute how many records match the given filters."""
    return len(filter_data(df, filters))

def save_filtered_data_to_csv(df, filters, output_file):
    filtered_df = filter_data(df, filters)
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")

df = load_data_to_pd_dataframe(input_file)
count = records_count_with_applied_filters(df, filters)
print(f"Number of records matching the filter: {count}")
save_filtered_data_to_csv(df, filters, output_file)


# Here I tried it again, if I dont have a mistake in my filter_data record, but it doesnt look like it.
# JUST TESTING purpose
# def number_of_filter_data_based_on_index_and_its_value(df, index, value):
#    print("Number of records is:", len(df[df[index] == value]))
# number_of_filter_data_based_on_index_and_its_value(df, "diagnosis_1", "Malignant")
